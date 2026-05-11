"""
Generation Studio — stage-by-stage character image pipeline.

Architecture
------------
  TOP SECTION  — character prefill helper only.
    Picking a char and clicking "Prefill All Stages" pushes that character
    into every stage's char picker.  Stages remain fully independent.

  EACH STAGE   — fully independent versioning.
    Own char / major / minor picker + "＋ Major" button.
    major = new design direction (user clicks "＋ Major" per stage)
    minor = auto-incremented when re-queuing an errored run (same intent,
            new attempt).

  DOWNSTREAM SOURCE PICKERS
    SD T-Pose, TRELLIS, Rig each have:
      Source stage  — which upstream stage to read image from
      Source version — specific done version to use (dropdown, default=latest)
    The URL is resolved at queue time from the selected version.

Schema
------
  Collections:
    manual_gen_stage_runs   — all stages except sd_tpose
    manual_gen_tpose_runs   — sd_tpose stage only (data isolation)
  Each document = one stage × one major.minor version run.

Pipeline
--------
  Stage 0: Flux      — text → character concept image
  Stage 1: Normalize — CPU resize to 512×512 (skipped if already correct size)
  Stage 2: SD T-Pose — SD1.5 + IP-Adapter (Flux identity) + OpenPose/Canny ControlNet
  Stage 3: TRELLIS   — 2D image → 3D mesh (GLB)
  Stage 4: Auto-Rig  — Skeleton rigging (CPU)
"""

import json
import os
import sys
import time
import uuid
from io import BytesIO
from typing import Optional

import gradio as gr

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_WORKER_DIR   = os.path.join(_PROJECT_ROOT, "worker")
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from lib.manual_gen_schema import (
    get_db,
    create_character, list_characters,
    create_run, get_run_for, ensure_run, auto_retry_run,
    list_chars, list_stage_majors, list_stage_minors,
    next_stage_major,
    version_str, save_run_params, mark_queued,
    get_latest_done_image_url, get_latest_done_run, get_run_any,
    STAGE_NAMES, COLLECTION,
)

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI = (os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")
             or "mongodb://kartik:Kartikg421@localhost:27017/?authSource=admin")
MONGO_DB  = os.getenv("MONGO_DB") or os.getenv("MONGODB_DB_NAME") or "World_builder"

def _resolve_redis():
    broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_BROKER_URL")
    if broker:
        from urllib.parse import urlparse
        p = urlparse(broker)
        return (p.hostname or "localhost"), (p.port or 6379), (p.password or None)
    return (os.getenv("REDIS_HOST", "localhost"),
            int(os.getenv("REDIS_PORT", 6379)),
            os.getenv("REDIS_PASSWORD") or None)

REDIS_HOST, REDIS_PORT, REDIS_PASSWORD = _resolve_redis()
S3_BUCKET   = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET", "sparkassets-us")
S3_REGION   = os.getenv("AWS_REGION", "us-east-1")
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
REDIS_QUEUE = "manual_gen_tasks"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    return get_db(MONGO_URI, MONGO_DB)

def _push_task(payload: dict, queue: str = REDIS_QUEUE) -> str:
    import redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                    db=0, decode_responses=True)
    payload.setdefault("task_id", str(uuid.uuid4()))
    payload.setdefault("timestamp", time.time())
    r.rpush(queue, json.dumps(payload))
    return payload["task_id"]

def _list_chars() -> list[str]:
    """Read character list from the registry + any existing stage runs."""
    try:
        return list_characters(_db()) or []
    except Exception:
        return []

def _list_majors(char: str, stage: str) -> list[int]:
    if not char:
        return [1]
    try:
        return list_stage_majors(_db(), char, stage) or [1]
    except Exception:
        return [1]

def _list_minors(char: str, stage: str, major) -> list[int]:
    if not char:
        return [0]
    try:
        return list_stage_minors(_db(), char, stage, int(major)) or [0]
    except Exception:
        return [0]

def _resolve_run(char: str, stage: str, major, minor) -> tuple[Optional[str], str]:
    """Return (run_id, display_info). Never creates a run."""
    if not char:
        return None, "No character selected."
    try:
        major, minor = int(major), int(minor)
        doc = get_run_for(_db(), char, stage, major, minor)
        if doc:
            sid = doc["_id"]
            st  = doc.get("status", "idle")
            return sid, f"{sid[:8]}…  v{major}.{minor}  [{st}]"
        return None, f"No run — {char}/{stage} v{major}.{minor}"
    except Exception as exc:
        return None, f"ERROR: {exc}"

def _get_run_doc(char: str, stage: str, major, minor) -> dict:
    """Load the run document; returns {} if not found."""
    if not char:
        return {}
    try:
        return get_run_for(_db(), char, stage, int(major), int(minor)) or {}
    except Exception:
        return {}

def _refresh_run(run_id: Optional[str]) -> tuple[str, str]:
    """Returns (status_text, image_url). Searches both collections."""
    if not run_id:
        return "idle", ""
    try:
        doc = get_run_any(_db(), run_id)
        if not doc:
            return "idle", ""
        st  = doc.get("status", "idle")
        url = doc.get("image_url") or ""
        if st == "done":    st = "✅ done"
        elif st == "error": st = f"❌ {doc.get('error', 'error')}"
        return st, url
    except Exception as exc:
        return f"❌ {exc}", ""

def _url_to_img(url: str, h: int = 400) -> str:
    if not url:
        return ("<div style='color:#999;padding:12px;text-align:center;"
                "border:1px dashed #ccc;border-radius:6px;font-size:13px;'>"
                "No image yet</div>")
    sep  = "&" if "?" in url else "?"
    bust = f"{url}{sep}t={int(time.time()*1000)}"
    return (f"<div style='text-align:center;'><img src='{bust}' "
            f"style='max-height:{h}px;max-width:100%;border-radius:6px;"
            f"object-fit:contain;'/></div>")

def _tok(text: str, is_sd: bool) -> str:
    w = len(text.split()) if text.strip() else 0
    if not is_sd:
        return f"Words: {w}  |  Flux T5 — no token limit"
    c = int(w * 1.3)
    return f"~{c}/77 CLIP tokens  {'⚠️ TRIM' if c > 77 else '✓'}"


# ── Collection routing (mirrors manual_gen_schema._coll_for_stage) ────────────

def _coll_for_stage(stage: str) -> str:
    """Return the correct MongoDB collection name for a stage."""
    return COLLECTION


# ── Source version helpers ─────────────────────────────────────────────────────

def _list_done_versions(char: str, stage: str) -> list[str]:
    """
    Return distinct done version strings for (char, stage), most recent first.
    e.g. ["2.1", "2.0", "1.0"]
    Automatically routes to the correct collection based on stage.
    """
    if not char or not stage:
        return []
    try:
        coll = _coll_for_stage(stage)
        docs = list(_db()[coll].find(
            {"char_label": char, "stage": stage, "status": "done"},
            {"major": 1, "minor": 1},
        ).sort("created_at", -1))
        seen, result = set(), []
        for d in docs:
            v = f"{d.get('major', 1)}.{d.get('minor', 0)}"
            if v not in seen:
                seen.add(v)
                result.append(v)
        return result
    except Exception:
        return []

def _get_src_url_for_ver(char: str, stage: str, ver: str) -> str:
    """Get image_url for a specific (char, stage, 'major.minor') version."""
    if not all([char, stage, ver]):
        return ""
    try:
        parts = ver.split(".", 1)
        major, minor = int(parts[0]), int(parts[1] if len(parts) > 1 else 0)
        doc = get_run_for(_db(), char, stage, major, minor)
        return (doc or {}).get("image_url") or ""
    except Exception:
        return ""

def _refresh_src_picker(char: str, src_stage: str):
    """
    Refresh source version dropdown for (char, src_stage).
    Returns (ver_dropdown_update, src_url, info_str).
    Defaults to the latest done version.
    """
    versions = _list_done_versions(char, src_stage)
    if versions:
        latest = versions[0]
        url    = _get_src_url_for_ver(char, src_stage, latest)
        return gr.update(choices=versions, value=latest), url, f"✓ {src_stage} v{latest}"
    return gr.update(choices=[], value=None), "", f"No done '{src_stage}' runs yet"

def _get_view_urls_for_ver(char: str, stage: str, ver: str) -> dict:
    """Return {front, side, back} URLs for (char, stage, 'major.minor')."""
    if not all([char, stage, ver]):
        return {"front": "", "side": "", "back": ""}
    try:
        from lib.manual_gen_schema import get_view_urls
        parts = ver.split(".", 1)
        major, minor = int(parts[0]), int(parts[1] if len(parts) > 1 else 0)
        return get_view_urls(_db(), char, stage, major, minor)
    except Exception:
        return {"front": "", "side": "", "back": ""}

def _view_availability_info(char: str, stage: str, ver: str) -> str:
    """Return human-readable availability string for all 3 views."""
    if not ver:
        return "No version selected"
    urls = _get_view_urls_for_ver(char, stage, ver)
    parts = []
    for view, label in [("front", "Front"), ("side", "Side"), ("back", "Back")]:
        parts.append(f"{label} {'✓' if urls[view] else '✗'}")
    return f"  |  ".join(parts)

def _on_src_ver(char: str, src_stage: str, ver: str):
    """User selected a specific source version. Returns (src_url, info_str)."""
    if not ver:
        return "", "No version selected"
    url = _get_src_url_for_ver(char, src_stage, ver)
    if url:
        return url, f"✓ {src_stage} v{ver}"
    return "", f"⚠️ {src_stage} v{ver} has no image"


# ══════════════════════════════════════════════════════════════════════════════
#  QUEUE FUNCTIONS  (one per stage)
#
#  All queue functions return (run_id, minor_update, info_update, status_text).
#  When the run had status="error", minor is auto-incremented and the
#  minor_update tells Gradio to refresh the minor picker.
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_run(char, stage, major, minor, prompt, neg, params):
    """
    Find or create the correct run to queue.
    Returns (run_id, new_minor_or_None, minor_update, info_update, error_str_or_None).
    """
    if not char:
        return None, None, gr.update(), gr.update(), "Pick a character first."
    major, minor = int(major), int(minor)
    db  = _db()
    run = get_run_for(db, char, stage, major, minor)

    if run:
        st = run.get("status", "idle")
        if st in ("queued", "running"):
            return run["_id"], None, gr.update(), gr.update(), f"⚠️ {stage} is already {st}."
        if st == "done":
            return run["_id"], None, gr.update(), gr.update(), \
                "⚠️ Already done — click ＋ Major to start a new design iteration."
        if st == "error":
            # Auto-create next minor for retry
            sid, new_n = auto_retry_run(db, char, stage, major, prompt, neg, params)
            minors = list_stage_minors(db, char, stage, major)
            info   = f"{sid[:8]}…  v{major}.{new_n}  [idle]"
            return sid, new_n, gr.update(choices=minors, value=new_n), info, None
        # status == "idle"
        sid = run["_id"]
    else:
        sid = create_run(db, char, stage, major, minor, prompt, neg, params)

    _, info = _resolve_run(char, stage, major, minor)
    return sid, None, gr.update(), info, None


def _q_flux(char, major, minor, prompt, neg, w, h, steps, guidance):
    """Queue the Flux stage generation."""
    stage  = "flux"
    params = {"width": int(w), "height": int(h),
               "steps": int(steps), "guidance_scale": float(guidance)}
    if not prompt.strip():
        return None, gr.update(), gr.update(), "⚠️ Prompt is empty."

    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, prompt, neg, params)
    if err:
        return sid, minor_upd, gr.update(), err

    db = _db()
    save_run_params(db, sid, prompt, neg, params, stage=stage)
    # Queue front view (primary — sets status to queued via mark_queued)
    tid_front = _push_task({"type": "flux", "session_id": sid, "stage": stage,
                             "char_label": char, "prompt": prompt, "negative": neg,
                             "params": params, "view": "front"})
    mark_queued(db, sid, stage=stage, task_id=tid_front)
    
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid_front[:8]}…"


def _q_normalize(char, major, minor, w, h, src_stage, src_ver):
    stage  = "normalize"
    params = {"resize_w": int(w), "resize_h": int(h)}

    src_url = (_get_src_url_for_ver(char, src_stage, src_ver) if src_ver
               else get_latest_done_image_url(_db(), char, src_stage) or "")
    if not src_url:
        return None, gr.update(), gr.update(), "error", _url_to_img(""), \
            f"No done image in '{src_stage}'. Run that stage first."

    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, "", "", params)
    if err:
        return sid, minor_upd, gr.update(), err, _url_to_img(""), ""

    try:
        import boto3
        from PIL import Image as PILImage
        import urllib.request

        with urllib.request.urlopen(src_url) as r:
            img = PILImage.open(BytesIO(r.read())).convert("RGB")

        tw, th = int(w), int(h)
        s3_key = None
        if img.size == (tw, th):
            # Already correct size — skip resize and re-upload, reuse source URL
            url = src_url
        else:
            img    = img.resize((tw, th), PILImage.LANCZOS)
            s3_key = f"manual_gen/{sid}/normalize_{tw}x{th}.png"
            buf    = BytesIO()
            img.save(buf, "PNG")
            buf.seek(0)
            _creds = {}
            if os.getenv("AWS_ACCESS_KEY_ID"):
                _creds = {"aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID"),
                          "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                          "aws_session_token":     os.getenv("AWS_SESSION_TOKEN")}
            boto3.client("s3", region_name=S3_REGION, **_creds).upload_fileobj(
                buf, S3_BUCKET, s3_key, ExtraArgs={"ContentType": "image/png"})
            url = f"{S3_BASE_URL}/{s3_key}"

        from lib.manual_gen_schema import update_run
        db = _db()
        update_run(db, sid, {"status": "done", "image_url": url,
                              "s3_key": s3_key or "",
                              "params": params, "completed_at": time.time(), "error": None})
        ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
        info = f"{sid[:8]}…  v{ver}  [done]"
        return sid, minor_upd, info, "✅ done", _url_to_img(url, 300), f"→ {int(w)}×{int(h)}"
    except Exception as exc:
        return sid, minor_upd, info_upd, f"error: {exc}", _url_to_img(""), str(exc)


def _q_sd(char, major, minor, stage, prompt, neg, params, src_stage, src_url):
    """
    src_url: already-resolved image URL from the source version picker.
    Falls back to latest done for src_stage if empty.
    Collection routing is automatic — stage is passed to mark_queued/save_run_params.
    """
    if not prompt.strip():
        return None, gr.update(), gr.update(), "⚠️ Prompt is empty."

    if not src_url:
        src_url = get_latest_done_image_url(_db(), char, src_stage) or ""
    if not src_url:
        return None, gr.update(), gr.update(), \
            f"No done image in '{src_stage}'. Run that stage first."

    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, prompt, neg, params)
    if err:
        return sid, minor_upd, gr.update(), err

    db = _db()
    save_run_params(db, sid, prompt, neg, params, stage=stage)
    tid = _push_task({"type": "sd_stage", "session_id": sid, "stage": stage,
                      "char_label": char, "prompt": prompt, "negative": neg,
                      "params": params, "input_stage": src_stage, "input_url": src_url})
    mark_queued(db, sid, stage=stage, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


def _q_trellis(char, major, minor, char_type, src_stage, src_ver):
    """
    Queue Trellis 3D task.
    Reads front/side/back view URLs from the selected source stage+version.
    Front view is required; side/back are optional enhancements.
    """
    stage = "trellis"
    db    = _db()

    # Resolve view URLs from source version (or fall back to latest done)
    if src_ver:
        view_urls = _get_view_urls_for_ver(char, src_stage, src_ver)
    else:
        # Fall back to latest done run
        latest = get_latest_done_run(db, char, src_stage)
        if latest:
            view_urls = {
                "front": latest.get("image_url") or "",
                "side":  latest.get("side_url")  or "",
                "back":  latest.get("back_url")  or "",
            }
        else:
            view_urls = {"front": "", "side": "", "back": ""}

    front_url = view_urls["front"]
    side_url  = view_urls["side"]
    back_url  = view_urls["back"]

    if not front_url:
        return None, gr.update(), gr.update(), \
            f"No front-view image in '{src_stage}' v{src_ver or 'latest'}. Run that stage first."

    params = {"char_type": char_type or "humanoid"}
    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, "", "", params)
    if err:
        return sid, minor_upd, gr.update(), err

    save_run_params(db, sid, "", "", params, stage=stage)
    tid = _push_task({"type": "trellis", "session_id": sid, "stage": stage,
                      "char_label": char, "char_type": char_type or "humanoid",
                      "input_front": front_url,
                      "input_side": side_url, "input_back": back_url,
                      "params": params})
    mark_queued(db, sid, stage=stage, task_id=tid)
    views_info = f"F{'✓' if front_url else '✗'} S{'✓' if side_url else '✗'} B{'✓' if back_url else '✗'}"
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  [{views_info}]  task={tid[:8]}…"


def _q_rig(char, major, minor, char_type, trellis_src_ver):
    stage = "rig"
    db    = _db()
    glb_url = (_get_src_url_for_ver(char, "trellis", trellis_src_ver) if trellis_src_ver
               else get_latest_done_image_url(db, char, "trellis") or "")
    if not glb_url:
        return None, gr.update(), gr.update(), "No done TRELLIS GLB. Run Trellis first."

    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, "", "", {"char_type": char_type})
    if err:
        return sid, minor_upd, gr.update(), err

    save_run_params(db, sid, "", "", {"char_type": char_type or "humanoid"}, stage=stage)
    tid = _push_task({"type": "rig", "session_id": sid, "stage": stage,
                      "char_label": char, "char_type": char_type or "humanoid",
                      "input_glb_url": glb_url},
                     queue="rig_tasks")
    mark_queued(db, sid, stage=stage, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


# ══════════════════════════════════════════════════════════════════════════════
#  PER-STAGE PICKER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_picker(initial_chars):
    """
    Build a compact char / major / minor row with an inline "＋ Major" button.
    Returns: (p_char, p_major, p_minor, p_new_major_btn, p_sid, p_info)
    """
    with gr.Row():
        p_char       = gr.Dropdown(choices=initial_chars, label="Character",
                                   allow_custom_value=False, scale=4)
        p_major      = gr.Dropdown(choices=[1], value=1, label="Major", scale=1)
        p_minor      = gr.Dropdown(choices=[0], value=0, label="Minor", scale=1)
        p_new_major  = gr.Button("＋ Major", size="sm", scale=1)
    p_info = gr.Textbox(label="Run", interactive=False, lines=1)
    p_sid  = gr.State(None)
    return p_char, p_major, p_minor, p_new_major, p_sid, p_info


def _make_src_picker(stage_choices, default_stage):
    """
    Build a source-stage + source-version picker row.
    Returns: (src_stage_dd, src_ver_dd, src_url_st, src_info_tb)
    """
    with gr.Row():
        src_stage = gr.Dropdown(choices=stage_choices, value=default_stage,
                                label="Source stage", scale=2)
        src_ver   = gr.Dropdown(choices=[], value=None, label="Source version",
                                allow_custom_value=False, scale=2,
                                info="default: latest done")
        src_url_st = gr.State("")
    src_info = gr.Textbox(label="Source", interactive=False, lines=1, scale=3)
    return src_stage, src_ver, src_url_st, src_info


def _wire_picker(stage_name, p_char, p_major, p_minor, p_new_major, p_sid, p_info,
                 stage_outputs: list, extract_fn):
    """
    Wire char/major/minor dropdowns and the "＋ Major" button.
    extract_fn(run_doc: dict) -> list matching stage_outputs.
    """

    def _on_char(char):
        majors = _list_majors(char, stage_name)
        m      = majors[-1]
        minors = _list_minors(char, stage_name, m)
        n      = minors[-1]
        sid, info = _resolve_run(char, stage_name, m, n)
        run       = _get_run_doc(char, stage_name, m, n)
        return ([gr.update(choices=majors, value=m),
                 gr.update(choices=minors, value=n),
                 sid, info]
                + extract_fn(run))

    def _on_major(char, major):
        if major is None: major = 1
        minors = _list_minors(char, stage_name, int(major))
        n      = minors[-1]
        sid, info = _resolve_run(char, stage_name, int(major), n)
        run       = _get_run_doc(char, stage_name, int(major), n)
        return ([gr.update(choices=minors, value=n), sid, info]
                + extract_fn(run))

    def _on_minor(char, major, minor):
        if major is None: major = 1
        if minor is None: minor = 0
        sid, info = _resolve_run(char, stage_name, int(major), int(minor))
        run       = _get_run_doc(char, stage_name, int(major), int(minor))
        return [sid, info] + extract_fn(run)

    def _do_new_major(char):
        if not char:
            return gr.update(), gr.update(), None, "Pick a character first."
        db    = _db()
        new_m = next_stage_major(db, char, stage_name)
        sid   = create_run(db, char, stage_name, new_m, 0)
        majors = _list_majors(char, stage_name)
        info   = f"{sid[:8]}…  v{new_m}.0  [idle]"
        return (gr.update(choices=majors, value=new_m),
                gr.update(choices=[0], value=0),
                sid, info)

    p_char.change(_on_char,   [p_char],
                  [p_major, p_minor, p_sid, p_info] + stage_outputs)
    p_major.change(_on_major, [p_char, p_major],
                   [p_minor, p_sid, p_info] + stage_outputs)
    p_minor.change(_on_minor, [p_char, p_major, p_minor],
                   [p_sid, p_info] + stage_outputs)
    p_new_major.click(_do_new_major, [p_char],
                      [p_major, p_minor, p_sid, p_info])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

def generation_studio_ui():

    with gr.Blocks() as tab:
        gr.Markdown("# 🎨 Generation Studio")

        stage_timer = gr.Timer(value=4, active=False)
        _chars = _list_chars()

        # ══════════════════════════════════════════════════════════════════════
        #  TOP: CHARACTER PREFILL HELPER
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("⚙️ Character Prefill", open=True):
            gr.Markdown(
                "_Pick a character and click **Prefill All Stages** to set every "
                "stage's character picker at once.  Each stage still has its own "
                "independent versioning._"
            )
            with gr.Row():
                g_char        = gr.Dropdown(
                    choices=_chars, value=(_chars[0] if _chars else None),
                    label="Character", allow_custom_value=False, scale=4)
                g_refresh_btn = gr.Button("⟳ Refresh Chars", size="sm", scale=1)
                g_prefill_btn = gr.Button("⬇ Prefill All Stages", variant="secondary", scale=2)

            with gr.Accordion("New Character", open=False):
                with gr.Row():
                    g_new_char_input = gr.Textbox(label="Character Label",
                                                  placeholder="e.g. knight_01", scale=3)
                    g_create_btn     = gr.Button("Create", variant="primary", scale=1)
                g_create_info = gr.Textbox(label="", interactive=False, lines=1)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 0: FLUX
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 0 — Flux (Text → Image)", open=True):
            fx_char, fx_major, fx_minor, fx_new_maj, fx_sid, fx_info = _make_picker(_chars)
            gr.Markdown("---")
            fx_prompt   = gr.Textbox(label="Prompt", lines=5,
                                     placeholder="full body character, T-pose, white background…")
            fx_tok      = gr.Textbox(label="", lines=1, interactive=False)
            fx_negative = gr.Textbox(label="Negative", lines=2,
                                     value="deformed, extra limbs, text, watermark, blurry, nsfw")
            with gr.Row():
                fx_w    = gr.Number(label="Width",    value=512, precision=0)
                fx_h    = gr.Number(label="Height",   value=512, precision=0)
                fx_steps= gr.Number(label="Steps",    value=4,   precision=0)
                fx_guid = gr.Number(label="Guidance", value=0.0)
            with gr.Row():
                fx_q_btn = gr.Button("Queue Flux (all 3 views)", variant="primary")
                fx_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                fx_r_btn = gr.Button("Refresh", size="sm")
            gr.Markdown("_Queues front + side + back views automatically. "
                        "Side/back append view suffix to your prompt._")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Front view**")
                    fx_img      = gr.HTML(value=_url_to_img(""))
                    fx_url      = gr.Textbox(label="Front URL", interactive=False)
                with gr.Column():
                    gr.Markdown("**Side view**")
                    fx_side_img = gr.HTML(value=_url_to_img(""))
                    fx_side_url = gr.Textbox(label="Side URL", interactive=False)
                with gr.Column():
                    gr.Markdown("**Back view**")
                    fx_back_img = gr.HTML(value=_url_to_img(""))
                    fx_back_url = gr.Textbox(label="Back URL", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 1: NORMALIZE
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 1 — Normalize (CPU, instant)", open=False):
            nm_char, nm_major, nm_minor, nm_new_maj, nm_sid, nm_info = _make_picker(_chars)
            gr.Markdown("---")
            nm_src_stage, nm_src_ver, nm_src_url_st, nm_src_info = _make_src_picker(
                ["flux", "sd_tpose"], "flux")
            with gr.Row():
                nm_w   = gr.Number(label="Width",  value=512, precision=0)
                nm_h   = gr.Number(label="Height", value=512, precision=0)
            with gr.Row():
                nm_btn    = gr.Button("Run Normalize", variant="primary")
                nm_status = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
            nm_img   = gr.HTML(value=_url_to_img("", 300))
            nm_info2 = gr.Textbox(label="Info", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 2: SD T-POSE LOCK (IP-Adapter + OpenPose/Canny ControlNet)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 2 — SD1.5 T-Pose Lock (IP-Adapter)", open=False):
            s1_char, s1_major, s1_minor, s1_new_maj, s1_sid, s1_info = _make_picker(_chars)
            gr.Markdown("---")
            gr.Markdown(
                "**IP-Adapter** encodes the Flux image to preserve character identity.  \n"
                "**OpenPose ControlNet** locks the output to a pre-baked T-pose skeleton.  \n"
                "**Canny ControlNet** adds edge structure from the T-pose template (not Flux)."
            )
            s1_src_stage, s1_src_ver, s1_src_url_st, s1_src_info = _make_src_picker(
                ["flux", "normalize"], "flux")
            s1_cat    = gr.Radio(choices=["humanoid", "quadruped"],
                                 value="humanoid", label="Character type")
            s1_prompt = gr.Textbox(label="Prompt (keep minimal — identity comes from IP-Adapter)", lines=3)
            s1_tok    = gr.Textbox(label="", lines=1, interactive=False)
            s1_neg    = gr.Textbox(label="Negative", lines=2,
                                   value="deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw")
            with gr.Row():
                s1_denoise = gr.Slider(0.05, 0.95, value=0.65, step=0.01, label="Denoise")
                s1_cfg     = gr.Slider(1.0, 15.0,  value=7.0,  step=0.5,  label="CFG")
                s1_steps   = gr.Number(label="Steps", value=25, precision=0)
            with gr.Row():
                s1_op_w    = gr.Slider(0.0, 1.5, value=1.00, step=0.05, label="OpenPose weight")
                s1_cn_w    = gr.Slider(0.0, 1.5, value=0.25, step=0.05, label="Canny weight")
                s1_ip_w    = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="IP-Adapter weight")
            _S3 = "https://sparkassets-us.s3.us-east-1.amazonaws.com/controlnet_refs"
            s1_openpose_ref = gr.Dropdown(
                choices=[
                    ("Default (active on S3)",       ""),
                    ("V1 — Hand-drawn",               f"{_S3}/tpose_v1_user.png"),
                    ("V2 — FBX extracted (X Bot)",    f"{_S3}/tpose_v2_fbx.png"),
                ],
                value="",
                label="T-Pose Skeleton",
                info="Select which OpenPose skeleton to use for this run",
            )
            with gr.Row():
                s1_q_btn = gr.Button("Queue T-Pose", variant="primary")
                s1_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                s1_r_btn = gr.Button("Refresh", size="sm")
            s1_img = gr.HTML(value=_url_to_img("", 350))
            s1_url = gr.Textbox(label="URL", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 3: TRELLIS 3D
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 3 — TRELLIS 3D Mesh", open=False):
            tr_char, tr_major, tr_minor, tr_new_maj, tr_sid, tr_info = _make_picker(_chars)
            gr.Markdown("---")
            tr_char_type = gr.Dropdown(
                choices=["humanoid", "quadruped", "bird", "fish"],
                value="humanoid", label="Character type")
            gr.Markdown("**Select source stage + version (side/back views auto-loaded if available):**")
            tr_src_stage, tr_src_ver, tr_src_url_st, tr_src_info = _make_src_picker(
                ["sd_tpose", "flux", "normalize"], "sd_tpose")
            tr_view_info = gr.Textbox(label="View availability", interactive=False, lines=1,
                                      info="Shows which views are present for selected version")
            with gr.Row():
                tr_q_btn = gr.Button("Queue TRELLIS", variant="primary")
                tr_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                tr_r_btn = gr.Button("Refresh", size="sm")
            tr_url = gr.Textbox(label="GLB URL (when done)", interactive=False)
            tr_3d_btn = gr.HTML(value="")

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 4: RIG
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 4 — Auto-Rig Pro (CPU)", open=False):
            rg_char, rg_major, rg_minor, rg_new_maj, rg_sid, rg_info = _make_picker(_chars)
            gr.Markdown("---")
            # Source = trellis GLB
            with gr.Row():
                rg_trellis_ver = gr.Dropdown(choices=[], value=None,
                                             label="Trellis source version",
                                             info="default: latest done", scale=2)
                rg_trellis_refresh = gr.Button("↻", size="sm", scale=0)
                rg_trellis_info = gr.Textbox(label="Source", interactive=False, lines=1, scale=3)
            rg_type = gr.Dropdown(choices=["humanoid", "quadruped", "bird", "fish"],
                                  value="humanoid", label="Character type")
            with gr.Row():
                rg_q_btn = gr.Button("Queue Rig", variant="primary")
                rg_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                rg_r_btn = gr.Button("Refresh", size="sm")
            rg_url = gr.Textbox(label="Rigged GLB URL (when done)", interactive=False)
            rg_3d_btn = gr.HTML(value="")


        # ══════════════════════════════════════════════════════════════════════
        #  EVENT WIRING
        # ══════════════════════════════════════════════════════════════════════

        # ── Token counters ────────────────────────────────────────────────────
        fx_prompt.input(lambda t: _tok(t, False), [fx_prompt], [fx_tok])
        s1_prompt.input(lambda t: _tok(t, True),  [s1_prompt], [s1_tok])

        # ── Global: Refresh char list ─────────────────────────────────────────
        def _do_refresh():
            chars = _list_chars()
            upd   = gr.update(choices=chars, value=(chars[0] if chars else None))
            return [upd] * 6

        g_refresh_btn.click(_do_refresh, [],
                            [g_char, fx_char, nm_char, s1_char, tr_char, rg_char])

        # ── Global: Create New Character ──────────────────────────────────────
        def _do_create(label):
            label = (label or "").strip()
            if not label:
                return [gr.update()] * 6 + ["Enter a character label first."]
            ok    = create_character(_db(), label)
            if not ok:
                return [gr.update()] * 6 + [f"❌ Failed to save '{label}' — check MongoDB connection."]
            chars = _list_chars()
            upd   = gr.update(choices=chars, value=label)
            return [upd] * 6 + [f"✓ Created '{label}'. Select it in any stage below and click ⬇ Prefill All Stages."]

        g_create_btn.click(_do_create, [g_new_char_input],
                           [g_char, fx_char, nm_char, s1_char, tr_char, rg_char, g_create_info])

        # ── Global: Prefill All Stages ────────────────────────────────────────
        # In Gradio 5, gr.update(value=X) does NOT trigger .change handlers,
        # so we must return ALL stage data directly — no cascading events.
        def _do_prefill(char):
            chars    = _list_chars()
            char_upd = gr.update(choices=chars, value=char) if char else gr.update()

            def _load(stage_name, extract_fn):
                """Return (major_upd, minor_upd, sid, info, *stage_data)."""
                if not char:
                    return (gr.update(), gr.update(), None, "") + tuple(extract_fn({}))
                majors = _list_majors(char, stage_name)
                m      = majors[-1]
                minors = _list_minors(char, stage_name, m)
                n      = minors[-1]
                sid, info = _resolve_run(char, stage_name, m, n)
                run       = _get_run_doc(char, stage_name, m, n)
                return (gr.update(choices=majors, value=m),
                        gr.update(choices=minors, value=n),
                        sid, info) + tuple(extract_fn(run))

            # Source pickers — refresh for the default source stage of each downstream stage
            # _refresh_src_picker returns (ver_upd, src_url, info_str)
            nm_src = _refresh_src_picker(char, "flux")     if char else (gr.update(), "", "")
            s1_src = _refresh_src_picker(char, "flux")     if char else (gr.update(), "", "")
            tr_src = _refresh_src_picker(char, "sd_tpose") if char else (gr.update(), "", "")
            # Trellis view availability info
            if char:
                tr_ver_val = tr_src[0].get("value") if hasattr(tr_src[0], "get") else None
                tr_vinfo   = _view_availability_info(char, "sd_tpose", tr_ver_val or "")
            else:
                tr_vinfo = ""

            return list((
                # 5 char dropdowns
                char_upd, char_upd, char_upd, char_upd, char_upd,
                # flux (major, minor, sid, info + 15 data fields = 19)
                *_load("flux", _ex_flux),
                # normalize (4+4=8) + source picker (3) = 11
                *_load("normalize", _ex_normalize),
                *nm_src,
                # sd_tpose (4+13=17) + source picker (3) = 20
                *_load("sd_tpose", _ex_sd_tpose),
                *s1_src,
                # trellis (4+3=7) + source picker (3) + view_info (1) = 11
                *_load("trellis", _ex_trellis),
                *tr_src, tr_vinfo,
                # rig (4+3=7) = 7
                *_load("rig", _ex_rig),
            ))

        g_prefill_btn.click(_do_prefill, [g_char], [
            # char dropdowns (5)
            fx_char, nm_char, s1_char, tr_char, rg_char,
            # flux (4 picker + 15 data = 19)
            fx_major, fx_minor, fx_sid, fx_info,
            fx_prompt, fx_negative, fx_w, fx_h, fx_steps, fx_guid,
            fx_status, fx_url, fx_img, fx_side_url, fx_side_img, fx_back_url, fx_back_img,
            # normalize (4+4=8) + source picker (3) = 11
            nm_major, nm_minor, nm_sid, nm_info,
            nm_w, nm_h, nm_status, nm_img,
            nm_src_ver, nm_src_url_st, nm_src_info,
            # sd_tpose (4+13=17) + source picker (3) = 20
            s1_major, s1_minor, s1_sid, s1_info,
            s1_prompt, s1_neg, s1_denoise, s1_cfg, s1_steps,
            s1_op_w, s1_cn_w, s1_ip_w, s1_cat, s1_openpose_ref, s1_status, s1_url, s1_img,
            s1_src_ver, s1_src_url_st, s1_src_info,
            # trellis (4+3=7) + source picker (3) + view_info (1) = 11
            tr_major, tr_minor, tr_sid, tr_info,
            tr_char_type, tr_status, tr_url,
            tr_src_ver, tr_src_url_st, tr_src_info, tr_view_info,
            # rig (4+3=7) = 7
            rg_major, rg_minor, rg_sid, rg_info, rg_type, rg_status, rg_url,
        ])

        # ── Extract functions per stage ───────────────────────────────────────

        def _ex_flux(run):
            p = run.get("params") or {}
            front_url = run.get("image_url", "") or ""
            side_url  = run.get("side_url",   "") or ""
            back_url  = run.get("back_url",   "") or ""
            return [run.get("prompt", ""),
                    run.get("negative", "deformed, extra limbs, text, watermark, blurry, nsfw"),
                    p.get("width", 512), p.get("height", 512),
                    p.get("steps", 4), p.get("guidance_scale", 0.0),
                    run.get("status", "idle"),
                    front_url, _url_to_img(front_url),
                    side_url,  _url_to_img(side_url),
                    back_url,  _url_to_img(back_url),
                    ]

        def _ex_normalize(run):
            p = run.get("params") or {}
            return [p.get("resize_w", 512), p.get("resize_h", 512),
                    run.get("status", "idle"),
                    _url_to_img(run.get("image_url", "") or "", 300)]

        def _ex_sd_tpose(run):
            p = run.get("params") or {}
            return [run.get("prompt", ""),
                    run.get("negative", "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw"),
                    p.get("denoise", 0.65), p.get("cfg", 7.0), p.get("steps", 25),
                    p.get("openpose_weight", 1.00), p.get("canny_weight", 0.25),
                    p.get("ip_adapter_weight", 0.65),
                    p.get("category", "humanoid"),
                    p.get("openpose_ref_url", ""),
                    run.get("status", "idle"),
                    run.get("image_url", "") or "",
                    _url_to_img(run.get("image_url", "") or "", 350)]

        def _ex_trellis(run):
            p = run.get("params") or {}
            return [p.get("char_type", "humanoid"),
                    run.get("status", "idle"),
                    run.get("image_url", "") or ""]

        def _ex_rig(run):
            p = run.get("params") or {}
            return [p.get("char_type", "humanoid"),
                    run.get("status", "idle"),
                    run.get("image_url", "") or ""]

        # ── Wire per-stage pickers ─────────────────────────────────────────────

        _wire_picker("flux", fx_char, fx_major, fx_minor, fx_new_maj, fx_sid, fx_info,
                     [fx_prompt, fx_negative, fx_w, fx_h, fx_steps, fx_guid,
                      fx_status, fx_url, fx_img, fx_side_url, fx_side_img, fx_back_url, fx_back_img],
                     _ex_flux)

        _wire_picker("normalize", nm_char, nm_major, nm_minor, nm_new_maj, nm_sid, nm_info,
                     [nm_w, nm_h, nm_status, nm_img],
                     _ex_normalize)

        _wire_picker("sd_tpose", s1_char, s1_major, s1_minor, s1_new_maj, s1_sid, s1_info,
                     [s1_prompt, s1_neg, s1_denoise, s1_cfg, s1_steps,
                      s1_op_w, s1_cn_w, s1_ip_w, s1_cat, s1_openpose_ref,
                      s1_status, s1_url, s1_img],
                     _ex_sd_tpose)

        _wire_picker("trellis", tr_char, tr_major, tr_minor, tr_new_maj, tr_sid, tr_info,
                     [tr_char_type, tr_status, tr_url],
                     _ex_trellis)

        _wire_picker("rig", rg_char, rg_major, rg_minor, rg_new_maj, rg_sid, rg_info,
                     [rg_type, rg_status, rg_url],
                     _ex_rig)

        # ── Source picker wiring — refresh on src_stage change ─────────────────

        def _make_src_wiring(char_comp, src_stage_comp, src_ver_comp, src_url_st_comp, src_info_comp):
            """Wire source stage/version pickers for one downstream stage."""
            src_stage_comp.change(
                _refresh_src_picker,
                [char_comp, src_stage_comp],
                [src_ver_comp, src_url_st_comp, src_info_comp]
            )
            src_ver_comp.change(
                _on_src_ver,
                [char_comp, src_stage_comp, src_ver_comp],
                [src_url_st_comp, src_info_comp]
            )
            # Also refresh when char changes (char may have different done runs)
            char_comp.change(
                _refresh_src_picker,
                [char_comp, src_stage_comp],
                [src_ver_comp, src_url_st_comp, src_info_comp]
            )

        _make_src_wiring(nm_char, nm_src_stage, nm_src_ver, nm_src_url_st, nm_src_info)
        _make_src_wiring(s1_char, s1_src_stage, s1_src_ver, s1_src_url_st, s1_src_info)
        _make_src_wiring(tr_char, tr_src_stage,  tr_src_ver,  tr_src_url_st,  tr_src_info)

        # Trellis: also update view availability info when ver or stage changes
        def _tr_update_view_info(char, src_stage, ver):
            return _view_availability_info(char, src_stage, ver)

        tr_src_ver.change(_tr_update_view_info,
                          [tr_char, tr_src_stage, tr_src_ver],
                          [tr_view_info])
        tr_src_stage.change(
            lambda char, stage: _view_availability_info(char, stage, ""),
            [tr_char, tr_src_stage], [tr_view_info]
        )
        tr_char.change(
            lambda char: _view_availability_info(char, "sd_tpose", ""),
            [tr_char], [tr_view_info]
        )

        # Rig: source is always trellis, just version picker
        def _refresh_trellis_ver(char):
            versions = _list_done_versions(char, "trellis")
            if versions:
                latest = versions[0]
                return gr.update(choices=versions, value=latest), f"✓ trellis v{latest}"
            return gr.update(choices=[], value=None), "No done trellis runs yet"

        rg_char.change(_refresh_trellis_ver, [rg_char], [rg_trellis_ver, rg_trellis_info])
        rg_trellis_refresh.click(_refresh_trellis_ver, [rg_char], [rg_trellis_ver, rg_trellis_info])
        rg_trellis_ver.change(
            lambda char, ver: (_get_src_url_for_ver(char, "trellis", ver) if ver else "",
                               f"✓ trellis v{ver}" if ver else "No version selected"),
            [rg_char, rg_trellis_ver],
            [gr.State(), rg_trellis_info]  # url not needed in UI, just info
        )

        # ── Queue buttons ─────────────────────────────────────────────────────

        # Flux
        def _refresh_with_img(sid, h=400):
            st, url = _refresh_run(sid)
            return st, url, _url_to_img(url, h)

        (fx_q_btn.click(
            _q_flux,
            [fx_char, fx_major, fx_minor,
             fx_prompt, fx_negative, fx_w, fx_h, fx_steps, fx_guid],
            [fx_sid, fx_minor, fx_info, fx_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        def _refresh_flux(sid):
            """Refresh flux: returns front status/url/img + side url/img + back url/img."""
            if not sid:
                return "idle", "", _url_to_img(""), "", _url_to_img(""), "", _url_to_img("")
            try:
                doc = get_run_any(_db(), sid)
                if not doc:
                    return "idle", "", _url_to_img(""), "", _url_to_img(""), "", _url_to_img("")
                st         = doc.get("status", "idle")
                front_url  = doc.get("image_url") or ""
                side_url   = doc.get("side_url")  or ""
                back_url   = doc.get("back_url")  or ""
                if st == "done":    st = "✅ done"
                elif st == "error": st = f"❌ {doc.get('error', 'error')}"
                return (st, front_url, _url_to_img(front_url),
                        side_url,  _url_to_img(side_url),
                        back_url,  _url_to_img(back_url))
            except Exception as exc:
                return f"❌ {exc}", "", _url_to_img(""), "", _url_to_img(""), "", _url_to_img("")

        fx_r_btn.click(
            _refresh_flux,
            [fx_sid], [fx_status, fx_url, fx_img, fx_side_url, fx_side_img, fx_back_url, fx_back_img]
        )

        # Normalize
        nm_btn.click(
            _q_normalize,
            [nm_char, nm_major, nm_minor, nm_w, nm_h, nm_src_stage, nm_src_ver],
            [nm_sid, nm_minor, nm_info, nm_status, nm_img, nm_info2]
        )

        # SD T-Pose
        def _do_q_tp(char, major, minor, p, n, dn, cfg, st, opw, cnw, ipw, cat, ref_url, src_stage, src_url):
            params = {"denoise": float(dn), "cfg": float(cfg), "steps": int(st),
                      "openpose_weight": float(opw), "canny_weight": float(cnw),
                      "ip_adapter_weight": float(ipw), "category": cat}
            if ref_url:
                params["openpose_ref_url"] = ref_url
            return _q_sd(char, major, minor, "sd_tpose", p, n, params, src_stage, src_url)

        (s1_q_btn.click(
            _do_q_tp,
            [s1_char, s1_major, s1_minor,
             s1_prompt, s1_neg, s1_denoise, s1_cfg, s1_steps,
             s1_op_w, s1_cn_w, s1_ip_w, s1_cat, s1_openpose_ref,
             s1_src_stage, s1_src_url_st],
            [s1_sid, s1_minor, s1_info, s1_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        s1_r_btn.click(
            lambda sid: _refresh_with_img(sid, 350),
            [s1_sid], [s1_status, s1_url, s1_img]
        )

        # TRELLIS
        (tr_q_btn.click(
            _q_trellis,
            [tr_char, tr_major, tr_minor,
             tr_char_type, tr_src_stage, tr_src_ver],
            [tr_sid, tr_minor, tr_info, tr_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        tr_r_btn.click(
            lambda sid: _refresh_run(sid),
            [tr_sid], [tr_status, tr_url]
        )

        # Rig
        (rg_q_btn.click(
            _q_rig,
            [rg_char, rg_major, rg_minor, rg_type, rg_trellis_ver],
            [rg_sid, rg_minor, rg_info, rg_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        rg_r_btn.click(
            lambda sid: _refresh_run(sid),
            [rg_sid], [rg_status, rg_url]
        )

        # ── 3D viewer button helper ───────────────────────────────────────────
        def _viewer_btn(url: str) -> str:
            if not url:
                return ""
            link = f"https://3dviewer.net/#model={url}"
            return (
                f'<div style="display: flex; gap: 8px;">'
                f'<a href="{link}" target="_blank" rel="noopener noreferrer" '
                f'style="display:inline-block;padding:6px 14px;background:#2563eb;'
                f'color:#fff;border-radius:6px;text-decoration:none;font-size:13px;">'
                f'🧊 Open in 3dviewer.net</a>'
                f'<a href="{url}" download target="_blank" rel="noopener noreferrer" '
                f'style="display:inline-block;padding:6px 14px;background:#10b981;'
                f'color:#fff;border-radius:6px;text-decoration:none;font-size:13px;">'
                f'⬇️ Download GLB</a>'
                f'</div>'
            )

        tr_url.change(_viewer_btn, [tr_url], [tr_3d_btn])
        rg_url.change(_viewer_btn, [rg_url], [rg_3d_btn])

        # ── Auto-refresh timer ────────────────────────────────────────────────
        _ACTIVE = {"queued", "running"}

        def _tick(fx_s, s1_s, tr_s, rg_s):
            def _r(sid): return _refresh_run(sid)

            # Flux: get all 3 view URLs from doc
            fx_doc     = get_run_any(_db(), fx_s) if fx_s else {}
            fx_st      = fx_doc.get("status", "idle") if fx_doc else "idle"
            fx_u       = (fx_doc or {}).get("image_url") or ""
            fx_side_u  = (fx_doc or {}).get("side_url")  or ""
            fx_back_u  = (fx_doc or {}).get("back_url")  or ""
            if fx_st == "done":    fx_st = "✅ done"
            elif fx_st == "error": fx_st = f"❌ {(fx_doc or {}).get('error', 'error')}"

            s1_st, s1_u = _r(s1_s)
            tr_st, tr_u = _r(tr_s)
            rg_st, rg_u = _r(rg_s)

            still = any(s in _ACTIVE for s in [fx_st, s1_st, tr_st, rg_st])

            return (
                fx_st, fx_u, _url_to_img(fx_u),
                fx_side_u, _url_to_img(fx_side_u),
                fx_back_u, _url_to_img(fx_back_u),
                s1_st, s1_u, _url_to_img(s1_u, 350),
                tr_st, tr_u, _viewer_btn(tr_u),
                rg_st, rg_u, _viewer_btn(rg_u),
                gr.Timer(active=still),
            )

        stage_timer.tick(
            _tick,
            [fx_sid, s1_sid, tr_sid, rg_sid],
            [fx_status, fx_url, fx_img,
             fx_side_url, fx_side_img, fx_back_url, fx_back_img,
             s1_status, s1_url, s1_img,
             tr_status, tr_url, tr_3d_btn,
             rg_status, rg_url, rg_3d_btn,
             stage_timer],
        )

    return tab
