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
    SD1, SD2, Multiview, TRELLIS, Rig each have:
      Source stage  — which upstream stage to read image from
      Source version — specific done version to use (dropdown, default=latest)
    The URL is resolved at queue time from the selected version.

Schema
------
  Collection: manual_gen_stage_runs
  Each document = one stage × one major.minor version run.
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
    get_latest_done_image_url,
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

def _push_task(payload: dict) -> str:
    import redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                    db=0, decode_responses=True)
    payload.setdefault("task_id", str(uuid.uuid4()))
    payload.setdefault("timestamp", time.time())
    r.rpush(REDIS_QUEUE, json.dumps(payload))
    return payload["task_id"]

def _list_chars() -> list[str]:
    """Read character list from the registry + any existing stage runs."""
    try:
        return list_characters(_db()) or []
    except Exception as e:
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
    """Returns (status_text, image_url)."""
    if not run_id:
        return "idle", ""
    try:
        doc = _db()[COLLECTION].find_one({"_id": run_id})
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


# ── Source version helpers ─────────────────────────────────────────────────────

def _list_done_versions(char: str, stage: str) -> list[str]:
    """
    Return distinct done version strings for (char, stage), most recent first.
    e.g. ["2.1", "2.0", "1.0"]
    """
    if not char or not stage:
        return []
    try:
        docs = list(_db()[COLLECTION].find(
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
    save_run_params(db, sid, prompt, neg, params)
    tid = _push_task({"type": "flux", "session_id": sid, "stage": stage,
                      "char_label": char, "prompt": prompt, "negative": neg,
                      "params": params})
    mark_queued(db, sid, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


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
        img    = img.resize((int(w), int(h)), PILImage.LANCZOS)
        s3_key = f"manual_gen/{sid}/normalize_{int(w)}x{int(h)}.png"
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
        update_run(db, sid, {"status": "done", "image_url": url, "s3_key": s3_key,
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
    save_run_params(db, sid, prompt, neg, params)
    tid = _push_task({"type": "sd_stage", "session_id": sid, "stage": stage,
                      "char_label": char, "prompt": prompt, "negative": neg,
                      "params": params, "input_stage": src_stage, "input_url": src_url})
    mark_queued(db, sid, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


def _q_multiview(char, major, minor, view, prompt, neg, denoise, cfg, src_stage, src_url):
    stage  = f"multiview_{view}"
    params = {"denoise": float(denoise), "cfg": float(cfg), "steps": 20}
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
    save_run_params(db, sid, prompt, neg, params)
    tid = _push_task({"type": "multiview", "session_id": sid, "stage": stage,
                      "char_label": char, "view": view, "prompt": prompt,
                      "negative": neg, "params": params,
                      "input_stage": src_stage, "input_url": src_url})
    mark_queued(db, sid, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


def _q_trellis(char, major, minor, front_stage, front_url, side_stage, side_url, back_stage, back_url):
    stage = "trellis"
    # Resolve any missing URLs from latest done
    db = _db()
    if not front_url:
        front_url = get_latest_done_image_url(db, char, front_stage) or ""
    if not side_url:
        side_url  = get_latest_done_image_url(db, char, side_stage)  or ""
    if not back_url:
        back_url  = get_latest_done_image_url(db, char, back_stage)  or ""

    if not front_url:
        return None, gr.update(), gr.update(), \
            f"No done image in '{front_stage}'. Run that stage first."

    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, "", "", {})
    if err:
        return sid, minor_upd, gr.update(), err

    tid = _push_task({"type": "trellis", "session_id": sid, "stage": stage,
                      "char_label": char, "input_front": front_url,
                      "input_side": side_url, "input_back": back_url})
    mark_queued(db, sid, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


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

    save_run_params(db, sid, "", "", {"char_type": char_type or "humanoid"})
    tid = _push_task({"type": "rig", "session_id": sid, "stage": stage,
                      "char_label": char, "char_type": char_type or "humanoid",
                      "input_glb_url": glb_url})
    mark_queued(db, sid, task_id=tid)
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
                fx_q_btn = gr.Button("Queue Flux", variant="primary")
                fx_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                fx_r_btn = gr.Button("Refresh", size="sm")
            fx_img = gr.HTML(value=_url_to_img(""))
            fx_url = gr.Textbox(label="Image URL", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 1: NORMALIZE
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 1 — Normalize (CPU, instant)", open=False):
            nm_char, nm_major, nm_minor, nm_new_maj, nm_sid, nm_info = _make_picker(_chars)
            gr.Markdown("---")
            nm_src_stage, nm_src_ver, nm_src_url_st, nm_src_info = _make_src_picker(
                ["flux", "sd_stage1", "sd_stage2"], "flux")
            with gr.Row():
                nm_w   = gr.Number(label="Width",  value=512, precision=0)
                nm_h   = gr.Number(label="Height", value=512, precision=0)
            with gr.Row():
                nm_btn    = gr.Button("Run Normalize", variant="primary")
                nm_status = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
            nm_img   = gr.HTML(value=_url_to_img("", 300))
            nm_info2 = gr.Textbox(label="Info", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 2: SD STAGE 1 — POSE LOCK
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 2 — SD1.5 ControlNet Pose Lock", open=False):
            s1_char, s1_major, s1_minor, s1_new_maj, s1_sid, s1_info = _make_picker(_chars)
            gr.Markdown("---")
            s1_src_stage, s1_src_ver, s1_src_url_st, s1_src_info = _make_src_picker(
                ["flux", "normalize"], "flux")
            s1_cat    = gr.Radio(choices=["humanoid", "quadruped"],
                                 value="humanoid", label="Character type")
            s1_prompt = gr.Textbox(label="Prompt (keep minimal)", lines=3)
            s1_tok    = gr.Textbox(label="", lines=1, interactive=False)
            s1_neg    = gr.Textbox(label="Negative", lines=2,
                                   value="deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw")
            with gr.Row():
                s1_denoise = gr.Slider(0.05, 0.50, value=0.20, step=0.01, label="Denoise")
                s1_cfg     = gr.Slider(1.0, 15.0,  value=5.5,  step=0.5,  label="CFG")
                s1_steps   = gr.Number(label="Steps", value=20, precision=0)
            with gr.Row():
                s1_op_w = gr.Slider(0.0, 1.5, value=0.85, step=0.05, label="OpenPose weight")
                s1_cn_w = gr.Slider(0.0, 1.5, value=0.55, step=0.05, label="Canny weight")
            with gr.Row():
                s1_q_btn = gr.Button("Queue SD Stage 1", variant="primary")
                s1_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                s1_r_btn = gr.Button("Refresh", size="sm")
            s1_img = gr.HTML(value=_url_to_img("", 350))
            s1_url = gr.Textbox(label="URL", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 3: SD STAGE 2 — DETAIL PASS
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 3 — SD1.5 Detail Pass", open=False):
            s2_char, s2_major, s2_minor, s2_new_maj, s2_sid, s2_info = _make_picker(_chars)
            gr.Markdown("---")
            s2_src_stage, s2_src_ver, s2_src_url_st, s2_src_info = _make_src_picker(
                ["sd_stage1", "flux", "normalize"], "sd_stage1")
            s2_prompt = gr.Textbox(label="Prompt", lines=3)
            s2_tok    = gr.Textbox(label="", lines=1, interactive=False)
            s2_neg    = gr.Textbox(label="Negative", lines=2,
                                   value="background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw")
            with gr.Row():
                s2_denoise = gr.Slider(0.10, 0.70, value=0.35, step=0.01, label="Denoise")
                s2_cfg     = gr.Slider(1.0, 15.0,  value=7.0,  step=0.5,  label="CFG")
                s2_steps   = gr.Number(label="Steps", value=20, precision=0)
            with gr.Row():
                s2_q_btn = gr.Button("Queue SD Stage 2", variant="primary")
                s2_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                s2_r_btn = gr.Button("Refresh", size="sm")
            s2_img = gr.HTML(value=_url_to_img("", 350))
            s2_url = gr.Textbox(label="URL", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 4: MULTI-VIEW
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 4 — Multi-view Generation", open=False):
            mv_char, mv_major, mv_minor, mv_new_maj, mv_sid, mv_info = _make_picker(_chars)
            gr.Markdown("---")
            # Shared source picker — same source for both side and back
            mv_src_stage, mv_src_ver, mv_src_url_st, mv_src_info = _make_src_picker(
                ["flux", "sd_stage1", "sd_stage2"], "flux")
            with gr.Row():
                mv_denoise = gr.Slider(0.30, 0.70, value=0.45, step=0.01, label="Denoise")
                mv_cfg     = gr.Slider(1.0, 15.0,  value=7.0,  step=0.5,  label="CFG")
            with gr.Row():
                with gr.Column():
                    mv_side_prompt = gr.Textbox(label="Side view prompt", lines=3)
                    mv_side_tok    = gr.Textbox(label="", lines=1, interactive=False)
                    mv_side_btn    = gr.Button("Queue Side View", variant="primary")
                    mv_side_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_side_r      = gr.Button("Refresh", size="sm")
                    mv_side_img    = gr.HTML(value=_url_to_img("", 300))
                    mv_side_sid    = gr.State(None)
                with gr.Column():
                    mv_back_prompt = gr.Textbox(label="Back view prompt", lines=3)
                    mv_back_tok    = gr.Textbox(label="", lines=1, interactive=False)
                    mv_back_btn    = gr.Button("Queue Back View", variant="primary")
                    mv_back_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_back_r      = gr.Button("Refresh", size="sm")
                    mv_back_img    = gr.HTML(value=_url_to_img("", 300))
                    mv_back_sid    = gr.State(None)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 5: TRELLIS 3D
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 5 — TRELLIS 3D Mesh", open=False):
            tr_char, tr_major, tr_minor, tr_new_maj, tr_sid, tr_info = _make_picker(_chars)
            gr.Markdown("---")
            gr.Markdown("**Select source version for each view:**")
            with gr.Row():
                tr_front_stage, tr_front_ver, tr_front_url_st, tr_front_info = _make_src_picker(
                    ["sd_stage2", "flux", "sd_stage1"], "sd_stage2")
            with gr.Row():
                tr_side_stage, tr_side_ver, tr_side_url_st, tr_side_info = _make_src_picker(
                    ["multiview_side", "flux"], "multiview_side")
            with gr.Row():
                tr_back_stage, tr_back_ver, tr_back_url_st, tr_back_info = _make_src_picker(
                    ["multiview_back", "flux"], "multiview_back")
            with gr.Row():
                tr_q_btn = gr.Button("Queue TRELLIS", variant="primary")
                tr_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                tr_r_btn = gr.Button("Refresh", size="sm")
            tr_url = gr.Textbox(label="GLB URL (when done)", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 6: RIG
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 6 — Auto-Rig Pro (CPU)", open=False):
            rg_char, rg_major, rg_minor, rg_new_maj, rg_sid, rg_info = _make_picker(_chars)
            gr.Markdown("---")
            # Source = trellis GLB
            with gr.Row():
                rg_trellis_ver = gr.Dropdown(choices=[], value=None,
                                             label="Trellis source version",
                                             info="default: latest done", scale=2)
                rg_trellis_info = gr.Textbox(label="Source", interactive=False, lines=1, scale=3)
            rg_type = gr.Dropdown(choices=["humanoid", "quadruped", "bird", "fish"],
                                  value="humanoid", label="Character type")
            with gr.Row():
                rg_q_btn = gr.Button("Queue Rig", variant="primary")
                rg_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                rg_r_btn = gr.Button("Refresh", size="sm")
            rg_url = gr.Textbox(label="Rigged GLB URL (when done)", interactive=False)


        # ══════════════════════════════════════════════════════════════════════
        #  EVENT WIRING
        # ══════════════════════════════════════════════════════════════════════

        # ── Token counters ────────────────────────────────────────────────────
        fx_prompt.input(lambda t: _tok(t, False), [fx_prompt], [fx_tok])
        s1_prompt.input(lambda t: _tok(t, True),  [s1_prompt], [s1_tok])
        s2_prompt.input(lambda t: _tok(t, True),  [s2_prompt], [s2_tok])
        mv_side_prompt.input(lambda t: _tok(t, True), [mv_side_prompt], [mv_side_tok])
        mv_back_prompt.input(lambda t: _tok(t, True), [mv_back_prompt], [mv_back_tok])

        # ── Global: Refresh char list ─────────────────────────────────────────
        def _do_refresh():
            chars = _list_chars()
            upd   = gr.update(choices=chars, value=(chars[0] if chars else None))
            return [upd] * 8

        g_refresh_btn.click(_do_refresh, [],
                            [g_char, fx_char, nm_char, s1_char,
                             s2_char, mv_char, tr_char, rg_char])

        # ── Global: Create New Character ──────────────────────────────────────
        def _do_create(label):
            label = (label or "").strip()
            if not label:
                return [gr.update()] * 8 + ["Enter a character label first."]
            ok    = create_character(_db(), label)
            if not ok:
                return [gr.update()] * 8 + [f"❌ Failed to save '{label}' — check MongoDB connection."]
            chars = _list_chars()
            upd   = gr.update(choices=chars, value=label)
            return [upd] * 8 + [f"✓ Created '{label}'. Select it in any stage below and click ⬇ Prefill All Stages."]

        g_create_btn.click(_do_create, [g_new_char_input],
                           [g_char, fx_char, nm_char, s1_char,
                            s2_char, mv_char, tr_char, rg_char, g_create_info])

        # ── Global: Prefill All Stages ────────────────────────────────────────
        # In Gradio 5, gr.update(value=X) does NOT trigger .change handlers,
        # so we must return ALL stage data directly — no cascading events.
        def _do_prefill(char):
            chars   = _list_chars()
            char_upd = gr.update(choices=chars, value=char) if char else gr.update()

            def _load(stage_name, extract_fn):
                """Return (major_upd, minor_upd, sid, info, *stage_data)."""
                if not char:
                    return gr.update(), gr.update(), None, "", *extract_fn({})
                majors = _list_majors(char, stage_name)
                m      = majors[-1]
                minors = _list_minors(char, stage_name, m)
                n      = minors[-1]
                sid, info = _resolve_run(char, stage_name, m, n)
                run       = _get_run_doc(char, stage_name, m, n)
                return (gr.update(choices=majors, value=m),
                        gr.update(choices=minors, value=n),
                        sid, info, *extract_fn(run))

            # MV is special — shared major/minor controls two sub-stages
            if char:
                mv_majors = _list_majors(char, "multiview_side")
                mv_m      = mv_majors[-1]
                mv_minors = _list_minors(char, "multiview_side", mv_m)
                mv_n      = mv_minors[-1]
                mv_st     = _mv_state(char, mv_m, mv_n)
            else:
                mv_majors, mv_m, mv_minors, mv_n = [1], 1, [0], 0
                mv_st = (None, None, "", "", 0.45, 7.0,
                         "idle", _url_to_img("", 300), "", "idle", _url_to_img("", 300))

            return [
                # 7 char dropdowns
                char_upd, char_upd, char_upd, char_upd, char_upd, char_upd, char_upd,
                # flux  (major, minor, sid, info + 9 data fields = 13)
                *_load("flux",        _ex_flux),
                # normalize (8)
                *_load("normalize",   _ex_normalize),
                # sd_stage1 (15)
                *_load("sd_stage1",   _ex_sd1),
                # sd_stage2 (12)
                *_load("sd_stage2",   _ex_sd2),
                # multiview (major, minor + 11 mv_state fields = 13)
                gr.update(choices=mv_majors, value=mv_m),
                gr.update(choices=mv_minors, value=mv_n),
                *mv_st,
                # trellis (6)
                *_load("trellis",     _ex_trellis),
                # rig (7)
                *_load("rig",         _ex_rig),
            ]

        g_prefill_btn.click(_do_prefill, [g_char], [
            # char dropdowns (7)
            fx_char, nm_char, s1_char, s2_char, mv_char, tr_char, rg_char,
            # flux (13)
            fx_major, fx_minor, fx_sid, fx_info,
            fx_prompt, fx_negative, fx_w, fx_h, fx_steps, fx_guid, fx_status, fx_url, fx_img,
            # normalize (8)
            nm_major, nm_minor, nm_sid, nm_info,
            nm_w, nm_h, nm_status, nm_img,
            # sd_stage1 (15)
            s1_major, s1_minor, s1_sid, s1_info,
            s1_prompt, s1_neg, s1_denoise, s1_cfg, s1_steps, s1_op_w, s1_cn_w, s1_cat, s1_status, s1_url, s1_img,
            # sd_stage2 (12)
            s2_major, s2_minor, s2_sid, s2_info,
            s2_prompt, s2_neg, s2_denoise, s2_cfg, s2_steps, s2_status, s2_url, s2_img,
            # multiview (13)
            mv_major, mv_minor,
            mv_side_sid, mv_back_sid, mv_info,
            mv_side_prompt, mv_denoise, mv_cfg, mv_side_status, mv_side_img,
            mv_back_prompt, mv_back_status, mv_back_img,
            # trellis (6)
            tr_major, tr_minor, tr_sid, tr_info, tr_status, tr_url,
            # rig (7)
            rg_major, rg_minor, rg_sid, rg_info, rg_type, rg_status, rg_url,
        ])

        # ── Extract functions per stage ───────────────────────────────────────

        def _ex_flux(run):
            p = run.get("params") or {}
            return [run.get("prompt", ""),
                    run.get("negative", "deformed, extra limbs, text, watermark, blurry, nsfw"),
                    p.get("width", 512), p.get("height", 512),
                    p.get("steps", 4), p.get("guidance_scale", 0.0),
                    run.get("status", "idle"),
                    run.get("image_url", "") or "",
                    _url_to_img(run.get("image_url", "") or "")]

        def _ex_normalize(run):
            p = run.get("params") or {}
            return [p.get("resize_w", 512), p.get("resize_h", 512),
                    run.get("status", "idle"),
                    _url_to_img(run.get("image_url", "") or "", 300)]

        def _ex_sd1(run):
            p = run.get("params") or {}
            return [run.get("prompt", ""),
                    run.get("negative", "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw"),
                    p.get("denoise", 0.20), p.get("cfg", 5.5), p.get("steps", 20),
                    p.get("openpose_weight", 0.85), p.get("canny_weight", 0.55),
                    p.get("category", "humanoid"),
                    run.get("status", "idle"),
                    run.get("image_url", "") or "",
                    _url_to_img(run.get("image_url", "") or "", 350)]

        def _ex_sd2(run):
            p = run.get("params") or {}
            return [run.get("prompt", ""),
                    run.get("negative", "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw"),
                    p.get("denoise", 0.35), p.get("cfg", 7.0), p.get("steps", 20),
                    run.get("status", "idle"),
                    run.get("image_url", "") or "",
                    _url_to_img(run.get("image_url", "") or "", 350)]

        def _ex_trellis(run):
            return [run.get("status", "idle"), run.get("image_url", "") or ""]

        def _ex_rig(run):
            p = run.get("params") or {}
            return [p.get("char_type", "humanoid"),
                    run.get("status", "idle"),
                    run.get("image_url", "") or ""]

        # ── Wire per-stage pickers ─────────────────────────────────────────────

        _wire_picker("flux", fx_char, fx_major, fx_minor, fx_new_maj, fx_sid, fx_info,
                     [fx_prompt, fx_negative, fx_w, fx_h, fx_steps, fx_guid,
                      fx_status, fx_url, fx_img],
                     _ex_flux)

        _wire_picker("normalize", nm_char, nm_major, nm_minor, nm_new_maj, nm_sid, nm_info,
                     [nm_w, nm_h, nm_status, nm_img],
                     _ex_normalize)

        _wire_picker("sd_stage1", s1_char, s1_major, s1_minor, s1_new_maj, s1_sid, s1_info,
                     [s1_prompt, s1_neg, s1_denoise, s1_cfg, s1_steps,
                      s1_op_w, s1_cn_w, s1_cat, s1_status, s1_url, s1_img],
                     _ex_sd1)

        _wire_picker("sd_stage2", s2_char, s2_major, s2_minor, s2_new_maj, s2_sid, s2_info,
                     [s2_prompt, s2_neg, s2_denoise, s2_cfg, s2_steps,
                      s2_status, s2_url, s2_img],
                     _ex_sd2)

        _wire_picker("trellis", tr_char, tr_major, tr_minor, tr_new_maj, tr_sid, tr_info,
                     [tr_status, tr_url],
                     _ex_trellis)

        _wire_picker("rig", rg_char, rg_major, rg_minor, rg_new_maj, rg_sid, rg_info,
                     [rg_type, rg_status, rg_url],
                     _ex_rig)

        # ── Wire Multiview (custom — controls both side + back sub-stages) ─────

        def _mv_state(char, major, minor):
            """Return all state fields for the multiview shared picker."""
            sid_s, _ = _resolve_run(char, "multiview_side", int(major), int(minor))
            sid_b, _ = _resolve_run(char, "multiview_back", int(major), int(minor))
            info     = (f"Side: {sid_s[:8] if sid_s else 'none'}  "
                        f"Back: {sid_b[:8] if sid_b else 'none'}")
            run_s = _get_run_doc(char, "multiview_side", int(major), int(minor))
            run_b = _get_run_doc(char, "multiview_back", int(major), int(minor))
            ps    = run_s.get("params") or {}
            return (sid_s, sid_b, info,
                    run_s.get("prompt", ""), ps.get("denoise", 0.45), ps.get("cfg", 7.0),
                    run_s.get("status", "idle"), _url_to_img(run_s.get("image_url", "") or "", 300),
                    run_b.get("prompt", ""),
                    run_b.get("status", "idle"), _url_to_img(run_b.get("image_url", "") or "", 300))

        _mv_shared = [mv_side_sid, mv_back_sid, mv_info,
                      mv_side_prompt, mv_denoise, mv_cfg,
                      mv_side_status, mv_side_img,
                      mv_back_prompt,
                      mv_back_status, mv_back_img]   # 11 outputs

        def _on_mv_char(char):
            majors = _list_majors(char, "multiview_side")
            m      = majors[-1]
            minors = _list_minors(char, "multiview_side", m)
            n      = minors[-1]
            return (gr.update(choices=majors, value=m),
                    gr.update(choices=minors, value=n)) + _mv_state(char, m, n)

        def _on_mv_major(char, major):
            if major is None: major = 1
            minors = _list_minors(char, "multiview_side", int(major))
            n      = minors[-1]
            return (gr.update(choices=minors, value=n),) + _mv_state(char, major, n)

        def _on_mv_minor(char, major, minor):
            if major is None: major = 1
            if minor is None: minor = 0
            return _mv_state(char, major, minor)

        def _mv_new_major(char):
            if not char:
                return gr.update(), gr.update(), None, None, "Pick a character first."
            db    = _db()
            new_m = next_stage_major(db, char, "multiview_side")
            sid_s = create_run(db, char, "multiview_side", new_m, 0)
            sid_b = create_run(db, char, "multiview_back", new_m, 0)
            majors = _list_majors(char, "multiview_side")
            return (gr.update(choices=majors, value=new_m),
                    gr.update(choices=[0], value=0),
                    sid_s, sid_b, f"Created v{new_m}.0 for side + back")

        mv_char.change(_on_mv_char,  [mv_char],
                       [mv_major, mv_minor] + _mv_shared)
        mv_major.change(_on_mv_major, [mv_char, mv_major],
                        [mv_minor] + _mv_shared)
        mv_minor.change(_on_mv_minor, [mv_char, mv_major, mv_minor],
                        _mv_shared)
        mv_new_maj.click(_mv_new_major, [mv_char],
                         [mv_major, mv_minor, mv_side_sid, mv_back_sid, mv_info])

        # ── Source picker wiring — refresh on src_stage change ─────────────────
        # Each downstream stage: when src_stage changes → reload version list
        # When src_ver changes → update url state
        # When char changes → reload version list (chained onto existing char handler)

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
        _make_src_wiring(s2_char, s2_src_stage, s2_src_ver, s2_src_url_st, s2_src_info)
        _make_src_wiring(mv_char, mv_src_stage, mv_src_ver, mv_src_url_st, mv_src_info)
        _make_src_wiring(tr_char, tr_front_stage, tr_front_ver, tr_front_url_st, tr_front_info)
        _make_src_wiring(tr_char, tr_side_stage,  tr_side_ver,  tr_side_url_st,  tr_side_info)
        _make_src_wiring(tr_char, tr_back_stage,  tr_back_ver,  tr_back_url_st,  tr_back_info)

        # Rig: source is always trellis, just version picker
        def _refresh_trellis_ver(char):
            versions = _list_done_versions(char, "trellis")
            if versions:
                latest = versions[0]
                return gr.update(choices=versions, value=latest), f"✓ trellis v{latest}"
            return gr.update(choices=[], value=None), "No done trellis runs yet"

        rg_char.change(_refresh_trellis_ver, [rg_char], [rg_trellis_ver, rg_trellis_info])
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

        fx_r_btn.click(
            _refresh_with_img,
            [fx_sid], [fx_status, fx_url, fx_img]
        )

        # Normalize
        nm_btn.click(
            _q_normalize,
            [nm_char, nm_major, nm_minor, nm_w, nm_h, nm_src_stage, nm_src_ver],
            [nm_sid, nm_minor, nm_info, nm_status, nm_img, nm_info2]
        )

        # SD Stage 1
        def _do_q_s1(char, major, minor, p, n, dn, cfg, st, opw, cnw, cat, src_stage, src_url):
            params = {"denoise": float(dn), "cfg": float(cfg), "steps": int(st),
                      "openpose_weight": float(opw), "canny_weight": float(cnw),
                      "category": cat}
            return _q_sd(char, major, minor, "sd_stage1", p, n, params, src_stage, src_url)

        (s1_q_btn.click(
            _do_q_s1,
            [s1_char, s1_major, s1_minor,
             s1_prompt, s1_neg, s1_denoise, s1_cfg, s1_steps,
             s1_op_w, s1_cn_w, s1_cat, s1_src_stage, s1_src_url_st],
            [s1_sid, s1_minor, s1_info, s1_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        s1_r_btn.click(
            lambda sid: _refresh_with_img(sid, 350),
            [s1_sid], [s1_status, s1_url, s1_img]
        )

        # SD Stage 2
        def _do_q_s2(char, major, minor, p, n, dn, cfg, st, src_stage, src_url):
            params = {"denoise": float(dn), "cfg": float(cfg), "steps": int(st)}
            return _q_sd(char, major, minor, "sd_stage2", p, n, params, src_stage, src_url)

        (s2_q_btn.click(
            _do_q_s2,
            [s2_char, s2_major, s2_minor,
             s2_prompt, s2_neg, s2_denoise, s2_cfg, s2_steps, s2_src_stage, s2_src_url_st],
            [s2_sid, s2_minor, s2_info, s2_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        s2_r_btn.click(
            lambda sid: _refresh_with_img(sid, 350),
            [s2_sid], [s2_status, s2_url, s2_img]
        )

        # Multiview Side
        (mv_side_btn.click(
            lambda char, maj, minor, p, dn, cfg, ss, su:
                _q_multiview(char, maj, minor, "side", p, "", dn, cfg, ss, su),
            [mv_char, mv_major, mv_minor, mv_side_prompt, mv_denoise, mv_cfg,
             mv_src_stage, mv_src_url_st],
            [mv_side_sid, mv_minor, mv_info, mv_side_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        mv_side_r.click(
            lambda sid: (_refresh_run(sid)[0], _url_to_img(_refresh_run(sid)[1], 300)),
            [mv_side_sid], [mv_side_status, mv_side_img]
        )

        # Multiview Back
        (mv_back_btn.click(
            lambda char, maj, minor, p, dn, cfg, ss, su:
                _q_multiview(char, maj, minor, "back", p, "", dn, cfg, ss, su),
            [mv_char, mv_major, mv_minor, mv_back_prompt, mv_denoise, mv_cfg,
             mv_src_stage, mv_src_url_st],
            [mv_back_sid, mv_minor, mv_info, mv_back_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        mv_back_r.click(
            lambda sid: (_refresh_run(sid)[0], _url_to_img(_refresh_run(sid)[1], 300)),
            [mv_back_sid], [mv_back_status, mv_back_img]
        )

        # TRELLIS
        (tr_q_btn.click(
            _q_trellis,
            [tr_char, tr_major, tr_minor,
             tr_front_stage, tr_front_url_st,
             tr_side_stage,  tr_side_url_st,
             tr_back_stage,  tr_back_url_st],
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

        # ── Auto-refresh timer ────────────────────────────────────────────────
        _ACTIVE = {"queued", "running"}

        def _tick(fx_s, s1_s, s2_s, ms_s, mb_s, tr_s, rg_s):
            def _r(sid): return _refresh_run(sid)

            fx_st, fx_u  = _r(fx_s)
            s1_st, s1_u  = _r(s1_s)
            s2_st, s2_u  = _r(s2_s)
            ms_st, ms_u  = _r(ms_s)
            mb_st, mb_u  = _r(mb_s)
            tr_st, tr_u  = _r(tr_s)
            rg_st, rg_u  = _r(rg_s)

            still = any(s in _ACTIVE for s in
                        [fx_st, s1_st, s2_st, ms_st, mb_st, tr_st, rg_st])

            return (
                fx_st, fx_u, _url_to_img(fx_u),
                s1_st, s1_u, _url_to_img(s1_u, 350),
                s2_st, s2_u, _url_to_img(s2_u, 350),
                ms_st, _url_to_img(ms_u, 300),
                mb_st, _url_to_img(mb_u, 300),
                tr_st, tr_u,
                rg_st, rg_u,
                gr.Timer(active=still),
            )

        stage_timer.tick(
            _tick,
            [fx_sid, s1_sid, s2_sid, mv_side_sid, mv_back_sid, tr_sid, rg_sid],
            [fx_status, fx_url, fx_img,
             s1_status, s1_url, s1_img,
             s2_status, s2_url, s2_img,
             mv_side_status, mv_side_img,
             mv_back_status, mv_back_img,
             tr_status, tr_url,
             rg_status, rg_url,
             stage_timer],
        )

    return tab
