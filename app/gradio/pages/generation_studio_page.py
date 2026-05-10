"""
Generation Studio — stage-by-stage character image pipeline.

Version scheme
--------------
  major.minor  (both integers stored in MongoDB)

  major  — increments when the user starts a new design direction
           → "New Major Version" button
  minor  — increments for retries / small tweaks on the same intent
           → "New Minor Version" button  (or auto on re-queue of errored stage)

  Display: major dropdown (1, 2, 3 …) + minor dropdown (0, 1, 2 …)

Session management
------------------
  Single global session — all stages work on the same selected session.
  No per-stage pickers. To switch version, use the global major/minor dropdowns.
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
    get_db, create_session, get_session, get_session_for,
    list_chars, list_major_versions, list_minor_versions,
    next_major, next_minor, version_str, _parse_major_minor,
    save_stage_prompts, mark_queued, get_stage_image_url,
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

REDIS_QUEUE_MANUAL = "manual_gen_tasks"


# ── MongoDB / Redis helpers ───────────────────────────────────────────────────

def _db():
    return get_db(MONGO_URI, MONGO_DB)


def _push_task(payload: dict, queue: str = REDIS_QUEUE_MANUAL) -> str:
    import redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                    db=0, decode_responses=True)
    payload.setdefault("task_id", str(uuid.uuid4()))
    payload.setdefault("timestamp", time.time())
    r.rpush(queue, json.dumps(payload))
    return payload["task_id"]


# ── Version helpers (UI layer) ────────────────────────────────────────────────

def _list_chars() -> list[str]:
    try:
        return list_chars(_db()) or []
    except Exception:
        return []


def _list_majors(char: str) -> list[int]:
    try:
        return list_major_versions(_db(), char) or [1]
    except Exception:
        return [1]


def _list_minors(char: str, major: int) -> list[int]:
    try:
        return list_minor_versions(_db(), char, int(major)) or [0]
    except Exception:
        return [0]


def _get_or_load_session(char: str, major: int, minor: int) -> tuple[Optional[str], str]:
    """Return (session_id, info_str) for the given char/major/minor."""
    if not char:
        return None, "No character selected."
    try:
        doc = get_session_for(_db(), char, int(major), int(minor))
        if doc:
            sid = doc["_id"]
            stages = doc.get("stages", {})
            done   = sum(1 for s in stages.values() if s.get("status") == "done")
            return sid, f"Session {sid[:8]}…  [{done}/{len(STAGE_NAMES)} stages done]  v{major}.{minor}"
        return None, f"No session for {char} v{major}.{minor} — create one below."
    except Exception as exc:
        return None, f"ERROR: {exc}"


# ── Session state loader (flat dict → UI values) ──────────────────────────────

def _load_session_state(char: str, major: int, minor: int) -> dict:
    defaults = {
        "flux_prompt":    "",
        "flux_negative":  "deformed, extra limbs, text, watermark, blurry, nsfw",
        "flux_width":     512, "flux_height": 512, "flux_steps": 4, "flux_guidance": 0.0,
        "flux_status":    "idle", "flux_image_url": "",
        "norm_w":         512, "norm_h": 512,
        "norm_status":    "idle", "norm_image_url": "",
        "sd1_prompt":     "",
        "sd1_negative":   "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
        "sd1_denoise":    0.20, "sd1_cfg": 5.5, "sd1_steps": 20,
        "sd1_openpose_w": 0.85, "sd1_canny_w": 0.55, "sd1_category": "humanoid",
        "sd1_status":     "idle", "sd1_image_url": "",
        "sd2_prompt":     "",
        "sd2_negative":   "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw",
        "sd2_denoise":    0.35, "sd2_cfg": 7.0, "sd2_steps": 20,
        "sd2_status":     "idle", "sd2_image_url": "",
        "mv_side_prompt": "", "mv_back_prompt": "",
        "mv_denoise":     0.45, "mv_cfg": 7.0,
        "mv_side_status": "idle", "mv_back_status": "idle",
        "mv_side_url":    "", "mv_back_url": "",
        "trellis_status": "idle", "trellis_url": "",
        "rig_status":     "idle", "rig_url":     "", "rig_char_type": "humanoid",
    }
    try:
        doc = get_session_for(_db(), char, int(major), int(minor))
        if not doc:
            return defaults
        stages = doc.get("stages", {})

        def _s(stage, key, fallback=None):
            return (stages.get(stage) or {}).get(key, fallback)

        def _p(stage, key, fallback=None):
            return ((stages.get(stage) or {}).get("params") or {}).get(key, fallback)

        return {
            "flux_prompt":    _s("flux", "prompt", ""),
            "flux_negative":  _s("flux", "negative", defaults["flux_negative"]),
            "flux_width":     _p("flux", "width",          512),
            "flux_height":    _p("flux", "height",         512),
            "flux_steps":     _p("flux", "steps",          4),
            "flux_guidance":  _p("flux", "guidance_scale", 0.0),
            "flux_status":    _s("flux", "status",         "idle"),
            "flux_image_url": _s("flux", "image_url",      ""),
            "norm_w":         _s("normalize", "resize_w",  512),
            "norm_h":         _s("normalize", "resize_h",  512),
            "norm_status":    _s("normalize", "status",    "idle"),
            "norm_image_url": _s("normalize", "image_url", ""),
            "sd1_prompt":     _s("sd_stage1", "prompt",   ""),
            "sd1_negative":   _s("sd_stage1", "negative", defaults["sd1_negative"]),
            "sd1_denoise":    _p("sd_stage1", "denoise",        0.20),
            "sd1_cfg":        _p("sd_stage1", "cfg",            5.5),
            "sd1_steps":      _p("sd_stage1", "steps",          20),
            "sd1_openpose_w": _p("sd_stage1", "openpose_weight", 0.85),
            "sd1_canny_w":    _p("sd_stage1", "canny_weight",    0.55),
            "sd1_category":   _p("sd_stage1", "category",       "humanoid"),
            "sd1_status":     _s("sd_stage1", "status",         "idle"),
            "sd1_image_url":  _s("sd_stage1", "image_url",      ""),
            "sd2_prompt":     _s("sd_stage2", "prompt",   ""),
            "sd2_negative":   _s("sd_stage2", "negative", defaults["sd2_negative"]),
            "sd2_denoise":    _p("sd_stage2", "denoise", 0.35),
            "sd2_cfg":        _p("sd_stage2", "cfg",     7.0),
            "sd2_steps":      _p("sd_stage2", "steps",   20),
            "sd2_status":     _s("sd_stage2", "status",  "idle"),
            "sd2_image_url":  _s("sd_stage2", "image_url", ""),
            "mv_side_prompt": _s("multiview_side", "prompt", ""),
            "mv_back_prompt": _s("multiview_back", "prompt", ""),
            "mv_denoise":     _p("multiview_side", "denoise", 0.45),
            "mv_cfg":         _p("multiview_side", "cfg",     7.0),
            "mv_side_status": _s("multiview_side", "status",    "idle"),
            "mv_back_status": _s("multiview_back", "status",    "idle"),
            "mv_side_url":    _s("multiview_side", "image_url", ""),
            "mv_back_url":    _s("multiview_back", "image_url", ""),
            "trellis_status": _s("trellis", "status",    "idle"),
            "trellis_url":    _s("trellis", "image_url", ""),
            "rig_status":     _s("rig", "status",    "idle"),
            "rig_url":        _s("rig", "image_url", ""),
            "rig_char_type":  _s("rig", "char_type", "humanoid") or "humanoid",
        }
    except Exception as exc:
        defaults["flux_status"] = f"ERROR: {exc}"
        return defaults


# ── Stage queue functions ─────────────────────────────────────────────────────

def _require_session(session_id: Optional[str]) -> Optional[str]:
    """Return error string if session_id is empty, else None."""
    if not session_id:
        return "No session loaded. Select or create an asset first."
    return None


def _stage_status(session_id: str, stage: str) -> str:
    """Return current status string for a stage (from MongoDB)."""
    try:
        db  = _db()
        doc = db[COLLECTION].find_one({"_id": session_id}, {f"stages.{stage}.status": 1})
        if not doc:
            return "idle"
        return (doc.get("stages") or {}).get(stage, {}).get("status", "idle")
    except Exception:
        return "idle"


def _ensure_session(session_id: Optional[str], char: str, major: int, minor: int):
    """
    Return the session_id to use:
    - If session_id is already set, use it.
    - Otherwise find existing session for (char, major, minor) or create one.
    Never creates a duplicate.
    """
    if session_id:
        return session_id
    db  = _db()
    doc = get_session_for(db, char, int(major), int(minor))
    if doc:
        return doc["_id"]
    return create_session(db, char, int(major), int(minor))


def _queue_flux(session_id, char, major, minor,
                prompt, negative, width, height, steps, guidance) -> tuple:
    """Returns (new_session_id, status_msg)."""
    if not prompt.strip():
        return session_id, "⚠️ Prompt is empty."
    try:
        db  = _db()
        sid = _ensure_session(session_id, char, major, minor)

        # Guard: if flux already done/running, suggest minor version
        cur_st = _stage_status(sid, "flux")
        if cur_st in ("queued", "running"):
            return sid, f"⚠️ Flux is {cur_st}. Wait for it to finish."
        if cur_st == "done":
            return sid, ("⚠️ Flux already done for this version. "
                         "Use 'New Minor Version' to retry or 'New Major Version' for a new design.")

        save_stage_prompts(db, sid, "flux", prompt, negative,
                           {"width": int(width), "height": int(height),
                            "steps": int(steps), "guidance_scale": float(guidance)})
        payload = {
            "type": "flux", "session_id": sid, "stage": "flux",
            "prompt": prompt, "negative": negative,
            "params": {"width": int(width), "height": int(height),
                       "steps": int(steps), "guidance_scale": float(guidance)},
        }
        task_id = _push_task(payload)
        mark_queued(db, sid, "flux", task_id)
        return sid, f"queued ✓  v{major}.{minor}  task={task_id[:8]}…"
    except Exception as exc:
        return session_id, f"ERROR: {exc}"


def _run_normalize(session_id, resize_w, resize_h, input_stage) -> tuple:
    """Returns (status, image_html, info_msg)."""
    err = _require_session(session_id)
    if err:
        return "idle", _url_to_img(""), err
    try:
        import boto3
        from PIL import Image as PILImage

        db      = _db()
        src_url = get_stage_image_url(db, session_id, input_stage)
        if not src_url:
            return "error", _url_to_img(""), f"Stage '{input_stage}' has no image yet."

        import urllib.request
        with urllib.request.urlopen(src_url) as resp:
            img_bytes = resp.read()
        img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
        w, h = int(resize_w), int(resize_h)
        img  = img.resize((w, h), PILImage.LANCZOS)

        s3_key = f"manual_gen/{session_id}/normalize_{w}x{h}.png"
        buf    = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        _creds = {}
        _key = os.getenv("AWS_ACCESS_KEY_ID")
        if _key:
            _creds = {"aws_access_key_id": _key,
                      "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                      "aws_session_token":     os.getenv("AWS_SESSION_TOKEN")}
        s3 = boto3.client("s3", region_name=S3_REGION, **_creds)
        s3.upload_fileobj(buf, S3_BUCKET, s3_key, ExtraArgs={"ContentType": "image/png"})
        image_url = f"{S3_BASE_URL}/{s3_key}"

        from lib.manual_gen_schema import update_stage
        update_stage(db, session_id, "normalize", {
            "status": "done", "resize_w": w, "resize_h": h,
            "input_stage": input_stage, "image_url": image_url, "s3_key": s3_key, "error": None,
        })
        return "done", _url_to_img(image_url, 300), f"Resized {input_stage} → {w}×{h}"
    except Exception as exc:
        return f"error: {exc}", _url_to_img(""), str(exc)


def _queue_sd(session_id, stage, prompt, negative, params, input_source) -> tuple:
    """Returns (task_id, status_msg)."""
    err = _require_session(session_id)
    if err:
        return "", err
    if not prompt.strip():
        return "", "⚠️ Prompt is empty."
    try:
        db        = _db()
        input_url = get_stage_image_url(db, session_id, input_source)
        if not input_url:
            return "", f"No image in '{input_source}'. Run that stage first."
        save_stage_prompts(db, session_id, stage, prompt, negative, params)
        payload = {
            "type": "sd_stage", "session_id": session_id, "stage": stage,
            "prompt": prompt, "negative": negative, "params": params,
            "input_stage": input_source, "input_url": input_url,
        }
        task_id = _push_task(payload)
        mark_queued(db, session_id, stage, task_id)
        return task_id, f"queued ✓  task={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _queue_multiview(session_id, side_or_back, prompt, negative,
                     denoise, cfg, input_source) -> tuple:
    """Returns (task_id, status_msg)."""
    stage = f"multiview_{side_or_back}"
    err   = _require_session(session_id)
    if err:
        return "", err
    if not prompt.strip():
        return "", "⚠️ Prompt is empty."
    try:
        db     = _db()
        params = {"denoise": float(denoise), "cfg": float(cfg), "steps": 20}
        input_url = get_stage_image_url(db, session_id, input_source)
        if not input_url:
            return "", f"No image in '{input_source}'. Run that stage first."
        save_stage_prompts(db, session_id, stage, prompt, "", params)
        payload = {
            "type": "multiview", "session_id": session_id, "stage": stage,
            "view": side_or_back, "prompt": prompt, "negative": negative,
            "params": params, "input_stage": input_source, "input_url": input_url,
        }
        task_id = _push_task(payload)
        mark_queued(db, session_id, stage, task_id)
        return task_id, f"queued ✓  task={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _queue_trellis(session_id, front_src, side_src, back_src) -> tuple:
    err = _require_session(session_id)
    if err:
        return "", err
    try:
        db        = _db()
        front_url = get_stage_image_url(db, session_id, front_src) or ""
        side_url  = get_stage_image_url(db, session_id, side_src)  or ""
        back_url  = get_stage_image_url(db, session_id, back_src)  or ""
        if not front_url:
            return "", f"No image in '{front_src}'. Run that stage first."
        payload = {
            "type": "trellis", "session_id": session_id, "stage": "trellis",
            "input_front": front_url, "input_side": side_url, "input_back": back_url,
        }
        task_id = _push_task(payload)
        mark_queued(db, session_id, "trellis", task_id)
        return task_id, f"queued ✓  task={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _queue_rig(session_id, char_type) -> tuple:
    err = _require_session(session_id)
    if err:
        return "", err
    try:
        db      = _db()
        glb_url = get_stage_image_url(db, session_id, "trellis")
        if not glb_url:
            return "", "Trellis has no GLB yet. Run TRELLIS first."
        payload = {
            "type": "rig", "session_id": session_id, "stage": "rig",
            "char_type": char_type or "humanoid", "input_glb_url": glb_url,
        }
        task_id = _push_task(payload)
        mark_queued(db, session_id, "rig", task_id)
        return task_id, f"queued ✓  task={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _refresh_stage(session_id: Optional[str], stage: str) -> tuple[str, str]:
    """Returns (status_str, image_url)."""
    if not session_id:
        return "idle", ""
    try:
        db  = _db()
        doc = db[COLLECTION].find_one({"_id": session_id}, {f"stages.{stage}": 1})
        if not doc:
            return "idle", ""
        s      = (doc.get("stages") or {}).get(stage, {})
        status = s.get("status", "idle")
        url    = s.get("image_url") or ""
        err    = s.get("error")
        if status == "done":
            status = "✅ done"
        elif status == "error":
            status = f"❌ {err or 'error'}"
        return status, url
    except Exception as exc:
        return f"❌ {exc}", ""


# ── Token counter ─────────────────────────────────────────────────────────────

def count_tokens(text: str, is_sd: bool) -> str:
    words    = len(text.split()) if text.strip() else 0
    clip_est = int(words * 1.3)
    if not is_sd:
        return f"Words: {words}  |  Flux T5 encoder — no token limit"
    color = "⚠️" if clip_est > 77 else "✓"
    return f"Est. CLIP tokens: ~{clip_est}/77  {color}{'  TRIM NEEDED' if clip_est > 77 else ' OK'}"


# ── Image display helper ──────────────────────────────────────────────────────

def _url_to_img(url: str, height: int = 400) -> str:
    if not url:
        return ("<div style='color:#888;padding:16px;text-align:center;"
                "border:1px dashed #ccc;border-radius:8px;'>No image yet</div>")
    sep = "&" if "?" in url else "?"
    bust = f"{url}{sep}t={int(time.time()*1000)}"
    return (f"<div style='text-align:center;'>"
            f"<img src='{bust}' style='max-height:{height}px;max-width:100%;"
            f"border-radius:6px;object-fit:contain;' /></div>")


# ══════════════════════════════════════════════════════════════════════════════
#  Main UI
# ══════════════════════════════════════════════════════════════════════════════

def generation_studio_ui():

    with gr.Blocks() as tab:
        gr.Markdown("# 🎨 Generation Studio")

        # ── Single global session state ───────────────────────────────────────
        session_id_state = gr.State(None)
        stage_timer      = gr.Timer(value=4, active=False)

        # ── Asset / Version management ────────────────────────────────────────
        with gr.Accordion("Asset", open=True):
            _initial_chars = _list_chars()

            with gr.Row():
                char_dd = gr.Dropdown(
                    label="Character",
                    choices=_initial_chars,
                    value=(_initial_chars[0] if _initial_chars else None),
                    allow_custom_value=False,
                    scale=3,
                )
                major_dd = gr.Dropdown(
                    label="Major (design)",
                    choices=[1], value=1,
                    allow_custom_value=False,
                    scale=1,
                )
                minor_dd = gr.Dropdown(
                    label="Minor (retry)",
                    choices=[0], value=0,
                    allow_custom_value=False,
                    scale=1,
                )
                refresh_chars_btn = gr.Button("⟳", size="sm", scale=0)

            with gr.Row():
                new_major_btn = gr.Button("＋ New Major Version", variant="primary", scale=1)
                new_minor_btn = gr.Button("＋ New Minor Version", scale=1)
            gr.Markdown(
                "_**Major** = new design/prompt direction · "
                "**Minor** = retry / small tweak on same design_"
            )

            with gr.Tab("New Character"):
                with gr.Row():
                    new_char_input = gr.Textbox(
                        label="New Character Label",
                        placeholder="e.g. college_student_01",
                        scale=3,
                    )
                    create_char_btn = gr.Button("Create", variant="primary", scale=1)

            session_info = gr.Textbox(label="Active Session", interactive=False, lines=1)

        # ── Stage 0: Flux ─────────────────────────────────────────────────────
        with gr.Accordion("Stage 0 — Flux (Text → Image)", open=True):
            flux_prompt = gr.Textbox(
                label="Prompt", lines=5, interactive=True,
                placeholder="full body character, T-pose, white background...",
            )
            flux_token_info = gr.Textbox(label="", lines=1, interactive=False)
            flux_negative   = gr.Textbox(
                label="Negative", lines=2, interactive=True,
                value="deformed, extra limbs, text, watermark, blurry, nsfw",
            )
            with gr.Row():
                flux_width    = gr.Number(label="Width",    value=512, precision=0)
                flux_height   = gr.Number(label="Height",   value=512, precision=0)
                flux_steps    = gr.Number(label="Steps",    value=4,   precision=0)
                flux_guidance = gr.Number(label="Guidance", value=0.0)
            with gr.Row():
                flux_queue_btn   = gr.Button("Queue Flux", variant="primary")
                flux_status      = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                flux_refresh_btn = gr.Button("Refresh", size="sm")
            flux_img = gr.HTML(label="Flux Output", value=_url_to_img(""))
            flux_url = gr.Textbox(label="Image URL", interactive=False)

        # ── Stage 1: Normalize ────────────────────────────────────────────────
        with gr.Accordion("Stage 1 — Normalize (CPU)", open=False):
            gr.Markdown("Resize Flux output to SD-friendly resolution.")
            with gr.Row():
                norm_w      = gr.Number(label="Width",  value=512, precision=0)
                norm_h      = gr.Number(label="Height", value=512, precision=0)
                norm_source = gr.Dropdown(choices=["flux"], value="flux",
                                          label="Input from", interactive=True)
            with gr.Row():
                norm_run_btn = gr.Button("Run Normalize", variant="primary")
                norm_status  = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
            norm_img  = gr.HTML(label="Output", value=_url_to_img("", 300))
            norm_info = gr.Textbox(label="Info", interactive=False)

        # ── Stage 2: SD Stage 1 ───────────────────────────────────────────────
        with gr.Accordion("Stage 2 — SD1.5 ControlNet Pose Lock", open=False):
            gr.Markdown("Light touch (denoise ~0.20). Corrects pose without destroying Flux design.")
            with gr.Row():
                sd1_category     = gr.Radio(choices=["humanoid", "quadruped"],
                                            value="humanoid", label="Character type")
                sd1_input_source = gr.Dropdown(choices=["flux", "normalize"],
                                               value="flux", label="Init image from")
            sd1_prompt = gr.Textbox(label="Prompt (keep minimal)", lines=3, interactive=True)
            sd1_token_info = gr.Textbox(label="", lines=1, interactive=False)
            sd1_negative   = gr.Textbox(
                label="Negative", lines=2, interactive=True,
                value="deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
            )
            with gr.Row():
                sd1_denoise    = gr.Slider(0.05, 0.50, value=0.20, step=0.01, label="Denoise")
                sd1_cfg        = gr.Slider(1.0, 15.0,  value=5.5,  step=0.5,  label="CFG")
                sd1_steps      = gr.Number(label="Steps", value=20, precision=0)
            with gr.Row():
                sd1_openpose_w = gr.Slider(0.0, 1.5, value=0.85, step=0.05, label="OpenPose weight")
                sd1_canny_w    = gr.Slider(0.0, 1.5, value=0.55, step=0.05, label="Canny weight")
            with gr.Row():
                sd1_queue_btn   = gr.Button("Queue SD Stage 1", variant="primary")
                sd1_status      = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                sd1_refresh_btn = gr.Button("Refresh", size="sm")
            sd1_img = gr.HTML(label="Stage 1 Output", value=_url_to_img("", 350))
            sd1_url = gr.Textbox(label="URL", interactive=False)

        # ── Stage 3: SD Stage 2 ───────────────────────────────────────────────
        with gr.Accordion("Stage 3 — SD1.5 Detail Pass", open=False):
            sd2_input_source = gr.Dropdown(
                choices=["sd_stage1", "flux"], value="sd_stage1", label="Init image from"
            )
            sd2_prompt = gr.Textbox(label="Prompt", lines=3, interactive=True)
            sd2_token_info = gr.Textbox(label="", lines=1, interactive=False)
            sd2_negative   = gr.Textbox(
                label="Negative", lines=2, interactive=True,
                value="background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw",
            )
            with gr.Row():
                sd2_denoise = gr.Slider(0.10, 0.70, value=0.35, step=0.01, label="Denoise")
                sd2_cfg     = gr.Slider(1.0, 15.0,  value=7.0,  step=0.5,  label="CFG")
                sd2_steps   = gr.Number(label="Steps", value=20, precision=0)
            with gr.Row():
                sd2_queue_btn   = gr.Button("Queue SD Stage 2", variant="primary")
                sd2_status      = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                sd2_refresh_btn = gr.Button("Refresh", size="sm")
            sd2_img = gr.HTML(label="Stage 2 Output", value=_url_to_img("", 350))
            sd2_url = gr.Textbox(label="URL", interactive=False)

        # ── Stage 4: Multi-view ───────────────────────────────────────────────
        with gr.Accordion("Stage 4 — Multi-view Generation", open=False):
            gr.Markdown("Use Flux output as init for best consistency.")
            mv_input_source = gr.Dropdown(
                choices=["flux", "sd_stage1", "sd_stage2"], value="flux",
                label="Init image from",
            )
            with gr.Row():
                with gr.Column():
                    mv_side_prompt = gr.Textbox(label="Side view prompt", lines=3)
                    mv_side_token  = gr.Textbox(label="", lines=1, interactive=False)
                    mv_side_btn    = gr.Button("Queue Side View", variant="primary")
                    mv_side_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_side_refresh = gr.Button("Refresh", size="sm")
                    mv_side_img    = gr.HTML(value=_url_to_img("", 300))
                with gr.Column():
                    mv_back_prompt = gr.Textbox(label="Back view prompt", lines=3)
                    mv_back_token  = gr.Textbox(label="", lines=1, interactive=False)
                    mv_back_btn    = gr.Button("Queue Back View", variant="primary")
                    mv_back_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_back_refresh = gr.Button("Refresh", size="sm")
                    mv_back_img    = gr.HTML(value=_url_to_img("", 300))
            with gr.Row():
                mv_denoise = gr.Slider(0.30, 0.70, value=0.45, step=0.01, label="Denoise")
                mv_cfg     = gr.Slider(1.0, 15.0,  value=7.0,  step=0.5,  label="CFG")

        # ── Stage 5: TRELLIS ─────────────────────────────────────────────────
        with gr.Accordion("Stage 5 — TRELLIS 3D Mesh", open=False):
            with gr.Row():
                trellis_front_src = gr.Dropdown(
                    choices=["sd_stage2", "flux", "sd_stage1"], value="sd_stage2",
                    label="Front image from"
                )
                trellis_side_src  = gr.Dropdown(
                    choices=["multiview_side", "flux"], value="multiview_side",
                    label="Side image from"
                )
                trellis_back_src  = gr.Dropdown(
                    choices=["multiview_back", "flux"], value="multiview_back",
                    label="Back image from"
                )
            with gr.Row():
                trellis_btn     = gr.Button("Queue TRELLIS", variant="primary")
                trellis_status  = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                trellis_refresh = gr.Button("Refresh", size="sm")
            trellis_url = gr.Textbox(label="GLB URL (when done)", interactive=False)

        # ── Stage 6: Rig ──────────────────────────────────────────────────────
        with gr.Accordion("Stage 6 — Auto-Rig Pro (CPU)", open=False):
            rig_char_type = gr.Dropdown(
                choices=["humanoid", "quadruped", "bird", "fish"],
                value="humanoid", label="Character type"
            )
            with gr.Row():
                rig_queue_btn   = gr.Button("Queue Rig", variant="primary")
                rig_status      = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                rig_refresh_btn = gr.Button("Refresh", size="sm")
            rig_url = gr.Textbox(label="Rigged GLB URL (when done)", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  LOAD HELPER — maps all outputs
        # ══════════════════════════════════════════════════════════════════════

        def _do_load(char, major, minor):
            """Load session state for (char, major, minor) → update all UI components."""
            major = int(major) if major else 1
            minor = int(minor) if minor else 0
            sid, info = _get_or_load_session(char, major, minor)
            st = _load_session_state(char, major, minor)
            return (
                sid, info,
                # flux
                st["flux_prompt"], st["flux_negative"],
                st["flux_width"], st["flux_height"], st["flux_steps"], st["flux_guidance"],
                st["flux_status"], st["flux_image_url"], _url_to_img(st["flux_image_url"], 400),
                # normalize
                st["norm_w"], st["norm_h"], st["norm_status"], _url_to_img(st["norm_image_url"], 300),
                # sd1
                st["sd1_prompt"], st["sd1_negative"],
                st["sd1_denoise"], st["sd1_cfg"], st["sd1_steps"],
                st["sd1_openpose_w"], st["sd1_canny_w"], st["sd1_category"],
                st["sd1_status"], st["sd1_image_url"], _url_to_img(st["sd1_image_url"], 350),
                # sd2
                st["sd2_prompt"], st["sd2_negative"],
                st["sd2_denoise"], st["sd2_cfg"], st["sd2_steps"],
                st["sd2_status"], st["sd2_image_url"], _url_to_img(st["sd2_image_url"], 350),
                # multiview
                st["mv_side_prompt"], st["mv_back_prompt"], st["mv_denoise"], st["mv_cfg"],
                st["mv_side_status"], _url_to_img(st["mv_side_url"], 300),
                st["mv_back_status"], _url_to_img(st["mv_back_url"], 300),
                # trellis + rig
                st["trellis_status"], st["trellis_url"],
                st["rig_status"], st["rig_url"],
            )

        _load_outputs = [
            session_id_state, session_info,
            flux_prompt, flux_negative, flux_width, flux_height, flux_steps, flux_guidance,
            flux_status, flux_url, flux_img,
            norm_w, norm_h, norm_status, norm_img,
            sd1_prompt, sd1_negative, sd1_denoise, sd1_cfg, sd1_steps,
            sd1_openpose_w, sd1_canny_w, sd1_category,
            sd1_status, sd1_url, sd1_img,
            sd2_prompt, sd2_negative, sd2_denoise, sd2_cfg, sd2_steps,
            sd2_status, sd2_url, sd2_img,
            mv_side_prompt, mv_back_prompt, mv_denoise, mv_cfg,
            mv_side_status, mv_side_img, mv_back_status, mv_back_img,
            trellis_status, trellis_url,
            rig_status, rig_url,
        ]

        # ══════════════════════════════════════════════════════════════════════
        #  VERSION PICKER EVENTS
        # ══════════════════════════════════════════════════════════════════════

        def _on_char_change(char):
            majors = _list_majors(char) if char else [1]
            latest_major = majors[-1]
            minors = _list_minors(char, latest_major) if char else [0]
            latest_minor = minors[-1]
            return (
                gr.update(choices=majors, value=latest_major),
                gr.update(choices=minors, value=latest_minor),
                *_do_load(char, latest_major, latest_minor),
            )

        def _on_major_change(char, major):
            if not char or major is None:
                return gr.update(), *_do_load(char, 1, 0)
            major  = int(major)
            minors = _list_minors(char, major)
            latest = minors[-1]
            return (
                gr.update(choices=minors, value=latest),
                *_do_load(char, major, latest),
            )

        def _on_minor_change(char, major, minor):
            return _do_load(char, int(major) if major else 1, int(minor) if minor else 0)

        char_dd.change(
            _on_char_change, [char_dd],
            [major_dd, minor_dd, *_load_outputs],
        )
        major_dd.change(
            _on_major_change, [char_dd, major_dd],
            [minor_dd, *_load_outputs],
        )
        minor_dd.change(
            _on_minor_change, [char_dd, major_dd, minor_dd],
            _load_outputs,
        )

        # ── New Major Version ─────────────────────────────────────────────────
        def _do_new_major(char):
            if not char:
                return gr.update(), gr.update(), None, "Pick a character first."
            db    = _db()
            new_m = next_major(db, char)
            create_session(db, char, new_m, 0)
            majors = _list_majors(char)
            return (
                gr.update(choices=majors, value=new_m),
                gr.update(choices=[0], value=0),
                *_do_load(char, new_m, 0),
            )

        new_major_btn.click(
            _do_new_major, [char_dd],
            [major_dd, minor_dd, *_load_outputs],
        )

        # ── New Minor Version ─────────────────────────────────────────────────
        def _do_new_minor(char, major):
            if not char or major is None:
                return gr.update(), None, "Pick a character and major version first."
            major = int(major)
            db    = _db()

            # Copy prompts from current session into the new minor session
            cur_doc = get_session_for(db, char, major, max(_list_minors(char, major)))
            new_n   = next_minor(db, char, major)
            new_sid = create_session(db, char, major, new_n)

            # Copy stage prompts/params from previous session
            if cur_doc:
                from lib.manual_gen_schema import update_stage as _us
                for stage_name in STAGE_NAMES:
                    old_s = (cur_doc.get("stages") or {}).get(stage_name, {})
                    if old_s.get("prompt") or old_s.get("params"):
                        _us(db, new_sid, stage_name, {
                            "prompt":   old_s.get("prompt", ""),
                            "negative": old_s.get("negative", ""),
                            "params":   old_s.get("params", {}),
                        })

            minors = _list_minors(char, major)
            return (
                gr.update(choices=minors, value=new_n),
                *_do_load(char, major, new_n),
            )

        new_minor_btn.click(
            _do_new_minor, [char_dd, major_dd],
            [minor_dd, *_load_outputs],
        )

        # ── Refresh character list ────────────────────────────────────────────
        def _do_refresh_chars():
            chars = _list_chars()
            return gr.update(choices=chars, value=(chars[0] if chars else None))

        refresh_chars_btn.click(_do_refresh_chars, [], [char_dd])

        # ── Create new character ──────────────────────────────────────────────
        def _do_create_char(new_char):
            new_char = (new_char or "").strip()
            if not new_char:
                return gr.update(), gr.update(), gr.update(), "Enter a character label first."
            db = _db()
            create_session(db, new_char, 1, 0)
            chars = _list_chars()
            return (
                gr.update(choices=chars, value=new_char),
                gr.update(choices=[1], value=1),
                gr.update(choices=[0], value=0),
                f"Created {new_char} v1.0 — select it in the dropdowns above.",
            )

        create_char_btn.click(
            _do_create_char, [new_char_input],
            [char_dd, major_dd, minor_dd, session_info],
        )

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE EVENT WIRING
        # ══════════════════════════════════════════════════════════════════════

        flux_prompt.input(lambda t: count_tokens(t, False), [flux_prompt], [flux_token_info])
        sd1_prompt.input(lambda t: count_tokens(t, True),  [sd1_prompt],  [sd1_token_info])
        sd2_prompt.input(lambda t: count_tokens(t, True),  [sd2_prompt],  [sd2_token_info])
        mv_side_prompt.input(lambda t: count_tokens(t, True), [mv_side_prompt], [mv_side_token])
        mv_back_prompt.input(lambda t: count_tokens(t, True), [mv_back_prompt], [mv_back_token])

        # ── Flux ──────────────────────────────────────────────────────────────
        def _do_queue_flux(sid, char, major, minor, prompt, neg, w, h, steps, guidance):
            new_sid, status = _queue_flux(sid, char, major, minor, prompt, neg, w, h, steps, guidance)
            return new_sid, status

        (flux_queue_btn.click(
            _do_queue_flux,
            [session_id_state, char_dd, major_dd, minor_dd,
             flux_prompt, flux_negative, flux_width, flux_height, flux_steps, flux_guidance],
            [session_id_state, flux_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        def _do_refresh_flux(sid):
            st, url = _refresh_stage(sid, "flux")
            return st, url, _url_to_img(url, 400)

        flux_refresh_btn.click(_do_refresh_flux, [session_id_state], [flux_status, flux_url, flux_img])

        # ── Normalize ─────────────────────────────────────────────────────────
        norm_run_btn.click(
            _run_normalize,
            [session_id_state, norm_w, norm_h, norm_source],
            [norm_status, norm_img, norm_info],
        )

        # ── SD Stage 1 ────────────────────────────────────────────────────────
        def _do_queue_sd1(sid, prompt, neg, denoise, cfg, steps, op_w, cn_w, cat, src):
            _, status = _queue_sd(sid, "sd_stage1", prompt, neg,
                                  {"denoise": float(denoise), "cfg": float(cfg),
                                   "steps": int(steps), "openpose_weight": float(op_w),
                                   "canny_weight": float(cn_w), "category": cat},
                                  src)
            return status

        (sd1_queue_btn.click(
            _do_queue_sd1,
            [session_id_state, sd1_prompt, sd1_negative, sd1_denoise, sd1_cfg,
             sd1_steps, sd1_openpose_w, sd1_canny_w, sd1_category, sd1_input_source],
            [sd1_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        def _do_refresh_sd1(sid):
            st, url = _refresh_stage(sid, "sd_stage1")
            return st, url, _url_to_img(url, 350)

        sd1_refresh_btn.click(_do_refresh_sd1, [session_id_state], [sd1_status, sd1_url, sd1_img])

        # ── SD Stage 2 ────────────────────────────────────────────────────────
        def _do_queue_sd2(sid, prompt, neg, denoise, cfg, steps, src):
            _, status = _queue_sd(sid, "sd_stage2", prompt, neg,
                                  {"denoise": float(denoise), "cfg": float(cfg),
                                   "steps": int(steps)}, src)
            return status

        (sd2_queue_btn.click(
            _do_queue_sd2,
            [session_id_state, sd2_prompt, sd2_negative, sd2_denoise, sd2_cfg,
             sd2_steps, sd2_input_source],
            [sd2_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        def _do_refresh_sd2(sid):
            st, url = _refresh_stage(sid, "sd_stage2")
            return st, url, _url_to_img(url, 350)

        sd2_refresh_btn.click(_do_refresh_sd2, [session_id_state], [sd2_status, sd2_url, sd2_img])

        # ── Multi-view ────────────────────────────────────────────────────────
        def _do_queue_mv_side(sid, prompt, denoise, cfg, src):
            _, status = _queue_multiview(sid, "side", prompt, "", denoise, cfg, src)
            return status

        (mv_side_btn.click(
            _do_queue_mv_side,
            [session_id_state, mv_side_prompt, mv_denoise, mv_cfg, mv_input_source],
            [mv_side_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        mv_side_refresh.click(
            lambda sid: (_refresh_stage(sid, "multiview_side")[0],
                         _url_to_img(_refresh_stage(sid, "multiview_side")[1], 300)),
            [session_id_state], [mv_side_status, mv_side_img],
        )

        def _do_queue_mv_back(sid, prompt, denoise, cfg, src):
            _, status = _queue_multiview(sid, "back", prompt, "", denoise, cfg, src)
            return status

        (mv_back_btn.click(
            _do_queue_mv_back,
            [session_id_state, mv_back_prompt, mv_denoise, mv_cfg, mv_input_source],
            [mv_back_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        mv_back_refresh.click(
            lambda sid: (_refresh_stage(sid, "multiview_back")[0],
                         _url_to_img(_refresh_stage(sid, "multiview_back")[1], 300)),
            [session_id_state], [mv_back_status, mv_back_img],
        )

        # ── TRELLIS ───────────────────────────────────────────────────────────
        def _do_queue_trellis(sid, front, side, back):
            _, status = _queue_trellis(sid, front, side, back)
            return status

        (trellis_btn.click(
            _do_queue_trellis,
            [session_id_state, trellis_front_src, trellis_side_src, trellis_back_src],
            [trellis_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        trellis_refresh.click(
            lambda sid: _refresh_stage(sid, "trellis"),
            [session_id_state], [trellis_status, trellis_url],
        )

        # ── Rig ───────────────────────────────────────────────────────────────
        def _do_queue_rig(sid, char_type):
            _, status = _queue_rig(sid, char_type)
            return status

        (rig_queue_btn.click(
            _do_queue_rig,
            [session_id_state, rig_char_type],
            [rig_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        rig_refresh_btn.click(
            lambda sid: _refresh_stage(sid, "rig"),
            [session_id_state], [rig_status, rig_url],
        )

        # ══════════════════════════════════════════════════════════════════════
        #  AUTO-REFRESH TIMER
        # ══════════════════════════════════════════════════════════════════════

        _ACTIVE = {"queued", "running"}

        def _tick_all(sid):
            if not sid:
                return (*([gr.update()] * 17), gr.Timer(active=False))

            fx_st, fx_url = _refresh_stage(sid, "flux")
            s1_st, s1_url = _refresh_stage(sid, "sd_stage1")
            s2_st, s2_url = _refresh_stage(sid, "sd_stage2")
            ms_st, ms_url = _refresh_stage(sid, "multiview_side")
            mb_st, mb_url = _refresh_stage(sid, "multiview_back")
            tr_st, tr_url = _refresh_stage(sid, "trellis")
            rg_st, rg_url = _refresh_stage(sid, "rig")

            still = any(s in _ACTIVE for s in [fx_st, s1_st, s2_st, ms_st, mb_st, tr_st, rg_st])

            return (
                fx_st, fx_url, _url_to_img(fx_url, 400),
                s1_st, s1_url, _url_to_img(s1_url, 350),
                s2_st, s2_url, _url_to_img(s2_url, 350),
                ms_st, _url_to_img(ms_url, 300),
                mb_st, _url_to_img(mb_url, 300),
                tr_st, tr_url,
                rg_st, rg_url,
                gr.Timer(active=still),
            )

        stage_timer.tick(
            _tick_all, [session_id_state],
            [flux_status, flux_url, flux_img,
             sd1_status, sd1_url, sd1_img,
             sd2_status, sd2_url, sd2_img,
             mv_side_status, mv_side_img,
             mv_back_status, mv_back_img,
             trellis_status, trellis_url,
             rig_status, rig_url,
             stage_timer],
        )

        # ── Auto-load on page open ────────────────────────────────────────────
        tab.load(_do_load, [char_dd, major_dd, minor_dd], _load_outputs)

    return tab
