"""
generation_studio_page.py — Generation Studio tab (port 7860)
=============================================================
Manual, stage-by-stage image generation for game characters.

Stages:
  0 — Flux concept (text → image, T5 encoder, no token limit)
  1 — Normalize (CPU-side PIL resize, no GPU queue, instant)
  2 — SD1.5 Stage 1 ControlNet pose lock (CLIP 77-token limit)
  3 — SD1.5 Stage 2 detail pass (CLIP 77-token limit)
  4 — Multi-view side + back (SD img2img, CLIP 77-token limit)
  5 — TRELLIS 3D mesh (front + side + back → model_tasks queue)
"""

import json
import os
import sys
import threading
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
    get_db, create_session, get_session, list_sessions, list_versions,
    save_stage_prompts, mark_queued, get_stage_image_url,
    STAGE_NAMES, COLLECTION,
)

# ── Config ────────────────────────────────────────────────────────────────────
# Accept both MONGO_URI / MONGODB_URL (systemd service uses the latter).
MONGO_URI   = (os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")
               or "mongodb://kartik:Kartikg421@localhost:27017/?authSource=admin")
MONGO_DB    = os.getenv("MONGO_DB") or os.getenv("MONGODB_DB_NAME") or "World_builder"

# Redis host: prefer parsing CELERY_BROKER_URL, then env var, then localhost.
def _resolve_redis():
    broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_BROKER_URL")
    if broker:
        from urllib.parse import urlparse
        p = urlparse(broker)
        return (p.hostname or "localhost"), (p.port or 6379)
    return os.getenv("REDIS_HOST", "localhost"), int(os.getenv("REDIS_PORT", 6379))

REDIS_HOST, REDIS_PORT = _resolve_redis()
S3_BUCKET   = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET", "sparkassets-us")
S3_REGION   = os.getenv("AWS_REGION", "us-east-1")
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

REDIS_QUEUE_MANUAL = "manual_gen_tasks"
REDIS_QUEUE_MODEL  = "model_tasks"

# GPU instance — IAM role on the CPU instance handles auth (no explicit keys needed)
GPU_INSTANCE_ID = os.getenv("GPU_INSTANCE_ID", "i-0d6b9d6d34ccc053d")
GPU_REGION      = os.getenv("GPU_REGION", "us-east-1")


# ── MongoDB helper ────────────────────────────────────────────────────────────

def _db():
    return get_db(MONGO_URI, MONGO_DB)


# ── GPU start (fire-and-forget, uses IAM role — no credentials in code) ───────

def _ensure_gpu_running_bg():
    """No-op — GPU instance management is manual. Start the GPU from AWS console before generating."""
    pass


# ── Redis push ────────────────────────────────────────────────────────────────

def _push_task(payload: dict, queue: str = REDIS_QUEUE_MANUAL,
               check_gpu: bool = True) -> str:
    """Push task to Redis. Fires GPU start check in background if check_gpu=True."""
    import redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    payload.setdefault("task_id", str(uuid.uuid4()))
    payload.setdefault("timestamp", time.time())
    r.rpush(queue, json.dumps(payload))
    if check_gpu:
        _ensure_gpu_running_bg()
    return payload["task_id"]


# ── Token counter ─────────────────────────────────────────────────────────────

def count_tokens(text: str, is_sd: bool) -> str:
    words    = len(text.split()) if text.strip() else 0
    clip_est = int(words * 1.3)
    if not is_sd:
        return f"Words: {words}  |  Flux T5 encoder — no token limit"
    color = "⚠️" if clip_est > 77 else "✓"
    return (
        f"Est. CLIP tokens: ~{clip_est}/77  {color}"
        f"{'  TRIM NEEDED' if clip_est > 77 else ' OK'}"
    )


def _count_flux(text: str) -> str:
    return count_tokens(text, is_sd=False)


def _count_sd(text: str) -> str:
    return count_tokens(text, is_sd=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Backend logic
# ══════════════════════════════════════════════════════════════════════════════

def _get_or_create_session(char_label: str, version: str) -> tuple:
    """Get existing session — returns (None, msg) if no character selected.
    Only creates a new session when explicitly requested via 'Create' / 'New Version'.
    """
    if not char_label:
        return None, "No asset selected. Pick one from 'Existing Asset' or create one in 'New Asset'."
    try:
        db   = _db()
        docs = list(db[COLLECTION].find(
            {"char_label": char_label, "version": version},
            {"_id": 1, "stages": 1},
        ).sort("created_at", -1).limit(1))
        if docs:
            session_id = docs[0]["_id"]
            stages     = docs[0].get("stages", {})
            done_count = sum(1 for s in stages.values() if s.get("status") == "done")
            return session_id, f"{session_id}  [{done_count}/{len(STAGE_NAMES)} stages done]"
        return None, f"No session for {char_label}/{version} — use 'Create' or '+ New Version'."
    except Exception as exc:
        return None, f"ERROR: {exc}"


def _load_session_state(char_label: str, version: str) -> dict:
    """Load all stage prompts, params, statuses, image_urls from MongoDB.

    Returns a flat dict with keys consumed by the Gradio wiring below.
    """
    defaults = {
        # flux
        "flux_prompt":     "",
        "flux_negative":   "deformed, extra limbs, text, watermark, blurry, nsfw",
        "flux_width":      768,
        "flux_height":     1024,
        "flux_steps":      4,
        "flux_guidance":   0.0,
        "flux_status":     "idle",
        "flux_image_url":  "",
        # normalize
        "norm_w":          512,
        "norm_h":          512,
        "norm_status":     "idle",
        "norm_image_url":  "",
        # sd_stage1
        "sd1_prompt":      "",
        "sd1_negative":    "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
        "sd1_denoise":     0.20,
        "sd1_cfg":         5.5,
        "sd1_steps":       20,
        "sd1_openpose_w":  0.85,
        "sd1_canny_w":     0.55,
        "sd1_category":    "humanoid",
        "sd1_status":      "idle",
        "sd1_image_url":   "",
        # sd_stage2
        "sd2_prompt":      "",
        "sd2_negative":    "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw",
        "sd2_denoise":     0.35,
        "sd2_cfg":         7.0,
        "sd2_steps":       20,
        "sd2_status":      "idle",
        "sd2_image_url":   "",
        # multiview
        "mv_side_prompt":  "",
        "mv_back_prompt":  "",
        "mv_denoise":      0.45,
        "mv_cfg":          7.0,
        "mv_side_status":  "idle",
        "mv_back_status":  "idle",
        "mv_side_url":     "",
        "mv_back_url":     "",
        # trellis
        "trellis_status":  "idle",
        "trellis_url":     "",
    }
    try:
        db   = _db()
        docs = list(db[COLLECTION].find(
            {"char_label": char_label, "version": version},
        ).sort("created_at", -1).limit(1))
        if not docs:
            return defaults
        doc    = docs[0]
        stages = doc.get("stages", {})

        def _s(stage, key, fallback=None):
            return (stages.get(stage) or {}).get(key, fallback)

        def _p(stage, key, fallback=None):
            return ((stages.get(stage) or {}).get("params") or {}).get(key, fallback)

        return {
            # flux
            "flux_prompt":    _s("flux", "prompt", ""),
            "flux_negative":  _s("flux", "negative", defaults["flux_negative"]),
            "flux_width":     _p("flux", "width",          768),
            "flux_height":    _p("flux", "height",         1024),
            "flux_steps":     _p("flux", "steps",          4),
            "flux_guidance":  _p("flux", "guidance_scale", 0.0),
            "flux_status":    _s("flux", "status",         "idle"),
            "flux_image_url": _s("flux", "image_url",      ""),
            # normalize
            "norm_w":         _s("normalize", "resize_w",  512),
            "norm_h":         _s("normalize", "resize_h",  512),
            "norm_status":    _s("normalize", "status",    "idle"),
            "norm_image_url": _s("normalize", "image_url", ""),
            # sd_stage1
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
            # sd_stage2
            "sd2_prompt":     _s("sd_stage2", "prompt",   ""),
            "sd2_negative":   _s("sd_stage2", "negative", defaults["sd2_negative"]),
            "sd2_denoise":    _p("sd_stage2", "denoise", 0.35),
            "sd2_cfg":        _p("sd_stage2", "cfg",     7.0),
            "sd2_steps":      _p("sd_stage2", "steps",   20),
            "sd2_status":     _s("sd_stage2", "status",  "idle"),
            "sd2_image_url":  _s("sd_stage2", "image_url", ""),
            # multiview
            "mv_side_prompt": _s("multiview_side", "prompt", ""),
            "mv_back_prompt": _s("multiview_back", "prompt", ""),
            "mv_denoise":     _p("multiview_side", "denoise", 0.45),
            "mv_cfg":         _p("multiview_side", "cfg",     7.0),
            "mv_side_status": _s("multiview_side", "status",    "idle"),
            "mv_back_status": _s("multiview_back", "status",    "idle"),
            "mv_side_url":    _s("multiview_side", "image_url", ""),
            "mv_back_url":    _s("multiview_back", "image_url", ""),
            # trellis
            "trellis_status": _s("trellis", "status",    "idle"),
            "trellis_url":    _s("trellis", "image_url", ""),
        }
    except Exception as exc:
        defaults["flux_status"] = f"ERROR loading: {exc}"
        return defaults


def _save_all_prompts(
    session_id,
    flux_prompt, flux_neg, flux_w, flux_h, flux_steps, flux_guidance,
    sd1_prompt, sd1_neg, sd1_denoise, sd1_cfg, sd1_category, sd1_steps,
    sd1_openpose_w, sd1_canny_w,
    sd2_prompt, sd2_neg, sd2_denoise, sd2_cfg, sd2_steps,
    mv_side_prompt, mv_back_prompt, mv_denoise, mv_cfg,
) -> str:
    if not session_id:
        return "No session loaded. Use 'Load Session' first."
    try:
        db = _db()
        save_stage_prompts(db, session_id, "flux",
                           flux_prompt, flux_neg,
                           {"width": int(flux_w), "height": int(flux_h),
                            "steps": int(flux_steps), "guidance_scale": float(flux_guidance)})
        save_stage_prompts(db, session_id, "sd_stage1",
                           sd1_prompt, sd1_neg,
                           {"denoise": float(sd1_denoise), "cfg": float(sd1_cfg),
                            "steps": int(sd1_steps), "category": sd1_category,
                            "openpose_weight": float(sd1_openpose_w),
                            "canny_weight": float(sd1_canny_w)})
        save_stage_prompts(db, session_id, "sd_stage2",
                           sd2_prompt, sd2_neg,
                           {"denoise": float(sd2_denoise), "cfg": float(sd2_cfg),
                            "steps": int(sd2_steps)})
        save_stage_prompts(db, session_id, "multiview_side",
                           mv_side_prompt, "",
                           {"denoise": float(mv_denoise), "cfg": float(mv_cfg), "steps": 20})
        save_stage_prompts(db, session_id, "multiview_back",
                           mv_back_prompt, "",
                           {"denoise": float(mv_denoise), "cfg": float(mv_cfg), "steps": 20})
        return f"Saved all prompts for session {session_id}"
    except Exception as exc:
        return f"ERROR: {exc}"


def _queue_flux(session_id, prompt, negative, width, height, steps, guidance) -> tuple:
    """Queue Flux task. Returns (task_id_or_error, status_msg)."""
    if not session_id:
        return "", "No session loaded. Click 'Load Session' first."
    if not prompt.strip():
        return "", "Prompt is empty — please enter a prompt."
    try:
        db = _db()
        save_stage_prompts(db, session_id, "flux", prompt, negative,
                           {"width": int(width), "height": int(height),
                            "steps": int(steps), "guidance_scale": float(guidance)})
        payload = {
            "type":       "flux",
            "session_id": session_id,
            "stage":      "flux",
            "prompt":     prompt,
            "negative":   negative,
            "params": {
                "width":          int(width),
                "height":         int(height),
                "steps":          int(steps),
                "guidance_scale": float(guidance),
            },
        }
        task_id = _push_task(payload, check_gpu=True)
        mark_queued(db, session_id, "flux", task_id)
        return task_id, f"queued ✓  task_id={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _run_normalize_cpu(session_id: str, resize_w: int, resize_h: int,
                       input_stage: str) -> tuple:
    """Run normalize on CPU. Download from S3, resize with PIL, re-upload.
    Returns (status_str, image_url, info_msg).
    """
    if not session_id:
        return "No session loaded.", "", ""
    try:
        import boto3
        from PIL import Image as PILImage

        db = _db()
        src_url = get_stage_image_url(db, session_id, input_stage)
        if not src_url:
            return "error", "", f"Source stage '{input_stage}' has no image yet."

        # Download
        import urllib.request
        with urllib.request.urlopen(src_url) as resp:
            img_bytes = resp.read()
        img = PILImage.open(BytesIO(img_bytes)).convert("RGB")

        # Resize
        w, h = int(resize_w), int(resize_h)
        img  = img.resize((w, h), PILImage.LANCZOS)

        # Re-upload to S3
        s3_key = f"manual_gen/{session_id}/normalize_{w}x{h}.png"
        buf    = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Use IAM role if no explicit keys — same pattern as aws_service.py
        _creds = {}
        _key = os.getenv("AWS_ACCESS_KEY_ID")
        if _key:
            _creds = {
                "aws_access_key_id":     _key,
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "aws_session_token":     os.getenv("AWS_SESSION_TOKEN"),
            }
        s3 = boto3.client("s3", region_name=S3_REGION, **_creds)
        s3.upload_fileobj(buf, S3_BUCKET, s3_key,
                          ExtraArgs={"ContentType": "image/png"})
        image_url = f"{S3_BASE_URL}/{s3_key}"

        # Persist to MongoDB
        from lib.manual_gen_schema import update_stage
        update_stage(db, session_id, "normalize", {
            "status":    "done",
            "resize_w":  w,
            "resize_h":  h,
            "input_stage": input_stage,
            "image_url": image_url,
            "s3_key":    s3_key,
            "error":     None,
        })
        return "done", image_url, f"Resized {input_stage} output → {w}x{h}  uploaded: {s3_key}"
    except Exception as exc:
        return f"error: {exc}", "", str(exc)


def _queue_sd(session_id, stage, prompt, negative, params: dict,
              input_source: str) -> tuple:
    """Queue sd_stage1 or sd_stage2 task. Returns (task_id, status_msg)."""
    if not session_id:
        return "", "No session loaded. Click 'Load Session' first."
    if not prompt.strip():
        return "", "Prompt is empty — please enter a prompt."
    try:
        db = _db()
        save_stage_prompts(db, session_id, stage, prompt, negative, params)
        # Resolve input image URL
        input_url = get_stage_image_url(db, session_id, input_source)
        if not input_url:
            return "", f"No image found in source stage '{input_source}'. Run that stage first."
        payload = {
            "type":        "sd_stage",
            "session_id":  session_id,
            "stage":       stage,
            "prompt":      prompt,
            "negative":    negative,
            "params":      params,
            "input_stage": input_source,
            "input_url":   input_url,
        }
        task_id = _push_task(payload, check_gpu=True)
        mark_queued(db, session_id, stage, task_id)
        return task_id, f"queued ✓  task_id={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _queue_multiview(session_id, side_or_back: str, prompt, negative,
                     denoise, cfg, input_source: str) -> tuple:
    """Queue multiview_side or multiview_back task. Returns (task_id, status_msg)."""
    stage = f"multiview_{side_or_back}"
    if not session_id:
        return "", "No session loaded. Click 'Load Session' first."
    if not prompt.strip():
        return "", "Prompt is empty — please enter a prompt."
    try:
        db     = _db()
        params = {"denoise": float(denoise), "cfg": float(cfg), "steps": 20}
        save_stage_prompts(db, session_id, stage, prompt, negative, params)
        input_url = get_stage_image_url(db, session_id, input_source)
        if not input_url:
            return "", f"No image found in source stage '{input_source}'. Run that stage first."
        payload = {
            "type":        "multiview",
            "session_id":  session_id,
            "stage":       stage,
            "view":        side_or_back,
            "prompt":      prompt,
            "negative":    negative,
            "params":      params,
            "input_stage": input_source,
            "input_url":   input_url,
        }
        task_id = _push_task(payload, check_gpu=True)
        mark_queued(db, session_id, stage, task_id)
        return task_id, f"queued ✓  task_id={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _queue_trellis(session_id, front_src, side_src, back_src) -> tuple:
    """Queue TRELLIS task to model_tasks queue. Returns (task_id, status_msg)."""
    if not session_id:
        return "", "No session loaded. Click 'Load Session' first."
    try:
        db        = _db()
        front_url = get_stage_image_url(db, session_id, front_src) or ""
        side_url  = get_stage_image_url(db, session_id, side_src)  or ""
        back_url  = get_stage_image_url(db, session_id, back_src)  or ""

        if not any([front_url, side_url, back_url]):
            return "", "No source images found. Run at least one image stage first."

        payload = {
            "type":       "trellis",
            "session_id": session_id,
            "stage":      "trellis",
            "input_front": front_url,
            "input_side":  side_url,
            "input_back":  back_url,
            "params": {
                "input_front": front_src,
                "input_side":  side_src,
                "input_back":  back_src,
            },
        }
        task_id = _push_task(payload, queue=REDIS_QUEUE_MODEL, check_gpu=True)
        mark_queued(db, session_id, "trellis", task_id)
        return task_id, f"queued ✓  task_id={task_id[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


def _refresh_stage(session_id: Optional[str], stage: str) -> tuple:
    """Poll MongoDB for stage status. Returns (status_str, image_url)."""
    if not session_id:
        return "no session", ""
    try:
        db  = _db()
        doc = db[COLLECTION].find_one(
            {"_id": session_id},
            {f"stages.{stage}": 1},
        )
        if not doc:
            return "session not found", ""
        sdata = (doc.get("stages") or {}).get(stage, {})
        status = sdata.get("status", "idle")
        url    = sdata.get("image_url") or ""
        err    = sdata.get("error")
        if err and status == "error":
            status = f"error: {err}"
        return status, url or ""
    except Exception as exc:
        return f"ERROR: {exc}", ""


def _list_versions(char_label: str) -> list:
    """Get all versions for a char_label from MongoDB."""
    try:
        return list_versions(_db(), char_label) or ["v1"]
    except Exception:
        return ["v1"]


def _list_chars() -> list:
    """Return all distinct char_labels in the manual_gen_sessions collection."""
    try:
        labels = _db()[COLLECTION].distinct("char_label")
        return sorted([l for l in labels if l]) or []
    except Exception:
        return []


def _next_version(char_label: str) -> str:
    """Compute the next version string (v1 → v2 → v3 …)."""
    versions = _list_versions(char_label)
    nums = []
    for v in versions:
        try:
            nums.append(int(v.lstrip("v")))
        except ValueError:
            pass
    return f"v{max(nums) + 1}" if nums else "v1"


# ══════════════════════════════════════════════════════════════════════════════
#  UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _url_to_img(url: str, height: int = 400) -> str:
    """Return an HTML <img> tag so the *browser* fetches the image directly.

    Gradio 5.x downloads image URLs server-side (SSRF protection) which
    crashes on S3 redirects. Using gr.HTML + <img src> sidesteps this entirely.
    Returns empty-state HTML when url is falsy.
    """
    if not url:
        return (
            "<div style='color:#888;padding:16px;text-align:center;"
            "border:1px dashed #ccc;border-radius:8px;'>"
            "No image yet</div>"
        )
    import time
    sep = "&" if "?" in url else "?"
    cache_busted_url = f"{url}{sep}t={int(time.time()*1000)}"
    return (
        f"<div style='text-align:center;'>"
        f"<img src='{cache_busted_url}' style='max-height:{height}px;max-width:100%;"
        f"border-radius:6px;object-fit:contain;' />"
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main UI entry-point
# ══════════════════════════════════════════════════════════════════════════════

def generation_studio_ui():
    """Build and return the Generation Studio Gradio Blocks tab."""

    with gr.Blocks() as tab:
        gr.Markdown("# 🎨 Generation Studio")
        gr.Markdown(
            "Manual stage-by-stage generation. "
            "Each stage is independent — trigger in any order. "
            "**Queuing any GPU stage automatically starts the GPU instance.**"
        )

        # ── Hidden state ──────────────────────────────────────────────────────
        session_id_state  = gr.State(None)
        # Per-stage session states — each stage can target any asset/version independently
        flux_sid_state   = gr.State(None)
        norm_sid_state   = gr.State(None)
        sd1_sid_state    = gr.State(None)
        sd2_sid_state    = gr.State(None)
        mv_sid_state     = gr.State(None)
        trel_sid_state   = gr.State(None)

        # ── Helper: compact per-stage asset picker (auto-loads on change) ──────
        def _picker(initial):
            with gr.Row():
                c = gr.Dropdown(choices=initial, value=(initial[0] if initial else None),
                                label='Character', scale=2, allow_custom_value=False)
                v = gr.Dropdown(choices=['v1'], value='v1', label='Version', scale=1)
                i = gr.Textbox(label='Session', interactive=False, scale=4, lines=1)
            return c, v, i

        # ══════════════════════════════════════════════════════════════════════
        #  SESSION — pick an existing asset, or create a new one.
        #  Selecting a character/version auto-loads its prompts and images.
        #  Queueing any stage auto-saves prompts (no separate save button).
        # ══════════════════════════════════════════════════════════════════════
        _initial_chars = _list_chars()
        with gr.Accordion("Asset", open=True):
            with gr.Tabs():
                with gr.Tab("Existing Asset"):
                    with gr.Row():
                        char_label = gr.Dropdown(
                            choices=_initial_chars,
                            value=(_initial_chars[0] if _initial_chars else None),
                            label="Character", allow_custom_value=False, scale=2,
                        )
                        version_dd = gr.Dropdown(choices=["v1"], value="v1", label="Version", scale=1)
                        new_version_btn = gr.Button("+ New Version", size="sm")
                        refresh_chars_btn = gr.Button("⟳", size="sm", scale=0)
                with gr.Tab("New Asset"):
                    with gr.Row():
                        new_char_input = gr.Textbox(
                            label="New Character Label",
                            placeholder="e.g. knight_001",
                            scale=3,
                        )
                        create_asset_btn = gr.Button("Create", variant="primary", scale=1)
                    gr.Markdown(
                        "_Creates a fresh `v1` session with empty prompts. "
                        "Once created, switch to the **Existing Asset** tab — "
                        "your new character will be selected automatically._"
                    )
            session_info = gr.Textbox(label="Session", interactive=False, lines=1)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 0: Flux
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 0 — Flux Concept (Text → Image)", open=True):
            gr.Markdown("**Asset:**")
            flux_char, flux_ver, flux_sid_info = _picker(_initial_chars)
            gr.Markdown("---")
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
                flux_width    = gr.Number(label="Width",    value=768,  precision=0)
                flux_height   = gr.Number(label="Height",   value=1024, precision=0)
                flux_steps    = gr.Number(label="Steps",    value=4,    precision=0)
                flux_guidance = gr.Number(label="Guidance", value=0.0)
            with gr.Row():
                flux_queue_btn  = gr.Button("Queue Flux", variant="primary")
                flux_status     = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                flux_refresh_btn = gr.Button("Refresh", size="sm")
            flux_img = gr.HTML(label="Flux Output", value=_url_to_img(""))
            flux_url = gr.Textbox(label="Image URL", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 1: Normalize
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 1 — Normalize (CPU, instant)", open=False):
            gr.Markdown("**Asset:**")
            norm_char, norm_ver, norm_sid_info = _picker(_initial_chars)
            gr.Markdown("---")
            gr.Markdown("Runs immediately on CPU. Resizes Flux output to SD-friendly size.")
            with gr.Row():
                norm_w      = gr.Number(label="Width",  value=512, precision=0)
                norm_h      = gr.Number(label="Height", value=512, precision=0)
                norm_source = gr.Dropdown(
                    choices=["flux"], value="flux", label="Input from", interactive=True,
                )
            with gr.Row():
                norm_run_btn = gr.Button("Run Normalize", variant="primary")
                norm_status  = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
            norm_img  = gr.HTML(label="Normalized Output", value=_url_to_img("", 300))
            norm_info = gr.Textbox(label="Info", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 2: SD Stage 1 — Pose Lock
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 2 — SD1.5 ControlNet Pose Lock", open=False):
            gr.Markdown("**Asset:**")
            sd1_char, sd1_ver, sd1_sid_info = _picker(_initial_chars)
            gr.Markdown("---")
            gr.Markdown(
                "**Very light touch** (denoise 0.20). "
                "Only corrects pose. Flux provides the design."
            )
            with gr.Row():
                sd1_category    = gr.Radio(
                    choices=["humanoid", "quadruped"], value="humanoid", label="Character type",
                )
                sd1_input_source = gr.Dropdown(
                    choices=["flux", "normalize"], value="flux", label="Init image from",
                )
            sd1_prompt = gr.Textbox(
                label="Prompt (keep minimal)", lines=3, interactive=True,
                placeholder="same character, T-pose, arms extended, front view, orthographic, white background",
            )
            sd1_token_info = gr.Textbox(label="", lines=1, interactive=False)
            sd1_negative   = gr.Textbox(
                label="Negative", lines=2, interactive=True,
                value="deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
            )
            with gr.Row():
                sd1_denoise = gr.Slider(0.05, 0.50, value=0.20, step=0.01, label="Denoise")
                sd1_cfg     = gr.Slider(1.0, 15.0,  value=5.5,  step=0.5,  label="CFG")
                sd1_steps   = gr.Number(label="Steps", value=20, precision=0)
            with gr.Row():
                sd1_openpose_w = gr.Slider(0.0, 1.5, value=0.85, step=0.05, label="OpenPose weight (humanoid)")
                sd1_canny_w    = gr.Slider(0.0, 1.5, value=0.55, step=0.05, label="Canny weight")
            with gr.Row():
                sd1_queue_btn   = gr.Button("Queue SD Stage 1", variant="primary")
                sd1_status      = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                sd1_refresh_btn = gr.Button("Refresh", size="sm")
            sd1_img = gr.HTML(label="Stage 1 Output", value=_url_to_img("", 350))
            sd1_url = gr.Textbox(label="URL", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 3: SD Stage 2 — Detail Pass
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 3 — SD1.5 Detail Pass", open=False):
            gr.Markdown("**Asset:**")
            sd2_char, sd2_ver, sd2_sid_info = _picker(_initial_chars)
            gr.Markdown("---")
            sd2_input_source = gr.Dropdown(
                choices=["sd_stage1", "flux"], value="sd_stage1", label="Init image from",
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

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 4: Multi-view
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 4 — Multi-view Generation", open=False):
            gr.Markdown("**Asset:**")
            mv_char, mv_ver, mv_sid_info = _picker(_initial_chars)
            gr.Markdown("---")
            gr.Markdown(
                "**IMPORTANT**: Use Flux output (not SD) as init for best consistency."
            )
            mv_input_source = gr.Dropdown(
                choices=["flux", "sd_stage1", "sd_stage2"], value="flux",
                label="Init image from",
            )
            with gr.Row():
                with gr.Column():
                    mv_side_prompt = gr.Textbox(
                        label="Side view prompt", lines=3, interactive=True,
                        placeholder="same character, side view, profile...",
                    )
                    mv_side_token = gr.Textbox(label="", lines=1, interactive=False)
                    mv_side_btn   = gr.Button("Queue Side View", variant="primary")
                    mv_side_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_side_refresh = gr.Button("Refresh", size="sm")
                    mv_side_img   = gr.HTML(label="Side View", value=_url_to_img("", 300))
                with gr.Column():
                    mv_back_prompt = gr.Textbox(
                        label="Back view prompt", lines=3, interactive=True,
                        placeholder="same character, back view, rear view...",
                    )
                    mv_back_token = gr.Textbox(label="", lines=1, interactive=False)
                    mv_back_btn   = gr.Button("Queue Back View", variant="primary")
                    mv_back_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_back_refresh = gr.Button("Refresh", size="sm")
                    mv_back_img   = gr.HTML(label="Back View", value=_url_to_img("", 300))
            mv_denoise = gr.Slider(0.30, 0.70, value=0.45, step=0.01, label="Denoise (both views)")
            mv_cfg     = gr.Slider(1.0, 15.0,  value=7.0,  step=0.5,  label="CFG (both views)")

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 5: TRELLIS 3D
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 5 — TRELLIS 3D Mesh", open=False):
            gr.Markdown("**Asset:**")
            trel_char, trel_ver, trel_sid_info = _picker(_initial_chars)
            gr.Markdown("---")
            gr.Markdown("Sends front+side+back to TRELLIS worker via `model_tasks` queue.")
            with gr.Row():
                trellis_front_src = gr.Dropdown(
                    choices=["sd_stage2", "flux", "sd_stage1"], value="sd_stage2",
                    label="Front image from",
                )
                trellis_side_src  = gr.Dropdown(
                    choices=["multiview_side", "flux"], value="multiview_side",
                    label="Side image from",
                )
                trellis_back_src  = gr.Dropdown(
                    choices=["multiview_back", "flux"], value="multiview_back",
                    label="Back image from",
                )
            with gr.Row():
                trellis_btn     = gr.Button("Queue TRELLIS", variant="primary")
                trellis_status  = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                trellis_refresh = gr.Button("Refresh", size="sm")
            trellis_url = gr.Textbox(label="3D Mesh URL (when done)", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  EVENT WIRING
        # ══════════════════════════════════════════════════════════════════════

        # ── Live token counts ─────────────────────────────────────────────────
        flux_prompt.input(_count_flux,   [flux_prompt],     [flux_token_info])
        sd1_prompt.input(_count_sd,      [sd1_prompt],      [sd1_token_info])
        sd2_prompt.input(_count_sd,      [sd2_prompt],      [sd2_token_info])
        mv_side_prompt.input(_count_sd,  [mv_side_prompt],  [mv_side_token])
        mv_back_prompt.input(_count_sd,  [mv_back_prompt],  [mv_back_token])

        # ── Session: Load ─────────────────────────────────────────────────────
        def _do_load(char, ver):
            sid, info = _get_or_create_session(char, ver)
            state = _load_session_state(char, ver)
            return (
                sid,
                info,
                # flux
                state["flux_prompt"],
                state["flux_negative"],
                state["flux_width"],
                state["flux_height"],
                state["flux_steps"],
                state["flux_guidance"],
                state["flux_status"],
                state["flux_image_url"],
                _url_to_img(state["flux_image_url"], 400),
                # normalize
                state["norm_w"],
                state["norm_h"],
                state["norm_status"],
                _url_to_img(state["norm_image_url"], 300),
                # sd_stage1
                state["sd1_prompt"],
                state["sd1_negative"],
                state["sd1_denoise"],
                state["sd1_cfg"],
                state["sd1_steps"],
                state["sd1_openpose_w"],
                state["sd1_canny_w"],
                state["sd1_category"],
                state["sd1_status"],
                state["sd1_image_url"],
                _url_to_img(state["sd1_image_url"], 350),
                # sd_stage2
                state["sd2_prompt"],
                state["sd2_negative"],
                state["sd2_denoise"],
                state["sd2_cfg"],
                state["sd2_steps"],
                state["sd2_status"],
                state["sd2_image_url"],
                _url_to_img(state["sd2_image_url"], 350),
                # multiview
                state["mv_side_prompt"],
                state["mv_back_prompt"],
                state["mv_denoise"],
                state["mv_cfg"],
                state["mv_side_status"],
                _url_to_img(state["mv_side_url"], 300),
                state["mv_back_status"],
                _url_to_img(state["mv_back_url"], 300),
                # trellis
                state["trellis_status"],
                state["trellis_url"],
                # stage session_id states (all same as global)
                sid, sid, sid, sid, sid, sid,
                # stage char/ver cascade
                gr.update(value=char), gr.update(choices=_list_versions(char) if char else ['v1'], value=ver),
                gr.update(value=char), gr.update(choices=_list_versions(char) if char else ['v1'], value=ver),
                gr.update(value=char), gr.update(choices=_list_versions(char) if char else ['v1'], value=ver),
                gr.update(value=char), gr.update(choices=_list_versions(char) if char else ['v1'], value=ver),
                gr.update(value=char), gr.update(choices=_list_versions(char) if char else ['v1'], value=ver),
                gr.update(value=char), gr.update(choices=_list_versions(char) if char else ['v1'], value=ver),
            )

        _load_outputs = [
            session_id_state, session_info,
            # flux
            flux_prompt, flux_negative, flux_width, flux_height, flux_steps, flux_guidance,
            flux_status, flux_url, flux_img,
            # normalize
            norm_w, norm_h, norm_status, norm_img,
            # sd_stage1
            sd1_prompt, sd1_negative, sd1_denoise, sd1_cfg, sd1_steps,
            sd1_openpose_w, sd1_canny_w, sd1_category,
            sd1_status, sd1_url, sd1_img,
            # sd_stage2
            sd2_prompt, sd2_negative, sd2_denoise, sd2_cfg, sd2_steps,
            sd2_status, sd2_url, sd2_img,
            # multiview
            mv_side_prompt, mv_back_prompt, mv_denoise, mv_cfg,
            mv_side_status, mv_side_img, mv_back_status, mv_back_img,
            # trellis
            trellis_status, trellis_url,
            # stage session_id states — all set to global session on cascade
            flux_sid_state, norm_sid_state, sd1_sid_state,
            sd2_sid_state, mv_sid_state, trel_sid_state,
            # stage char/ver dropdowns — cascaded from global picker
            flux_char, flux_ver, norm_char, norm_ver,
            sd1_char, sd1_ver, sd2_char, sd2_ver,
            mv_char, mv_ver, trel_char, trel_ver,
        ]

        # ── Auto-load whenever character or version dropdown changes ──────────
        # When char_label changes: refresh versions, pick the latest, then load it.
        def _on_char_change(char):
            versions = _list_versions(char) if char else ["v1"]
            latest   = versions[-1] if versions else "v1"
            return gr.update(choices=versions, value=latest), *_do_load(char, latest)

        char_label.change(
            _on_char_change,
            [char_label],
            [version_dd, *_load_outputs],
        )
        version_dd.change(_do_load, [char_label, version_dd], _load_outputs)

        # ── + New Version: clones char, makes vN+1, loads empty stages ────────
        def _do_new_version(char):
            if not char:
                return gr.update(), None, "Pick a character first."
            new_ver  = _next_version(char)
            create_session(_db(), char, new_ver)
            versions = _list_versions(char)
            # Reload everything for the new (empty) version
            return gr.update(choices=versions, value=new_ver), *_do_load(char, new_ver)

        new_version_btn.click(
            _do_new_version,
            [char_label],
            [version_dd, *_load_outputs],
        )

        # ── Refresh character list (after creating new asset elsewhere) ───────
        def _refresh_chars():
            chars = _list_chars()
            return gr.update(choices=chars, value=(chars[0] if chars else None))

        refresh_chars_btn.click(_refresh_chars, [], [char_label])

        # ── Create New Asset (creates char + v1, switches dropdown to it) ─────
        def _do_create_asset(new_char):
            new_char = (new_char or "").strip()
            if not new_char:
                return gr.update(), gr.update(), "Enter a character label first."
            create_session(_db(), new_char, "v1")
            chars = _list_chars()
            return (
                gr.update(choices=chars, value=new_char),    # char_label dropdown
                gr.update(choices=["v1"], value="v1"),        # version dropdown
                f"Created {new_char} (v1). Switch to 'Existing Asset' tab to start.",
            )

        create_asset_btn.click(
            _do_create_asset,
            [new_char_input],
            [char_label, version_dd, session_info],
        )

        # ── Flux: Queue + Refresh ─────────────────────────────────────────────
        def _do_queue_flux(sid, prompt, negative, width, height, steps, guidance):
            _, status = _queue_flux(sid, prompt, negative, width, height, steps, guidance)
            return status

        flux_queue_btn.click(
            _do_queue_flux,
            [flux_sid_state, flux_prompt, flux_negative, flux_width, flux_height,
             flux_steps, flux_guidance],
            [flux_status],
        )

        def _do_refresh_flux(sid):
            status, url = _refresh_stage(sid, "flux")
            return status, url, _url_to_img(url, 400)

        flux_refresh_btn.click(
            _do_refresh_flux,
            [flux_sid_state],
            [flux_status, flux_url, flux_img],
        )

        # ── Normalize: Run ────────────────────────────────────────────────────
        def _do_normalize(sid, w, h, src):
            status, url, info = _run_normalize_cpu(sid, int(w), int(h), src)
            return status, _url_to_img(url, 300), info

        norm_run_btn.click(
            _do_normalize,
            [norm_sid_state, norm_w, norm_h, norm_source],
            [norm_status, norm_img, norm_info],
        )

        # ── SD Stage 1: Queue + Refresh ───────────────────────────────────────
        def _do_queue_sd1(sid, prompt, negative, denoise, cfg, steps,
                          openpose_w, canny_w, category, input_src):
            params = {
                "denoise":         float(denoise),
                "cfg":             float(cfg),
                "steps":           int(steps),
                "openpose_weight": float(openpose_w),
                "canny_weight":    float(canny_w),
                "category":        category,
            }
            _, status = _queue_sd(sid, "sd_stage1", prompt, negative, params, input_src)
            return status

        sd1_queue_btn.click(
            _do_queue_sd1,
            [sd1_sid_state, sd1_prompt, sd1_negative, sd1_denoise, sd1_cfg, sd1_steps,
             sd1_openpose_w, sd1_canny_w, sd1_category, sd1_input_source],
            [sd1_status],
        )

        def _do_refresh_sd1(sid):
            status, url = _refresh_stage(sid, "sd_stage1")
            return status, url, _url_to_img(url, 350)

        sd1_refresh_btn.click(
            _do_refresh_sd1,
            [sd1_sid_state],
            [sd1_status, sd1_url, sd1_img],
        )

        # ── SD Stage 2: Queue + Refresh ───────────────────────────────────────
        def _do_queue_sd2(sid, prompt, negative, denoise, cfg, steps, input_src):
            params = {
                "denoise": float(denoise),
                "cfg":     float(cfg),
                "steps":   int(steps),
            }
            _, status = _queue_sd(sid, "sd_stage2", prompt, negative, params, input_src)
            return status

        sd2_queue_btn.click(
            _do_queue_sd2,
            [sd2_sid_state, sd2_prompt, sd2_negative, sd2_denoise, sd2_cfg,
             sd2_steps, sd2_input_source],
            [sd2_status],
        )

        def _do_refresh_sd2(sid):
            status, url = _refresh_stage(sid, "sd_stage2")
            return status, url, _url_to_img(url, 350)

        sd2_refresh_btn.click(
            _do_refresh_sd2,
            [sd2_sid_state],
            [sd2_status, sd2_url, sd2_img],
        )

        # ── Multi-view Side: Queue + Refresh ──────────────────────────────────
        def _do_queue_mv_side(sid, prompt, denoise, cfg, input_src):
            _, status = _queue_multiview(sid, "side", prompt, "", denoise, cfg, input_src)
            return status

        mv_side_btn.click(
            _do_queue_mv_side,
            [mv_sid_state, mv_side_prompt, mv_denoise, mv_cfg, mv_input_source],
            [mv_side_status],
        )

        def _do_refresh_mv_side(sid):
            status, url = _refresh_stage(sid, "multiview_side")
            return status, _url_to_img(url, 300)

        mv_side_refresh.click(
            _do_refresh_mv_side,
            [mv_sid_state],
            [mv_side_status, mv_side_img],
        )

        # ── Multi-view Back: Queue + Refresh ──────────────────────────────────
        def _do_queue_mv_back(sid, prompt, denoise, cfg, input_src):
            _, status = _queue_multiview(sid, "back", prompt, "", denoise, cfg, input_src)
            return status

        mv_back_btn.click(
            _do_queue_mv_back,
            [mv_sid_state, mv_back_prompt, mv_denoise, mv_cfg, mv_input_source],
            [mv_back_status],
        )

        def _do_refresh_mv_back(sid):
            status, url = _refresh_stage(sid, "multiview_back")
            return status, _url_to_img(url, 300)

        mv_back_refresh.click(
            _do_refresh_mv_back,
            [mv_sid_state],
            [mv_back_status, mv_back_img],
        )

        # ── TRELLIS: Queue + Refresh ──────────────────────────────────────────
        def _do_queue_trellis(sid, front_src, side_src, back_src):
            _, status = _queue_trellis(sid, front_src, side_src, back_src)
            return status

        trellis_btn.click(
            _do_queue_trellis,
            [trel_sid_state, trellis_front_src, trellis_side_src, trellis_back_src],
            [trellis_status],
        )

        def _do_refresh_trellis(sid):
            status, url = _refresh_stage(sid, "trellis")
            return status, url

        trellis_refresh.click(
            _do_refresh_trellis,
            [trel_sid_state],
            [trellis_status, trellis_url],
        )

        # ══════════════════════════════════════════════════════════════════════
        #  PER-STAGE AUTO-LOAD: each stage's char/ver picker independently loads
        #  that stage's prompts/params/image without affecting other stages.
        # ══════════════════════════════════════════════════════════════════════

        def _stage_load(char, ver, stage_keys):
            """Generic per-stage loader. stage_keys is the list of state-dict
            keys (in output order) to extract from _load_session_state."""
            sid, info = _get_or_create_session(char, ver)
            state = _load_session_state(char, ver)
            return [sid, info] + [state[k] for k in stage_keys]

        # ── Flux per-stage ────────────────────────────────────────────────────
        _flux_keys = ["flux_prompt", "flux_negative", "flux_width", "flux_height",
                      "flux_steps", "flux_guidance", "flux_status", "flux_image_url"]
        _flux_outputs = [flux_sid_state, flux_sid_info,
                         flux_prompt, flux_negative, flux_width, flux_height,
                         flux_steps, flux_guidance, flux_status, flux_url, flux_img]

        def _load_flux(char, ver):
            base = _stage_load(char, ver, _flux_keys)
            base.append(_url_to_img(base[-1], 400))  # flux_img from flux_image_url
            return tuple(base)

        def _on_flux_char(char):
            versions = _list_versions(char) if char else ["v1"]
            latest = versions[-1] if versions else "v1"
            return (gr.update(choices=versions, value=latest), *_load_flux(char, latest))

        flux_char.change(_on_flux_char, [flux_char], [flux_ver, *_flux_outputs])
        flux_ver.change(_load_flux, [flux_char, flux_ver], _flux_outputs)

        # ── Normalize per-stage ───────────────────────────────────────────────
        _norm_keys = ["norm_w", "norm_h", "norm_status", "norm_image_url"]
        _norm_outputs = [norm_sid_state, norm_sid_info,
                         norm_w, norm_h, norm_status, norm_img]

        def _load_norm(char, ver):
            base = _stage_load(char, ver, _norm_keys)
            base[-1] = _url_to_img(base[-1], 300)  # replace url with html
            return tuple(base)

        def _on_norm_char(char):
            versions = _list_versions(char) if char else ["v1"]
            latest = versions[-1] if versions else "v1"
            return (gr.update(choices=versions, value=latest), *_load_norm(char, latest))

        norm_char.change(_on_norm_char, [norm_char], [norm_ver, *_norm_outputs])
        norm_ver.change(_load_norm, [norm_char, norm_ver], _norm_outputs)

        # ── SD Stage 1 per-stage ──────────────────────────────────────────────
        _sd1_keys = ["sd1_prompt", "sd1_negative", "sd1_denoise", "sd1_cfg", "sd1_steps",
                     "sd1_openpose_w", "sd1_canny_w", "sd1_category",
                     "sd1_status", "sd1_image_url"]
        _sd1_outputs = [sd1_sid_state, sd1_sid_info,
                        sd1_prompt, sd1_negative, sd1_denoise, sd1_cfg, sd1_steps,
                        sd1_openpose_w, sd1_canny_w, sd1_category,
                        sd1_status, sd1_url, sd1_img]

        def _load_sd1(char, ver):
            base = _stage_load(char, ver, _sd1_keys)
            base.append(_url_to_img(base[-1], 350))  # sd1_img
            return tuple(base)

        def _on_sd1_char(char):
            versions = _list_versions(char) if char else ["v1"]
            latest = versions[-1] if versions else "v1"
            return (gr.update(choices=versions, value=latest), *_load_sd1(char, latest))

        sd1_char.change(_on_sd1_char, [sd1_char], [sd1_ver, *_sd1_outputs])
        sd1_ver.change(_load_sd1, [sd1_char, sd1_ver], _sd1_outputs)

        # ── SD Stage 2 per-stage ──────────────────────────────────────────────
        _sd2_keys = ["sd2_prompt", "sd2_negative", "sd2_denoise", "sd2_cfg", "sd2_steps",
                     "sd2_status", "sd2_image_url"]
        _sd2_outputs = [sd2_sid_state, sd2_sid_info,
                        sd2_prompt, sd2_negative, sd2_denoise, sd2_cfg, sd2_steps,
                        sd2_status, sd2_url, sd2_img]

        def _load_sd2(char, ver):
            base = _stage_load(char, ver, _sd2_keys)
            base.append(_url_to_img(base[-1], 350))
            return tuple(base)

        def _on_sd2_char(char):
            versions = _list_versions(char) if char else ["v1"]
            latest = versions[-1] if versions else "v1"
            return (gr.update(choices=versions, value=latest), *_load_sd2(char, latest))

        sd2_char.change(_on_sd2_char, [sd2_char], [sd2_ver, *_sd2_outputs])
        sd2_ver.change(_load_sd2, [sd2_char, sd2_ver], _sd2_outputs)

        # ── Multi-view per-stage ──────────────────────────────────────────────
        _mv_outputs = [mv_sid_state, mv_sid_info,
                       mv_side_prompt, mv_back_prompt, mv_denoise, mv_cfg,
                       mv_side_status, mv_side_img, mv_back_status, mv_back_img]

        def _load_mv(char, ver):
            sid, info = _get_or_create_session(char, ver)
            state = _load_session_state(char, ver)
            return (sid, info,
                    state["mv_side_prompt"], state["mv_back_prompt"],
                    state["mv_denoise"], state["mv_cfg"],
                    state["mv_side_status"], _url_to_img(state["mv_side_url"], 300),
                    state["mv_back_status"], _url_to_img(state["mv_back_url"], 300))

        def _on_mv_char(char):
            versions = _list_versions(char) if char else ["v1"]
            latest = versions[-1] if versions else "v1"
            return (gr.update(choices=versions, value=latest), *_load_mv(char, latest))

        mv_char.change(_on_mv_char, [mv_char], [mv_ver, *_mv_outputs])
        mv_ver.change(_load_mv, [mv_char, mv_ver], _mv_outputs)

        # ── TRELLIS per-stage ─────────────────────────────────────────────────
        _trel_outputs = [trel_sid_state, trel_sid_info, trellis_status, trellis_url]

        def _load_trel(char, ver):
            sid, info = _get_or_create_session(char, ver)
            state = _load_session_state(char, ver)
            return (sid, info, state["trellis_status"], state["trellis_url"])

        def _on_trel_char(char):
            versions = _list_versions(char) if char else ["v1"]
            latest = versions[-1] if versions else "v1"
            return (gr.update(choices=versions, value=latest), *_load_trel(char, latest))

        trel_char.change(_on_trel_char, [trel_char], [trel_ver, *_trel_outputs])
        trel_ver.change(_load_trel, [trel_char, trel_ver], _trel_outputs)

        # ── Auto-load session on page open ────────────────────────────────────
        tab.load(_do_load, [char_label, version_dd], _load_outputs)

    return tab
