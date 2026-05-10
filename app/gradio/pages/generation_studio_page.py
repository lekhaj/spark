"""
Generation Studio — stage-by-stage character image pipeline.

Architecture
------------
  TOP SECTION  — prefill helper only.
    Pick char + major + minor, click "Prefill All Stages" to populate every stage form.
    Also: New Major Version / New Minor Version / New Character buttons.

  EACH STAGE   — fully independent.
    Own char / major / minor picker → own session_id.
    Changing the picker auto-loads that stage's prompts + image from MongoDB.
    Queue sends only that stage to the GPU.
    Stages can target completely different assets/versions simultaneously.

Version scheme
--------------
  major.minor  (ints stored in MongoDB)
  major = new design direction (user-triggered)
  minor = retry / small tweak (auto on re-queue of errored stage, or manual)
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
    get_db, create_session, get_session_for,
    list_chars, list_major_versions, list_minor_versions,
    next_major, next_minor, version_str,
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
    try:
        return list_chars(_db()) or []
    except Exception:
        return []

def _list_majors(char: str) -> list[int]:
    if not char:
        return [1]
    try:
        return list_major_versions(_db(), char) or [1]
    except Exception:
        return [1]

def _list_minors(char: str, major) -> list[int]:
    if not char:
        return [0]
    try:
        return list_minor_versions(_db(), char, int(major)) or [0]
    except Exception:
        return [0]

def _resolve_session(char: str, major, minor) -> tuple[Optional[str], str]:
    """Return (session_id, display_info). Never creates a session."""
    if not char:
        return None, "No character selected."
    try:
        major, minor = int(major), int(minor)
        doc = get_session_for(_db(), char, major, minor)
        if doc:
            sid    = doc["_id"]
            stages = doc.get("stages", {})
            done   = sum(1 for s in stages.values() if s.get("status") == "done")
            return sid, f"{sid[:8]}…  [{done}/{len(STAGE_NAMES)} done]  v{major}.{minor}"
        return None, f"No session for {char} v{major}.{minor}"
    except Exception as exc:
        return None, f"ERROR: {exc}"

def _ensure_session(char: str, major, minor) -> str:
    """Get existing session or create one. Never duplicates."""
    db  = _db()
    doc = get_session_for(db, char, int(major), int(minor))
    if doc:
        return doc["_id"]
    return create_session(db, char, int(major), int(minor))

def _get_stage_state(char: str, major, minor) -> dict:
    """Load all stage data from MongoDB for this (char, major, minor)."""
    D = {  # defaults
        "flux_prompt": "", "flux_negative": "deformed, extra limbs, text, watermark, blurry, nsfw",
        "flux_width": 512, "flux_height": 512, "flux_steps": 4, "flux_guidance": 0.0,
        "flux_status": "idle", "flux_image_url": "",
        "norm_w": 512, "norm_h": 512, "norm_status": "idle", "norm_image_url": "",
        "sd1_prompt": "", "sd1_negative": "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
        "sd1_denoise": 0.20, "sd1_cfg": 5.5, "sd1_steps": 20,
        "sd1_openpose_w": 0.85, "sd1_canny_w": 0.55, "sd1_category": "humanoid",
        "sd1_status": "idle", "sd1_image_url": "",
        "sd2_prompt": "", "sd2_negative": "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw",
        "sd2_denoise": 0.35, "sd2_cfg": 7.0, "sd2_steps": 20,
        "sd2_status": "idle", "sd2_image_url": "",
        "mv_side_prompt": "", "mv_back_prompt": "",
        "mv_denoise": 0.45, "mv_cfg": 7.0,
        "mv_side_status": "idle", "mv_side_url": "",
        "mv_back_status": "idle", "mv_back_url": "",
        "trellis_status": "idle", "trellis_url": "",
        "rig_status": "idle", "rig_url": "", "rig_char_type": "humanoid",
    }
    if not char:
        return D
    try:
        doc = get_session_for(_db(), char, int(major), int(minor))
        if not doc:
            return D
        stages = doc.get("stages", {})
        def _s(st, k, fb=None): return (stages.get(st) or {}).get(k, fb)
        def _p(st, k, fb=None): return ((stages.get(st) or {}).get("params") or {}).get(k, fb)
        return {
            "flux_prompt":    _s("flux","prompt",""),
            "flux_negative":  _s("flux","negative", D["flux_negative"]),
            "flux_width":     _p("flux","width",512),
            "flux_height":    _p("flux","height",512),
            "flux_steps":     _p("flux","steps",4),
            "flux_guidance":  _p("flux","guidance_scale",0.0),
            "flux_status":    _s("flux","status","idle"),
            "flux_image_url": _s("flux","image_url","") or "",
            "norm_w":         _s("normalize","resize_w",512),
            "norm_h":         _s("normalize","resize_h",512),
            "norm_status":    _s("normalize","status","idle"),
            "norm_image_url": _s("normalize","image_url","") or "",
            "sd1_prompt":     _s("sd_stage1","prompt",""),
            "sd1_negative":   _s("sd_stage1","negative", D["sd1_negative"]),
            "sd1_denoise":    _p("sd_stage1","denoise",0.20),
            "sd1_cfg":        _p("sd_stage1","cfg",5.5),
            "sd1_steps":      _p("sd_stage1","steps",20),
            "sd1_openpose_w": _p("sd_stage1","openpose_weight",0.85),
            "sd1_canny_w":    _p("sd_stage1","canny_weight",0.55),
            "sd1_category":   _p("sd_stage1","category","humanoid"),
            "sd1_status":     _s("sd_stage1","status","idle"),
            "sd1_image_url":  _s("sd_stage1","image_url","") or "",
            "sd2_prompt":     _s("sd_stage2","prompt",""),
            "sd2_negative":   _s("sd_stage2","negative", D["sd2_negative"]),
            "sd2_denoise":    _p("sd_stage2","denoise",0.35),
            "sd2_cfg":        _p("sd_stage2","cfg",7.0),
            "sd2_steps":      _p("sd_stage2","steps",20),
            "sd2_status":     _s("sd_stage2","status","idle"),
            "sd2_image_url":  _s("sd_stage2","image_url","") or "",
            "mv_side_prompt": _s("multiview_side","prompt",""),
            "mv_back_prompt": _s("multiview_back","prompt",""),
            "mv_denoise":     _p("multiview_side","denoise",0.45),
            "mv_cfg":         _p("multiview_side","cfg",7.0),
            "mv_side_status": _s("multiview_side","status","idle"),
            "mv_side_url":    _s("multiview_side","image_url","") or "",
            "mv_back_status": _s("multiview_back","status","idle"),
            "mv_back_url":    _s("multiview_back","image_url","") or "",
            "trellis_status": _s("trellis","status","idle"),
            "trellis_url":    _s("trellis","image_url","") or "",
            "rig_status":     _s("rig","status","idle"),
            "rig_url":        _s("rig","image_url","") or "",
            "rig_char_type":  _s("rig","char_type","humanoid") or "humanoid",
        }
    except Exception as exc:
        D["flux_status"] = f"ERROR: {exc}"
        return D

def _refresh_stage(sid: Optional[str], stage: str) -> tuple[str, str]:
    if not sid:
        return "idle", ""
    try:
        doc = _db()[COLLECTION].find_one({"_id": sid}, {f"stages.{stage}": 1})
        if not doc:
            return "idle", ""
        s   = (doc.get("stages") or {}).get(stage, {})
        st  = s.get("status", "idle")
        url = s.get("image_url") or ""
        if st == "done":     st = "✅ done"
        elif st == "error":  st = f"❌ {s.get('error','error')}"
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


# ══════════════════════════════════════════════════════════════════════════════
#  QUEUE FUNCTIONS  (one per stage)
# ══════════════════════════════════════════════════════════════════════════════

def _q_flux(sid, char, major, minor, prompt, neg, w, h, steps, guidance):
    if not prompt.strip():
        return sid, "⚠️ Prompt is empty."
    try:
        sid = _ensure_session(char, major, minor)
        # Guard re-queue
        doc = _db()[COLLECTION].find_one({"_id": sid}, {"stages.flux.status": 1})
        st  = (doc or {}).get("stages", {}).get("flux", {}).get("status", "idle")
        if st in ("queued", "running"):
            return sid, f"⚠️ Flux is {st}."
        if st == "done":
            return sid, "⚠️ Already done — use New Minor Version to retry."
        db = _db()
        save_stage_prompts(db, sid, "flux", prompt, neg,
                           {"width":int(w),"height":int(h),"steps":int(steps),
                            "guidance_scale":float(guidance)})
        tid = _push_task({"type":"flux","session_id":sid,"stage":"flux",
                          "prompt":prompt,"negative":neg,
                          "params":{"width":int(w),"height":int(h),
                                    "steps":int(steps),"guidance_scale":float(guidance)}})
        mark_queued(db, sid, "flux", tid)
        return sid, f"queued ✓  v{major}.{minor}  task={tid[:8]}…"
    except Exception as exc:
        return sid, f"ERROR: {exc}"

def _q_normalize(sid, w, h, src):
    if not sid:
        return "idle", _url_to_img(""), "No session — select asset first."
    try:
        import boto3
        from PIL import Image as PILImage
        db      = _db()
        src_url = get_stage_image_url(db, sid, src)
        if not src_url:
            return "error", _url_to_img(""), f"Stage '{src}' has no image yet."
        import urllib.request
        with urllib.request.urlopen(src_url) as r:
            img = PILImage.open(BytesIO(r.read())).convert("RGB")
        img    = img.resize((int(w), int(h)), PILImage.LANCZOS)
        s3_key = f"manual_gen/{sid}/normalize_{int(w)}x{int(h)}.png"
        buf    = BytesIO(); img.save(buf, "PNG"); buf.seek(0)
        _creds = {}
        if os.getenv("AWS_ACCESS_KEY_ID"):
            _creds = {"aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID"),
                      "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                      "aws_session_token":     os.getenv("AWS_SESSION_TOKEN")}
        boto3.client("s3", region_name=S3_REGION, **_creds).upload_fileobj(
            buf, S3_BUCKET, s3_key, ExtraArgs={"ContentType":"image/png"})
        url = f"{S3_BASE_URL}/{s3_key}"
        from lib.manual_gen_schema import update_stage
        update_stage(db, sid, "normalize",
                     {"status":"done","resize_w":int(w),"resize_h":int(h),
                      "input_stage":src,"image_url":url,"s3_key":s3_key,"error":None})
        return "done", _url_to_img(url, 300), f"→ {int(w)}×{int(h)}"
    except Exception as exc:
        return f"error: {exc}", _url_to_img(""), str(exc)

def _q_sd(sid, stage, prompt, neg, params, src):
    if not sid:   return "", "No session — select asset first."
    if not prompt.strip(): return "", "⚠️ Prompt is empty."
    try:
        db  = _db()
        url = get_stage_image_url(db, sid, src)
        if not url: return "", f"No image in '{src}'. Run that stage first."
        save_stage_prompts(db, sid, stage, prompt, neg, params)
        tid = _push_task({"type":"sd_stage","session_id":sid,"stage":stage,
                          "prompt":prompt,"negative":neg,"params":params,
                          "input_stage":src,"input_url":url})
        mark_queued(db, sid, stage, tid)
        return tid, f"queued ✓  task={tid[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"

def _q_multiview(sid, view, prompt, neg, denoise, cfg, src):
    stage = f"multiview_{view}"
    if not sid:   return "", "No session — select asset first."
    if not prompt.strip(): return "", "⚠️ Prompt is empty."
    try:
        db     = _db()
        params = {"denoise":float(denoise),"cfg":float(cfg),"steps":20}
        url    = get_stage_image_url(db, sid, src)
        if not url: return "", f"No image in '{src}'. Run that stage first."
        save_stage_prompts(db, sid, stage, prompt, "", params)
        tid = _push_task({"type":"multiview","session_id":sid,"stage":stage,
                          "view":view,"prompt":prompt,"negative":neg,"params":params,
                          "input_stage":src,"input_url":url})
        mark_queued(db, sid, stage, tid)
        return tid, f"queued ✓  task={tid[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"

def _q_trellis(sid, front_src, side_src, back_src):
    if not sid: return "", "No session — select asset first."
    try:
        db = _db()
        fu = get_stage_image_url(db, sid, front_src) or ""
        su = get_stage_image_url(db, sid, side_src)  or ""
        bu = get_stage_image_url(db, sid, back_src)  or ""
        if not fu: return "", f"No image in '{front_src}'. Run that stage first."
        tid = _push_task({"type":"trellis","session_id":sid,"stage":"trellis",
                          "input_front":fu,"input_side":su,"input_back":bu})
        mark_queued(db, sid, "trellis", tid)
        return tid, f"queued ✓  task={tid[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"

def _q_rig(sid, char_type):
    if not sid: return "", "No session — select asset first."
    try:
        db  = _db()
        glb = get_stage_image_url(db, sid, "trellis")
        if not glb: return "", "Trellis has no GLB yet."
        tid = _push_task({"type":"rig","session_id":sid,"stage":"rig",
                          "char_type":char_type or "humanoid","input_glb_url":glb})
        mark_queued(db, sid, "rig", tid)
        return tid, f"queued ✓  task={tid[:8]}…"
    except Exception as exc:
        return "", f"ERROR: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS — per-stage picker + event wiring
# ══════════════════════════════════════════════════════════════════════════════

def _make_picker(initial_chars):
    """Build a compact char/major/minor picker row. Returns (char, major, minor, sid_state, info)."""
    with gr.Row():
        p_char  = gr.Dropdown(choices=initial_chars, label="Character",
                              allow_custom_value=False, scale=3)
        p_major = gr.Dropdown(choices=[1], value=1, label="Major", scale=1)
        p_minor = gr.Dropdown(choices=[0], value=0, label="Minor", scale=1)
    p_info = gr.Textbox(label="Session", interactive=False, lines=1)
    p_sid  = gr.State(None)
    return p_char, p_major, p_minor, p_sid, p_info


def _wire_picker(p_char, p_major, p_minor, p_sid, p_info,
                 stage_outputs: list, extract_fn):
    """
    Wire char/major/minor dropdowns so changing any of them:
      1. Refreshes dependent dropdowns
      2. Loads the stage's data from MongoDB
      3. Updates stage_outputs via extract_fn(state_dict) -> list

    extract_fn: takes the full state dict → returns list matching stage_outputs
    """
    def _on_char(char):
        majors  = _list_majors(char)
        m       = majors[-1]
        minors  = _list_minors(char, m)
        n       = minors[-1]
        sid, info = _resolve_session(char, m, n)
        st      = _get_stage_state(char, m, n)
        return [gr.update(choices=majors, value=m),
                gr.update(choices=minors, value=n),
                sid, info] + extract_fn(st)

    def _on_major(char, major):
        if major is None: major = 1
        minors  = _list_minors(char, int(major))
        n       = minors[-1]
        sid, info = _resolve_session(char, int(major), n)
        st      = _get_stage_state(char, int(major), n)
        return [gr.update(choices=minors, value=n),
                sid, info] + extract_fn(st)

    def _on_minor(char, major, minor):
        if major is None: major = 1
        if minor is None: minor = 0
        sid, info = _resolve_session(char, int(major), int(minor))
        st        = _get_stage_state(char, int(major), int(minor))
        return [sid, info] + extract_fn(st)

    p_char.change(_on_char,  [p_char],
                  [p_major, p_minor, p_sid, p_info] + stage_outputs)
    p_major.change(_on_major, [p_char, p_major],
                   [p_minor, p_sid, p_info] + stage_outputs)
    p_minor.change(_on_minor, [p_char, p_major, p_minor],
                   [p_sid, p_info] + stage_outputs)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

def generation_studio_ui():

    with gr.Blocks() as tab:
        gr.Markdown("# 🎨 Generation Studio")

        stage_timer = gr.Timer(value=4, active=False)
        _chars = _list_chars()

        # ══════════════════════════════════════════════════════════════════════
        #  TOP: PREFILL + VERSION MANAGEMENT
        #  Selecting char/major/minor here and clicking "Prefill All Stages"
        #  pushes the data into every stage form below. Stages remain independent.
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("⚙️ Prefill / Version Management", open=True):
            gr.Markdown(
                "_Select a character and version, then click **Prefill All Stages** "
                "to populate every stage form. Each stage can still be changed independently._"
            )
            with gr.Row():
                g_char  = gr.Dropdown(choices=_chars, value=(_chars[0] if _chars else None),
                                      label="Character", allow_custom_value=False, scale=3)
                g_major = gr.Dropdown(choices=[1], value=1, label="Major (design)", scale=1)
                g_minor = gr.Dropdown(choices=[0], value=0, label="Minor (retry)",  scale=1)
                g_refresh_btn = gr.Button("⟳ Refresh List", size="sm", scale=1)

            with gr.Row():
                g_new_major_btn  = gr.Button("＋ New Major Version", variant="primary", scale=1)
                g_new_minor_btn  = gr.Button("＋ New Minor Version", scale=1)
                g_prefill_btn    = gr.Button("⬇ Prefill All Stages", variant="secondary", scale=1)

            with gr.Accordion("New Character", open=False):
                with gr.Row():
                    g_new_char_input = gr.Textbox(label="Character Label",
                                                  placeholder="e.g. knight_01", scale=3)
                    g_create_btn     = gr.Button("Create v1.0", variant="primary", scale=1)

            g_info = gr.Textbox(label="Selected Session", interactive=False, lines=1)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 0: FLUX
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 0 — Flux (Text → Image)", open=True):
            fx_char, fx_major, fx_minor, fx_sid, fx_info = _make_picker(_chars)
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
            nm_char, nm_major, nm_minor, nm_sid, nm_info = _make_picker(_chars)
            gr.Markdown("---")
            with gr.Row():
                nm_w   = gr.Number(label="Width",  value=512, precision=0)
                nm_h   = gr.Number(label="Height", value=512, precision=0)
                nm_src = gr.Dropdown(choices=["flux"], value="flux",
                                     label="Input from", interactive=True)
            with gr.Row():
                nm_btn    = gr.Button("Run Normalize", variant="primary")
                nm_status = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
            nm_img  = gr.HTML(value=_url_to_img("", 300))
            nm_info2= gr.Textbox(label="Info", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 2: SD STAGE 1 — POSE LOCK
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 2 — SD1.5 ControlNet Pose Lock", open=False):
            s1_char, s1_major, s1_minor, s1_sid, s1_info = _make_picker(_chars)
            gr.Markdown("---")
            with gr.Row():
                s1_cat = gr.Radio(choices=["humanoid","quadruped"],
                                  value="humanoid", label="Character type")
                s1_src = gr.Dropdown(choices=["flux","normalize"],
                                     value="flux", label="Init image from")
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
            s2_char, s2_major, s2_minor, s2_sid, s2_info = _make_picker(_chars)
            gr.Markdown("---")
            s2_src    = gr.Dropdown(choices=["sd_stage1","flux"], value="sd_stage1",
                                    label="Init image from")
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
            mv_char, mv_major, mv_minor, mv_sid, mv_info = _make_picker(_chars)
            gr.Markdown("---")
            mv_src = gr.Dropdown(choices=["flux","sd_stage1","sd_stage2"],
                                 value="flux", label="Init image from")
            with gr.Row():
                with gr.Column():
                    mv_side_prompt = gr.Textbox(label="Side view prompt", lines=3)
                    mv_side_tok    = gr.Textbox(label="", lines=1, interactive=False)
                    mv_side_btn    = gr.Button("Queue Side View", variant="primary")
                    mv_side_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_side_r      = gr.Button("Refresh", size="sm")
                    mv_side_img    = gr.HTML(value=_url_to_img("", 300))
                with gr.Column():
                    mv_back_prompt = gr.Textbox(label="Back view prompt", lines=3)
                    mv_back_tok    = gr.Textbox(label="", lines=1, interactive=False)
                    mv_back_btn    = gr.Button("Queue Back View", variant="primary")
                    mv_back_status = gr.Textbox(label="Status", value="idle", interactive=False)
                    mv_back_r      = gr.Button("Refresh", size="sm")
                    mv_back_img    = gr.HTML(value=_url_to_img("", 300))
            with gr.Row():
                mv_denoise = gr.Slider(0.30, 0.70, value=0.45, step=0.01, label="Denoise")
                mv_cfg     = gr.Slider(1.0, 15.0,  value=7.0,  step=0.5,  label="CFG")

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 5: TRELLIS 3D
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 5 — TRELLIS 3D Mesh", open=False):
            tr_char, tr_major, tr_minor, tr_sid, tr_info = _make_picker(_chars)
            gr.Markdown("---")
            with gr.Row():
                tr_front = gr.Dropdown(choices=["sd_stage2","flux","sd_stage1"],
                                       value="sd_stage2", label="Front from")
                tr_side  = gr.Dropdown(choices=["multiview_side","flux"],
                                       value="multiview_side", label="Side from")
                tr_back  = gr.Dropdown(choices=["multiview_back","flux"],
                                       value="multiview_back", label="Back from")
            with gr.Row():
                tr_q_btn = gr.Button("Queue TRELLIS", variant="primary")
                tr_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                tr_r_btn = gr.Button("Refresh", size="sm")
            tr_url = gr.Textbox(label="GLB URL (when done)", interactive=False)

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 6: RIG
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 6 — Auto-Rig Pro (CPU)", open=False):
            rg_char, rg_major, rg_minor, rg_sid, rg_info = _make_picker(_chars)
            gr.Markdown("---")
            rg_type = gr.Dropdown(choices=["humanoid","quadruped","bird","fish"],
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

        # ── Global top section: char/major/minor cascade ──────────────────────
        def _g_on_char(char):
            majors = _list_majors(char)
            m      = majors[-1]
            minors = _list_minors(char, m)
            n      = minors[-1]
            sid, info = _resolve_session(char, m, n)
            return (gr.update(choices=majors, value=m),
                    gr.update(choices=minors, value=n), info)

        def _g_on_major(char, major):
            if major is None: major = 1
            minors = _list_minors(char, int(major))
            n      = minors[-1]
            sid, info = _resolve_session(char, int(major), n)
            return gr.update(choices=minors, value=n), info

        def _g_on_minor(char, major, minor):
            sid, info = _resolve_session(char,
                                         int(major) if major else 1,
                                         int(minor) if minor else 0)
            return info

        g_char.change(_g_on_char,  [g_char],              [g_major, g_minor, g_info])
        g_major.change(_g_on_major,[g_char, g_major],      [g_minor, g_info])
        g_minor.change(_g_on_minor,[g_char, g_major, g_minor], [g_info])

        # ── Global: Refresh char list ─────────────────────────────────────────
        def _do_refresh():
            chars = _list_chars()
            upd   = gr.update(choices=chars, value=(chars[0] if chars else None))
            # refresh all 7 stage pickers too
            return [upd] * 8  # g_char + 7 stage chars

        g_refresh_btn.click(_do_refresh, [],
                            [g_char, fx_char, nm_char, s1_char,
                             s2_char, mv_char, tr_char, rg_char])

        # ── Global: New Major Version ─────────────────────────────────────────
        def _do_new_major(char):
            if not char:
                return gr.update(), gr.update(), "Pick a character first."
            db    = _db()
            new_m = next_major(db, char)
            create_session(db, char, new_m, 0)
            majors = _list_majors(char)
            _, info = _resolve_session(char, new_m, 0)
            return gr.update(choices=majors, value=new_m), gr.update(choices=[0], value=0), info

        g_new_major_btn.click(_do_new_major, [g_char], [g_major, g_minor, g_info])

        # ── Global: New Minor Version ─────────────────────────────────────────
        def _do_new_minor(char, major):
            if not char or major is None:
                return gr.update(), "Pick a character and major first."
            major   = int(major)
            db      = _db()
            cur_doc = get_session_for(db, char, major, max(_list_minors(char, major)))
            new_n   = next_minor(db, char, major)
            new_sid = create_session(db, char, major, new_n)
            # Copy prompts from previous session
            if cur_doc:
                from lib.manual_gen_schema import update_stage as _us
                for sn in STAGE_NAMES:
                    old = (cur_doc.get("stages") or {}).get(sn, {})
                    if old.get("prompt") or old.get("params"):
                        _us(db, new_sid, sn, {"prompt":  old.get("prompt",""),
                                              "negative":old.get("negative",""),
                                              "params":  old.get("params",{})})
            minors = _list_minors(char, major)
            _, info = _resolve_session(char, major, new_n)
            return gr.update(choices=minors, value=new_n), info

        g_new_minor_btn.click(_do_new_minor, [g_char, g_major], [g_minor, g_info])

        # ── Global: Create New Character ──────────────────────────────────────
        def _do_create(label):
            label = (label or "").strip()
            if not label:
                return gr.update(), gr.update(), gr.update(), "Enter a label."
            create_session(_db(), label, 1, 0)
            chars = _list_chars()
            return (gr.update(choices=chars, value=label),
                    gr.update(choices=[1], value=1),
                    gr.update(choices=[0], value=0),
                    f"Created {label} v1.0")

        g_create_btn.click(_do_create, [g_new_char_input],
                           [g_char, g_major, g_minor, g_info])

        # ── Global: Prefill All Stages ────────────────────────────────────────
        # Pushes global char/major/minor into all 7 stage pickers at once,
        # then loads each stage's data.
        def _do_prefill(char, major, minor):
            if not char:
                return [gr.update()] * 56  # 7 stages × 8 outputs each
            major = int(major) if major else 1
            minor = int(minor) if minor else 0
            majors = _list_majors(char)
            minors = _list_minors(char, major)
            maj_upd = gr.update(choices=majors, value=major)
            min_upd = gr.update(choices=minors, value=minor)
            sid, info = _resolve_session(char, major, minor)
            st  = _get_stage_state(char, major, minor)
            c   = gr.update(value=char)

            return [
                # Flux (8)
                c, maj_upd, min_upd, sid, info,
                st["flux_prompt"], st["flux_negative"],
                _url_to_img(st["flux_image_url"]),
                # Normalize (8)
                c, maj_upd, min_upd, sid, info,
                st["norm_w"], st["norm_h"],
                _url_to_img(st["norm_image_url"], 300),
                # SD1 (8)
                c, maj_upd, min_upd, sid, info,
                st["sd1_prompt"], st["sd1_negative"],
                _url_to_img(st["sd1_image_url"], 350),
                # SD2 (8)
                c, maj_upd, min_upd, sid, info,
                st["sd2_prompt"], st["sd2_negative"],
                _url_to_img(st["sd2_image_url"], 350),
                # Multiview (8)
                c, maj_upd, min_upd, sid, info,
                st["mv_side_prompt"], st["mv_back_prompt"],
                _url_to_img(st["mv_side_url"], 300),
                # Trellis (6)
                c, maj_upd, min_upd, sid, info, st["trellis_url"],
                # Rig (7)
                c, maj_upd, min_upd, sid, info,
                st["rig_char_type"], st["rig_url"],
            ]

        g_prefill_btn.click(
            _do_prefill, [g_char, g_major, g_minor],
            [
                fx_char, fx_major, fx_minor, fx_sid, fx_info,
                fx_prompt, fx_negative, fx_img,
                nm_char, nm_major, nm_minor, nm_sid, nm_info,
                nm_w, nm_h, nm_img,
                s1_char, s1_major, s1_minor, s1_sid, s1_info,
                s1_prompt, s1_neg, s1_img,
                s2_char, s2_major, s2_minor, s2_sid, s2_info,
                s2_prompt, s2_neg, s2_img,
                mv_char, mv_major, mv_minor, mv_sid, mv_info,
                mv_side_prompt, mv_back_prompt, mv_side_img,
                tr_char, tr_major, tr_minor, tr_sid, tr_info, tr_url,
                rg_char, rg_major, rg_minor, rg_sid, rg_info,
                rg_type, rg_url,
            ]
        )

        # ── Per-stage picker wiring ───────────────────────────────────────────
        # Each stage independently loads its own data when its picker changes.

        _wire_picker(fx_char, fx_major, fx_minor, fx_sid, fx_info,
                     [fx_prompt, fx_negative, fx_w, fx_h, fx_steps, fx_guid,
                      fx_status, fx_url, fx_img],
                     lambda st: [st["flux_prompt"], st["flux_negative"],
                                 st["flux_width"], st["flux_height"],
                                 st["flux_steps"], st["flux_guidance"],
                                 st["flux_status"], st["flux_image_url"],
                                 _url_to_img(st["flux_image_url"])])

        _wire_picker(nm_char, nm_major, nm_minor, nm_sid, nm_info,
                     [nm_w, nm_h, nm_status, nm_img],
                     lambda st: [st["norm_w"], st["norm_h"],
                                 st["norm_status"],
                                 _url_to_img(st["norm_image_url"], 300)])

        _wire_picker(s1_char, s1_major, s1_minor, s1_sid, s1_info,
                     [s1_prompt, s1_neg, s1_denoise, s1_cfg, s1_steps,
                      s1_op_w, s1_cn_w, s1_cat, s1_status, s1_url, s1_img],
                     lambda st: [st["sd1_prompt"], st["sd1_negative"],
                                 st["sd1_denoise"], st["sd1_cfg"], st["sd1_steps"],
                                 st["sd1_openpose_w"], st["sd1_canny_w"],
                                 st["sd1_category"],
                                 st["sd1_status"], st["sd1_image_url"],
                                 _url_to_img(st["sd1_image_url"], 350)])

        _wire_picker(s2_char, s2_major, s2_minor, s2_sid, s2_info,
                     [s2_prompt, s2_neg, s2_denoise, s2_cfg, s2_steps,
                      s2_status, s2_url, s2_img],
                     lambda st: [st["sd2_prompt"], st["sd2_negative"],
                                 st["sd2_denoise"], st["sd2_cfg"], st["sd2_steps"],
                                 st["sd2_status"], st["sd2_image_url"],
                                 _url_to_img(st["sd2_image_url"], 350)])

        _wire_picker(mv_char, mv_major, mv_minor, mv_sid, mv_info,
                     [mv_side_prompt, mv_back_prompt, mv_denoise, mv_cfg,
                      mv_side_status, mv_side_img, mv_back_status, mv_back_img],
                     lambda st: [st["mv_side_prompt"], st["mv_back_prompt"],
                                 st["mv_denoise"], st["mv_cfg"],
                                 st["mv_side_status"], _url_to_img(st["mv_side_url"], 300),
                                 st["mv_back_status"], _url_to_img(st["mv_back_url"], 300)])

        _wire_picker(tr_char, tr_major, tr_minor, tr_sid, tr_info,
                     [tr_status, tr_url],
                     lambda st: [st["trellis_status"], st["trellis_url"]])

        _wire_picker(rg_char, rg_major, rg_minor, rg_sid, rg_info,
                     [rg_type, rg_status, rg_url],
                     lambda st: [st["rig_char_type"], st["rig_status"], st["rig_url"]])

        # ── Queue buttons ─────────────────────────────────────────────────────

        # Flux
        (fx_q_btn.click(
            lambda sid, char, maj, min_, p, n, w, h, s, g:
                _q_flux(sid, char, maj, min_, p, n, w, h, s, g),
            [fx_sid, fx_char, fx_major, fx_minor,
             fx_prompt, fx_negative, fx_w, fx_h, fx_steps, fx_guid],
            [fx_sid, fx_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        fx_r_btn.click(
            lambda sid: (*_refresh_stage(sid,"flux"),
                         _url_to_img(_refresh_stage(sid,"flux")[1])),
            [fx_sid], [fx_status, fx_url, fx_img]
        )

        # Normalize
        nm_btn.click(_q_normalize,
                     [nm_sid, nm_w, nm_h, nm_src],
                     [nm_status, nm_img, nm_info2])

        # SD Stage 1
        def _do_q_s1(sid, p, n, dn, cfg, st, opw, cnw, cat, src):
            return _q_sd(sid, "sd_stage1", p, n,
                         {"denoise":float(dn),"cfg":float(cfg),"steps":int(st),
                          "openpose_weight":float(opw),"canny_weight":float(cnw),
                          "category":cat}, src)[1]

        (s1_q_btn.click(_do_q_s1,
                        [s1_sid, s1_prompt, s1_neg, s1_denoise, s1_cfg,
                         s1_steps, s1_op_w, s1_cn_w, s1_cat, s1_src],
                        [s1_status])
         .then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        s1_r_btn.click(
            lambda sid: (*_refresh_stage(sid,"sd_stage1"),
                         _url_to_img(_refresh_stage(sid,"sd_stage1")[1], 350)),
            [s1_sid], [s1_status, s1_url, s1_img]
        )

        # SD Stage 2
        def _do_q_s2(sid, p, n, dn, cfg, st, src):
            return _q_sd(sid, "sd_stage2", p, n,
                         {"denoise":float(dn),"cfg":float(cfg),"steps":int(st)}, src)[1]

        (s2_q_btn.click(_do_q_s2,
                        [s2_sid, s2_prompt, s2_neg, s2_denoise, s2_cfg, s2_steps, s2_src],
                        [s2_status])
         .then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        s2_r_btn.click(
            lambda sid: (*_refresh_stage(sid,"sd_stage2"),
                         _url_to_img(_refresh_stage(sid,"sd_stage2")[1], 350)),
            [s2_sid], [s2_status, s2_url, s2_img]
        )

        # Multiview side
        (mv_side_btn.click(
            lambda sid, p, dn, cfg, src: _q_multiview(sid,"side",p,"",dn,cfg,src)[1],
            [mv_sid, mv_side_prompt, mv_denoise, mv_cfg, mv_src],
            [mv_side_status]
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        mv_side_r.click(
            lambda sid: (_refresh_stage(sid,"multiview_side")[0],
                         _url_to_img(_refresh_stage(sid,"multiview_side")[1], 300)),
            [mv_sid], [mv_side_status, mv_side_img]
        )

        # Multiview back
        (mv_back_btn.click(
            lambda sid, p, dn, cfg, src: _q_multiview(sid,"back",p,"",dn,cfg,src)[1],
            [mv_sid, mv_back_prompt, mv_denoise, mv_cfg, mv_src],
            [mv_back_status]
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        mv_back_r.click(
            lambda sid: (_refresh_stage(sid,"multiview_back")[0],
                         _url_to_img(_refresh_stage(sid,"multiview_back")[1], 300)),
            [mv_sid], [mv_back_status, mv_back_img]
        )

        # TRELLIS
        (tr_q_btn.click(
            lambda sid, f, s, b: _q_trellis(sid, f, s, b)[1],
            [tr_sid, tr_front, tr_side, tr_back],
            [tr_status]
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        tr_r_btn.click(
            lambda sid: _refresh_stage(sid, "trellis"),
            [tr_sid], [tr_status, tr_url]
        )

        # Rig
        (rg_q_btn.click(
            lambda sid, t: _q_rig(sid, t)[1],
            [rg_sid, rg_type],
            [rg_status]
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        rg_r_btn.click(
            lambda sid: _refresh_stage(sid, "rig"),
            [rg_sid], [rg_status, rg_url]
        )

        # ── Auto-refresh timer ────────────────────────────────────────────────
        # Polls the last-queued stage's session. Simplified: polls fx_sid
        # (or whichever has an active task — use the first non-None sid found).
        _ACTIVE = {"queued", "running"}

        def _tick(fx_s, nm_s, s1_s, s2_s, mv_s, tr_s, rg_s):
            """Poll all stage sessions and update statuses."""
            def _r(sid, stage): return _refresh_stage(sid, stage)

            fx_st, fx_u = _r(fx_s, "flux")
            s1_st, s1_u = _r(s1_s, "sd_stage1")
            s2_st, s2_u = _r(s2_s, "sd_stage2")
            ms_st, ms_u = _r(mv_s, "multiview_side")
            mb_st, mb_u = _r(mv_s, "multiview_back")
            tr_st, tr_u = _r(tr_s, "trellis")
            rg_st, rg_u = _r(rg_s, "rig")

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
            [fx_sid, nm_sid, s1_sid, s2_sid, mv_sid, tr_sid, rg_sid],
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
