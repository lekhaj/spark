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
REDIS_QUEUE = os.getenv("MANUAL_GEN_QUEUE", "manual_gen_tasks")


# Map queue-name → (instance-id, label) so the orchestrator and the
# AutoShutdown admin UI both know which physical GPU serves each queue.
#
# Override via env var GPU_INSTANCE_MAP, comma-separated:
#   GPU_INSTANCE_MAP=manual_gen_tasks=i-aaa:L4 g6.2xlarge,manual_gen_tasks_spot=i-bbb:L40S g6e.2xlarge
def _parse_gpu_instance_map() -> dict[str, dict]:
    raw = os.getenv("GPU_INSTANCE_MAP", "").strip()
    if raw:
        out = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry or "=" not in entry:
                continue
            queue, rest = entry.split("=", 1)
            iid, _, label = rest.partition(":")
            q = queue.strip()
            # Infer prewarm: True for spot queues (which run spark-prewarm.service),
            # False for legacy on-demand instances (don't expect sentinel).
            prewarm = q.endswith("_spot")
            out[q] = {"id": iid.strip(), "label": (label or iid).strip(), "prewarm": prewarm}
        return out
    # Defaults — leave id="" for instances that should be resolved at runtime
    # (the launcher will discover by tag Project=spark-gpu).
    # "prewarm" flag: True if the instance runs spark-prewarm.service and
    # publishes prewarm:ready:<iid>. Legacy/manually-managed instances should
    # set prewarm=False so the status panel doesn't show "Prewarming forever".
    return {
        "manual_gen_tasks":      {"id": "",
                                  "label": "Default L4 (set GPU_INSTANCE_MAP to override)",
                                  "prewarm": False},
        "manual_gen_tasks_spot": {"id": os.getenv("AWS_GPU_INSTANCE_ID", "").strip(),
                                  "label": "L40S g6e.2xlarge (spot — auto-resolved)",
                                  "prewarm": True},
    }

GPU_INSTANCE_MAP = _parse_gpu_instance_map()

# Redis key used to persist the UI's GPU target across gradio worker
# processes — the in-process `global REDIS_QUEUE` mutation doesn't reliably
# propagate when gradio spawns multiple workers, so we round-trip via Redis.
KEY_UI_QUEUE = "ui:active_queue"

def _iid_for_queue(queue: str) -> str | None:
    """Return the instance-id that serves the given queue (or None if unmapped)."""
    return (GPU_INSTANCE_MAP.get(queue) or {}).get("id")

def _active_queue_from_redis() -> str:
    """Read the UI-selected queue from Redis; fall back to env default."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                        db=0, decode_responses=True)
        q = r.get(KEY_UI_QUEUE)
        if q and q in GPU_INSTANCE_MAP:
            return q
    except Exception:
        pass
    return REDIS_QUEUE


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    return get_db(MONGO_URI, MONGO_DB)

def _push_task(payload: dict, queue: str = None) -> str:
    """
    Orchestrator-aware task push.

    Flow (per user-requested orchestration):
      1. ensure_gpu_ready() — start/launch the spot if needed
      2. If spot was cold (reason in launched/started) AND user is targeting
         the spot queue:
           a. snapshot autoshutdown:enabled flag
           b. disable autoshutdown (only if it was enabled)
           c. block until prewarm:ready:<iid>=1  (≤30 min)
      3. push task to the resolved queue
      4. if we disabled autoshutdown, spawn a background watcher that polls
         Mongo for status=done|error on this task and re-enables the flag.
    Subsequent tasks on a warm spot see prewarm:ready already set and skip
    the whole block — this only fires on the FIRST task after a cold launch.
    """
    import logging
    log = logging.getLogger("generation_studio")

    # Resolve at call-time. Use Redis-backed UI selection so the choice
    # propagates across gradio worker processes (the in-process REDIS_QUEUE
    # global was unreliable when gradio runs > 1 worker).
    if queue is None:
        queue = _active_queue_from_redis()

    # Warn if targeting the default (non-spot) queue — spark_l4 is deleted.
    if not queue.endswith("_spot"):
        gr.Warning(
            "⚠️ spark_l4 instance has been DELETED. "
            "Switch GPU target to 'L40S Spot' — fixed-instance calls will fail.",
            duration=10,
        )

    import redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                    db=0, decode_responses=True)

    # ── 1. ensure GPU is at least running ─────────────────────────────────
    # On capacity failures (typical for spot), retry up to 5 times at 60s
    # intervals (~5 min total) before giving up and warning the user.
    reason = "disabled"
    try:
        from lib.gpu_launcher import ensure_gpu_ready
        ok, reason = ensure_gpu_ready()
        if not ok and reason != "disabled":
            capacity_err = "capacity" in reason.lower() or "insufficient" in reason.lower()
            if capacity_err:
                gr.Info("⏳ Spot capacity unavailable — retrying for up to 5 min...", duration=8)
                import time as _time
                for attempt in range(2, 7):  # attempts 2..6 → 5 retries
                    _time.sleep(60)
                    log.info("GPU start retry %d/5 (prev reason=%s)", attempt - 1, reason)
                    ok, reason = ensure_gpu_ready()
                    if ok:
                        gr.Info(f"✅ GPU started on retry {attempt - 1}/5", duration=6)
                        break
                if not ok:
                    gr.Warning(
                        "⚠️ GPU spot capacity unavailable after 5 retries (~5 min). "
                        "AWS has no g6e.2xlarge capacity in us-east-1d right now. "
                        "Task queued — will run when capacity returns, or switch to on-demand.",
                        duration=20,
                    )
                    log.warning("GPU not ready after capacity retries (reason=%s)", reason)
            else:
                log.warning("GPU not ready before queueing (reason=%s) — task will wait.", reason)
    except Exception as _e:
        log.debug("gpu_launcher unavailable: %s", _e)

    # ── 2. cold-launch protection ─────────────────────────────────────────
    # Only applies when the spot was just brought up (launched/started) AND
    # the user is targeting the spot queue. Already-warm path (reason="running")
    # skips everything below and goes straight to rpush.
    # Only wait for prewarm when a brand-new instance was built from AMI.
    # "started" = was stopped and restarted — EBS preserved, page cache warm,
    # no need to block on prewarm sentinel.
    cold = reason == "launched"
    targets_spot = queue.endswith("_spot") or os.getenv("MANUAL_GEN_QUEUE", "") == queue
    was_enabled = False
    disabled_here = False

    # Resolve the GPU instance serving this queue (per-instance keys)
    target_iid = _iid_for_queue(queue) or os.getenv("AWS_GPU_INSTANCE_ID", "").strip()

    if cold and targets_spot:
        try:
            from lib.autoshutdown_ctl import (
                autoshutdown_was_enabled, disable_autoshutdown,
                wait_for_prewarm, is_prewarm_ready,
            )
            # Cheap fast-path: maybe sentinel is already published from a
            # previous boot (cache still warm). Skip the wait if so.
            if target_iid and not is_prewarm_ready(r, target_iid):
                was_enabled = autoshutdown_was_enabled(r, target_iid)
                if was_enabled:
                    disable_autoshutdown(r, target_iid)
                    disabled_here = True
                log.info("Cold spot %s detected (reason=%s) — waiting for prewarm sentinel",
                         target_iid, reason)
                ok_warm = wait_for_prewarm(r, target_iid, timeout=1800, poll=10)
                if not ok_warm:
                    log.warning("Prewarm wait timed out — queueing anyway; first inference may be slow")
                # Clear the pending flag — status panel now shows Ready
                try:
                    r.delete(f"prewarm:pending:{target_iid}")
                except Exception:
                    pass
        except Exception as e:
            log.warning("Cold-launch protection failed (continuing without it): %s", e)

    # ── 3. push task ──────────────────────────────────────────────────────
    payload.setdefault("task_id", str(uuid.uuid4()))
    payload.setdefault("timestamp", time.time())
    r.rpush(queue, json.dumps(payload))

    # ── 4. background restore-on-completion ───────────────────────────────
    if disabled_here:
        import threading
        t = threading.Thread(
            target=_watch_and_restore_autoshutdown,
            args=(payload["task_id"], was_enabled, target_iid),
            name=f"AsRestore-{payload['task_id'][:8]}",
            daemon=True,
        )
        t.start()

    return payload["task_id"]


def _watch_and_restore_autoshutdown(task_id: str, was_enabled: bool,
                                    instance_id: str | None = None,
                                    timeout: int = 3600, poll: int = 15) -> None:
    """
    Poll Mongo until the given task hits status=done|error, then re-enable
    AutoShutdown via Redis. Runs in a daemon thread spawned by _push_task().

    Timeout (1h) is a safety belt — if Mongo never reports done, we restore
    anyway so the spot can eventually stop itself.
    """
    import logging
    log = logging.getLogger("generation_studio")
    try:
        import redis
        from lib.autoshutdown_ctl import restore_autoshutdown
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                        db=0, decode_responses=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                doc = get_run_any(_db(), task_id)
                st = (doc or {}).get("status", "")
                if st in ("done", "error"):
                    log.info("Task %s reached status=%s — restoring AutoShutdown for %s",
                             task_id[:8], st, instance_id or "global")
                    restore_autoshutdown(r, was_enabled, instance_id)
                    return
            except Exception as e:
                log.debug("watcher mongo poll failed: %s", e)
            time.sleep(poll)
        log.warning("Task %s never completed within %ds — restoring AutoShutdown for %s anyway",
                    task_id[:8], timeout, instance_id or "global")
        restore_autoshutdown(r, was_enabled, instance_id)
    except Exception as e:
        log.warning("AutoShutdown restore watcher crashed: %s", e)

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
            sid = str(doc["_id"])
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
            return str(run["_id"]), None, gr.update(), gr.update(), f"⚠️ {stage} is already {st}."
        if st == "done":
            # Auto-create next minor — same as error retry so user can keep running
            sid, new_n = auto_retry_run(db, char, stage, major, prompt, neg, params)
            minors = list_stage_minors(db, char, stage, major)
            info   = f"{sid[:8]}…  v{major}.{new_n}  [idle]"
            return sid, new_n, gr.update(choices=minors, value=new_n), info, None
        if st == "error":
            # Auto-create next minor for retry
            sid, new_n = auto_retry_run(db, char, stage, major, prompt, neg, params)
            minors = list_stage_minors(db, char, stage, major)
            info   = f"{sid[:8]}…  v{major}.{new_n}  [idle]"
            return sid, new_n, gr.update(choices=minors, value=new_n), info, None
        # status == "idle"
        sid = str(run["_id"])
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
    _ver_minor = new_n if new_n is not None else int(minor)
    # Queue front view (primary — sets status to queued via mark_queued)
    tid_front = _push_task({"type": "flux", "session_id": sid, "stage": stage,
                             "char_name": char, "major": int(major), "minor": _ver_minor,
                             "prompt": prompt, "negative": neg,
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
            _char_slug = char.replace(" ", "_").lower()
            _ver_minor = new_n if new_n is not None else int(minor)
            _maj       = int(major)
            s3_key = f"chars/{_char_slug}/v{_maj}.{_ver_minor}/{_char_slug}_{_maj}_{_ver_minor}_normalize.png"
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
    _ver_minor = new_n if new_n is not None else int(minor)
    tid = _push_task({"type": "sd_stage", "session_id": sid, "stage": stage,
                      "char_name": char, "major": int(major), "minor": _ver_minor,
                      "prompt": prompt, "negative": neg,
                      "params": params, "input_stage": src_stage, "input_url": src_url})
    mark_queued(db, sid, stage=stage, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


def _q_flux_pose(char, major, minor,
                 prompt, neg,
                 use_control, control_mode, auto_extract, control_image_url,
                 width, height, steps, guidance_scale, true_cfg_scale,
                 cn_scale, cn_start, cn_end, seed,
                 src_stage, src_url):
    """
    Queue a FLUX.1-dev + ControlNet-Union-Pro-2.0 task.

    Three modes:
      use_control=False                 → pure text-to-image (no conditioning)
      use_control=True, auto_extract=T  → extract control image from src_url
      use_control=True, auto_extract=F  → control_image_url must be supplied
    """
    if not prompt or not prompt.strip():
        return None, gr.update(), gr.update(), "⚠️ Prompt is empty."

    stage = "flux_pose"
    use_control = bool(use_control)

    if use_control and auto_extract and not control_image_url:
        if not src_url:
            src_url = get_latest_done_image_url(_db(), char, src_stage) or ""
        if not src_url:
            return None, gr.update(), gr.update(), \
                f"Conditioning is ON but no done image in '{src_stage}'. "\
                "Either turn off conditioning, run that stage first, or supply Control image URL."

    if use_control and not auto_extract and not control_image_url:
        return None, gr.update(), gr.update(), \
            "Conditioning is ON with auto-extract off — supply a Control image URL."

    params = {
        "use_control":                   use_control,
        "control_mode":                  (control_mode or "pose").lower(),
        "auto_extract":                  bool(auto_extract),
        "controlnet_conditioning_scale": float(cn_scale),
        "control_guidance_start":        float(cn_start),
        "control_guidance_end":          float(cn_end),
        "steps":                         int(steps),
        "guidance_scale":                float(guidance_scale),
        "true_cfg_scale":                float(true_cfg_scale),
        "width":                         int(width),
        "height":                        int(height),
        "seed":                          int(seed),
    }
    if use_control and control_image_url:
        params["control_image_url"] = control_image_url

    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, prompt, neg, params)
    if err:
        return sid, minor_upd, gr.update(), err

    db = _db()
    save_run_params(db, sid, prompt, neg, params, stage=stage)
    _ver_minor = new_n if new_n is not None else int(minor)

    payload = {"type": "flux_pose", "session_id": sid, "stage": stage,
               "char_name": char, "major": int(major), "minor": _ver_minor,
               "prompt": prompt, "negative": neg,
               "params": params,
               "input_stage": src_stage if use_control else "",
               "input_url":   (src_url if use_control else "") or ""}
    if use_control and control_image_url:
        payload["control_image_url"] = control_image_url

    tid = _push_task(payload)
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
    _ver_minor = new_n if new_n is not None else int(minor)
    tid = _push_task({"type": "trellis", "session_id": sid, "stage": stage,
                      "char_name": char, "major": int(major), "minor": _ver_minor,
                      "char_type": char_type or "humanoid",
                      "input_front": front_url,
                      "input_side": side_url, "input_back": back_url,
                      "params": params})
    mark_queued(db, sid, stage=stage, task_id=tid)
    views_info = f"F{'✓' if front_url else '✗'} S{'✓' if side_url else '✗'} B{'✓' if back_url else '✗'}"
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  [{views_info}]  task={tid[:8]}…"


def _q_pixal3d(char, major, minor, char_type, src_stage, src_ver):
    """
    Queue Pixal3D 3D task.
    Single-image input (no multi-view); reads front-view URL from source stage/version.
    """
    stage = "pixal3d"
    db    = _db()

    if src_ver:
        view_urls = _get_view_urls_for_ver(char, src_stage, src_ver)
    else:
        latest = get_latest_done_run(db, char, src_stage)
        view_urls = {"front": (latest or {}).get("image_url") or ""}

    front_url = view_urls.get("front") or ""
    if not front_url:
        return None, gr.update(), gr.update(), \
            f"No front-view image in '{src_stage}' v{src_ver or 'latest'}. Run that stage first."

    params = {"char_type": char_type or "humanoid"}
    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, "", "", params)
    if err:
        return sid, minor_upd, gr.update(), err

    save_run_params(db, sid, "", "", params, stage=stage)
    _ver_minor = new_n if new_n is not None else int(minor)
    tid = _push_task({"type": "pixal3d", "session_id": sid, "stage": stage,
                      "char_name": char, "major": int(major), "minor": _ver_minor,
                      "char_type": char_type or "humanoid",
                      "input_front": front_url,
                      "params": params})
    mark_queued(db, sid, stage=stage, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


def _q_hunyuan3d(char, major, minor, char_type, src_stage, src_ver):
    """
    Queue Hunyuan3D-2.0 task.
    Single-image input; reads front-view URL from source stage/version.
    """
    stage = "hunyuan3d"
    db    = _db()

    if src_ver:
        view_urls = _get_view_urls_for_ver(char, src_stage, src_ver)
    else:
        latest = get_latest_done_run(db, char, src_stage)
        view_urls = {"front": (latest or {}).get("image_url") or ""}

    front_url = view_urls.get("front") or ""
    if not front_url:
        return None, gr.update(), gr.update(), \
            f"No front-view image in '{src_stage}' v{src_ver or 'latest'}. Run that stage first."

    params = {"char_type": char_type or "humanoid"}
    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, "", "", params)
    if err:
        return sid, minor_upd, gr.update(), err

    save_run_params(db, sid, "", "", params, stage=stage)
    _ver_minor = new_n if new_n is not None else int(minor)
    tid = _push_task({"type": "hunyuan3d", "session_id": sid, "stage": stage,
                      "char_name": char, "major": int(major), "minor": _ver_minor,
                      "char_type": char_type or "humanoid",
                      "input_front": front_url,
                      "params": params})
    mark_queued(db, sid, stage=stage, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    return sid, minor_upd, info, f"queued ✓  v{ver}  task={tid[:8]}…"


def _q_mesh_lod(char, major, minor, src_stage, src_ver,
                profiles, ratio_b, ratio_c, voxel_d,
                quadriflow, qf_target, bake_res):
    """Queue Mesh-LOD task on its own CPU queue (mesh_lod_tasks)."""
    stage = "mesh_lod"
    db    = _db()

    src_url = (_get_src_url_for_ver(char, src_stage, src_ver) if src_ver
               else get_latest_done_image_url(db, char, src_stage) or "")
    if not src_url:
        return None, gr.update(), gr.update(), f"No done GLB in '{src_stage}'.", \
               "", "", "", "", "", "", "", "", "", "", "", ""

    if not profiles:
        profiles = ["a", "b", "c", "d"]

    params = {
        "ratio_b":           float(ratio_b),
        "ratio_c":           float(ratio_c),
        "voxel_size_d":      float(voxel_d),
        "quadriflow":        bool(quadriflow),
        "quadriflow_target": int(qf_target),
        "bake_resolution":   int(bake_res),
    }
    sid, new_n, minor_upd, info_upd, err = _prepare_run(
        char, stage, major, minor, "", "", params)
    if err:
        return sid, minor_upd, gr.update(), err, \
               "", "", "", "", "", "", "", "", "", "", "", ""

    save_run_params(db, sid, "", "", params, stage=stage)
    _ver_minor = new_n if new_n is not None else int(minor)
    tid = _push_task({
        "type": "mesh_lod", "session_id": sid, "stage": stage,
        "char_name": char, "major": int(major), "minor": _ver_minor,
        "source_url": src_url,
        "profiles": profiles,
        "params": params,
    }, queue="mesh_lod_tasks")
    mark_queued(db, sid, stage=stage, task_id=tid)
    ver  = f"{major}.{new_n}" if new_n is not None else f"{major}.{minor}"
    info = f"{sid[:8]}…  v{ver}  [queued]"
    status = f"queued ✓  v{ver}  profiles={','.join(profiles)}  task={tid[:8]}…"
    # Reset result fields
    return sid, minor_upd, info, status, \
           "", "", "", "", "", "", "", "", "", "", "", ""


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
    _ver_minor = new_n if new_n is not None else int(minor)
    tid = _push_task({"type": "rig", "session_id": sid, "stage": stage,
                      "char_name": char, "major": int(major), "minor": _ver_minor,
                      "char_type": char_type or "humanoid",
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

        # GPU target selector — picks which Redis queue tasks are pushed to.
        # Each queue is consumed by exactly one manual_gen_worker (running on
        # a specific GPU instance). Defaults to whatever MANUAL_GEN_QUEUE is
        # set to in the CPU env (.env). Changing here updates REDIS_QUEUE at
        # process scope until the gradio app restarts.
        _queue_choices = [
            ("Default GPU (manual_gen_tasks)", "manual_gen_tasks"),
            ("L40S Spot (manual_gen_tasks_spot)", "manual_gen_tasks_spot"),
        ]
        with gr.Row():
            gpu_target = gr.Dropdown(
                choices=_queue_choices,
                value=REDIS_QUEUE,
                label="🖥️ Target GPU queue",
                scale=4,
                info="Routes Queue clicks to the chosen GPU. Defaults from MANUAL_GEN_QUEUE env.",
            )
            gpu_target_info = gr.Textbox(
                value=f"Active queue: {REDIS_QUEUE}",
                label="", interactive=False, scale=2,
            )

        def _set_target_queue(q):
            # Persist to Redis so the choice survives across gradio workers
            # and is the source of truth for _push_task().
            global REDIS_QUEUE
            REDIS_QUEUE = q
            try:
                import redis
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                password=REDIS_PASSWORD, db=0, decode_responses=True)
                r.set(KEY_UI_QUEUE, q)
            except Exception as e:
                return f"⚠ Set locally but Redis persist failed: {e}"
            iid = _iid_for_queue(q) or "(no id)"
            return f"✓ Active: {q}  →  {iid}"
        gpu_target.change(_set_target_queue, inputs=gpu_target, outputs=gpu_target_info)

        # On page load, sync the dropdown to whatever's in Redis (or env default)
        def _load_target_queue():
            q = _active_queue_from_redis()
            iid = _iid_for_queue(q) or "(no id)"
            return q, f"✓ Active: {q}  →  {iid}"
        tab.load(_load_target_queue, inputs=[], outputs=[gpu_target, gpu_target_info])

        # ── GPU lifecycle status (live, auto-refresh) ────────────────────────
        # One row per known instance. Shows EC2 state + prewarm sentinel so
        # you can see at a glance whether queueing now will be fast (Ready)
        # or slow (Prewarming / Stopped → boot).
        with gr.Accordion("📡 GPU Status (live)", open=True):
            gr.Markdown(
                "_Phase = `EC2 state` + `prewarm sentinel`. Auto-refreshes every 10 s. "
                "Queue when status reads **🟢 Ready** to avoid the cold first-inference wait._"
            )
            _status_rows: list[tuple[str, dict, gr.Markdown]] = []
            for q_name, meta in GPU_INSTANCE_MAP.items():
                with gr.Row():
                    label_md = gr.Markdown(
                        f"**{meta['label']}**  \n"
                        f"`{meta['id'] or '(no id)'}`  ·  queue: `{q_name}`",
                    )
                    status_md = gr.Markdown("⏳ Loading…")
                _status_rows.append((meta["id"], meta, status_md))

            def _render_one(iid, prewarm_expected: bool):
                if not iid:
                    return "❌ Not configured (set `AWS_GPU_INSTANCE_ID` or `GPU_INSTANCE_MAP`)"
                try:
                    import redis
                    from lib.gpu_launcher import get_gpu_status
                    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                    password=REDIS_PASSWORD, db=0, decode_responses=True)
                    s = get_gpu_status(iid, r=r)
                    label = s["phase_label"]
                    # Override "Prewarming" → "Ready" for instances that don't
                    # publish prewarm:ready (legacy / non-orchestrated GPUs)
                    if s["phase"] == "prewarming" and not prewarm_expected:
                        label = "🟢 Ready (no prewarm — legacy)"
                    bits = [label]
                    if s["instance_type"]:
                        bits.append(f"`{s['instance_type']}`")
                    if s["public_ip"]:
                        bits.append(f"IP: `{s['public_ip']}`")
                    if s["detail"]:
                        bits.append(f"_{s['detail']}_")
                    return "  ·  ".join(bits)
                except Exception as e:
                    return f"⚠ status error: {e}"

            def _render_all():
                return [_render_one(iid, meta.get("prewarm", True))
                        for iid, meta, _md in _status_rows]

            status_refresh_btn = gr.Button("🔄 Refresh now", size="sm")
            status_refresh_btn.click(
                _render_all, inputs=[],
                outputs=[md for _iid, _m, md in _status_rows],
            )

            # Auto-refresh every 10 s while the tab is open
            status_timer = gr.Timer(value=10, active=True)
            status_timer.tick(
                _render_all, inputs=[],
                outputs=[md for _iid, _m, md in _status_rows],
            )

            # Initial fill on page load
            tab.load(
                _render_all, inputs=[],
                outputs=[md for _iid, _m, md in _status_rows],
            )

        # ── Per-instance AutoShutdown admin ──────────────────────────────────
        # Each GPU instance has its own Redis-controlled AutoShutdown enabled
        # flag and idle threshold. The per-instance key takes precedence over
        # the global key; the GPU-side auto_shutdown.py reads the per-instance
        # key first and only falls back to global if unset.
        with gr.Accordion("🛠️ GPU AutoShutdown Settings (per-instance)", open=True):
            gr.Markdown(
                "_Each GPU has its own Redis-driven enable flag and idle timer. "
                "Disabling pauses the self-stop loop on that GPU; the orchestrator "
                "still toggles these automatically during cold launch + first inference._"
            )
            _as_rows: list[tuple[str, dict, gr.Checkbox, gr.Number, gr.Textbox]] = []

            def _get_as_redis():
                import redis
                return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,
                                   db=0, decode_responses=True)

            def _refresh_one(iid):
                try:
                    from lib.autoshutdown_ctl import get_instance_config
                    cfg = get_instance_config(_get_as_redis(), iid)
                    return (cfg["enabled"], cfg["idle_minutes"],
                            f"✓ enabled={cfg['enabled']}  idle={cfg['idle_minutes']}min  ({cfg['source']})")
                except Exception as e:
                    return (True, 15, f"⚠ {e}")

            def _save_one(iid, enabled, minutes):
                try:
                    from lib.autoshutdown_ctl import set_autoshutdown_enabled, set_idle_minutes
                    r = _get_as_redis()
                    set_autoshutdown_enabled(r, iid, bool(enabled))
                    set_idle_minutes(r, iid, int(minutes))
                    return f"✓ saved: enabled={bool(enabled)} idle={int(minutes)}min"
                except Exception as e:
                    return f"❌ {e}"

            for q_name, meta in GPU_INSTANCE_MAP.items():
                iid = meta["id"]; label = meta["label"]
                with gr.Row():
                    gr.Markdown(f"**{label}**  \n`{iid}`  (queue: `{q_name}`)")
                    en_cb = gr.Checkbox(label="AutoShutdown enabled", value=True, scale=1)
                    idle_n = gr.Number(label="Idle minutes", value=15, precision=0,
                                       minimum=1, maximum=720, scale=1)
                    status = gr.Textbox(label="Status", interactive=False, scale=2)
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("↻ Read", size="sm")
                        save_btn    = gr.Button("💾 Save", size="sm", variant="primary")
                # Wire callbacks (closure over iid via default-arg trick)
                refresh_btn.click(
                    lambda _iid=iid: _refresh_one(_iid),
                    inputs=[], outputs=[en_cb, idle_n, status],
                )
                save_btn.click(
                    lambda en, mn, _iid=iid: _save_one(_iid, en, mn),
                    inputs=[en_cb, idle_n], outputs=[status],
                )
                _as_rows.append((iid, meta, en_cb, idle_n, status))

            # Auto-populate on page load
            def _load_all():
                out = []
                for iid, _meta, _en, _idle, _st in _as_rows:
                    en, mn, st = _refresh_one(iid)
                    out.extend([en, mn, st])
                return out
            tab.load(
                _load_all, inputs=[],
                outputs=[w for _i, _m, en, idle, st in _as_rows for w in (en, idle, st)],
            )

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
        #  STAGE 1b: FLUX POSE — FLUX.1-dev + ControlNet-Union-Pro-2.0
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 1b — Flux Pose (ControlNet-Union-Pro-2.0)", open=False):
            fp_char, fp_major, fp_minor, fp_new_maj, fp_sid, fp_info = _make_picker(_chars)
            gr.Markdown("---")
            gr.Markdown(
                "**Model**: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` on top of "
                "`black-forest-labs/FLUX.1-dev`.  Independent stage — runs with just a "
                "text prompt, or optionally with a control image (pose / edges / depth / …) "
                "extracted from a source image or supplied directly."
            )
            fp_use_control = gr.Checkbox(
                value=False,
                label="Use ControlNet conditioning",
                info="Off: pure text-to-image (free pose). On: lock pose/structure "
                     "via a control image — required for reliable T-pose.",
            )
            with gr.Group(visible=False) as fp_ctrl_group:
                fp_control_mode = gr.Dropdown(
                    choices=["pose", "canny", "soft_edge", "depth", "blur", "gray", "low_quality"],
                    value="pose", label="Control mode",
                    info="'pose' = OpenPose skeleton. Use it with a T-pose preset below "
                         "for clean front-facing limb separation.",
                )
                _CN_S3 = "https://sparkassets-us.s3.us-east-1.amazonaws.com/controlnet_refs"
                fp_tpose_preset = gr.Dropdown(
                    choices=[
                        ("Canonical (recommended — generated from COCO-18 keypoints)", ""),
                        ("T-Pose V2 — FBX extracted (S3 copy)",                         f"{_CN_S3}/tpose_v2_fbx.png"),
                        ("T-Pose V1 — Hand-drawn",                                      f"{_CN_S3}/tpose_v1_user.png"),
                        ("T-Pose — OpenPose default",                                   f"{_CN_S3}/tpose_openpose.png"),
                        ("T-Pose — FBX 512",                                            f"{_CN_S3}/tpose_fbx_512.png"),
                    ],
                    value="",   # empty → worker loads bundled tpose_canonical.png
                    label="T-Pose skeleton preset",
                    info="Default is the canonical OpenPose-format T-pose generated "
                         "from COCO-18 keypoints (anatomically exact, ships with the "
                         "worker, no network fetch). Pick an S3 URL only if you want "
                         "a different curated skeleton.",
                )
                fp_src_stage, fp_src_ver, fp_src_url_st, fp_src_info = _make_src_picker(
                    ["flux", "normalize", "sd_tpose", "flux_pose"], "flux")
                with gr.Row():
                    fp_auto_extract = gr.Checkbox(
                        value=False, label="Auto-extract control image from source",
                        info="Off (default): use the T-pose preset or URL below. "
                             "On: extract control map from the source-stage image.",
                    )
                    fp_control_image_url = gr.Textbox(
                        label="Control image URL (leave blank to use bundled skeleton)",
                        value="",
                        placeholder="https://...png",
                    )
                # Preset → URL textbox
                fp_tpose_preset.change(
                    lambda v: v if v else gr.update(),
                    [fp_tpose_preset], [fp_control_image_url],
                )
            fp_use_control.change(
                lambda on: gr.update(visible=bool(on)),
                [fp_use_control], [fp_ctrl_group],
            )
            fp_prompt = gr.Textbox(
                label="Prompt", lines=3,
                placeholder="A heroic warrior in full plate armor, standing tall, full body, "
                            "white studio background, sharp focus, 4k",
            )
            fp_neg = gr.Textbox(
                label="Negative prompt (used when True CFG > 1.0)", lines=2,
                value="deformed, extra limbs, mutated hands, text, watermark, blurry, low quality",
            )
            with gr.Row():
                fp_width  = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                fp_height = gr.Slider(512, 1536, value=1024, step=64, label="Height")
                fp_steps  = gr.Slider(8, 50, value=28, step=1, label="Steps")
            with gr.Row():
                fp_guid    = gr.Slider(1.0, 10.0, value=3.5, step=0.1,
                                       label="Guidance scale (distilled)")
                fp_true_cfg = gr.Slider(1.0, 10.0, value=1.0, step=0.1,
                                        label="True CFG (>1 enables negative prompt)")
                fp_seed    = gr.Number(value=-1, precision=0,
                                       label="Seed (-1 = random)")
            with gr.Row():
                fp_cn_scale = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                        label="ControlNet conditioning scale")
                fp_cn_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                        label="Control guidance start")
                fp_cn_end   = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                        label="Control guidance end")
            with gr.Row():
                fp_q_btn  = gr.Button("Queue Flux Pose", variant="primary")
                fp_status = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                fp_r_btn  = gr.Button("Refresh", size="sm")
            fp_img = gr.HTML(value=_url_to_img("", 400))
            fp_url = gr.Textbox(label="URL", interactive=False)

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
        #  STAGE 3b: PIXAL3D 3D  (alternative to TRELLIS — TencentARC/Pixal3D)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 3b — Pixal3D 3D Mesh", open=False):
            px_char, px_major, px_minor, px_new_maj, px_sid, px_info = _make_picker(_chars)
            gr.Markdown("---")
            px_char_type = gr.Dropdown(
                choices=["humanoid", "quadruped", "bird", "fish"],
                value="humanoid", label="Character type")
            gr.Markdown("**Select source stage + version (single front-view image):**")
            px_src_stage, px_src_ver, px_src_url_st, px_src_info = _make_src_picker(
                ["sd_tpose", "flux", "normalize"], "sd_tpose")
            with gr.Row():
                px_q_btn = gr.Button("Queue Pixal3D", variant="primary")
                px_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                px_r_btn = gr.Button("Refresh", size="sm")
            px_url = gr.Textbox(label="GLB URL (when done)", interactive=False)
            px_3d_btn = gr.HTML(value="")

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 3c: HUNYUAN3D (alternative to TRELLIS/Pixal3D)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 3c — Hunyuan3D-2.0 3D Mesh (PBR)", open=False):
            hy_char, hy_major, hy_minor, hy_new_maj, hy_sid, hy_info = _make_picker(_chars)
            gr.Markdown("---")
            hy_char_type = gr.Dropdown(
                choices=["humanoid", "quadruped", "bird", "fish"],
                value="humanoid", label="Character type")
            gr.Markdown("**Select source stage + version (single front-view image):**")
            hy_src_stage, hy_src_ver, hy_src_url_st, hy_src_info = _make_src_picker(
                ["sd_tpose", "flux", "normalize"], "sd_tpose")
            with gr.Row():
                hy_q_btn = gr.Button("Queue Hunyuan3D", variant="primary")
                hy_status= gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                hy_r_btn = gr.Button("Refresh", size="sm")
            hy_url = gr.Textbox(label="GLB URL (when done)", interactive=False)
            hy_3d_btn = gr.HTML(value="")

        # ══════════════════════════════════════════════════════════════════════
        #  STAGE 3d: MESH LOD (CPU)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Stage 3d — Mesh LOD / poly-count optimization (CPU)", open=False):
            ml_char, ml_major, ml_minor, ml_new_maj, ml_sid, ml_info = _make_picker(_chars)
            gr.Markdown(
                "_Generates up to 4 LODs from the source GLB:_ \n"
                "* **A — Preserve** _(merge + clean only; no decimation, lossless)_\n"
                "* **B — Decimate safe** _(ratio ≈ 0.4, UV preserved)_\n"
                "* **C — Decimate aggressive** _(ratio ≈ 0.15 + planar dissolve)_\n"
                "* **D — Voxel remesh + texture rebake** _(target ~5-10k tris, can take 2-3 min)_"
            )
            gr.Markdown("**Select source stage + version (must be a finished GLB):**")
            ml_src_stage, ml_src_ver, ml_src_url_st, ml_src_info = _make_src_picker(
                ["trellis", "pixal3d", "hunyuan3d"], "pixal3d")
            with gr.Row():
                ml_profiles = gr.CheckboxGroup(
                    choices=[("A — Preserve",     "a"),
                             ("B — Safe decimate", "b"),
                             ("C — Aggressive",    "c"),
                             ("D — Voxel + bake",  "d")],
                    value=["a", "b", "c", "d"],
                    label="LOD profiles to generate")
            with gr.Row():
                ml_ratio_b = gr.Slider(0.05, 1.0, value=0.40,
                                       label="B ratio (keep fraction)", scale=1)
                ml_ratio_c = gr.Slider(0.05, 1.0, value=0.15,
                                       label="C ratio (keep fraction)", scale=1)
                ml_voxel_d = gr.Slider(0.005, 0.10, value=0.030, step=0.005,
                                       label="D voxel size", scale=1)
            with gr.Row():
                ml_quadriflow = gr.Checkbox(value=False, label="D: QuadriFlow (slow, prettier topology)")
                ml_qf_target  = gr.Number(value=5000, precision=0, label="QuadriFlow target faces")
                ml_bake_res   = gr.Dropdown(choices=[512, 1024, 2048], value=1024,
                                            label="D bake resolution (px)")
            with gr.Row():
                ml_q_btn = gr.Button("Queue Mesh LOD", variant="primary")
                ml_status = gr.Textbox(label="Status", value="idle", interactive=False, scale=2)
                ml_r_btn = gr.Button("Refresh", size="sm")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**LOD A — Preserve**")
                    ml_a_info = gr.Textbox(label="", interactive=False, lines=1)
                    ml_a_url  = gr.Textbox(label="URL", interactive=False)
                    ml_a_3d   = gr.HTML(value="")
                with gr.Column():
                    gr.Markdown("**LOD B — Safe decimate**")
                    ml_b_info = gr.Textbox(label="", interactive=False, lines=1)
                    ml_b_url  = gr.Textbox(label="URL", interactive=False)
                    ml_b_3d   = gr.HTML(value="")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**LOD C — Aggressive**")
                    ml_c_info = gr.Textbox(label="", interactive=False, lines=1)
                    ml_c_url  = gr.Textbox(label="URL", interactive=False)
                    ml_c_3d   = gr.HTML(value="")
                with gr.Column():
                    gr.Markdown("**LOD D — Voxel + rebake**")
                    ml_d_info = gr.Textbox(label="", interactive=False, lines=1)
                    ml_d_url  = gr.Textbox(label="URL", interactive=False)
                    ml_d_3d   = gr.HTML(value="")

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
            return [upd] * 9

        g_refresh_btn.click(_do_refresh, [],
                            [g_char, fx_char, nm_char, fp_char, s1_char, tr_char, px_char, ml_char, rg_char])

        # ── Global: Create New Character ──────────────────────────────────────
        def _do_create(label):
            label = (label or "").strip()
            if not label:
                return [gr.update()] * 9 + ["Enter a character label first."]
            ok    = create_character(_db(), label)
            if not ok:
                return [gr.update()] * 9 + [f"❌ Failed to save '{label}' — check MongoDB connection."]
            chars = _list_chars()
            upd   = gr.update(choices=chars, value=label)
            return [upd] * 9 + [f"✓ Created '{label}'. Select it in any stage below and click ⬇ Prefill All Stages."]

        g_create_btn.click(_do_create, [g_new_char_input],
                           [g_char, fx_char, nm_char, fp_char, s1_char, tr_char, px_char, ml_char, rg_char, g_create_info])

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
            px_src = _refresh_src_picker(char, "sd_tpose") if char else (gr.update(), "", "")
            hy_src = _refresh_src_picker(char, "sd_tpose") if char else (gr.update(), "", "")
            # Trellis view availability info
            if char:
                tr_ver_val = tr_src[0].get("value") if hasattr(tr_src[0], "get") else None
                tr_vinfo   = _view_availability_info(char, "sd_tpose", tr_ver_val or "")
            else:
                tr_vinfo = ""

            return list((
                # 7 char dropdowns
                char_upd, char_upd, char_upd, char_upd, char_upd, char_upd, char_upd,
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
                # pixal3d (4+3=7) + source picker (3) = 10
                *_load("pixal3d", _ex_pixal3d),
                *px_src,
                # hunyuan3d (4+3=7) + source picker (3) = 10
                *_load("hunyuan3d", _ex_hunyuan3d),
                *hy_src,
                # rig (4+3=7) = 7
                *_load("rig", _ex_rig),
            ))

        g_prefill_btn.click(_do_prefill, [g_char], [
            # char dropdowns (7)
            fx_char, nm_char, s1_char, tr_char, px_char, hy_char, rg_char,
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
            # pixal3d (4+3=7) + source picker (3) = 10
            px_major, px_minor, px_sid, px_info,
            px_char_type, px_status, px_url,
            px_src_ver, px_src_url_st, px_src_info,
            # hunyuan3d (4+3=7) + source picker (3) = 10
            hy_major, hy_minor, hy_sid, hy_info,
            hy_char_type, hy_status, hy_url,
            hy_src_ver, hy_src_url_st, hy_src_info,
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

        def _ex_pixal3d(run):
            p = run.get("params") or {}
            return [p.get("char_type", "humanoid"),
                    run.get("status", "idle"),
                    run.get("image_url", "") or ""]

        def _ex_hunyuan3d(run):
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

        def _ex_flux_pose(run):
            p = run.get("params") or {}
            url = run.get("image_url", "") or ""
            return [run.get("prompt", ""),
                    run.get("negative", "deformed, extra limbs, mutated hands, text, "
                                        "watermark, blurry, low quality"),
                    bool(p.get("use_control", False)),
                    p.get("control_mode", "pose"),
                    bool(p.get("auto_extract", True)),
                    p.get("control_image_url", ""),
                    p.get("width", 1024), p.get("height", 1024),
                    p.get("steps", 28),
                    p.get("guidance_scale", 3.5),
                    p.get("true_cfg_scale", 1.0),
                    p.get("controlnet_conditioning_scale", 0.7),
                    p.get("control_guidance_start", 0.0),
                    p.get("control_guidance_end", 0.8),
                    p.get("seed", -1),
                    run.get("status", "idle"),
                    url,
                    _url_to_img(url, 400)]

        _wire_picker("flux_pose", fp_char, fp_major, fp_minor, fp_new_maj, fp_sid, fp_info,
                     [fp_prompt, fp_neg,
                      fp_use_control, fp_control_mode, fp_auto_extract, fp_control_image_url,
                      fp_width, fp_height, fp_steps, fp_guid, fp_true_cfg,
                      fp_cn_scale, fp_cn_start, fp_cn_end, fp_seed,
                      fp_status, fp_url, fp_img],
                     _ex_flux_pose)

        _wire_picker("trellis", tr_char, tr_major, tr_minor, tr_new_maj, tr_sid, tr_info,
                     [tr_char_type, tr_status, tr_url],
                     _ex_trellis)

        _wire_picker("pixal3d", px_char, px_major, px_minor, px_new_maj, px_sid, px_info,
                     [px_char_type, px_status, px_url],
                     _ex_pixal3d)

        _wire_picker("hunyuan3d", hy_char, hy_major, hy_minor, hy_new_maj, hy_sid, hy_info,
                     [hy_char_type, hy_status, hy_url],
                     _ex_hunyuan3d)

        def _ex_mesh_lod(run):
            st = run.get("status", "idle")
            results = []
            for p in ("a", "b", "c", "d"):
                url   = run.get(f"lod_{p}_url",   "") or ""
                tris  = run.get(f"lod_{p}_tris",  "") or ""
                bts   = run.get(f"lod_{p}_bytes", 0) or 0
                err   = run.get(f"lod_{p}_error", "") or ""
                info  = (f"❌ {err[:80]}" if err else
                         (f"tris={tris} size={bts/1024:.0f}KB" if url else "—"))
                viewer = _viewer_btn(url) if url else ""
                results.extend([url, info, viewer])
            return [st, *results]

        _wire_picker("mesh_lod", ml_char, ml_major, ml_minor, ml_new_maj, ml_sid, ml_info,
                     [ml_status,
                      ml_a_url, ml_a_info, ml_a_3d,
                      ml_b_url, ml_b_info, ml_b_3d,
                      ml_c_url, ml_c_info, ml_c_3d,
                      ml_d_url, ml_d_info, ml_d_3d],
                     _ex_mesh_lod)

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
        _make_src_wiring(fp_char, fp_src_stage, fp_src_ver, fp_src_url_st, fp_src_info)
        _make_src_wiring(s1_char, s1_src_stage, s1_src_ver, s1_src_url_st, s1_src_info)
        _make_src_wiring(tr_char, tr_src_stage,  tr_src_ver,  tr_src_url_st,  tr_src_info)
        _make_src_wiring(px_char, px_src_stage,  px_src_ver,  px_src_url_st,  px_src_info)
        _make_src_wiring(hy_char, hy_src_stage,  hy_src_ver,  hy_src_url_st,  hy_src_info)
        _make_src_wiring(ml_char, ml_src_stage,  ml_src_ver,  ml_src_url_st,  ml_src_info)

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

        # Flux Pose (FLUX.1-dev + ControlNet-Union-Pro-2.0)
        (fp_q_btn.click(
            _q_flux_pose,
            [fp_char, fp_major, fp_minor,
             fp_prompt, fp_neg,
             fp_use_control, fp_control_mode, fp_auto_extract, fp_control_image_url,
             fp_width, fp_height, fp_steps, fp_guid, fp_true_cfg,
             fp_cn_scale, fp_cn_start, fp_cn_end, fp_seed,
             fp_src_stage, fp_src_url_st],
            [fp_sid, fp_minor, fp_info, fp_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        fp_r_btn.click(
            lambda sid: _refresh_with_img(sid, 400),
            [fp_sid], [fp_status, fp_url, fp_img]
        )

        # Flux Pose source-stage picker → refresh available versions
        # flux_pose source picker wiring is handled by _make_src_wiring above
        # (covers src_stage.change, src_ver.change, and char.change).

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

        # Pixal3D
        (px_q_btn.click(
            _q_pixal3d,
            [px_char, px_major, px_minor,
             px_char_type, px_src_stage, px_src_ver],
            [px_sid, px_minor, px_info, px_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        px_r_btn.click(
            lambda sid: _refresh_run(sid),
            [px_sid], [px_status, px_url]
        )

        # Hunyuan3D
        (hy_q_btn.click(
            _q_hunyuan3d,
            [hy_char, hy_major, hy_minor,
             hy_char_type, hy_src_stage, hy_src_ver],
            [hy_sid, hy_minor, hy_info, hy_status],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        hy_r_btn.click(
            lambda sid: _refresh_run(sid),
            [hy_sid], [hy_status, hy_url]
        )

        # Mesh LOD
        (ml_q_btn.click(
            _q_mesh_lod,
            [ml_char, ml_major, ml_minor,
             ml_src_stage, ml_src_ver,
             ml_profiles,
             ml_ratio_b, ml_ratio_c, ml_voxel_d,
             ml_quadriflow, ml_qf_target, ml_bake_res],
            [ml_sid, ml_minor, ml_info, ml_status,
             ml_a_url, ml_a_info, ml_a_3d,
             ml_b_url, ml_b_info, ml_b_3d,
             ml_c_url, ml_c_info, ml_c_3d,
             ml_d_url, ml_d_info, ml_d_3d],
        ).then(lambda: gr.Timer(active=True), outputs=[stage_timer]))

        def _refresh_mesh_lod(sid):
            doc = get_run_any(_db(), sid) if sid else None
            if not doc:
                return ("idle",
                        "", "", "",
                        "", "", "",
                        "", "", "",
                        "", "", "")
            st = doc.get("status", "?")
            results = []
            for p in ("a", "b", "c", "d"):
                url   = doc.get(f"lod_{p}_url",   "")
                tris  = doc.get(f"lod_{p}_tris",  "")
                bts   = doc.get(f"lod_{p}_bytes", 0)
                err   = doc.get(f"lod_{p}_error", "")
                info  = (f"❌ {err[:80]}" if err else
                         (f"tris={tris} size={bts/1024:.0f}KB" if url else "—"))
                viewer = _viewer_btn(url) if url else ""
                results.extend([url, info, viewer])
            return (st, *results)

        ml_r_btn.click(
            _refresh_mesh_lod,
            [ml_sid],
            [ml_status,
             ml_a_url, ml_a_info, ml_a_3d,
             ml_b_url, ml_b_info, ml_b_3d,
             ml_c_url, ml_c_info, ml_c_3d,
             ml_d_url, ml_d_info, ml_d_3d],
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
        # Builds three buttons per GLB url: View (Babylon, in-app), Download
        # (direct GLB), and a fallback to 3dviewer.net.
        # The "View" path goes to /static/babylon_viewer.html on the same
        # gradio origin so we don't need to know the public host name.
        def _viewer_btn(url: str) -> str:
            if not url:
                return ""
            import urllib.parse as _up
            from html import escape as _esc
            viewer_url = "/static/babylon_viewer.html?model=" + _up.quote(url, safe="")
            fallback   = "https://3dviewer.net/#model=" + url
            u = _esc(url, quote=True)
            return (
                '<div style="display:flex; gap:8px; flex-wrap:wrap;">'
                f'<a href="{_esc(viewer_url, quote=True)}" target="_blank" rel="noopener noreferrer" '
                'style="display:inline-block;padding:6px 14px;background:#2563eb;'
                'color:#fff;border-radius:6px;text-decoration:none;font-size:13px;">'
                '🧊 View in Babylon</a>'
                f'<a href="{u}" download target="_blank" rel="noopener noreferrer" '
                'style="display:inline-block;padding:6px 14px;background:#10b981;'
                'color:#fff;border-radius:6px;text-decoration:none;font-size:13px;">'
                '⬇️ Download GLB</a>'
                f'<a href="{_esc(fallback, quote=True)}" target="_blank" rel="noopener noreferrer" '
                'style="display:inline-block;padding:6px 14px;background:#6b7280;'
                'color:#fff;border-radius:6px;text-decoration:none;font-size:13px;">'
                '🌐 3dviewer.net</a>'
                '</div>'
            )

        # Multi-LOD viewer link — opens Babylon with all 4 LODs and a switcher
        def _multi_lod_btn(lod_a: str, lod_b: str, lod_c: str, lod_d: str) -> str:
            urls = {"a": lod_a, "b": lod_b, "c": lod_c, "d": lod_d}
            urls = {k: v for k, v in urls.items() if v}
            if not urls:
                return ""
            import urllib.parse as _up
            from html import escape as _esc
            qs = "&".join(f"lod_{k}=" + _up.quote(v, safe="") for k, v in urls.items())
            viewer_url = "/static/babylon_viewer.html?" + qs
            return (
                f'<div style="margin-top:8px;">'
                f'<a href="{_esc(viewer_url, quote=True)}" target="_blank" rel="noopener noreferrer" '
                'style="display:inline-block;padding:8px 16px;background:#a855f7;'
                'color:#fff;border-radius:6px;text-decoration:none;font-size:13px;font-weight:600;">'
                '🎚️ Open all LODs in Babylon (compare)</a>'
                '</div>'
            )

        tr_url.change(_viewer_btn, [tr_url], [tr_3d_btn])
        px_url.change(_viewer_btn, [px_url], [px_3d_btn])
        rg_url.change(_viewer_btn, [rg_url], [rg_3d_btn])
        ml_a_url.change(_viewer_btn, [ml_a_url], [ml_a_3d])
        ml_b_url.change(_viewer_btn, [ml_b_url], [ml_b_3d])
        ml_c_url.change(_viewer_btn, [ml_c_url], [ml_c_3d])
        ml_d_url.change(_viewer_btn, [ml_d_url], [ml_d_3d])

        # ── Auto-refresh timer ────────────────────────────────────────────────
        _ACTIVE = {"queued", "running"}

        def _tick(fx_s, s1_s, tr_s, px_s, rg_s):
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
            px_st, px_u = _r(px_s)
            rg_st, rg_u = _r(rg_s)

            still = any(s in _ACTIVE for s in [fx_st, s1_st, tr_st, px_st, rg_st])

            return (
                fx_st, fx_u, _url_to_img(fx_u),
                fx_side_u, _url_to_img(fx_side_u),
                fx_back_u, _url_to_img(fx_back_u),
                s1_st, s1_u, _url_to_img(s1_u, 350),
                tr_st, tr_u, _viewer_btn(tr_u),
                px_st, px_u, _viewer_btn(px_u),
                rg_st, rg_u, _viewer_btn(rg_u),
                gr.Timer(active=still),
            )

        stage_timer.tick(
            _tick,
            [fx_sid, s1_sid, tr_sid, px_sid, rg_sid],
            [fx_status, fx_url, fx_img,
             fx_side_url, fx_side_img, fx_back_url, fx_back_img,
             s1_status, s1_url, s1_img,
             tr_status, tr_url, tr_3d_btn,
             px_status, px_url, px_3d_btn,
             rg_status, rg_url, rg_3d_btn,
             stage_timer],
        )

    return tab
