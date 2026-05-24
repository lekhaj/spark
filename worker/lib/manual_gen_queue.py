"""
manual_gen_queue.py — Pure queue functions for manual generation stages
=========================================================================

Why this file exists
--------------------
The Gradio UI's `_q_<stage>` functions in
``app/gradio/pages/generation_studio_page.py`` mix three responsibilities:

  1. Mongo write (run document creation / version bumping)
  2. Redis push (queue a worker task)
  3. Gradio UI feedback (``gr.update(...)`` tuples, ``gr.Info`` toasts)

This module extracts (1) and (2) as pure functions so the same logic can be
called from:

  - The existing Gradio UI (no change in PR-A — handlers still work)
  - The new FastAPI REST routes in ``app/routes/manual_gen_routes.py``
  - Any future CLI / cron / batch caller

No ``gradio`` import here. No UI strings. Pure data in, typed dataclass out.

Scope
-----
PR-A covers ``flux``, ``hunyuan3d``, ``rig`` as the proof-of-pattern stages.
These three form a clean text → image → mesh → rig path that demos cleanly.
The remaining stages (normalize, flux_pose, sd_tpose, trellis, pixal3d,
mesh_lod) get added in a follow-up once the pattern is approved.

Status feedback
---------------
Pure functions accept an optional ``on_status`` callback ``(level, message)``
so a caller (Gradio) can render toasts without coupling the queue logic to a
particular UI. REST callers pass None — status appears in the returned
QueueResult and via subsequent reads on the run document.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional

import redis

from . import manual_gen_schema as mgs

log = logging.getLogger("manual_gen_queue")

# ─── Types ────────────────────────────────────────────────────────────────────

StatusCallback = Optional[Callable[[str, str], None]]  # (level, message)


@dataclass
class QueueResult:
    """Returned by every ``queue_<stage>`` function."""
    run_id: str
    task_id: str
    stage: str
    char_label: str
    major: int
    minor: int
    version: str  # "M.N"
    status: str  # always "queued" on success
    queue: str  # which Redis queue it was pushed to

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedRun:
    """Internal: result of resolving / creating a run before queueing."""
    run_id: str
    major: int
    minor: int
    is_new_minor: bool  # True if we had to auto-bump the minor version


# ─── Redis / queue resolution (env-driven, no UI deps) ────────────────────────

def _redis_client() -> redis.Redis:
    """Build a Redis client from env. Matches the gradio module's defaults."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD") or None
    return redis.Redis(
        host=host, port=port, password=password, db=0, decode_responses=True,
    )


_ACTIVE_QUEUE_KEY = "manual_gen:active_queue"
_DEFAULT_QUEUE_ENV = "MANUAL_GEN_QUEUE"
_DEFAULT_QUEUE_FALLBACK = "manual_gen_tasks_spot"


def resolve_active_queue(r: Optional[redis.Redis] = None) -> str:
    """
    Resolve the currently-selected manual_gen queue.

    Mirrors ``_active_queue_from_redis`` in the gradio module: if the UI has
    set a queue, use it; otherwise fall back to the env default; finally fall
    back to the spot queue (since spark_l4 is deleted).
    """
    r = r or _redis_client()
    try:
        val = r.get(_ACTIVE_QUEUE_KEY)
        if val:
            return val
    except Exception as e:
        log.debug("Redis active-queue lookup failed: %s", e)
    return os.getenv(_DEFAULT_QUEUE_ENV, _DEFAULT_QUEUE_FALLBACK)


def push_task_minimal(
    payload: Dict[str, Any],
    queue: Optional[str] = None,
    r: Optional[redis.Redis] = None,
    ensure_gpu: bool = True,
    on_status: StatusCallback = None,
) -> str:
    """
    Push a task onto the manual_gen Redis queue and return the task_id.

    Pure version of ``_push_task`` from ``generation_studio_page.py`` —
    no Gradio imports, no toast calls. Optional cold-start prewarm wait is
    intentionally omitted here; the worker handles slow first-inference
    gracefully on its own.

    Parameters
    ----------
    payload : dict
        Task body. A ``task_id`` is generated and added if not present.
    queue : str, optional
        Target queue name. Defaults to ``resolve_active_queue()``.
    r : redis.Redis, optional
        Reusable Redis client. One is built if omitted.
    ensure_gpu : bool
        If True, call ``ensure_gpu_ready()`` from lib.gpu_launcher before
        pushing. Fails silently to logs if the launcher isn't importable.
    on_status : callable, optional
        Receives ``(level, message)`` for UI feedback. ``level`` is one of
        ``info``, ``warn``, ``error``.

    Returns
    -------
    task_id : str
        UUID assigned to the queued task.
    """
    r = r or _redis_client()
    queue = queue or resolve_active_queue(r)

    if ensure_gpu:
        try:
            from lib.gpu_launcher import ensure_gpu_ready  # type: ignore
            ok, reason = ensure_gpu_ready()
            if not ok and reason != "disabled":
                msg = f"GPU not ready (reason={reason}); task will wait in queue."
                log.warning(msg)
                if on_status:
                    on_status("warn", msg)
        except Exception as e:
            log.debug("gpu_launcher unavailable: %s", e)

    if "task_id" not in payload:
        payload["task_id"] = str(uuid.uuid4())
    payload.setdefault("queued_at", time.time())

    try:
        r.rpush(queue, json.dumps(payload))
    except Exception as e:
        log.error("Redis rpush failed: %s", e)
        if on_status:
            on_status("error", f"Failed to queue task: {e}")
        raise

    log.info("Queued task %s on %s (type=%s)",
             payload["task_id"][:8], queue, payload.get("type", "?"))
    return payload["task_id"]


# ─── Run preparation (Mongo-side) ─────────────────────────────────────────────

class StageBusyError(Exception):
    """Raised when a run is already queued or running and cannot be re-queued."""
    def __init__(self, run_id: str, status: str, stage: str):
        super().__init__(f"{stage} is already {status} (run {run_id[:8]}…)")
        self.run_id = run_id
        self.status = status


def prepare_run(
    db,
    char_label: str,
    stage: str,
    major: int,
    minor: int,
    prompt: str = "",
    negative: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> PreparedRun:
    """
    Find or create the right run document to queue.

    Mirrors ``_prepare_run`` in the gradio module but returns a typed
    PreparedRun instead of Gradio update tuples.

    Lifecycle handling:
      - existing run, status idle  → reuse
      - existing run, status done  → auto-bump to next minor (new run)
      - existing run, status error → auto-bump to next minor (new run)
      - existing run, status queued/running → raise StageBusyError
      - no existing run            → create at (major, minor)
    """
    if not char_label:
        raise ValueError("char_label is required")
    major = int(major)
    minor = int(minor)
    params = params or {}

    # Make sure the character is registered (idempotent)
    mgs.create_character(db, char_label)

    existing = mgs.get_run_for(db, char_label, stage, major, minor)
    if existing:
        status = existing.get("status", "idle")
        if status in ("queued", "running"):
            raise StageBusyError(str(existing["_id"]), status, stage)
        if status in ("done", "error"):
            run_id, new_minor = mgs.auto_retry_run(
                db, char_label, stage, major,
                prompt=prompt, negative=negative, params=params,
            )
            return PreparedRun(run_id=run_id, major=major, minor=new_minor,
                               is_new_minor=True)
        # status == "idle" — reuse
        return PreparedRun(run_id=str(existing["_id"]), major=major,
                           minor=minor, is_new_minor=False)

    # Brand new
    run_id = mgs.create_run(
        db, char_label, stage, major, minor,
        prompt=prompt, negative=negative, params=params,
    )
    return PreparedRun(run_id=run_id, major=major, minor=minor,
                       is_new_minor=False)


# ─── Stage queueing — proof-of-pattern subset ─────────────────────────────────
# PR-A ships these three. The same pattern extends to the other 6 stages
# (normalize, flux_pose, sd_tpose, trellis, pixal3d, mesh_lod) — see the
# corresponding `_q_*` functions in generation_studio_page.py for the params.

def queue_flux(
    db,
    char_label: str,
    major: int,
    minor: int,
    prompt: str,
    negative: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    guidance: float = 4.0,
    queue: Optional[str] = None,
    on_status: StatusCallback = None,
) -> QueueResult:
    """
    Queue a FLUX text-to-image generation.

    Equivalent to the gradio ``_q_flux`` handler but returns a QueueResult.
    """
    stage = "flux"
    if not prompt.strip():
        raise ValueError("prompt is empty")

    params = {
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "guidance_scale": float(guidance),
    }
    prep = prepare_run(db, char_label, stage, major, minor,
                       prompt=prompt, negative=negative, params=params)
    mgs.save_run_params(db, prep.run_id, prompt, negative, params, stage=stage)

    r = _redis_client()
    queue_name = queue or resolve_active_queue(r)
    task_id = push_task_minimal(
        payload={
            "type": "flux",
            "session_id": prep.run_id,
            "stage": stage,
            "char_name": char_label,
            "major": prep.major,
            "minor": prep.minor,
            "prompt": prompt,
            "negative": negative,
            "params": params,
            "view": "front",
        },
        queue=queue_name,
        r=r,
        on_status=on_status,
    )
    mgs.mark_queued(db, prep.run_id, stage=stage, task_id=task_id)

    return QueueResult(
        run_id=prep.run_id, task_id=task_id, stage=stage,
        char_label=char_label, major=prep.major, minor=prep.minor,
        version=f"{prep.major}.{prep.minor}", status="queued", queue=queue_name,
    )


def queue_hunyuan3d(
    db,
    char_label: str,
    major: int,
    minor: int,
    char_type: str = "humanoid",
    src_stage: str = "flux",
    src_ver: str = "",
    queue: Optional[str] = None,
    on_status: StatusCallback = None,
) -> QueueResult:
    """
    Queue a Hunyuan3D image→3D job. ``src_ver`` is "M.N" of the source run;
    if empty, falls back to the latest 'done' image for ``src_stage``.
    """
    stage = "hunyuan3d"
    src_url = _resolve_source_image_url(db, char_label, src_stage, src_ver)
    if not src_url:
        raise ValueError(
            f"No 'done' image found for {char_label} stage={src_stage} "
            f"version={src_ver or 'latest'}"
        )

    params = {"char_type": char_type, "src_stage": src_stage, "src_ver": src_ver}
    prep = prepare_run(db, char_label, stage, major, minor, params=params)
    mgs.save_run_params(db, prep.run_id, "", "", params, stage=stage)

    r = _redis_client()
    queue_name = queue or resolve_active_queue(r)
    task_id = push_task_minimal(
        payload={
            "type": "hunyuan3d",
            "session_id": prep.run_id,
            "stage": stage,
            "char_name": char_label,
            "major": prep.major,
            "minor": prep.minor,
            "params": params,
            "input_stage": src_stage,
            "input_url": src_url,
            "char_type": char_type,
        },
        queue=queue_name,
        r=r,
        on_status=on_status,
    )
    mgs.mark_queued(db, prep.run_id, stage=stage, task_id=task_id)

    return QueueResult(
        run_id=prep.run_id, task_id=task_id, stage=stage,
        char_label=char_label, major=prep.major, minor=prep.minor,
        version=f"{prep.major}.{prep.minor}", status="queued", queue=queue_name,
    )


def queue_rig(
    db,
    char_label: str,
    major: int,
    minor: int,
    char_type: str = "humanoid",
    trellis_src_ver: str = "",
    queue: Optional[str] = None,
    on_status: StatusCallback = None,
) -> QueueResult:
    """
    Queue an Auto-Rig Pro rigging job over the most recent (or specified)
    trellis output. ``trellis_src_ver`` is "M.N"; if empty, uses latest done.
    """
    stage = "rig"
    src_url = _resolve_source_image_url(db, char_label, "trellis", trellis_src_ver)
    if not src_url:
        raise ValueError(
            f"No 'done' trellis output found for {char_label} version="
            f"{trellis_src_ver or 'latest'}"
        )

    params = {"char_type": char_type, "trellis_src_ver": trellis_src_ver}
    prep = prepare_run(db, char_label, stage, major, minor, params=params)
    mgs.save_run_params(db, prep.run_id, "", "", params, stage=stage)

    r = _redis_client()
    queue_name = queue or resolve_active_queue(r)
    task_id = push_task_minimal(
        payload={
            "type": "rig",
            "session_id": prep.run_id,
            "stage": stage,
            "char_name": char_label,
            "major": prep.major,
            "minor": prep.minor,
            "params": params,
            "input_stage": "trellis",
            "input_url": src_url,
            "char_type": char_type,
        },
        queue=queue_name,
        r=r,
        on_status=on_status,
    )
    mgs.mark_queued(db, prep.run_id, stage=stage, task_id=task_id)

    return QueueResult(
        run_id=prep.run_id, task_id=task_id, stage=stage,
        char_label=char_label, major=prep.major, minor=prep.minor,
        version=f"{prep.major}.{prep.minor}", status="queued", queue=queue_name,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_source_image_url(
    db, char_label: str, src_stage: str, src_ver: str,
) -> str:
    """
    Look up the source image URL. If ``src_ver`` is "M.N", fetch that exact
    run; otherwise return the latest 'done' image for the stage.
    """
    if src_ver:
        try:
            major_str, minor_str = src_ver.split(".")
            doc = mgs.get_run_for(db, char_label, src_stage,
                                  int(major_str), int(minor_str)) or {}
            return doc.get("image_url") or ""
        except (ValueError, KeyError):
            pass
    return mgs.get_latest_done_image_url(db, char_label, src_stage) or ""


# Re-export for convenience
__all__ = [
    "QueueResult",
    "PreparedRun",
    "StageBusyError",
    "StatusCallback",
    "prepare_run",
    "push_task_minimal",
    "resolve_active_queue",
    "queue_flux",
    "queue_hunyuan3d",
    "queue_rig",
]
