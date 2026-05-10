"""
manual_gen_schema.py — MongoDB CRUD for manual_gen_sessions
============================================================

Collection: manual_gen_sessions  (in World_builder DB)

Each session tracks the full multi-stage generation pipeline for a single
character version: Flux concept → normalize → SD Stage 1 → SD Stage 2 →
multiview side/back → TRELLIS 3D.

Stage lifecycle:
    idle → queued → running → done
                            → error
"""

import os
import time
import uuid
from typing import Optional

import pymongo

# ── Connection defaults ───────────────────────────────────────────────────────
# Reads from env so the same module works from local (public IP) and on-instance (localhost).
MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "World_builder")

# Collection name — centralised here so callers never hard-code it.
COLLECTION = "manual_gen_sessions"

# ── Valid stage names ─────────────────────────────────────────────────────────
STAGE_NAMES = (
    "flux",
    "normalize",
    "sd_stage1",
    "sd_stage2",
    "multiview_side",
    "multiview_back",
    "trellis",
    "rig",
)


# ── Connection ────────────────────────────────────────────────────────────────

def get_db(uri: str = MONGO_URI, db_name: str = MONGO_DB, timeout_ms: int = 10_000):
    """Return a pymongo Database handle.

    Creates a fresh MongoClient each call — callers that need a long-lived
    connection should cache the result themselves.
    """
    client = pymongo.MongoClient(
        uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
    )
    return client[db_name]


# ── Default stage documents ───────────────────────────────────────────────────

def _default_stage_doc(prompt: str = "", negative: str = "", params: dict = None,
                        input_stage: str = "") -> dict:
    """Return a blank StageDoc with all required keys present."""
    return {
        "status":       "idle",
        "prompt":       prompt,
        "negative":     negative,
        "params":       params or {},
        "input_stage":  input_stage,
        "image_url":    None,
        "s3_key":       None,
        "task_id":      None,
        "queued_at":    None,
        "started_at":   None,
        "completed_at": None,
        "error":        None,
    }


def _default_normalize_doc() -> dict:
    """Return a blank NormalizeDoc (CPU-side, no GPU queue)."""
    return {
        "status":      "idle",
        "resize_w":    512,
        "resize_h":    512,
        "input_stage": "flux",
        "image_url":   None,
        "s3_key":      None,
        "error":       None,
    }


def _default_rig_doc() -> dict:
    """Return a blank RigDoc (Blender ARP, CPU-only, queued via manual_gen_tasks)."""
    return {
        "status":       "idle",
        "char_type":    "humanoid",
        "input_stage":  "trellis",
        "image_url":    None,   # stores the rigged GLB URL when done
        "s3_key":       None,
        "task_id":      None,
        "queued_at":    None,
        "started_at":   None,
        "completed_at": None,
        "error":        None,
    }


def _default_stages() -> dict:
    """Return a fresh stages dict populated with all per-stage defaults.

    Keys match STAGE_NAMES exactly. Params mirror the pipeline's tuned settings
    as of the current iteration; they can be overridden per-session via
    save_stage_prompts().
    """
    return {
        "flux": _default_stage_doc(
            params={
                "width":          768,
                "height":         1024,
                "steps":          4,
                "guidance_scale": 0.0,
            },
        ),
        "normalize": _default_normalize_doc(),
        "sd_stage1": _default_stage_doc(
            params={
                "denoise":          0.20,
                "cfg":              5.5,
                "steps":            20,
                "openpose_weight":  0.85,
                "canny_weight":     0.55,
                "category":         "humanoid",
            },
            input_stage="flux",
        ),
        "sd_stage2": _default_stage_doc(
            params={
                "denoise": 0.35,
                "cfg":     7.0,
                "steps":   20,
            },
            input_stage="sd_stage1",
        ),
        "multiview_side": _default_stage_doc(
            params={
                "denoise": 0.45,
                "cfg":     7.0,
                "steps":   20,
            },
            input_stage="flux",
        ),
        "multiview_back": _default_stage_doc(
            params={
                "denoise": 0.45,
                "cfg":     7.0,
                "steps":   20,
            },
            input_stage="flux",
        ),
        "trellis": _default_stage_doc(
            params={
                "input_front": "sd_stage2",
                "input_side":  "multiview_side",
                "input_back":  "multiview_back",
            },
        ),
        "rig": _default_rig_doc(),
    }


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(db, char_label: str, version: str) -> str:
    """Insert a new session document and return its session_id (UUID string).

    Args:
        db:         pymongo Database handle (from get_db()).
        char_label: User-defined character identifier, e.g. ``"wei_liang"``.
        version:    Version string, e.g. ``"v1"``.

    Returns:
        The new session's ``_id`` (UUID4 string).
    """
    now = time.time()
    session_id = str(uuid.uuid4())
    doc = {
        "_id":        session_id,
        "char_label": char_label,
        "version":    version,
        "created_at": now,
        "updated_at": now,
        "stages":     _default_stages(),
    }
    db[COLLECTION].insert_one(doc)
    return session_id


def get_session(db, session_id: str) -> Optional[dict]:
    """Return the session document for *session_id*, or ``None`` if not found."""
    return db[COLLECTION].find_one({"_id": session_id})


def list_sessions(db, char_label: str = None) -> list:
    """Return all sessions, optionally filtered by *char_label*.

    Results are sorted by ``created_at`` descending (most recent first).

    Args:
        db:         pymongo Database handle.
        char_label: When provided, only sessions for this character are returned.

    Returns:
        List of session documents (dicts).
    """
    filt = {"char_label": char_label} if char_label else {}
    return list(db[COLLECTION].find(filt).sort("created_at", pymongo.DESCENDING))


def list_versions(db, char_label: str) -> list:
    """Return the list of version strings that exist for *char_label*.

    Sorted ascending so callers can easily determine the latest version.

    Args:
        db:         pymongo Database handle.
        char_label: Character identifier to query.

    Returns:
        List of version strings, e.g. ``["v1", "v2", "v3"]``.
    """
    docs = db[COLLECTION].find(
        {"char_label": char_label},
        {"version": 1},
    ).sort("created_at", pymongo.ASCENDING)
    return [d["version"] for d in docs]


# ── Stage field updates ───────────────────────────────────────────────────────

def _validate_stage(stage: str):
    """Raise ValueError if *stage* is not a recognised stage name."""
    if stage not in STAGE_NAMES:
        raise ValueError(
            f"Unknown stage '{stage}'. Must be one of: {', '.join(STAGE_NAMES)}"
        )


def update_stage(db, session_id: str, stage: str, fields: dict):
    """Apply a partial update to a single stage using dot-notation ``$set``.

    Also bumps the top-level ``updated_at`` timestamp.

    Args:
        db:         pymongo Database handle.
        session_id: Target session ``_id``.
        stage:      Stage name (must be in STAGE_NAMES).
        fields:     Mapping of field name → new value, applied under
                    ``stages.<stage>.*``.

    Raises:
        ValueError: If *stage* is not recognised.
    """
    _validate_stage(stage)
    set_payload = {f"stages.{stage}.{k}": v for k, v in fields.items()}
    set_payload["updated_at"] = time.time()
    db[COLLECTION].update_one(
        {"_id": session_id},
        {"$set": set_payload},
    )


def save_stage_prompts(db, session_id: str, stage: str,
                       prompt: str, negative: str, params: dict):
    """Persist prompt, negative prompt, and generation params for *stage*.

    Does **not** change the stage ``status`` — safe to call before queuing.

    Args:
        db:         pymongo Database handle.
        session_id: Target session ``_id``.
        stage:      Stage name.
        prompt:     Positive prompt text.
        negative:   Negative prompt text.
        params:     Dict of generation parameters (width, steps, denoise, etc.).
    """
    update_stage(db, session_id, stage, {
        "prompt":   prompt,
        "negative": negative,
        "params":   params,
    })


def mark_queued(db, session_id: str, stage: str, task_id: str):
    """Transition *stage* to ``queued`` and record the Redis *task_id*.

    Args:
        db:         pymongo Database handle.
        session_id: Target session ``_id``.
        stage:      Stage name.
        task_id:    UUID of the task pushed to Redis.
    """
    update_stage(db, session_id, stage, {
        "status":    "queued",
        "task_id":   task_id,
        "queued_at": time.time(),
        # Clear any previous error if re-queuing after failure.
        "error":     None,
    })


def mark_running(db, session_id: str, stage: str):
    """Transition *stage* to ``running`` and record ``started_at``.

    Called by the GPU worker immediately after popping the task.

    Args:
        db:         pymongo Database handle.
        session_id: Target session ``_id``.
        stage:      Stage name.
    """
    update_stage(db, session_id, stage, {
        "status":     "running",
        "started_at": time.time(),
    })


def mark_done(db, session_id: str, stage: str, image_url: str, s3_key: str):
    """Transition *stage* to ``done`` and record the output image location.

    Args:
        db:         pymongo Database handle.
        session_id: Target session ``_id``.
        stage:      Stage name.
        image_url:  Public HTTPS URL of the generated image on S3.
        s3_key:     S3 object key (without bucket prefix).
    """
    update_stage(db, session_id, stage, {
        "status":       "done",
        "image_url":    image_url,
        "s3_key":       s3_key,
        "completed_at": time.time(),
        "error":        None,
    })


def mark_error(db, session_id: str, stage: str, error: str):
    """Transition *stage* to ``error`` and record the error message.

    Args:
        db:         pymongo Database handle.
        session_id: Target session ``_id``.
        stage:      Stage name.
        error:      Human-readable error string (typically ``str(exc)``).
    """
    update_stage(db, session_id, stage, {
        "status":       "error",
        "completed_at": time.time(),
        "error":        error,
    })


# ── Convenience read helpers ──────────────────────────────────────────────────

def get_stage_image_url(db, session_id: str, stage: str) -> Optional[str]:
    """Return the ``image_url`` for a completed stage, or ``None``.

    Args:
        db:         pymongo Database handle.
        session_id: Target session ``_id``.
        stage:      Stage name.

    Returns:
        Public S3 URL string if the stage is ``done`` and has an image,
        otherwise ``None``.
    """
    _validate_stage(stage)
    doc = db[COLLECTION].find_one(
        {"_id": session_id},
        {f"stages.{stage}.image_url": 1},
    )
    if doc is None:
        return None
    return (doc.get("stages") or {}).get(stage, {}).get("image_url")
