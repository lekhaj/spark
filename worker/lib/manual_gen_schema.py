"""
manual_gen_schema.py — MongoDB CRUD for manual_gen_stage_runs
=============================================================

Collection: manual_gen_stage_runs  (in World_builder DB)

Data model
----------
Each document = one stage × one version run.  Stages are versioned
independently — Flux can be at v1.2 while SD Stage 1 is still at v1.0.

Version scheme
--------------
  major.minor  (ints stored explicitly)
  Display format: "1.0", "2.3" etc.

  major — user-triggered design iteration (user clicks "+ New Major" per stage)
  minor — auto-incremented on retry when the previous run for that major
          had status "error" (same intent, different attempt).

Stage lifecycle:
    idle → queued → running → done
                            → error

Backward compat
---------------
The old "manual_gen_sessions" collection (one document with all stages embedded)
is no longer written to, but result_consumer.mark_done / mark_running / mark_error
keep the same call signature so nothing in the consumer needs changing.
"""

import os
import time
import uuid
from typing import Optional

import pymongo

# ── Connection defaults ───────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "World_builder")

COLLECTION = "manual_gen_stage_runs"

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
    client = pymongo.MongoClient(
        uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
    )
    return client[db_name]


# ── Version helpers ───────────────────────────────────────────────────────────

def version_str(major: int, minor: int) -> str:
    """Format a major.minor pair as a display string: "1.0", "2.3" …"""
    return f"{major}.{minor}"


# ── Document factory ──────────────────────────────────────────────────────────

def _new_run_doc(char_label: str, stage: str, major: int, minor: int,
                 prompt: str = "", negative: str = "", params: dict = None) -> dict:
    now = time.time()
    return {
        "char_label":   char_label,
        "stage":        stage,
        "major":        int(major),
        "minor":        int(minor),
        "version":      version_str(major, minor),
        "status":       "idle",
        "prompt":       prompt,
        "negative":     negative,
        "params":       params or {},
        "input_stage":  "",
        "image_url":    None,
        "s3_key":       None,
        "task_id":      None,
        "queued_at":    None,
        "started_at":   None,
        "completed_at": None,
        "error":        None,
        "created_at":   now,
        "updated_at":   now,
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_run(
    db,
    char_label: str,
    stage: str,
    major: int,
    minor: int = 0,
    prompt: str = "",
    negative: str = "",
    params: dict = None,
) -> str:
    """
    Insert a new stage-run document and return its run_id (UUID string).

    Args:
        db:         pymongo Database handle.
        char_label: Character identifier, e.g. "college_student".
        stage:      Stage name, one of STAGE_NAMES.
        major:      Major version number (design iteration).
        minor:      Minor version number (retry within an iteration).
        prompt:     Initial prompt (can be updated later via save_run_params).
        negative:   Initial negative prompt.
        params:     Initial params dict.

    Returns:
        The new run's _id (UUID4 string).
    """
    run_id = str(uuid.uuid4())
    doc    = {"_id": run_id,
              **_new_run_doc(char_label, stage, major, minor, prompt, negative, params)}
    db[COLLECTION].insert_one(doc)
    return run_id


def get_run(db, run_id: str) -> Optional[dict]:
    """Return the full run document by _id, or None."""
    return db[COLLECTION].find_one({"_id": run_id})


def get_run_for(
    db,
    char_label: str,
    stage: str,
    major: int,
    minor: int,
) -> Optional[dict]:
    """
    Return the most-recently-created run for (char_label, stage, major, minor).
    Returns None if no such run exists.
    """
    return db[COLLECTION].find_one(
        {"char_label": char_label, "stage": stage,
         "major": int(major), "minor": int(minor)},
        sort=[("created_at", pymongo.DESCENDING)],
    )


def get_latest_done_run(db, char_label: str, stage: str) -> Optional[dict]:
    """Return the most recent run with status='done' for char+stage (any version)."""
    return db[COLLECTION].find_one(
        {"char_label": char_label, "stage": stage, "status": "done"},
        sort=[("created_at", pymongo.DESCENDING)],
    )


def ensure_run(
    db,
    char_label: str,
    stage: str,
    major: int,
    minor: int,
) -> str:
    """
    Return the existing run_id for (char, stage, major, minor), or create one.
    Never creates duplicates.
    """
    doc = get_run_for(db, char_label, stage, major, minor)
    if doc:
        return doc["_id"]
    return create_run(db, char_label, stage, major, minor)


def auto_retry_run(
    db,
    char_label: str,
    stage: str,
    major: int,
    prompt: str = "",
    negative: str = "",
    params: dict = None,
) -> tuple[str, int]:
    """
    Create a new minor version for a retry (called automatically when the
    current run for this major has status='error').

    Returns:
        (run_id, new_minor)
    """
    new_minor = next_stage_minor(db, char_label, stage, major)
    run_id    = create_run(db, char_label, stage, major, new_minor,
                           prompt, negative, params)
    return run_id, new_minor


def list_chars(db) -> list[str]:
    """Return all distinct char_labels sorted alphabetically."""
    labels = db[COLLECTION].distinct("char_label")
    return sorted([l for l in labels if l])


def list_stages_for_char(db, char_label: str) -> list[str]:
    """Return stage names that have at least one run for char_label."""
    stages = db[COLLECTION].distinct("stage", {"char_label": char_label})
    return sorted([s for s in stages if s])


def list_stage_majors(db, char_label: str, stage: str) -> list[int]:
    """Return sorted list of distinct major version numbers for (char, stage)."""
    docs   = db[COLLECTION].find(
        {"char_label": char_label, "stage": stage},
        {"major": 1},
    )
    majors = {int(d["major"]) for d in docs if "major" in d}
    return sorted(majors) or [1]


def list_stage_minors(db, char_label: str, stage: str, major: int) -> list[int]:
    """Return sorted list of minor version numbers for (char, stage, major)."""
    docs   = db[COLLECTION].find(
        {"char_label": char_label, "stage": stage, "major": int(major)},
        {"minor": 1},
    )
    minors = {int(d["minor"]) for d in docs if "minor" in d}
    return sorted(minors) or [0]


def next_stage_major(db, char_label: str, stage: str) -> int:
    """Return the next unused major version number for (char, stage)."""
    majors = list_stage_majors(db, char_label, stage)
    return (max(majors) + 1) if majors else 1


def next_stage_minor(db, char_label: str, stage: str, major: int) -> int:
    """Return the next unused minor version number for (char, stage, major)."""
    minors = list_stage_minors(db, char_label, stage, major)
    return (max(minors) + 1) if minors else 1


# ── Field updates ─────────────────────────────────────────────────────────────

def update_run(db, run_id: str, fields: dict) -> None:
    """Apply arbitrary field updates to a run document."""
    set_payload = {**fields, "updated_at": time.time()}
    db[COLLECTION].update_one({"_id": run_id}, {"$set": set_payload})


def save_run_params(
    db,
    run_id: str,
    prompt: str,
    negative: str,
    params: dict,
) -> None:
    """Persist prompt / negative / params to an existing run."""
    update_run(db, run_id, {
        "prompt":   prompt,
        "negative": negative,
        "params":   params,
    })


# ── Status transitions ────────────────────────────────────────────────────────
# The `stage` parameter is accepted for backward compatibility with
# result_consumer.py which passes it positionally — it is ignored here
# because each run document is already for a specific stage.

def mark_queued(db, run_id: str, stage: str = "", task_id: str = "") -> None:
    update_run(db, run_id, {
        "status":    "queued",
        "task_id":   task_id,
        "queued_at": time.time(),
        "error":     None,
    })


def mark_running(db, run_id: str, stage: str = "") -> None:
    update_run(db, run_id, {
        "status":     "running",
        "started_at": time.time(),
    })


def mark_done(
    db,
    run_id: str,
    stage: str = "",
    image_url: str = "",
    s3_key: str = "",
) -> None:
    update_run(db, run_id, {
        "status":       "done",
        "image_url":    image_url,
        "s3_key":       s3_key,
        "completed_at": time.time(),
        "error":        None,
    })


def mark_error(db, run_id: str, stage: str = "", error: str = "") -> None:
    update_run(db, run_id, {
        "status":       "error",
        "error":        error,
        "completed_at": time.time(),
    })


# ── Convenience read helpers ──────────────────────────────────────────────────

def get_run_image_url(db, run_id: str) -> Optional[str]:
    """Return the image_url field from a run document, or None."""
    doc = db[COLLECTION].find_one({"_id": run_id}, {"image_url": 1})
    return (doc or {}).get("image_url")


def get_latest_done_image_url(db, char_label: str, stage: str) -> Optional[str]:
    """Return image_url from the most recent done run for char+stage."""
    doc = get_latest_done_run(db, char_label, stage)
    return (doc or {}).get("image_url")
