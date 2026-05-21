"""
manual_gen_schema.py — MongoDB CRUD for manual generation stage runs
=====================================================================

Collections
-----------
  manual_gen_stage_runs   — all stages except sd_tpose (flux, normalize, trellis, rig, …)
  manual_gen_tpose_runs   — sd_tpose stage only (separate for query isolation)
  manual_gen_characters   — parent asset registry (char label → created_at)

Stage routing
-------------
  All CRUD functions accept an optional `coll` parameter.
  If omitted, `_coll_for_stage(stage)` auto-routes:
    "sd_tpose"  → TPOSE_COLLECTION
    anything else → COLLECTION

  This means callers that pass an explicit stage never need to think about
  which collection to use — the routing is automatic.

Version scheme
--------------
  major.minor  (ints stored explicitly)
  major — user-triggered design iteration (user clicks "+ New Major" per stage)
  minor — auto-incremented on retry when the previous run had status "error"

Stage lifecycle:  idle → queued → running → done
                                           → error
"""

import os
import time
import uuid
from typing import Optional

import pymongo

# ── Connection defaults ───────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "World_builder")

# ── Collections ───────────────────────────────────────────────────────────────
COLLECTION            = "manual_gen_stage_runs"    # all stages except sd_tpose
TPOSE_COLLECTION      = "manual_gen_tpose_runs"    # sd_tpose stage only
CHARACTERS_COLLECTION = "manual_gen_characters"    # parent asset registry

# ── Valid stage names ─────────────────────────────────────────────────────────
STAGE_NAMES = (
    "flux",
    "normalize",
    "flux_pose",        # FLUX.1-dev + ControlNet-Union-Pro-2.0 pose-conditioned gen
    "sd_tpose",         # SD1.5 + ControlNet T-pose with IP-Adapter identity lock
    # "sd_stage1",      # legacy — superseded by sd_tpose
    # "sd_stage2",      # legacy — detail pass removed (handled in T-pose stage)
    # "multiview_side", # future — multi-view generation (not yet active)
    # "multiview_back", # future — multi-view generation (not yet active)
    "trellis",
    "pixal3d",          # TencentARC/Pixal3D — alternative image→3D (isolated conda env)
    "mesh_lod",         # CPU stage — generates 4 LOD GLBs from a source GLB (Blender + gltfpack)
    "rig",
)


# ── Collection routing ────────────────────────────────────────────────────────

def _coll_for_stage(stage: str) -> str:
    """Return the correct collection name for a given stage."""
    return TPOSE_COLLECTION if stage == "sd_tpose" else COLLECTION


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
    coll: str = None,
) -> str:
    """Insert a new stage-run document and return its run_id (UUID string)."""
    coll   = coll or _coll_for_stage(stage)
    run_id = str(uuid.uuid4())
    doc    = {"_id": run_id,
              **_new_run_doc(char_label, stage, major, minor, prompt, negative, params)}
    db[coll].insert_one(doc)
    return run_id


def get_run(db, run_id: str, stage: str = "", coll: str = None) -> Optional[dict]:
    """Return the full run document by _id, or None.
    If stage is unknown, searches all known collections."""
    if coll:
        return db[coll].find_one({"_id": run_id})
    if stage:
        return db[_coll_for_stage(stage)].find_one({"_id": run_id})
    # Unknown stage — try both collections
    doc = db[TPOSE_COLLECTION].find_one({"_id": run_id})
    return doc if doc else db[COLLECTION].find_one({"_id": run_id})


def get_run_any(db, run_id: str) -> Optional[dict]:
    """Search all collections for a run_id. Used by UI refresh handlers."""
    doc = db[TPOSE_COLLECTION].find_one({"_id": run_id})
    return doc if doc else db[COLLECTION].find_one({"_id": run_id})


def get_run_for(
    db,
    char_label: str,
    stage: str,
    major: int,
    minor: int,
    coll: str = None,
) -> Optional[dict]:
    """Return the most-recently-created run for (char_label, stage, major, minor)."""
    coll = coll or _coll_for_stage(stage)
    return db[coll].find_one(
        {"char_label": char_label, "stage": stage,
         "major": int(major), "minor": int(minor)},
        sort=[("created_at", pymongo.DESCENDING)],
    )


def get_latest_done_run(db, char_label: str, stage: str, coll: str = None) -> Optional[dict]:
    """Return the most recent run with status='done' for char+stage (any version)."""
    coll = coll or _coll_for_stage(stage)
    return db[coll].find_one(
        {"char_label": char_label, "stage": stage, "status": "done"},
        sort=[("created_at", pymongo.DESCENDING)],
    )


def ensure_run(
    db,
    char_label: str,
    stage: str,
    major: int,
    minor: int,
    coll: str = None,
) -> str:
    """Return existing run_id for (char, stage, major, minor), or create one."""
    coll = coll or _coll_for_stage(stage)
    doc  = get_run_for(db, char_label, stage, major, minor, coll=coll)
    if doc:
        return doc["_id"]
    return create_run(db, char_label, stage, major, minor, coll=coll)


def auto_retry_run(
    db,
    char_label: str,
    stage: str,
    major: int,
    prompt: str = "",
    negative: str = "",
    params: dict = None,
    coll: str = None,
) -> tuple[str, int]:
    """Create a new minor version for a retry. Returns (run_id, new_minor)."""
    coll      = coll or _coll_for_stage(stage)
    new_minor = next_stage_minor(db, char_label, stage, major, coll=coll)
    run_id    = create_run(db, char_label, stage, major, new_minor,
                           prompt, negative, params, coll=coll)
    return run_id, new_minor


# ── Character registry ────────────────────────────────────────────────────────

def create_character(db, label: str) -> bool:
    """Register a character label. Idempotent."""
    try:
        db[CHARACTERS_COLLECTION].update_one(
            {"_id": label},
            {"$setOnInsert": {"_id": label, "created_at": time.time()}},
            upsert=True,
        )
        return True
    except Exception:
        return False


def list_characters(db) -> list[str]:
    """Return all character labels from registry + stage runs (backward compat)."""
    try:
        from_registry = {d["_id"] for d in db[CHARACTERS_COLLECTION].find({}, {"_id": 1})}
    except Exception:
        from_registry = set()
    try:
        from_runs = set(db[COLLECTION].distinct("char_label"))
    except Exception:
        from_runs = set()
    try:
        from_tpose = set(db[TPOSE_COLLECTION].distinct("char_label"))
    except Exception:
        from_tpose = set()
    merged = sorted([l for l in (from_registry | from_runs | from_tpose) if l])
    return merged


def list_chars(db) -> list[str]:
    """Alias for list_characters."""
    return list_characters(db)


# ── Version list helpers ──────────────────────────────────────────────────────

def list_stages_for_char(db, char_label: str) -> list[str]:
    stages = set(db[COLLECTION].distinct("stage", {"char_label": char_label}))
    stages |= set(db[TPOSE_COLLECTION].distinct("stage", {"char_label": char_label}))
    return sorted([s for s in stages if s])


def list_stage_majors(db, char_label: str, stage: str, coll: str = None) -> list[int]:
    coll  = coll or _coll_for_stage(stage)
    docs  = db[coll].find(
        {"char_label": char_label, "stage": stage},
        {"major": 1},
    )
    majors = {int(d["major"]) for d in docs if "major" in d}
    return sorted(majors) or [1]


def list_stage_minors(db, char_label: str, stage: str, major: int,
                      coll: str = None) -> list[int]:
    coll  = coll or _coll_for_stage(stage)
    docs  = db[coll].find(
        {"char_label": char_label, "stage": stage, "major": int(major)},
        {"minor": 1},
    )
    minors = {int(d["minor"]) for d in docs if "minor" in d}
    return sorted(minors) or [0]


def next_stage_major(db, char_label: str, stage: str, coll: str = None) -> int:
    coll   = coll or _coll_for_stage(stage)
    majors = list_stage_majors(db, char_label, stage, coll=coll)
    return (max(majors) + 1) if majors else 1


def next_stage_minor(db, char_label: str, stage: str, major: int,
                     coll: str = None) -> int:
    coll   = coll or _coll_for_stage(stage)
    minors = list_stage_minors(db, char_label, stage, major, coll=coll)
    return (max(minors) + 1) if minors else 1


# ── Field updates ─────────────────────────────────────────────────────────────

def update_run(db, run_id: str, fields: dict, coll: str = COLLECTION) -> None:
    """Apply arbitrary field updates to a run document."""
    set_payload = {**fields, "updated_at": time.time()}
    db[coll].update_one({"_id": run_id}, {"$set": set_payload})


def save_run_params(
    db, run_id: str, prompt: str, negative: str, params: dict,
    stage: str = "", coll: str = None,
) -> None:
    coll = coll or (_coll_for_stage(stage) if stage else COLLECTION)
    update_run(db, run_id, {"prompt": prompt, "negative": negative, "params": params}, coll=coll)


# ── Status transitions ────────────────────────────────────────────────────────

def mark_queued(db, run_id: str, stage: str = "", task_id: str = "") -> None:
    coll = _coll_for_stage(stage) if stage else COLLECTION
    update_run(db, run_id, {
        "status":    "queued",
        "task_id":   task_id,
        "queued_at": time.time(),
        "error":     None,
    }, coll=coll)


def mark_running(db, run_id: str, stage: str = "") -> None:
    coll = _coll_for_stage(stage) if stage else COLLECTION
    update_run(db, run_id, {
        "status":     "running",
        "started_at": time.time(),
    }, coll=coll)


def mark_done(
    db, run_id: str, stage: str = "",
    image_url: str = "", s3_key: str = "",
) -> None:
    coll = _coll_for_stage(stage) if stage else COLLECTION
    update_run(db, run_id, {
        "status":       "done",
        "image_url":    image_url,
        "s3_key":       s3_key,
        "completed_at": time.time(),
        "error":        None,
    }, coll=coll)


def mark_error(db, run_id: str, stage: str = "", error: str = "") -> None:
    coll = _coll_for_stage(stage) if stage else COLLECTION
    update_run(db, run_id, {
        "status":       "error",
        "error":        error,
        "completed_at": time.time(),
    }, coll=coll)


# ── Convenience read helpers ──────────────────────────────────────────────────

def get_run_image_url(db, run_id: str, stage: str = "", coll: str = None) -> Optional[str]:
    coll = coll or (_coll_for_stage(stage) if stage else COLLECTION)
    doc  = db[coll].find_one({"_id": run_id}, {"image_url": 1})
    return (doc or {}).get("image_url")


def get_latest_done_image_url(db, char_label: str, stage: str,
                              coll: str = None) -> Optional[str]:
    doc = get_latest_done_run(db, char_label, stage, coll=coll)
    return (doc or {}).get("image_url")


def get_view_urls(db, char_label: str, stage: str, major: int, minor: int,
                  coll: str = None) -> dict:
    """
    Return all view URLs for a given (char, stage, major, minor) run.
    Returns dict with keys: front, side, back (any may be empty string).
    front = image_url (primary, backward-compat field).
    """
    coll = coll or _coll_for_stage(stage)
    doc  = get_run_for(db, char_label, stage, major, minor, coll=coll) or {}
    return {
        "front": doc.get("image_url") or "",
        "side":  doc.get("side_url")  or "",
        "back":  doc.get("back_url")  or "",
    }
