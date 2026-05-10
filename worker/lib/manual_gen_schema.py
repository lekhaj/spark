"""
manual_gen_schema.py — MongoDB CRUD for manual_gen_sessions
============================================================

Collection: manual_gen_sessions  (in World_builder DB)

Version scheme
--------------
Each session has a (major, minor) integer pair stored explicitly.
Display format: "major.minor"  e.g.  1.0 / 1.1 / 2.0

  major — increments when the user explicitly starts a new design iteration
          (intentional prompt / concept change)
  minor — increments on automatic retries or small tweaks within an iteration
          (same intent, different run or error recovery)

Backward compatibility
  Old sessions stored version as "v1", "v2" etc.
  _parse_major_minor() maps these → (1,0), (2,0), ... so all helpers work.

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
MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "World_builder")

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
    client = pymongo.MongoClient(
        uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
    )
    return client[db_name]


# ── Version helpers ───────────────────────────────────────────────────────────

def _parse_major_minor(version: str) -> tuple[int, int]:
    """
    Parse any version string → (major, minor).

    Handles:
      "1.0"  → (1, 0)
      "2.3"  → (2, 3)
      "v1"   → (1, 0)   ← old format
      "v2"   → (2, 0)
      "3"    → (3, 0)
    """
    v = (version or "1").strip().lstrip("v")
    if "." in v:
        parts = v.split(".", 1)
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    try:
        return int(v), 0
    except ValueError:
        return 1, 0


def version_str(major: int, minor: int) -> str:
    """Format a major.minor pair as a display string: "1.0", "2.3" …"""
    return f"{major}.{minor}"


def _get_major_minor(doc: dict) -> tuple[int, int]:
    """Extract (major, minor) from a session document, handling old-format docs."""
    if "major" in doc and "minor" in doc:
        return int(doc["major"]), int(doc["minor"])
    return _parse_major_minor(doc.get("version", "1"))


# ── Default stage documents ───────────────────────────────────────────────────

def _default_stage_doc(prompt: str = "", negative: str = "", params: dict = None,
                        input_stage: str = "") -> dict:
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
    return {
        "status":       "idle",
        "char_type":    "humanoid",
        "input_stage":  "trellis",
        "image_url":    None,
        "s3_key":       None,
        "task_id":      None,
        "queued_at":    None,
        "started_at":   None,
        "completed_at": None,
        "error":        None,
    }


def _default_stages() -> dict:
    return {
        "flux": _default_stage_doc(
            params={"width": 768, "height": 1024, "steps": 4, "guidance_scale": 0.0},
        ),
        "normalize": _default_normalize_doc(),
        "sd_stage1": _default_stage_doc(
            params={"denoise": 0.20, "cfg": 5.5, "steps": 20,
                    "openpose_weight": 0.85, "canny_weight": 0.55, "category": "humanoid"},
            input_stage="flux",
        ),
        "sd_stage2": _default_stage_doc(
            params={"denoise": 0.35, "cfg": 7.0, "steps": 20},
            input_stage="sd_stage1",
        ),
        "multiview_side": _default_stage_doc(
            params={"denoise": 0.45, "cfg": 7.0, "steps": 20},
            input_stage="flux",
        ),
        "multiview_back": _default_stage_doc(
            params={"denoise": 0.45, "cfg": 7.0, "steps": 20},
            input_stage="flux",
        ),
        "trellis": _default_stage_doc(
            params={"input_front": "sd_stage2", "input_side": "multiview_side",
                    "input_back": "multiview_back"},
        ),
        "rig": _default_rig_doc(),
    }


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(db, char_label: str, major: int, minor: int = 0) -> str:
    """
    Insert a new session document and return its session_id (UUID string).

    Args:
        db:         pymongo Database handle.
        char_label: Character identifier, e.g. "college_student".
        major:      Major version number (design iteration).
        minor:      Minor version number (retry / tweak within iteration).

    Returns:
        The new session's _id (UUID4 string).
    """
    now        = time.time()
    session_id = str(uuid.uuid4())
    doc = {
        "_id":        session_id,
        "char_label": char_label,
        "major":      int(major),
        "minor":      int(minor),
        "version":    version_str(major, minor),   # "1.0", "2.3" etc.
        "created_at": now,
        "updated_at": now,
        "stages":     _default_stages(),
    }
    db[COLLECTION].insert_one(doc)
    return session_id


def get_session(db, session_id: str) -> Optional[dict]:
    return db[COLLECTION].find_one({"_id": session_id})


def get_session_for(db, char_label: str, major: int, minor: int) -> Optional[dict]:
    """
    Return the most-recently-created session for (char_label, major, minor).

    Handles both new documents (major/minor fields) and old-format docs
    (version = "v1" etc.) so nothing breaks after the schema upgrade.
    """
    # Try new-format docs first (fast indexed query)
    doc = db[COLLECTION].find_one(
        {"char_label": char_label, "major": int(major), "minor": int(minor)},
        sort=[("created_at", pymongo.DESCENDING)],
    )
    if doc:
        return doc

    # Fallback: old-format docs stored version as "v1", "1", etc.
    ver = version_str(major, minor)           # "1.0"
    old_ver = f"v{major}" if minor == 0 else None  # "v1" (only if minor==0)
    query_versions = [ver]
    if old_ver:
        query_versions.append(old_ver)

    doc = db[COLLECTION].find_one(
        {"char_label": char_label, "version": {"$in": query_versions}},
        sort=[("created_at", pymongo.DESCENDING)],
    )
    return doc


def list_sessions(db, char_label: str = None) -> list:
    filt = {"char_label": char_label} if char_label else {}
    return list(db[COLLECTION].find(filt).sort("created_at", pymongo.DESCENDING))


def list_major_versions(db, char_label: str) -> list[int]:
    """
    Return sorted list of distinct major version numbers for char_label.
    Works for both old ("v1") and new ("1.0") format docs.
    """
    docs = db[COLLECTION].find(
        {"char_label": char_label},
        {"major": 1, "version": 1},
    )
    majors = set()
    for doc in docs:
        m, _ = _get_major_minor(doc)
        majors.add(m)
    return sorted(majors) or [1]


def list_minor_versions(db, char_label: str, major: int) -> list[int]:
    """
    Return sorted list of minor version numbers for (char_label, major).
    """
    docs = db[COLLECTION].find(
        {"char_label": char_label},
        {"major": 1, "minor": 1, "version": 1},
    )
    minors = set()
    for doc in docs:
        m, n = _get_major_minor(doc)
        if m == int(major):
            minors.add(n)
    return sorted(minors) or [0]


def next_major(db, char_label: str) -> int:
    """Return the next unused major version number for char_label."""
    majors = list_major_versions(db, char_label)
    return (max(majors) + 1) if majors else 1


def next_minor(db, char_label: str, major: int) -> int:
    """Return the next unused minor version number for (char_label, major)."""
    minors = list_minor_versions(db, char_label, int(major))
    return (max(minors) + 1) if minors else 0


# ── Backward-compat alias (old code called create_session with version str) ───

def create_session_v1(db, char_label: str, version: str) -> str:
    """Legacy: accepts version as a string like 'v1' or '1.0'. Parses to major.minor."""
    major, minor = _parse_major_minor(version)
    return create_session(db, char_label, major, minor)


# ── Stage field updates ───────────────────────────────────────────────────────

def _validate_stage(stage: str):
    if stage not in STAGE_NAMES:
        raise ValueError(
            f"Unknown stage '{stage}'. Must be one of: {', '.join(STAGE_NAMES)}"
        )


def update_stage(db, session_id: str, stage: str, fields: dict):
    _validate_stage(stage)
    set_payload = {f"stages.{stage}.{k}": v for k, v in fields.items()}
    set_payload["updated_at"] = time.time()
    db[COLLECTION].update_one({"_id": session_id}, {"$set": set_payload})


def save_stage_prompts(db, session_id: str, stage: str,
                       prompt: str, negative: str, params: dict):
    update_stage(db, session_id, stage, {
        "prompt":   prompt,
        "negative": negative,
        "params":   params,
    })


def mark_queued(db, session_id: str, stage: str, task_id: str):
    update_stage(db, session_id, stage, {
        "status":    "queued",
        "task_id":   task_id,
        "queued_at": time.time(),
        "error":     None,
    })


def mark_running(db, session_id: str, stage: str):
    update_stage(db, session_id, stage, {
        "status":     "running",
        "started_at": time.time(),
    })


def mark_done(db, session_id: str, stage: str, image_url: str, s3_key: str):
    update_stage(db, session_id, stage, {
        "status":       "done",
        "image_url":    image_url,
        "s3_key":       s3_key,
        "completed_at": time.time(),
        "error":        None,
    })


def mark_error(db, session_id: str, stage: str, error: str):
    update_stage(db, session_id, stage, {
        "status":       "error",
        "completed_at": time.time(),
        "error":        error,
    })


# ── Convenience read helpers ──────────────────────────────────────────────────

def get_stage_image_url(db, session_id: str, stage: str) -> Optional[str]:
    _validate_stage(stage)
    doc = db[COLLECTION].find_one(
        {"_id": session_id},
        {f"stages.{stage}.image_url": 1},
    )
    if doc is None:
        return None
    return (doc.get("stages") or {}).get(stage, {}).get("image_url")


# ── List all distinct char_labels ─────────────────────────────────────────────

def list_chars(db) -> list[str]:
    """Return all distinct char_labels sorted alphabetically."""
    labels = db[COLLECTION].distinct("char_label")
    return sorted([l for l in labels if l])
