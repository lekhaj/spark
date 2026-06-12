"""
schema_routes.py — versioned JSON Schemas as data (CycleZero T00)
=================================================================

Game-structure specs (zones, missions, terrain, systems) validate against
JSON Schemas that are **editable data with versions**, stored in Mongo
collection ``spec_schemas`` — not files in a repo. Artifacts pin the schema
version they were validated against; packets render from the active version.

Invariants
----------
  - ``version`` is monotonically increasing per ``schema_key``.
  - Exactly one active version per key.
  - Versions are immutable once written; no deletes. Rollback = move the
    active pointer (``POST /schemas/{key}/activate/{version}``).
  - Every new version requires a one-line ``changelog``.
  - ``engine_bound`` schemas answer with ``engine_sync_required: true`` on
    new versions (consumed by T10's auto-draft; only the flag for now).

Routes mounted at ``/schemas`` (set in app/main.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from worker.lib import manual_gen_schema as mgs

log = logging.getLogger("schema_routes")

router = APIRouter()

COLLECTION = "spec_schemas"


def _db():
    """Open a Mongo connection. Overridden in tests (mongomock)."""
    return mgs.get_db()


# ─── Models ───────────────────────────────────────────────────────────────────

class SchemaCreate(BaseModel):
    json_schema: Dict[str, Any]
    title: Optional[str] = None
    engine_bound: Optional[bool] = None
    changelog: str = Field(min_length=1)


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {**doc}
    out.pop("_id", None)
    created = out.get("created_at")
    if isinstance(created, datetime):
        out["created_at"] = created.isoformat()
    return out


# ─── Reads ────────────────────────────────────────────────────────────────────

@router.get("")
def list_schemas() -> List[Dict[str, Any]]:
    """List schema keys with their active version + engine_bound flag."""
    col = _db()[COLLECTION]
    out = []
    for doc in col.find({"active": True}).sort("schema_key", 1):
        out.append({
            "schema_key":   doc["schema_key"],
            "version":      doc["version"],
            "title":        doc.get("title", doc["schema_key"]),
            "engine_bound": bool(doc.get("engine_bound", False)),
        })
    return out


@router.get("/{key}")
def get_versions(key: str) -> List[Dict[str, Any]]:
    """All versions for a key, newest first."""
    docs = list(_db()[COLLECTION].find({"schema_key": key}).sort("version", -1))
    if not docs:
        raise HTTPException(404, f"unknown schema key: {key}")
    return [_serialize(d) for d in docs]


@router.get("/{key}/active")
def get_active(key: str) -> Dict[str, Any]:
    doc = _db()[COLLECTION].find_one({"schema_key": key, "active": True})
    if not doc:
        raise HTTPException(404, f"no active version for schema key: {key}")
    return _serialize(doc)


@router.get("/{key}/v/{version}")
def get_version(key: str, version: int) -> Dict[str, Any]:
    doc = _db()[COLLECTION].find_one({"schema_key": key, "version": version})
    if not doc:
        raise HTTPException(404, f"{key} has no version {version}")
    return _serialize(doc)


# ─── Writes ───────────────────────────────────────────────────────────────────

@router.post("/{key}")
def post_version(key: str, body: SchemaCreate) -> Dict[str, Any]:
    """
    Append a new version: validates the payload is itself a valid
    draft 2020-12 schema, assigns version = max+1, marks it active.
    """
    try:
        Draft202012Validator.check_schema(body.json_schema)
    except SchemaError as e:
        raise HTTPException(422, f"json_schema is not a valid draft 2020-12 schema: {e.message}")

    col = _db()[COLLECTION]
    prev = col.find_one({"schema_key": key, "active": True})
    latest = col.find_one({"schema_key": key}, sort=[("version", -1)])
    version = (latest["version"] + 1) if latest else 1

    # New versions inherit title/engine_bound from the previous active
    # version unless explicitly overridden in the body.
    title = body.title if body.title is not None else (prev or {}).get("title", key)
    engine_bound = (
        body.engine_bound if body.engine_bound is not None
        else bool((prev or {}).get("engine_bound", False))
    )

    doc = {
        "schema_key":   key,
        "version":      version,
        "title":        title,
        "engine_bound": engine_bound,
        "json_schema":  body.json_schema,
        "changelog":    body.changelog,
        "created_at":   datetime.now(timezone.utc),
        "active":       True,
    }
    col.update_many({"schema_key": key, "active": True}, {"$set": {"active": False}})
    col.insert_one(doc)

    out = _serialize(doc)
    if engine_bound:
        out["engine_sync_required"] = True
    return out


@router.post("/{key}/activate/{version}")
def activate_version(key: str, version: int) -> Dict[str, Any]:
    """Roll back (or forward) the active pointer to an existing version."""
    col = _db()[COLLECTION]
    target = col.find_one({"schema_key": key, "version": version})
    if not target:
        raise HTTPException(404, f"{key} has no version {version}")
    col.update_many({"schema_key": key, "active": True}, {"$set": {"active": False}})
    col.update_one({"schema_key": key, "version": version}, {"$set": {"active": True}})
    target["active"] = True
    return _serialize(target)
