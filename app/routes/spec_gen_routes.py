"""
spec_gen_routes.py — versioned JSON-artifact runs (CycleZero T03+T04)
=====================================================================

JSON artifacts (zone specs, mission specs, …) get the same run model images
already have: versioned per (project, entity, stage). ``mode:"paste"`` means
the output arrives pre-generated — the user pasted an external LLM's reply;
the server validates it against the **pinned** schema version, versions it,
and stores it. ``mode:"agent"`` is reserved (501).

Versioning semantics (mirrors manual_gen major.minor):
  - first run for (project, entity, stage)            → 1.0
  - retry while major open                            → minor+1
  - create after an accepted run in current major     → major+1, minor 0
  - exactly one ``accepted`` run per major (accepting demotes siblings)

Validation pins: a run validates against the schema version captured at
creation — posting schema v2 later never invalidates a v1 artifact.
Invalid runs produce a **Fix Packet**: one copyable block the user pastes
back into the LLM chat to get corrected JSON.

Routes mounted at ``/spec-gen`` (set in app/main.py).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from jsonschema import Draft202012Validator

from worker.lib import manual_gen_schema as mgs

log = logging.getLogger("spec_gen_routes")

router = APIRouter()

COLLECTION = "spec_gen_runs"
SCHEMAS_COLLECTION = "spec_schemas"


def _db():
    """Open a Mongo connection. Overridden in tests (mongomock)."""
    return mgs.get_db()


# ─── Models ───────────────────────────────────────────────────────────────────

class SpecGenCreate(BaseModel):
    project_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)          # schema_key it validates against
    mode: Literal["paste", "agent"] = "paste"
    output: Dict[str, Any]
    input_ref: Optional[str] = None
    prompt_text: Optional[str] = None


class FeedbackBody(BaseModel):
    note: str = Field(min_length=1)


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {**doc}
    out.pop("_id", None)
    created = out.get("created_at")
    if isinstance(created, datetime):
        out["created_at"] = created.isoformat()
    return out


# ─── Validation (T04) ─────────────────────────────────────────────────────────

def _validate_output(db, stage: str, schema_version: int, output: Dict[str, Any]):
    """Validate output against the PINNED schema version. Returns error list."""
    schema_doc = db[SCHEMAS_COLLECTION].find_one(
        {"schema_key": stage, "version": schema_version}
    )
    if not schema_doc:
        return [{"json_path": "(root)", "message": f"pinned schema {stage} v{schema_version} no longer exists", "schema_path": ""}]
    validator = Draft202012Validator(schema_doc["json_schema"])
    errors = []
    for err in sorted(validator.iter_errors(output), key=lambda e: list(e.absolute_path)):
        errors.append({
            "json_path": "/" + "/".join(str(p) for p in err.absolute_path) if err.absolute_path else "(root)",
            "message": err.message,
            "schema_path": "/".join(str(p) for p in err.absolute_schema_path),
        })
    return errors


# ─── Create + reads (T03) ─────────────────────────────────────────────────────

@router.post("/runs")
def create_run(body: SpecGenCreate) -> Dict[str, Any]:
    if body.mode == "agent":
        raise HTTPException(501, "agent mode not enabled")

    db = _db()

    active_schema = db[SCHEMAS_COLLECTION].find_one(
        {"schema_key": body.stage, "active": True}
    )
    if not active_schema:
        raise HTTPException(404, f"no active schema for stage '{body.stage}' — seed it via /schemas first")

    scope = {"project_id": body.project_id, "entity_id": body.entity_id, "stage": body.stage}
    latest = db[COLLECTION].find_one(scope, sort=[("major", -1), ("minor", -1)])
    if latest is None:
        major, minor = 1, 0
    else:
        # T04 rule: an accept closes the major — next create starts major+1.
        major_accepted = db[COLLECTION].find_one(
            {**scope, "major": latest["major"], "status": "accepted"}
        )
        if major_accepted:
            major, minor = latest["major"] + 1, 0
        else:
            major, minor = latest["major"], latest["minor"] + 1

    schema_version = active_schema["version"]
    errors = _validate_output(db, body.stage, schema_version, body.output)

    doc = {
        "run_id":            uuid.uuid4().hex,
        "project_id":        body.project_id,
        "entity_id":         body.entity_id,
        "stage":             body.stage,
        "schema_version":    schema_version,
        "major":             major,
        "minor":             minor,
        "mode":              body.mode,
        "input_ref":         body.input_ref,
        "prompt_text":       body.prompt_text,
        "output":            body.output,
        "status":            "valid" if not errors else "invalid",
        "validation_errors": errors,
        "feedback":          [],
        "created_at":        datetime.now(timezone.utc),
    }
    db[COLLECTION].insert_one(doc)
    return _serialize(doc)


def _get_run(db, run_id: str) -> Dict[str, Any]:
    doc = db[COLLECTION].find_one({"run_id": run_id})
    if not doc:
        raise HTTPException(404, f"no such run: {run_id}")
    return doc


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    return _serialize(_get_run(_db(), run_id))


@router.get("/{project_id}/{entity_id}/{stage}/versions")
def list_versions(project_id: str, entity_id: str, stage: str) -> List[Dict[str, Any]]:
    docs = list(
        _db()[COLLECTION]
        .find({"project_id": project_id, "entity_id": entity_id, "stage": stage})
        .sort([("major", -1), ("minor", -1)])
    )
    return [_serialize(d) for d in docs]


@router.get("/{project_id}/entities")
def list_entities(project_id: str) -> List[Dict[str, Any]]:
    """Distinct entity_ids with their stages + latest run status each (T08 grid)."""
    db = _db()
    entities: Dict[str, Dict[str, Any]] = {}
    for doc in db[COLLECTION].find({"project_id": project_id}).sort(
        [("major", 1), ("minor", 1), ("created_at", 1)]
    ):
        ent = entities.setdefault(doc["entity_id"], {"entity_id": doc["entity_id"], "stages": {}})
        # Later docs overwrite — sort order guarantees we end on the latest.
        ent["stages"][doc["stage"]] = {
            "status":  doc["status"],
            "version": f"{doc['major']}.{doc['minor']}",
            "run_id":  doc["run_id"],
        }
    return sorted(entities.values(), key=lambda e: e["entity_id"])


# ─── Fix packet / feedback / accept / reject / revalidate (T04) ──────────────

FIX_PACKET_TEMPLATE = """Your previous JSON response had validation errors. Fix ONLY these errors and
resend the complete corrected JSON in one code block. Do not change any other
fields.

ERRORS:
{error_lines}

THE SCHEMA (authoritative):
{schema_block}

YOUR PREVIOUS RESPONSE:
{output_block}
"""


@router.get("/runs/{run_id}/fix-packet")
def fix_packet(run_id: str) -> Dict[str, str]:
    import json as _json

    db = _db()
    doc = _get_run(db, run_id)
    if doc["status"] != "invalid":
        raise HTTPException(409, f"fix packet only exists for invalid runs (status={doc['status']})")

    schema_doc = db[SCHEMAS_COLLECTION].find_one(
        {"schema_key": doc["stage"], "version": doc["schema_version"]}
    )
    error_lines = "\n".join(
        f"- {e['json_path']}: {e['message']}" for e in doc["validation_errors"]
    )
    text = FIX_PACKET_TEMPLATE.format(
        error_lines=error_lines,
        schema_block=_json.dumps(schema_doc["json_schema"] if schema_doc else {}, indent=2),
        output_block=_json.dumps(doc["output"], indent=2),
    )
    return {"text": text}


@router.post("/runs/{run_id}/feedback")
def add_feedback(run_id: str, body: FeedbackBody) -> Dict[str, Any]:
    db = _db()
    _get_run(db, run_id)
    entry = {"at": datetime.now(timezone.utc).isoformat(), "note": body.note}
    db[COLLECTION].update_one({"run_id": run_id}, {"$push": {"feedback": entry}})
    return _serialize(_get_run(db, run_id))


@router.post("/runs/{run_id}/accept")
def accept_run(run_id: str) -> Dict[str, Any]:
    db = _db()
    doc = _get_run(db, run_id)
    if doc["status"] != "valid":
        raise HTTPException(409, f"only valid runs can be accepted (status={doc['status']})")
    # One accepted per (project, entity, stage, major) — demote siblings.
    db[COLLECTION].update_many(
        {
            "project_id": doc["project_id"],
            "entity_id":  doc["entity_id"],
            "stage":      doc["stage"],
            "major":      doc["major"],
            "status":     "accepted",
        },
        {"$set": {"status": "rejected"}},
    )
    db[COLLECTION].update_one({"run_id": run_id}, {"$set": {"status": "accepted"}})
    return _serialize(_get_run(db, run_id))


@router.post("/runs/{run_id}/reject")
def reject_run(run_id: str) -> Dict[str, Any]:
    db = _db()
    doc = _get_run(db, run_id)
    if doc["status"] == "accepted":
        raise HTTPException(409, "accepted runs cannot be rejected — accept a different run instead")
    db[COLLECTION].update_one({"run_id": run_id}, {"$set": {"status": "rejected"}})
    return _serialize(_get_run(db, run_id))


@router.post("/runs/{run_id}/revalidate")
def revalidate_run(run_id: str) -> Dict[str, Any]:
    """Re-run validation against the PINNED version (e.g. after schema rollback)."""
    db = _db()
    doc = _get_run(db, run_id)
    errors = _validate_output(db, doc["stage"], doc["schema_version"], doc["output"])
    status = doc["status"]
    if status in ("valid", "invalid", "pending_validation"):
        status = "valid" if not errors else "invalid"
    db[COLLECTION].update_one(
        {"run_id": run_id},
        {"$set": {"validation_errors": errors, "status": status}},
    )
    return _serialize(_get_run(db, run_id))
