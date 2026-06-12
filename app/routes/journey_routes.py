"""
journey_routes.py — journeys + impact items (CycleZero U01/U04)
===============================================================

A **journey** is one user-initiated change ("add a stamina system") with an
ordered rail of steps and a set of **impact items** that must all be resolved
to complete it. Impact items are the feed cards in the hybrid shell; branch
workspaces resolve them. Everything persists in Mongo so a journey survives
reload / coming back tomorrow.

Rules:
  - Ripple items (``workspace: "none"``) are informational — they never block
    completion and auto-close when the journey completes.
  - The journey auto-completes (status, completed_at, all rail steps done)
    when its last open blocking item is resolved or dismissed.
  - ``escalate`` (U04) reroutes an open item to a heavier workspace — e.g. a
    revalidation that found errors hands off to the spec & code workspace.

Routes mounted at "" (paths carry their own prefixes: /journeys, /impact-items).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from worker.lib import manual_gen_schema as mgs

log = logging.getLogger("journey_routes")

router = APIRouter()

JOURNEYS = "journeys"
ITEMS = "impact_items"

IMPACTS = ("direct", "high", "review", "ripple")
WORKSPACES = ("spec_code", "asset", "story_review", "revalidate", "none")
RAIL_STATES = ("pending", "active", "done")


def _db():
    """Open a Mongo connection. Overridden in tests (mongomock)."""
    return mgs.get_db()


# ─── Models ───────────────────────────────────────────────────────────────────

class RailStep(BaseModel):
    label: str = Field(min_length=1)
    sub: str = ""
    state: Literal["pending", "active", "done"] = "pending"


class ItemTarget(BaseModel):
    entity_id: str = ""
    stage: str = ""


class ImpactItemCreate(BaseModel):
    impact: Literal["direct", "high", "review", "ripple"]
    icon: str = "•"
    title: str = Field(min_length=1)
    body: str = ""
    target: ItemTarget = ItemTarget()
    workspace: Literal["spec_code", "asset", "story_review", "revalidate", "none"] = "none"
    suggested_intent: str = ""


class JourneyCreate(BaseModel):
    project_id: str = Field(min_length=1)
    kind: Literal["system", "asset", "data", "mixed"]
    title: str = Field(min_length=1)
    user_intent: str = ""
    rail: List[RailStep] = []
    items: List[ImpactItemCreate] = []


class RailUpdate(BaseModel):
    state: Literal["pending", "active", "done"]


class ResolveBody(BaseModel):
    note: str = Field(min_length=1)


class DismissBody(BaseModel):
    reason: str = Field(min_length=1)


class EscalateBody(BaseModel):
    workspace: Literal["spec_code", "asset", "story_review", "revalidate"]


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {**doc}
    out.pop("_id", None)
    for k in ("created_at", "completed_at", "resolved_at"):
        if isinstance(out.get(k), datetime):
            out[k] = out[k].isoformat()
    return out


# ─── Journeys ─────────────────────────────────────────────────────────────────

@router.post("/journeys")
def create_journey(body: JourneyCreate) -> Dict[str, Any]:
    db = _db()
    jid = uuid.uuid4().hex
    rail = [s.model_dump() for s in body.rail]
    if rail and all(s["state"] == "pending" for s in rail):
        rail[0]["state"] = "active"
    journey = {
        "journey_id":   jid,
        "project_id":   body.project_id,
        "kind":         body.kind,
        "title":        body.title,
        "user_intent":  body.user_intent,
        "rail":         rail,
        "status":       "active",
        "created_at":   datetime.now(timezone.utc),
        "completed_at": None,
    }
    db[JOURNEYS].insert_one(journey)
    items = []
    for it in body.items:
        doc = {
            "item_id":          uuid.uuid4().hex,
            "journey_id":       jid,
            "project_id":       body.project_id,
            **it.model_dump(),
            "status":           "open",
            "resolution_note":  None,
            "resolved_at":      None,
        }
        db[ITEMS].insert_one(doc)
        items.append(_serialize(doc))
    out = _serialize(journey)
    out["items"] = items
    return out


@router.get("/journeys/{project_id}")
def list_journeys(project_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    q: Dict[str, Any] = {"project_id": project_id}
    if status:
        q["status"] = status
    docs = list(_db()[JOURNEYS].find(q).sort("created_at", -1))
    return [_serialize(d) for d in docs]


def _get_journey(db, jid: str) -> Dict[str, Any]:
    doc = db[JOURNEYS].find_one({"journey_id": jid})
    if not doc:
        raise HTTPException(404, f"no such journey: {jid}")
    return doc


@router.get("/journeys/id/{jid}")
def get_journey(jid: str) -> Dict[str, Any]:
    db = _db()
    out = _serialize(_get_journey(db, jid))
    out["items"] = [_serialize(d) for d in db[ITEMS].find({"journey_id": jid})]
    return out


@router.post("/journeys/{jid}/rail/{index}")
def update_rail_step(jid: str, index: int, body: RailUpdate) -> Dict[str, Any]:
    db = _db()
    doc = _get_journey(db, jid)
    if not (0 <= index < len(doc["rail"])):
        raise HTTPException(422, f"rail index {index} out of range (len={len(doc['rail'])})")
    db[JOURNEYS].update_one({"journey_id": jid}, {"$set": {f"rail.{index}.state": body.state}})
    return _serialize(_get_journey(db, jid))


# ─── Impact items ─────────────────────────────────────────────────────────────

def _get_item(db, item_id: str) -> Dict[str, Any]:
    doc = db[ITEMS].find_one({"item_id": item_id})
    if not doc:
        raise HTTPException(404, f"no such impact item: {item_id}")
    return doc


@router.get("/impact-items/{item_id}")
def get_item(item_id: str) -> Dict[str, Any]:
    return _serialize(_get_item(_db(), item_id))


def _maybe_complete_journey(db, jid: str) -> None:
    """Complete the journey when no blocking (non-ripple) item is still open."""
    blocking_open = db[ITEMS].count_documents(
        {"journey_id": jid, "status": "open", "workspace": {"$ne": "none"}}
    )
    if blocking_open:
        return
    journey = db[JOURNEYS].find_one({"journey_id": jid})
    if not journey or journey["status"] != "active":
        return
    rail = journey.get("rail", [])
    for step in rail:
        step["state"] = "done"
    db[JOURNEYS].update_one(
        {"journey_id": jid},
        {"$set": {"status": "complete", "completed_at": datetime.now(timezone.utc), "rail": rail}},
    )
    # Ripples were informational — close them with the journey.
    db[ITEMS].update_many(
        {"journey_id": jid, "status": "open", "workspace": "none"},
        {"$set": {"status": "resolved", "resolution_note": "journey complete",
                  "resolved_at": datetime.now(timezone.utc)}},
    )


def _close_item(db, item_id: str, status: str, note: str) -> Dict[str, Any]:
    doc = _get_item(db, item_id)
    if doc["status"] != "open":
        raise HTTPException(409, f"item already {doc['status']}")
    db[ITEMS].update_one(
        {"item_id": item_id},
        {"$set": {"status": status, "resolution_note": note,
                  "resolved_at": datetime.now(timezone.utc)}},
    )
    _maybe_complete_journey(db, doc["journey_id"])
    out = _serialize(_get_item(db, item_id))
    out["journey"] = _serialize(db[JOURNEYS].find_one({"journey_id": doc["journey_id"]}))
    return out


@router.post("/impact-items/{item_id}/resolve")
def resolve_item(item_id: str, body: ResolveBody) -> Dict[str, Any]:
    return _close_item(_db(), item_id, "resolved", body.note)


@router.post("/impact-items/{item_id}/dismiss")
def dismiss_item(item_id: str, body: DismissBody) -> Dict[str, Any]:
    """The "keep original — intentional" path: counts as closed for completion."""
    return _close_item(_db(), item_id, "dismissed", body.reason)


@router.post("/impact-items/{item_id}/escalate")
def escalate_item(item_id: str, body: EscalateBody) -> Dict[str, Any]:
    """U04: reroute an open item to a heavier workspace (e.g. revalidate → spec_code)."""
    db = _db()
    doc = _get_item(db, item_id)
    if doc["status"] != "open":
        raise HTTPException(409, f"only open items can be escalated (status={doc['status']})")
    db[ITEMS].update_one({"item_id": item_id}, {"$set": {"workspace": body.workspace}})
    return _serialize(_get_item(db, item_id))
