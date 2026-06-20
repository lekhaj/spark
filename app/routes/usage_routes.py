"""
usage_routes.py — studio event audit + usage/cost surface
=========================================================

Two distinct things, deliberately kept separate (per owner requirement):

1. **External LLM (Bedrock = AWS credits)** — Haiku 4.5 today. Real token counts
   priced into dollars. Source = the ``llm_usage`` collection written by
   ``usage_recorder`` at the provider boundary.

2. **Internal generation (our own GPU)** — Flux / 3D / rig. NO external dollars;
   tracked as run *counts* per stage, aggregated from the existing
   ``manual_gen_stage_runs`` collection. Never dollar-costed.

Plus a lightweight **event audit** (``studio_events``): what the user did / typed,
with high-level surface metadata. Ingest is best-effort and batched.

All writes are best-effort and must never 500 the UI. Admin reads (everyone's
activity / usage) are owner-gated; ``/usage/me`` is self-scoped.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.lib.identity import StudioUser, current_user, require_owner
from app.services.mongo_service import get_db

log = logging.getLogger("usage_routes")
router = APIRouter()

EVENTS = "studio_events"
LLM_USAGE = "llm_usage"
GPU_RUNS = "manual_gen_stage_runs"
GPU_TPOSE = "manual_gen_tpose_runs"

MAX_BATCH = 100
MAX_TEXT = 4000


# ── Event ingest ──────────────────────────────────────────────────────────────
class EventIn(BaseModel):
    type: str
    ts: Optional[float] = None          # client epoch ms (optional)
    route: Optional[str] = None
    game_slug: Optional[str] = None
    layer: Optional[str] = None
    entity_key: Optional[str] = None
    text: Optional[str] = None          # submitted text (truncated server-side)
    meta: Dict[str, Any] = Field(default_factory=dict)


class EventBatch(BaseModel):
    events: List[EventIn] = Field(default_factory=list)


# High-level "which feature/surface" map → answers "what files/functions at a
# high level" without the client guessing. Extend as new event types appear.
SURFACE = {
    "refiner.chat": "shell/refiner → refiner_routes.chat",
    "compile.run": "explore/Compile → cyclezero.compile_agent",
    "validate.run": "explore/World → cyclezero.validate_agent",
    "propose.run": "designer/DescribePanel → cyclezero.propose_agent",
    "inspector.save": "explore/inspectors → entities PATCH",
    "asset.run": "shell/AssetWorkspace → asset_run_routes",
    "nav": "router navigation",
}


@router.post("/events")
def ingest_events(batch: EventBatch, user: StudioUser = Depends(current_user)) -> Dict[str, Any]:
    """Bulk-insert client events. Best-effort; never raises into the UI."""
    db = get_db()
    if db is None or not batch.events:
        return {"ok": True, "inserted": 0}
    now = datetime.now(timezone.utc)
    docs = []
    for e in batch.events[:MAX_BATCH]:
        text = (e.text or "")[:MAX_TEXT] or None
        docs.append({
            "ts": now,
            "client_ts": e.ts,
            "type": e.type,
            "surface": SURFACE.get(e.type, e.meta.get("surface")),
            "route": e.route,
            "game_slug": e.game_slug,
            "layer": e.layer,
            "entity_key": e.entity_key,
            "text": text,
            "meta": e.meta or {},
            "uid": user.uid,
            "email": user.email,
        })
    try:
        db[EVENTS].insert_many(docs, ordered=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("event ingest failed: %s", exc)
        return {"ok": False, "inserted": 0}
    return {"ok": True, "inserted": len(docs)}


# ── Usage aggregation helpers ─────────────────────────────────────────────────
def _llm_match(email: Optional[str], days: int) -> Dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    m: Dict[str, Any] = {"ts": {"$gte": since}}
    if email:
        m["email"] = email.lower()
    return m


def _llm_summary(db, email: Optional[str], days: int) -> Dict[str, Any]:
    """Aggregate llm_usage into totals + by-model + by-day. External / $ costed."""
    match = _llm_match(email, days)
    base_fields = {
        "calls": {"$sum": 1},
        "input_tokens": {"$sum": "$input_tokens"},
        "output_tokens": {"$sum": "$output_tokens"},
        "cache_read_tokens": {"$sum": "$cache_read_tokens"},
        "cache_write_tokens": {"$sum": "$cache_write_tokens"},
        "total_tokens": {"$sum": "$total_tokens"},
        "cost_usd": {"$sum": "$cost_usd"},
    }
    coll = db[LLM_USAGE]

    def _run(group_id):
        rows = list(coll.aggregate([
            {"$match": match},
            {"$group": {"_id": group_id, **base_fields}},
            {"$sort": {"cost_usd": -1}},
        ]))
        for r in rows:
            r["cost_usd"] = round(r.get("cost_usd", 0.0), 4)
        return rows

    totals_rows = _run(None)
    totals = totals_rows[0] if totals_rows else {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0,
    }
    totals.pop("_id", None)
    return {
        "totals": totals,
        "by_model": _run("$model_family"),
        "by_agent": _run("$agent"),
        "by_day": list(coll.aggregate([
            {"$match": match},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts"}},
                        "cost_usd": {"$sum": "$cost_usd"},
                        "total_tokens": {"$sum": "$total_tokens"}}},
            {"$sort": {"_id": 1}},
        ])),
    }


def _gpu_summary(db, days: int) -> Dict[str, Any]:
    """Internal GPU generation — run counts per stage. No dollars (own hardware).
    Self-attribution isn't tracked on runs yet, so this is global for now."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    match = {"created_at": {"$gte": since}}
    by_stage: Dict[str, int] = {}
    total = 0
    for coll_name in (GPU_RUNS, GPU_TPOSE):
        try:
            for r in db[coll_name].aggregate([
                {"$match": match},
                {"$group": {"_id": "$stage", "runs": {"$sum": 1}}},
            ]):
                stage = r["_id"] or "unknown"
                by_stage[stage] = by_stage.get(stage, 0) + r["runs"]
                total += r["runs"]
        except Exception as exc:  # noqa: BLE001
            log.warning("gpu summary on %s failed: %s", coll_name, exc)
    return {
        "total_runs": total,
        "by_stage": [{"stage": k, "runs": v} for k, v in
                     sorted(by_stage.items(), key=lambda kv: -kv[1])],
        "note": "Internal GPU compute — no external cost.",
    }


@router.get("/usage/me")
def usage_me(user: StudioUser = Depends(current_user),
             days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """The caller's own usage: external Bedrock LLM (tokens + $) and a global GPU
    generation summary (counts, no $)."""
    db = get_db()
    if db is None:
        return {"llm": None, "gpu": None, "error": "db unavailable"}
    return {
        "email": user.email,
        "days": days,
        "llm": _llm_summary(db, user.email, days),
        "gpu": _gpu_summary(db, days),
    }


@router.get("/usage")
def usage_all(user: StudioUser = Depends(current_user),
              days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Owner-only: everyone's LLM spend (per-user leaderboard) + global GPU usage."""
    require_owner(user)
    db = get_db()
    if db is None:
        return {"error": "db unavailable"}
    since = datetime.now(timezone.utc) - timedelta(days=days)
    leaderboard = list(db[LLM_USAGE].aggregate([
        {"$match": {"ts": {"$gte": since}}},
        {"$group": {"_id": "$email", "calls": {"$sum": 1},
                    "input_tokens": {"$sum": "$input_tokens"},
                    "output_tokens": {"$sum": "$output_tokens"},
                    "cost_usd": {"$sum": "$cost_usd"}}},
        {"$sort": {"cost_usd": -1}},
    ]))
    for r in leaderboard:
        r["cost_usd"] = round(r.get("cost_usd", 0.0), 4)
    return {
        "days": days,
        "llm_total": _llm_summary(db, None, days),
        "by_user": leaderboard,
        "gpu": _gpu_summary(db, days),
    }


# ── AWS Cost Explorer reconciliation (authoritative billed $) ─────────────────
# Our `llm_usage` rows price tokens at *list* rates (live, per-call). The actual
# credit-spend AWS bills (after any negotiated/credit discounts) is only knowable
# from Cost Explorer — aggregate per-service/day, ~24h delayed. This endpoint
# surfaces the Bedrock line so the owner can reconcile our estimate vs reality.
def _ce_client():
    """Lazy boto3 Cost Explorer client. Reuses aws_service credential + timeout
    config; the `ce` API is global but addressed via us-east-1."""
    import boto3
    from app.services.aws_service import _BOTO_CONFIG, _boto3_kwargs
    return boto3.client("ce", region_name="us-east-1",
                        config=_BOTO_CONFIG, **_boto3_kwargs())


@router.get("/usage/aws-cost")
def aws_cost(user: StudioUser = Depends(current_user),
             days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Owner-only: AWS Cost Explorer's actual billed cost for Amazon Bedrock,
    daily, for the window. Authoritative $ (post-credit, ~24h delayed) to
    reconcile against our list-priced `llm_usage` estimate."""
    require_owner(user)
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=days)).isoformat()
    end = (today + timedelta(days=1)).isoformat()  # CE end is exclusive
    try:
        ce = _ce_client()
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="DAILY",
            Metrics=["NetUnblendedCost", "UnblendedCost"],
            Filter={"Dimensions": {"Key": "SERVICE",
                                   "Values": ["Amazon Bedrock"]}},
        )
    except Exception as exc:  # noqa: BLE001 — surface, never 500 the dashboard
        log.warning("aws cost explorer query failed: %s", exc)
        return {"available": False, "error": str(exc), "by_day": [], "total_usd": 0.0}

    by_day: List[Dict[str, Any]] = []
    total = 0.0
    currency = "USD"
    for r in resp.get("ResultsByTime", []):
        amt = r.get("Total", {}).get("NetUnblendedCost", {}) \
            or r.get("Total", {}).get("UnblendedCost", {})
        val = float(amt.get("Amount", 0.0) or 0.0)
        currency = amt.get("Unit", currency)
        total += val
        by_day.append({"date": r.get("TimePeriod", {}).get("Start"),
                       "cost_usd": round(val, 4)})
    return {
        "available": True,
        "service": "Amazon Bedrock",
        "days": days,
        "currency": currency,
        "total_usd": round(total, 4),
        "by_day": by_day,
        "note": "Net cost after credits; ~24h delayed, aggregate per service.",
    }


@router.get("/events")
def list_events(user: StudioUser = Depends(current_user),
                limit: int = Query(100, ge=1, le=500),
                email: Optional[str] = None,
                type: Optional[str] = None,
                game_slug: Optional[str] = None) -> Dict[str, Any]:
    """Owner-only event audit feed (most recent first), with simple filters."""
    require_owner(user)
    db = get_db()
    if db is None:
        return {"events": []}
    q: Dict[str, Any] = {}
    if email:
        q["email"] = email.lower()
    if type:
        q["type"] = type
    if game_slug:
        q["game_slug"] = game_slug
    rows = list(db[EVENTS].find(q, {"_id": 0}).sort("ts", -1).limit(limit))
    for r in rows:
        if isinstance(r.get("ts"), datetime):
            r["ts"] = r["ts"].isoformat()
    return {"events": rows}
