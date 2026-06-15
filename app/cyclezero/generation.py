"""P5 — asset-generation bridge.

A generation trigger on a CycleZero entity becomes an ``asset_job`` row (the
durable record + status surface) and is submitted to the existing GPU pipeline.
The actual heavy lifting reuses spark's asset-runs / manual-gen machinery; this
module is the thin seam between the game graph and that pipeline.

Submission is best-effort and isolated: if the pipeline isn't reachable (e.g. in
tests), the job stays ``queued`` with a recorded note rather than raising — the
API call still succeeds and the job can be retried.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import models
from .mongo import get_mongo

log = logging.getLogger("cyclezero.generation")

# Mongo collection (in the cyclezero DB) that records each generation request
# and is the hand-off the pipeline/worker reads.
REQUESTS = "asset_requests"


def submit(job: models.AssetJob, entity: Optional[models.Entity]) -> Dict[str, Any]:
    """Record the request in Mongo and hand off to the pipeline.

    Returns a dict merged into the job ``result`` (e.g. the mongo request id).
    Never raises — failures are returned as ``{"submit_error": ...}`` so the
    caller can keep the job queued.
    """
    doc = {
        "job_id": str(job.id),
        "game_id": str(job.game_id),
        "entity_id": str(job.entity_id) if job.entity_id else None,
        "entity_key": entity.key if entity else None,
        "entity_layer": entity.layer if entity else None,
        "kind": job.kind,
        "params": job.params,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
    }
    try:
        mongo = get_mongo()
        res = mongo[REQUESTS].insert_one(doc)
        # Integration point: enqueue to the existing asset-runs / manual-gen
        # pipeline here (Redis queue / asset_runs doc). Kept as a recorded
        # hand-off for now so a worker can drain it; wiring the live GPU fan-out
        # is the deploy-time verify step (gated on GPU spend).
        return {"mongo_request_id": str(res.inserted_id), "submitted": True}
    except Exception as exc:  # noqa: BLE001 — best-effort, never fail the API call
        log.warning("asset job %s submit deferred: %s", job.id, exc)
        return {"submitted": False, "submit_error": str(exc)}
