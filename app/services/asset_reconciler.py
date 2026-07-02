"""Background asset-run reconciler.

The cyclezero asset-run stage machine is **poll-driven**: `_refresh` only advances
(image → 3D fan-out → choose → rig → write-back) when something polls it. Until now
that "something" was a human hitting `GET …/jobs/{id}` (or a manual poller script),
so a generation would silently stall mid-pipeline if nobody watched it.

This task closes that gap: it periodically reconciles every in-flight asset job —
advancing the run and writing the finished GLB onto the owning entity — with no human
in the loop. Started in `app/main.py` lifespan next to the orchestrator/result-consumer.

It does NOT stop the GPU: that stays owned by the GPU worker's AutoShutdown (single
clock, no race). Tighten the idle window via the worker's `IDLE_SHUTDOWN_MINUTES`.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("asset_reconciler")

# How often to sweep in-flight jobs. Cheap (a single indexed query + best-effort
# reconcile of each), so a tight-ish interval keeps runs moving without polling load.
INTERVAL_S = int(os.getenv("RECONCILE_INTERVAL_S", "20"))

# Jobs in any of these states still have work to advance; "done"/"failed" are terminal.
_ACTIVE_STATES = ("queued", "running", "generating")


def _reconcile_once() -> int:
    """One sweep (sync; runs in a thread). Returns how many jobs were reconciled."""
    from app.cyclezero.db import get_session_factory
    from app.cyclezero import models
    from app.cyclezero.routes import reconcile_job

    db = get_session_factory()()
    n = 0
    try:
        jobs = (
            db.query(models.AssetJob)
            .filter(models.AssetJob.status.in_(_ACTIVE_STATES))
            .all()
        )
        for job in jobs:
            if not (job.result or {}).get("asset_run_id"):
                continue
            try:
                reconcile_job(db, job)
                n += 1
            except Exception:  # noqa: BLE001 — one bad job must not stall the rest
                logger.exception("reconcile failed for job %s", job.id)
                db.rollback()
    finally:
        db.close()
    return n


async def reconcile_main() -> None:
    """Background loop: sweep in-flight asset jobs every INTERVAL_S seconds."""
    logger.info("asset reconciler started (interval=%ss)", INTERVAL_S)
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Run the blocking DB/boto3 work in a thread so the event loop stays free.
            n = await loop.run_in_executor(None, _reconcile_once)
            if n:
                logger.debug("reconciled %d in-flight job(s)", n)
        except asyncio.CancelledError:
            logger.info("asset reconciler stopping")
            raise
        except Exception:  # noqa: BLE001 — never let the loop die
            logger.exception("asset reconciler sweep errored")
        await asyncio.sleep(INTERVAL_S)
