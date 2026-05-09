"""
result_consumer.py — CPU-side GPU result consumer
==================================================
Runs as an asyncio background task inside FastAPI (wired in app/main.py).

Reads from `spark_results` Redis queue (populated by GPU workers via
worker/lib/result_channel.py) and writes to MongoDB manual_gen_sessions.

This is the ONLY writer to manual_gen_sessions from the result path;
GPU workers never touch MongoDB directly for the manual-gen pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging

import redis as _redis

from app.config import settings
from app.services.mongo_service import get_db
from worker.lib.result_channel import RESULT_QUEUE

logger = logging.getLogger("result_consumer")

# How long BLPOP waits before returning None (keeps the loop responsive to cancel).
_BLPOP_TIMEOUT = 5


def _apply_result(db, msg: dict) -> None:
    """Translate one result message into a MongoDB update."""
    # Import schema helpers here (lazy) so startup doesn't fail if schema module
    # has a transient import issue.
    from worker.lib.manual_gen_schema import (
        mark_done,
        mark_error,
        mark_queued,
        mark_running,
    )

    session_id = msg.get("session_id", "")
    stage      = msg.get("stage",      "")
    status     = msg.get("status",     "")

    if not (session_id and stage and status):
        logger.warning(f"[result_consumer] Malformed result, skipping: {msg}")
        return

    try:
        if status == "running":
            mark_running(db, session_id, stage)

        elif status == "done":
            # Image stages return image_url / s3_key; 3D stages return glb_url / s3_key
            image_url = msg.get("image_url") or msg.get("glb_url", "")
            s3_key    = msg.get("s3_key", "")
            mark_done(db, session_id, stage, image_url, s3_key)

        elif status == "error":
            mark_error(db, session_id, stage, msg.get("error", "unknown error"))

        elif status == "queued":
            mark_queued(db, session_id, stage, msg.get("task_id", ""))

        else:
            logger.warning(f"[result_consumer] Unknown status '{status}': {msg}")

        logger.info(
            f"[result_consumer] Applied: session={session_id[:8]} "
            f"stage={stage} → {status}"
        )

    except Exception as exc:
        logger.error(
            f"[result_consumer] MongoDB update failed for "
            f"session={session_id[:8]} stage={stage}: {exc}",
            exc_info=True,
        )


async def result_consumer_main() -> None:
    """
    Async background loop.
    BLPOP from spark_results → update MongoDB manual_gen_sessions.
    """
    logger.info(f"[result_consumer] Started — watching queue: {RESULT_QUEUE}")

    r    = _redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    loop = asyncio.get_event_loop()

    while True:
        try:
            # Run blocking BLPOP in a thread so we don't block the event loop.
            raw = await loop.run_in_executor(
                None,
                lambda: r.blpop(RESULT_QUEUE, timeout=_BLPOP_TIMEOUT),
            )

            if raw is None:
                continue  # Timeout — loop again

            _, payload = raw

            try:
                msg = json.loads(payload)
            except json.JSONDecodeError:
                logger.error(f"[result_consumer] Bad JSON: {payload[:120]}")
                continue

            db = get_db()
            if db is None:
                logger.error("[result_consumer] MongoDB unavailable, re-queuing result")
                r.rpush(RESULT_QUEUE, payload)  # Put it back
                await asyncio.sleep(5)
                continue

            _apply_result(db, msg)

        except asyncio.CancelledError:
            logger.info("[result_consumer] Cancelled — shutting down.")
            raise

        except Exception as exc:
            logger.error(f"[result_consumer] Unexpected error: {exc}", exc_info=True)
            await asyncio.sleep(5)
