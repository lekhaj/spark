"""Stage-affinity task scheduling for the manual-gen worker.

The worker pulls one task at a time from a single Redis list (FIFO via BLPOP). The
orchestrator fans out per character, so stages interleave in the queue
(``trellis_A, pixal3d_A, hunyuan3d_A, trellis_B, …``). Running it in that order
reloads each model once per character.

This helper reorders the *pull* (not the output): after finishing a task of some
stage, it prefers the next queued task of the **same** stage, so the model/server
for that stage stays warm and loads once per batch instead of once per character.
It changes nothing about what gets produced — every task still runs, at full
quality; only the load/unload churn is removed.

Safe because: there is a single manual-gen consumer (no concurrent pop race), and
every task in this queue is independently runnable (the orchestrator only enqueues
a stage once its inputs exist; the dependent ``rig`` stage lives on another queue).
So reordering tasks within the queue cannot violate a dependency.

Pure module — no torch/model imports — so it unit-tests with a tiny fake Redis.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

log = logging.getLogger("stage_affinity")


def _stage_of(raw: Any, json_loads: Callable[[Any], Any]) -> Optional[str]:
    try:
        obj = json_loads(raw)
        if isinstance(obj, dict):
            return obj.get("stage")
    except Exception:  # noqa: BLE001 — malformed payloads fall back to FIFO handling
        return None
    return None


def peek_has_stage(
    r,
    queue: str,
    stage: Optional[str],
    *,
    json_loads: Callable[[Any], Any] = json.loads,
) -> bool:
    """True if at least one task of ``stage`` is currently queued.

    Read-only (LRANGE, no pop). The worker uses this as a *lookahead* after a
    task finishes: if another same-stage task is waiting, the handler can keep
    that stage's model/server resident (one load per batch) instead of evicting
    and reloading for the next character. Degrades to False (→ evict, the safe
    pre-existing behavior) on any Redis hiccup or malformed payload.
    """
    if not stage:
        return False
    try:
        items = r.lrange(queue, 0, -1)
    except Exception as exc:  # noqa: BLE001 — on any error, say "no more" → evict (safe)
        log.debug("peek lrange failed (%s); assuming no same-stage task", exc)
        return False
    for raw in items:
        if _stage_of(raw, json_loads) == stage:
            return True
    return False


def pop_next_task(
    r,
    queue: str,
    last_stage: Optional[str],
    *,
    timeout: int = 30,
    json_loads: Callable[[Any], Any] = json.loads,
) -> Optional[str]:
    """Return the next task payload to process, or None on idle timeout.

    Preference order:
      1. If ``last_stage`` is set, the FIRST queued task whose ``stage == last_stage``
         (removed atomically-enough for a single consumer via LREM). Keeps that
         stage's model warm.
      2. Otherwise the FIFO head via BLPOP(timeout) — preserves ordering, idle
         detection, and auto-shutdown behavior when the queue is empty.
    """
    if last_stage:
        try:
            items = r.lrange(queue, 0, -1)
        except Exception as exc:  # noqa: BLE001 — degrade to plain FIFO on any Redis hiccup
            log.debug("lrange failed (%s); falling back to FIFO", exc)
            items = []
        for raw in items:
            if _stage_of(raw, json_loads) == last_stage:
                try:
                    removed = r.lrem(queue, 1, raw)
                except Exception as exc:  # noqa: BLE001
                    log.debug("lrem failed (%s); falling back to FIFO", exc)
                    break
                if removed:
                    return raw
                # value already gone (shouldn't happen with one consumer) — keep scanning
        # no same-stage task pending → fall through to FIFO head

    res = r.blpop(queue, timeout=timeout)
    if res is None:
        return None
    # redis-py returns (key, value); with decode_responses=True both are str
    return res[1]
