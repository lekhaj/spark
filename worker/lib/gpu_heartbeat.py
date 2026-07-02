"""
gpu_heartbeat.py — the single "GPU is doing work right now" signal.

Why this exists
---------------
Idle-shutdown used to be judged purely from Redis *queue depth* + a coarse
in-process ``_active`` flag. That is blind to the two cases that actually matter:

  1. A long 3D **subprocess** (pixal3d / hunyuan3d) runs for 5–7 min with the
     queue already drained — "queue empty" looked idle even though the GPU was
     pinned. (This stopped a box mid-pipeline in production.)
  2. The CPU orchestrator (a *different* process on a *different* host) needs to
     know the GPU is busy without SSHing in to read worker state.

A heartbeat in Redis solves both: the worker ``touch()``es it on every task pop
and every ~30s while a task is in flight; any reader (GPU-side ``auto_shutdown``,
CPU-side orchestrator) treats a *fresh* heartbeat as "busy" regardless of queue
depth. It is the safety primitive both stop-paths key off.

Keys
----
  gpu:last_activity            — global (epoch float as string)
  gpu:last_activity:<iid>      — per-instance (epoch float as string)

Both carry a TTL so a crashed worker's stale heartbeat self-expires (after which
``seconds_since`` reports the TTL has elapsed → not busy → eligible to stop).

Fail-safe convention
--------------------
Readers must treat *unknown* (Redis error / missing key) as **busy** (return a
large "seconds since" only when the key is genuinely absent, but on a Redis
*error* return ``None`` so callers can choose not to stop). See ``seconds_since``.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("gpu_heartbeat")

KEY_GLOBAL = "gpu:last_activity"
KEY_IID = "gpu:last_activity:{instance_id}"

# Heartbeat key TTL. Must comfortably exceed the worker's touch interval (~30s)
# so a still-working box never lets its own heartbeat lapse, while a *dead*
# worker's heartbeat expires within this window and frees the box to stop.
DEFAULT_TTL_SECONDS = 1800


def touch(r, instance_id: Optional[str] = None, ttl: int = DEFAULT_TTL_SECONDS) -> None:
    """Record 'GPU active right now'. Safe to call from any thread; never raises."""
    now = str(time.time())
    try:
        r.set(KEY_GLOBAL, now, ex=ttl)
        if instance_id:
            r.set(KEY_IID.format(instance_id=instance_id), now, ex=ttl)
    except Exception as e:  # noqa: BLE001 — heartbeat must never crash the worker
        logger.debug("heartbeat touch failed: %s", e)


def _read_ts(r, key: str) -> Optional[float]:
    """Return the float epoch stored at key, or None (missing or unparseable)."""
    try:
        val = r.get(key)
    except Exception as e:  # noqa: BLE001
        logger.debug("heartbeat get(%s) failed: %s", key, e)
        raise  # let caller distinguish error from missing
    if val is None:
        return None
    s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def seconds_since(r, instance_id: Optional[str] = None) -> Optional[float]:
    """
    Seconds since the last heartbeat for ``instance_id`` (falling back to the
    global key).

    Returns:
      * a non-negative float   — time since the freshest known heartbeat
      * ``None``               — Redis was unreachable (caller should FAIL SAFE:
                                 treat as busy / do not stop)

    A genuinely *absent* key (never touched, or TTL-expired) returns a very large
    number (``float('inf')``) so it reads as "long idle" → eligible to stop.
    """
    keys = ([KEY_IID.format(instance_id=instance_id)] if instance_id else []) + [KEY_GLOBAL]
    newest: Optional[float] = None
    try:
        for k in keys:
            ts = _read_ts(r, k)
            if ts is not None and (newest is None or ts > newest):
                newest = ts
    except Exception:
        return None  # Redis error → unknown → caller fails safe (busy)
    if newest is None:
        return float("inf")  # no heartbeat anywhere → long idle
    return max(0.0, time.time() - newest)


def is_busy(r, instance_id: Optional[str], fresh_seconds: float) -> bool:
    """
    True if a heartbeat is fresher than ``fresh_seconds``.

    FAIL SAFE: on a Redis error (``seconds_since`` returns None) we report
    **busy=True** so neither stop-path ever kills a box we can't read.
    """
    s = seconds_since(r, instance_id)
    if s is None:
        return True  # unknown → assume busy
    return s < fresh_seconds
