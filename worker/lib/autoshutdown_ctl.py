"""
autoshutdown_ctl.py — CPU-side helpers for the GPU AutoShutdown Redis flag.

Two responsibilities:

1. snapshot/disable/restore the `autoshutdown:enabled` flag so the CPU can
   wrap a "spot launch + prewarm + first inference" sequence without the GPU
   stopping itself mid-warmup.

2. wait_for_prewarm() — block until the GPU's auto_shutdown.py publishes
   `prewarm:ready:<instance_id>=1` (which it does on first sentinel sight).

The disable/restore pair is only meaningful on the FIRST task after a fresh
spot launch. Once the GPU is warm and the worker has done one inference,
subsequent tasks see `prewarm:ready:*` already set and skip the wait
entirely — so this module is a no-op on the hot path.

Public API
----------
    autoshutdown_was_enabled(r) -> bool
    disable_autoshutdown(r) -> None
    restore_autoshutdown(r, was_enabled: bool) -> None
    is_prewarm_ready(r, instance_id) -> bool
    wait_for_prewarm(r, instance_id, timeout=1800, poll=10, progress_cb=None) -> bool

All functions are safe to call from any thread.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

logger = logging.getLogger("autoshutdown_ctl")

# Redis keys (mirrored from worker/workers/auto_shutdown.py)
# Per-instance keys take precedence; the unsuffixed forms are global fallbacks
# kept for backwards-compatibility.
KEY_ENABLED          = "autoshutdown:enabled"               # global
KEY_ENABLED_IID      = "autoshutdown:enabled:{instance_id}" # per-instance
KEY_IDLE_MINUTES     = "autoshutdown:idle_minutes"          # global
KEY_IDLE_MINUTES_IID = "autoshutdown:idle_minutes:{instance_id}"
KEY_PREWARM_READY    = "prewarm:ready:{instance_id}"


def _resolve_iid(instance_id: Optional[str]) -> Optional[str]:
    """Strip and validate the instance-id arg; treat empty as None."""
    if not instance_id:
        return None
    iid = instance_id.strip()
    return iid or None


# ── enable/disable flag (per-instance, with global fallback) ──────────────────

def autoshutdown_was_enabled(r, instance_id: Optional[str] = None) -> bool:
    """
    True if AutoShutdown was enabled for the given instance.

    Lookup order: per-instance key → global key → default True. Mirrors the
    GPU-side `_is_enabled()` semantics so snapshot/restore is round-trip exact.
    """
    iid = _resolve_iid(instance_id)
    keys = ([KEY_ENABLED_IID.format(instance_id=iid)] if iid else []) + [KEY_ENABLED]
    for k in keys:
        try:
            val = r.get(k)
        except Exception as e:
            logger.warning(f"redis get({k}) failed: {e}")
            continue
        if val is None:
            continue
        s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
        return s == "1"
    return True   # default: enabled


def disable_autoshutdown(r, instance_id: Optional[str] = None) -> None:
    """
    Set autoshutdown:enabled=0 for the given instance (or global if iid=None).
    GPU thread on that instance will skip self-stop while this is set.
    """
    iid = _resolve_iid(instance_id)
    key = KEY_ENABLED_IID.format(instance_id=iid) if iid else KEY_ENABLED
    try:
        r.set(key, "0")
        logger.info(f"AutoShutdown DISABLED via Redis flag ({key})")
    except Exception as e:
        logger.warning(f"redis set({key}=0) failed: {e}")


def restore_autoshutdown(r, was_enabled: bool, instance_id: Optional[str] = None) -> None:
    """
    Re-enable AutoShutdown for the given instance only if it was enabled
    before we touched it. No-op if the user had explicitly disabled it.
    """
    if not was_enabled:
        logger.info("AutoShutdown was already disabled before; not restoring")
        return
    iid = _resolve_iid(instance_id)
    key = KEY_ENABLED_IID.format(instance_id=iid) if iid else KEY_ENABLED
    try:
        r.set(key, "1")
        logger.info(f"AutoShutdown RE-ENABLED via Redis flag ({key})")
    except Exception as e:
        logger.warning(f"redis set({key}=1) failed: {e}")


# ── per-instance config setters/getters (for the UI panel) ────────────────────

def set_autoshutdown_enabled(r, instance_id: str, enabled: bool) -> None:
    """UI helper: explicitly set per-instance enabled flag to "0" or "1"."""
    iid = _resolve_iid(instance_id)
    if not iid:
        raise ValueError("instance_id is required")
    key = KEY_ENABLED_IID.format(instance_id=iid)
    r.set(key, "1" if enabled else "0")
    logger.info(f"UI: {key} = {'1' if enabled else '0'}")


def set_idle_minutes(r, instance_id: str, minutes: int) -> None:
    """UI helper: explicitly set per-instance idle threshold."""
    iid = _resolve_iid(instance_id)
    if not iid:
        raise ValueError("instance_id is required")
    if minutes < 1:
        raise ValueError("minutes must be >= 1")
    key = KEY_IDLE_MINUTES_IID.format(instance_id=iid)
    r.set(key, str(int(minutes)))
    logger.info(f"UI: {key} = {minutes}")


def get_instance_config(r, instance_id: str) -> dict:
    """Return current effective config for one instance (for UI display)."""
    iid = _resolve_iid(instance_id)
    if not iid:
        return {"enabled": True, "idle_minutes": 15, "source": "no-iid"}
    enabled = autoshutdown_was_enabled(r, iid)
    # Idle minutes: per-instance → global → default 15
    minutes = 15
    source_min = "default"
    for k, src in [
        (KEY_IDLE_MINUTES_IID.format(instance_id=iid), "per-instance"),
        (KEY_IDLE_MINUTES, "global"),
    ]:
        try:
            v = r.get(k)
        except Exception:
            continue
        if v is None:
            continue
        s = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
        try:
            minutes = int(s)
            source_min = src
            break
        except (TypeError, ValueError):
            continue
    return {"enabled": enabled, "idle_minutes": minutes, "source": source_min}


# ── prewarm sentinel ──────────────────────────────────────────────────────────

def is_prewarm_ready(r, instance_id: str) -> bool:
    """True if the GPU has published prewarm:ready:<iid>=1."""
    if not instance_id:
        return False
    try:
        val = r.get(KEY_PREWARM_READY.format(instance_id=instance_id))
    except Exception as e:
        logger.debug(f"is_prewarm_ready redis error: {e}")
        return False
    if val is None:
        return False
    s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
    return s == "1"


def wait_for_prewarm(
    r,
    instance_id: str,
    timeout: int = 1800,
    poll: int = 10,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """
    Block until prewarm:ready:<iid>=1 or timeout (seconds).

    Returns True if ready in time, False on timeout.

    progress_cb (if given) is called every `poll` seconds with
    (elapsed_seconds, timeout_seconds) so the caller can surface a
    "warming GPU…" status to the UI.
    """
    if not instance_id:
        logger.warning("wait_for_prewarm: instance_id missing — cannot wait")
        return False

    start = time.time()
    deadline = start + timeout
    last_log = 0.0

    while time.time() < deadline:
        if is_prewarm_ready(r, instance_id):
            elapsed = int(time.time() - start)
            logger.info(f"Prewarm READY for {instance_id} (waited {elapsed}s)")
            return True

        elapsed = int(time.time() - start)
        if progress_cb is not None:
            try:
                progress_cb(elapsed, timeout)
            except Exception:
                pass
        # Log at most every 60s so logs aren't spammed
        if elapsed - last_log >= 60:
            logger.info(f"Waiting for prewarm:ready:{instance_id} ({elapsed}/{timeout}s)")
            last_log = elapsed

        time.sleep(poll)

    logger.warning(f"Prewarm wait TIMED OUT after {timeout}s for {instance_id}")
    return False
