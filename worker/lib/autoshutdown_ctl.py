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
KEY_ENABLED       = "autoshutdown:enabled"
KEY_PREWARM_READY = "prewarm:ready:{instance_id}"


# ── enable/disable flag ───────────────────────────────────────────────────────

def autoshutdown_was_enabled(r) -> bool:
    """
    True if AutoShutdown was enabled (default = True if key unset).

    Mirrors the GPU-side `_is_enabled()` semantics so snapshot/restore
    is round-trip exact.
    """
    try:
        val = r.get(KEY_ENABLED)
    except Exception as e:
        logger.warning(f"redis get({KEY_ENABLED}) failed: {e} — assuming enabled")
        return True
    if val is None:
        return True
    s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
    return s == "1"


def disable_autoshutdown(r) -> None:
    """Set autoshutdown:enabled=0 (GPU thread will skip self-stop while this is set)."""
    try:
        r.set(KEY_ENABLED, "0")
        logger.info("AutoShutdown DISABLED via Redis flag (CPU orchestrator)")
    except Exception as e:
        logger.warning(f"redis set({KEY_ENABLED}=0) failed: {e}")


def restore_autoshutdown(r, was_enabled: bool) -> None:
    """
    Re-enable AutoShutdown only if it was enabled before we touched it.

    No-op if the user had explicitly disabled it — we never silently turn
    on a flag the user turned off.
    """
    if not was_enabled:
        logger.info("AutoShutdown was already disabled before; not restoring")
        return
    try:
        r.set(KEY_ENABLED, "1")
        logger.info("AutoShutdown RE-ENABLED via Redis flag (CPU orchestrator)")
    except Exception as e:
        logger.warning(f"redis set({KEY_ENABLED}=1) failed: {e}")


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
