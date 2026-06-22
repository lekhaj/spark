"""
gpu_launcher.py — bring a GPU online before queueing work (spot-first).

Design: TWO persistent root volumes, one per box (no migration)
======================================================================
Each GPU box owns its OWN 256 GB stack root in us-east-1d (conda envs, code,
~150 GB model cache): the spot's ``infra.SPOT_STACK_VOLUME_ID`` and the
on-demand's ``infra.ONDEMAND_STACK_VOLUME_ID``. They are independent duplicates.
We NEVER detach/migrate a root volume — doing so on a persistent spot drops its
request to a terminal ``failed`` state (the bug that bricked the original spot).

Failover is therefore trivial: just START the right box.
  * The spot is the steady-state primary (cheapest). It is persistent with
    interruption=stop → an AWS reclaim STOPS it (never terminates); its root
    survives and it restarts when capacity returns.
  * The on-demand is the failover. It has its own root, always startable.
  * Both roots have DeleteOnTermination=false (survive an explicit terminate).

There is NO golden AMI and NO snapshot backup (2026-06-14 directive). If the spot
is ever fully terminated, its root volume still survives (DoT=false) but a brand
new spot would have to be created + the root re-attached MANUALLY — that path is
not automated here; we just fail over to the on-demand and flag it.

The ladder (``ensure_gpu_ready``), in order:
    0. STICK: a box already running & recorded active → use it.
    1. SPOT preferred:
         running                    → use it.
         stopped/stopping           → start it (just start — its root is attached).
         start fails (capacity)     → fall back to on-demand.
         missing/terminated (rare)  → fall back to on-demand; flag for manual spot
                                       recovery (no AMI to auto-relaunch from).
    2. ON-DEMAND failover: start it (its own root is attached).
    3. Neither can run → (False, reason); orchestrator retries next poll.

Convergence back to spot: while the on-demand serves, the orchestrator STICKS to
it (step 0). Once it goes idle it stops (GPU-side autoshutdown / CPU backstop);
the next work batch goes spot-first again. Net steady state: spot, cheapest.

Two-pinned EIPs: each box owns its EIP permanently (``infra.GPU_BOXES``); we
(re)attach a box's OWN EIP on start and never steal the other's.

Activation: gated by ``GPU_AUTO_LAUNCH`` (default "0" → no-op, returns disabled).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger("gpu_launcher")

# Redis keys.
REDIS_ACTIVE_KEY = "gpu:active_instance_id"   # box the orchestrator manages now
REDIS_SPOT_RECOVER_KEY = "gpu:spot_recover"    # set when a spot needs MANUAL recovery

_TERMINAL = ("terminated", "shutting-down")
_SPOT_CAPACITY_ERRORS = (
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "MaxSpotInstanceCountExceeded",
    "InsufficientSpotInstanceCapacity",
)


# ── config access (import infra lazily to keep this module import-light) ────────

def _cfg():
    from app import infra
    return infra


# ── Feature flag ──────────────────────────────────────────────────────────────

def _enabled() -> bool:
    return os.getenv("GPU_AUTO_LAUNCH", "0") == "1"


# ── boto3 / redis helpers ──────────────────────────────────────────────────────

def _ec2_client():
    import boto3
    return boto3.client("ec2", region_name=os.getenv("AWS_REGION", "us-east-1"))


def _redis():
    try:
        import redis as _r
        client = _r.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=0, decode_responses=True,
        )
        client.ping()
        return client
    except Exception as e:
        logger.debug("redis unavailable: %s", e)
        return None


def _clean_id(v: str) -> str:
    """Sanitize an instance id from env. systemd ``EnvironmentFile`` does NOT strip
    inline ``#`` comments, so ``AWS_GPU_INSTANCE_ID=i-123  # note`` arrives with the
    comment attached. Take the first whitespace-delimited token (the id), dropping
    any trailing comment — an instance id never contains spaces."""
    return (v or "").split("#", 1)[0].split()[0] if (v or "").strip() else ""


def _spot_id() -> str:
    return _clean_id(os.getenv("AWS_GPU_SPOT_INSTANCE_ID", "")) or _cfg().SPOT_GPU_INSTANCE_ID


def _ondemand_id() -> str:
    return _clean_id(os.getenv("AWS_GPU_INSTANCE_ID", "")) or _cfg().ONDEMAND_GPU_INSTANCE_ID


def _err_code(e) -> str:
    return getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""


def _describe(ec2, iid: str) -> Optional[dict]:
    if not iid:
        return None
    try:
        resp = ec2.describe_instances(InstanceIds=[iid])
    except Exception as e:
        if "InvalidInstanceID" in str(e):
            return None
        raise
    for res in resp.get("Reservations", []):
        for inst in res.get("Instances", []):
            return inst
    return None


def _state(ec2, iid: str) -> str:
    inst = _describe(ec2, iid)
    return inst["State"]["Name"] if inst else "missing"


def _wait_state(ec2, iid: str, target: str, timeout: int, poll: int = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _state(ec2, iid) == target:
            return True
        time.sleep(poll)
    return False


# ── Core ladder ────────────────────────────────────────────────────────────---

def ensure_gpu_ready(timeout: Optional[int] = None) -> tuple[bool, str]:
    """Bring a GPU online (spot-first, on-demand failover). Returns (ready, reason)."""
    if not _enabled():
        return True, "disabled"

    timeout = int(timeout if timeout is not None else os.getenv("GPU_BOOT_TIMEOUT_S", "300"))

    try:
        ec2 = _ec2_client()
    except Exception as e:
        logger.warning("ensure_gpu_ready: boto3 unavailable: %s", e)
        return False, f"boto3-unavailable: {e}"

    # 0) STICK to the RECORDED active box if it's still running — don't re-flip its
    # EIP or re-record mid-job. Only an explicit Redis record counts here (not the
    # spot-id fallback); with nothing recorded we fall through to _try_spot so the
    # box gets properly finalized (EIP attached + recorded).
    active = _recorded_active_id()
    if active and _state(ec2, active) == "running":
        return True, "active-running"

    # 1) SPOT preferred.
    ok, reason = _try_spot(ec2, timeout)
    if ok:
        return True, reason
    logger.warning("spot unavailable (%s) — failing over to on-demand", reason)

    # 2) ON-DEMAND failover (just start; it has its own root).
    ok, reason2 = _try_ondemand(ec2, timeout)
    if ok:
        return True, reason2
    logger.error("on-demand failover also failed (%s)", reason2)
    return False, f"no-gpu-available: spot={reason} ondemand={reason2}"


def _try_spot(ec2, timeout: int) -> tuple[bool, str]:
    iid = _spot_id()
    state = _state(ec2, iid)
    logger.info("spot %s state=%s", iid, state)

    if state == "running":
        _finalize(ec2, iid)
        return True, "spot-running"

    # Missing/terminated → no AMI to auto-relaunch from. The root volume survived
    # (DoT=false) but re-creating the spot + re-attaching the root is manual.
    # Flag it and fail over to the on-demand for THIS work.
    if state in ("missing",) + _TERMINAL:
        _flag_spot_manual_recovery()
        return False, "spot-missing-needs-manual-recovery"

    if state == "stopping":
        if not _wait_state(ec2, iid, "stopped", timeout):
            return False, "timeout-spot-stopping"
        state = "stopped"

    if state == "stopped":
        try:
            ec2.start_instances(InstanceIds=[iid])
        except Exception as e:
            code = _err_code(e)
            if code in _SPOT_CAPACITY_ERRORS:
                return False, f"spot-capacity:{code}"
            return False, f"spot-start-failed: {e}"

    if _wait_state(ec2, iid, "running", timeout):
        _finalize(ec2, iid)
        return True, "spot-started"
    return False, "timeout-spot-pending"


def _try_ondemand(ec2, timeout: int) -> tuple[bool, str]:
    iid = _ondemand_id()
    state = _state(ec2, iid)
    logger.info("on-demand %s state=%s", iid, state)
    if state == "missing":
        return False, "ondemand-missing"
    if state == "running":
        _finalize(ec2, iid)
        return True, "ondemand-running"
    if state in _TERMINAL:
        return False, "ondemand-terminated"

    if state == "stopping":
        if not _wait_state(ec2, iid, "stopped", timeout):
            return False, "timeout-ondemand-stopping"

    try:
        ec2.start_instances(InstanceIds=[iid])
    except Exception as e:
        return False, f"ondemand-start-failed: {e}"

    if _wait_state(ec2, iid, "running", timeout):
        _finalize(ec2, iid)
        return True, "ondemand-started"
    return False, "timeout-ondemand-pending"


def _flag_spot_manual_recovery() -> None:
    """Record that the spot is gone and needs MANUAL recovery (no AMI to relaunch
    from under the no-backup model). The root volume survives (DoT=false); an
    operator must create a fresh persistent spot and attach it as /dev/xvda."""
    r = _redis()
    if r is not None:
        try:
            r.setex(REDIS_SPOT_RECOVER_KEY, 86400, "1")
        except Exception:
            pass
    logger.error(
        "SPOT is missing/terminated. Its root volume survived (DoT=false) but there "
        "is NO AMI to auto-relaunch from. Falling back to on-demand. To restore the "
        "spot: launch a new persistent (interruption=stop) g7e in subnet %s, stop it, "
        "swap its root to the surviving spot volume, set DoT=false, start it, then "
        "update SPOT_GPU_INSTANCE_ID.", _cfg().GPU_SUBNET_ID,
    )


def _finalize(ec2, iid: str) -> None:
    """Attach the box's OWN EIP and record it active."""
    box = _cfg().GPU_BOXES.get(iid, {})
    alloc = box.get("eip_alloc") or os.getenv("AWS_GPU_EIP_ALLOC_ID", "").strip()
    if alloc:
        try:
            ec2.associate_address(InstanceId=iid, AllocationId=alloc)
            logger.info("attached EIP %s to %s", alloc, iid)
        except Exception as e:
            logger.warning("associate_address failed for %s: %s", iid, e)
    _set_active(iid)


def _set_active(iid: str) -> None:
    r = _redis()
    if r is None:
        return
    try:
        r.set(REDIS_ACTIVE_KEY, iid)
        logger.info("recorded active GPU = %s", iid)
    except Exception as e:
        logger.warning("could not record active GPU: %s", e)


def _recorded_active_id() -> Optional[str]:
    """The instance id explicitly recorded active in Redis, or None if unset.
    Used by the STICK check — must NOT fall back to a default, or a never-finalized
    box would be treated as already-active and skip EIP attach."""
    r = _redis()
    if r is None:
        return None
    try:
        return r.get(REDIS_ACTIVE_KEY) or None
    except Exception:
        return None


def get_active_instance_id() -> str:
    """The box the orchestrator currently manages. Resolution order:
    Redis active key → the spot (steady-state primary)."""
    return _recorded_active_id() or _spot_id()


# ── UI status ──────────────────────────────────────────────────────────────--

def get_gpu_status(instance_id: str, r=None) -> dict:
    """Resolve a single GPU instance's lifecycle phase for the status panel."""
    out = {
        "iid": instance_id or "", "ec2_state": "missing", "prewarm_ready": False,
        "phase": "missing", "phase_label": "❌ Not configured",
        "public_ip": "", "instance_type": "", "launch_time": "", "detail": "",
    }
    if not instance_id:
        return out
    try:
        ec2 = _ec2_client()
        inst = _describe(ec2, instance_id)
    except Exception as e:
        out.update(ec2_state="error", phase="unknown", phase_label="⚠ EC2 lookup failed", detail=str(e)[:100])
        return out
    if inst is None:
        out["phase_label"] = "❌ Instance not found"
        return out

    state = inst["State"]["Name"]
    out["ec2_state"]     = state
    out["public_ip"]     = inst.get("PublicIpAddress", "") or ""
    out["instance_type"] = inst.get("InstanceType", "") or ""
    lt = inst.get("LaunchTime")
    out["launch_time"]   = lt.isoformat() if lt else ""

    if r is not None and state == "running":
        try:
            from lib.autoshutdown_ctl import is_prewarm_ready
            out["prewarm_ready"] = is_prewarm_ready(r, instance_id)
        except Exception:
            out["prewarm_ready"] = False

    if state == "running":
        if out["prewarm_ready"]:
            out.update(phase="ready", phase_label="🟢 Ready for inference")
        else:
            pending = False
            if r is not None:
                try:
                    pending = bool(r.get(f"prewarm:pending:{instance_id}"))
                except Exception:
                    pass
            out.update(**({"phase": "prewarming", "phase_label": "🟠 Prewarming model weights (~8-10 min)"}
                          if pending else {"phase": "ready", "phase_label": "🟢 Ready for inference"}))
    elif state == "pending":
        out.update(phase="booting", phase_label="🟡 Booting")
    elif state == "stopping":
        out.update(phase="stopping", phase_label="🟡 Stopping")
    elif state == "stopped":
        out.update(phase="stopped", phase_label="⏸  Stopped — will start on next queue")
    elif state == "terminated":
        out.update(phase="missing", phase_label="❌ Terminated")
    else:
        out.update(phase="unknown", phase_label=f"⚠ State: {state}")
    return out


# ── Stop ─────────────────────────────────────────────────────────────────────-

def stop_gpu(force: bool = False) -> tuple[bool, str]:
    """Stop the active GPU box."""
    if not _enabled() and not force:
        return False, "disabled"
    try:
        ec2 = _ec2_client()
    except Exception as e:
        return False, f"boto3-unavailable: {e}"
    iid = get_active_instance_id()
    if not iid:
        return False, "missing"
    try:
        ec2.stop_instances(InstanceIds=[iid])
        return True, "stopping"
    except Exception as e:
        return False, f"stop-failed: {e}"
