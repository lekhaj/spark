"""
gpu_launcher.py — auto-start the GPU instance from the CPU before queueing.

Activation
----------
Gated by env var ``GPU_AUTO_LAUNCH``. Default ``0`` (no-op) — code is inert
until you set ``GPU_AUTO_LAUNCH=1`` in the CPU env.

Public API
----------
    ensure_gpu_ready(timeout=120) -> tuple[bool, str]
        Returns (True, "running") if GPU was/became reachable in time.
        Returns (False, reason) if disabled, unreachable, or timed out.

Behaviour matrix
----------------
    running     → no-op, return True
    stopped     → StartInstances, poll until running
    pending     → poll until running
    stopping    → poll until stopped, then start
    terminated  → reserved for Phase 2 (request new spot from AMI).
                  Today: return False with reason="terminated".
    missing     → return False with reason="missing"

Lookup strategy
---------------
1. AWS_GPU_INSTANCE_ID env var (preferred, current production setup).
2. Tag-based: instances with ``Project=spark-gpu`` in state running/pending/stopped/stopping.

Environment vars
----------------
    GPU_AUTO_LAUNCH       "1" to activate. Default "0".
    AWS_GPU_INSTANCE_ID   instance to manage (e.g. i-0d6b9d6d34ccc053d).
    AWS_REGION            default "us-east-1".
    GPU_BOOT_TIMEOUT_S    seconds to wait for running state. Default 120.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger("gpu_launcher")

# ── Feature flag ──────────────────────────────────────────────────────────────

def _enabled() -> bool:
    return os.getenv("GPU_AUTO_LAUNCH", "0") == "1"


# ── boto3 helpers (imported lazily so module loads even without boto3) ────────

def _ec2_client():
    import boto3
    return boto3.client("ec2", region_name=os.getenv("AWS_REGION", "us-east-1"))


def _find_instance_id(ec2) -> Optional[str]:
    """Resolve the GPU instance id: env var first, then tag lookup."""
    iid = os.getenv("AWS_GPU_INSTANCE_ID", "").strip()
    if iid:
        return iid

    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:Project",          "Values": ["spark-gpu"]},
        {"Name": "instance-state-name",  "Values": ["running", "pending", "stopped", "stopping"]},
    ])
    for res in resp.get("Reservations", []):
        for inst in res.get("Instances", []):
            return inst["InstanceId"]
    return None


def _describe(ec2, iid: str) -> Optional[dict]:
    """Return the full Instance dict for iid, or None if not found."""
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


# ── Core API ──────────────────────────────────────────────────────────────────

def ensure_gpu_ready(timeout: Optional[int] = None) -> tuple[bool, str]:
    """
    Return (ready, reason).

    ready == True only when the GPU instance is fully `running`. Reason is a
    short human-readable status word ("running", "started", "timeout", etc).
    """
    if not _enabled():
        return True, "disabled"

    timeout = int(timeout if timeout is not None else os.getenv("GPU_BOOT_TIMEOUT_S", "120"))

    try:
        ec2 = _ec2_client()
    except Exception as e:
        logger.warning("ensure_gpu_ready: boto3 unavailable: %s", e)
        return False, f"boto3-unavailable: {e}"

    iid = _find_instance_id(ec2)
    if not iid:
        return False, "missing"

    inst = _describe(ec2, iid)
    if inst is None:
        return False, "missing"

    state = inst["State"]["Name"]
    logger.info("ensure_gpu_ready: %s state=%s", iid, state)

    if state == "running":
        return True, "running"

    if state == "terminated":
        # Phase 2: launch new spot from AMI. Not yet implemented.
        return False, "terminated"

    if state == "stopping":
        if not _wait_for(ec2, iid, "stopped", timeout):
            return False, "timeout-stopping"
        state = "stopped"

    if state == "stopped":
        try:
            ec2.start_instances(InstanceIds=[iid])
        except Exception as e:
            logger.warning("start_instances failed: %s", e)
            return False, f"start-failed: {e}"

    # state in ("pending", just-started)
    if _wait_for(ec2, iid, "running", timeout):
        return True, "started"
    return False, "timeout-pending"


def _wait_for(ec2, iid: str, target: str, timeout: int, poll: int = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = _describe(ec2, iid)
        if inst and inst["State"]["Name"] == target:
            return True
        time.sleep(poll)
    return False


# ── Optional: stop helper for explicit shutdown from CPU UI ───────────────────

def stop_gpu(force: bool = False) -> tuple[bool, str]:
    """Stop the GPU instance. Returns (ok, reason)."""
    if not _enabled() and not force:
        return False, "disabled"
    try:
        ec2 = _ec2_client()
    except Exception as e:
        return False, f"boto3-unavailable: {e}"
    iid = _find_instance_id(ec2)
    if not iid:
        return False, "missing"
    try:
        ec2.stop_instances(InstanceIds=[iid])
        return True, "stopping"
    except Exception as e:
        return False, f"stop-failed: {e}"
