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
    GPU_AUTO_LAUNCH         "1" to activate. Default "0".
    AWS_GPU_INSTANCE_ID     instance to manage (e.g. i-0d6b9d6d34ccc053d).
    AWS_REGION              default "us-east-1".
    GPU_BOOT_TIMEOUT_S      seconds to wait for running state. Default 180.

For launching a NEW persistent-spot from AMI when the previous one is
terminated by AWS (rare with persistent+stop, but handled):

    AWS_GPU_AMI_ID          AMI to boot from (e.g. ami-0d689e40322983537)
    AWS_GPU_INSTANCE_TYPE   default "g6e.2xlarge"
    AWS_GPU_SUBNET_ID       subnet (AZ-locked); default subnet-0c5b465f9ede9e6ce
    AWS_GPU_SG_ID           security group; default sg-0a4a561065082e3c9
    AWS_GPU_KEY_NAME        SSH key pair name; default us_cpu_key
    AWS_GPU_INSTANCE_PROFILE  IAM instance profile; default ec2_s3
    AWS_GPU_EIP_ALLOC_ID    optional Elastic IP allocation to attach
    AWS_GPU_PROJECT_TAG     tag value for "Project"; default "spark-gpu"
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
    inst = _describe(ec2, iid) if iid else None

    # No instance exists at all (e.g. first launch after manual termination,
    # or AWS reclaimed the persistent spot). Provision a fresh one from AMI
    # if we have the config to do so.
    if inst is None or inst["State"]["Name"] == "terminated":
        new_id, reason = _launch_spot_from_ami(ec2)
        if not new_id:
            return False, reason
        logger.info("ensure_gpu_ready: launched new spot %s (was %s)", new_id, iid or "missing")
        iid = new_id
        if _wait_for(ec2, iid, "running", timeout):
            _maybe_attach_eip(ec2, iid)
            return True, "launched"
        return False, "timeout-new-spot"

    state = inst["State"]["Name"]
    logger.info("ensure_gpu_ready: %s state=%s", iid, state)

    if state == "running":
        return True, "running"

    # terminated case is handled above (we never get here with state==terminated)

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

def _launch_spot_from_ami(ec2) -> tuple[Optional[str], str]:
    """RunInstances → persistent spot from AMI. Returns (instance_id, reason)."""
    ami = os.getenv("AWS_GPU_AMI_ID", "").strip()
    if not ami:
        return None, "no-ami-configured"

    instance_type    = os.getenv("AWS_GPU_INSTANCE_TYPE",   "g6e.2xlarge")
    subnet_id        = os.getenv("AWS_GPU_SUBNET_ID",       "subnet-0c5b465f9ede9e6ce")
    sg_id            = os.getenv("AWS_GPU_SG_ID",           "sg-0a4a561065082e3c9")
    key_name         = os.getenv("AWS_GPU_KEY_NAME",        "us_cpu_key")
    instance_profile = os.getenv("AWS_GPU_INSTANCE_PROFILE","ec2_s3")
    project_tag      = os.getenv("AWS_GPU_PROJECT_TAG",     "spark-gpu")

    logger.info(
        "launching spot from AMI %s (%s in %s)",
        ami, instance_type, subnet_id,
    )
    try:
        resp = ec2.run_instances(
            ImageId=ami,
            InstanceType=instance_type,
            MinCount=1, MaxCount=1,
            SubnetId=subnet_id,
            SecurityGroupIds=[sg_id],
            KeyName=key_name,
            IamInstanceProfile={"Name": instance_profile},
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "persistent",
                    "InstanceInterruptionBehavior": "stop",
                },
            },
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": [
                    {"Key": "Name",    "Value": "spark_gpu_spot"},
                    {"Key": "Project", "Value": project_tag},
                ]},
            ],
        )
        iid = resp["Instances"][0]["InstanceId"]
        return iid, "launched"
    except Exception as e:
        logger.error("run_instances failed: %s", e)
        return None, f"launch-failed: {e}"


def _maybe_attach_eip(ec2, iid: str) -> None:
    """Attach the configured Elastic IP, if any, to the given instance."""
    alloc = os.getenv("AWS_GPU_EIP_ALLOC_ID", "").strip()
    if not alloc:
        return
    try:
        ec2.associate_address(InstanceId=iid, AllocationId=alloc)
        logger.info("attached EIP %s to %s", alloc, iid)
    except Exception as e:
        logger.warning("associate_address failed for %s: %s", iid, e)


def get_gpu_status(instance_id: str, r=None) -> dict:
    """
    Resolve a single GPU instance's lifecycle phase for the UI status panel.

    Returns a dict:
      {
        "iid": "<instance-id>",
        "ec2_state": "running|stopped|pending|stopping|terminated|missing|error",
        "prewarm_ready": True | False,        # Redis prewarm:ready:<iid> == "1"
        "phase":   "ready|prewarming|booting|stopped|stopping|missing|unknown",
        "phase_label": "🟢 Ready for inference",
        "public_ip":  "52.91.128.47" or "",
        "instance_type": "g6e.2xlarge" or "",
        "launch_time":  "2026-05-16T14:21:37Z" or "",
        "detail":  "optional human note",
      }
    """
    out = {
        "iid": instance_id or "",
        "ec2_state": "missing",
        "prewarm_ready": False,
        "phase": "missing",
        "phase_label": "❌ Not configured",
        "public_ip": "",
        "instance_type": "",
        "launch_time": "",
        "detail": "",
    }
    if not instance_id:
        return out

    # ── EC2 describe ──────────────────────────────────────────────────────
    try:
        ec2 = _ec2_client()
        inst = _describe(ec2, instance_id)
    except Exception as e:
        out["ec2_state"] = "error"
        out["phase"] = "unknown"
        out["phase_label"] = "⚠ EC2 lookup failed"
        out["detail"] = str(e)[:100]
        return out

    if inst is None:
        out["phase_label"] = "❌ Instance not found"
        return out

    state = inst["State"]["Name"]
    out["ec2_state"]      = state
    out["public_ip"]      = inst.get("PublicIpAddress", "") or ""
    out["instance_type"]  = inst.get("InstanceType", "") or ""
    lt = inst.get("LaunchTime")
    out["launch_time"]    = lt.isoformat() if lt else ""

    # ── Prewarm sentinel via Redis ────────────────────────────────────────
    if r is not None and state == "running":
        try:
            from lib.autoshutdown_ctl import is_prewarm_ready
            out["prewarm_ready"] = is_prewarm_ready(r, instance_id)
        except Exception:
            out["prewarm_ready"] = False

    # ── Phase resolution ──────────────────────────────────────────────────
    if state == "running":
        if out["prewarm_ready"]:
            out["phase"]       = "ready"
            out["phase_label"] = "🟢 Ready for inference"
        else:
            out["phase"]       = "prewarming"
            out["phase_label"] = "🟠 Prewarming model weights (~15-20 min)"
    elif state == "pending":
        out["phase"]       = "booting"
        out["phase_label"] = "🟡 Booting"
    elif state == "stopping":
        out["phase"]       = "stopping"
        out["phase_label"] = "🟡 Stopping"
    elif state == "stopped":
        out["phase"]       = "stopped"
        out["phase_label"] = "⏸  Stopped — will start on next queue"
    elif state == "terminated":
        out["phase"]       = "missing"
        out["phase_label"] = "❌ Terminated"
    else:
        out["phase"]       = "unknown"
        out["phase_label"] = f"⚠ State: {state}"

    return out


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
