"""
gpu_launcher.py — ensure a GPU is running before queueing work.

Spot-first, on-demand-fallback ladder
-------------------------------------
``ensure_gpu_ready()`` is the single authority for bringing a GPU online. It is
called both at queue-time (``manual_gen_queue``) and from the CPU orchestrator
loop (``orchestrator_service``). The ladder, in order, with no edge cases:

    0. STICK: if a box is already running and serving (the recorded active
       instance), keep using it — never flip the EIP off a box that is mid-job.
    1. SPOT preferred (cheapest):
         running              → use it (attach EIP, mark active)
         stopped / stopping   → start it, wait for running
         can't start (spot capacity / start error) → fall back to on-demand
         terminated / missing → launch a NEW persistent spot from AMI in the
                                 BACKGROUND (non-blocking) AND fall back to
                                 on-demand for the current work; the fresh spot
                                 warms up and is preferred next work period.
    2. ON-DEMAND fallback: start the on-demand box, wait for running.
    3. On success: attach the Elastic IP to whichever box won (so the CPU always
       reaches a stable host) and record ``gpu:active_instance_id`` in Redis.

Both boxes run the GPU-side ``auto_shutdown`` watcher, so each independently
stops itself after the idle threshold (default 15 min, Redis-configurable from
the Gradio panel) when its queues are empty and nothing is processing — even the
parallel spot that was launched but never received work.

Activation
----------
Gated by ``GPU_AUTO_LAUNCH``. Default ``0`` (no-op) — returns (True, "disabled").

Environment vars
----------------
    GPU_AUTO_LAUNCH            "1" to activate. Default "0".
    AWS_GPU_SPOT_INSTANCE_ID   spot instance to prefer (e.g. i-083d…).
    AWS_GPU_INSTANCE_ID        on-demand fallback instance (e.g. i-05e857…).
    AWS_GPU_EIP_ALLOC_ID       Elastic IP allocation that rides the active box.
    AWS_REGION                 default "us-east-1".
    GPU_BOOT_TIMEOUT_S         seconds to wait for running state. Default 180.
    REDIS_HOST/REDIS_PORT/REDIS_PASSWORD  to record the active instance id.

For relaunching a reclaimed spot from AMI:
    AWS_GPU_AMI_ID            AMI to boot from (the baked g7e image).
    AWS_GPU_INSTANCE_TYPE     default "g7e.2xlarge".
    AWS_GPU_SUBNET_ID         subnet (AZ-locked); default subnet-0c5b465f9ede9e6ce.
    AWS_GPU_SG_ID             security group; default sg-0d2c95823f90b5b25.
    AWS_GPU_KEY_NAME          SSH key pair name; default us_cpu_key.
    AWS_GPU_INSTANCE_PROFILE  IAM instance profile; default ec2_s3.
    AWS_GPU_PROJECT_TAG       tag value for "Project"; default "spark-gpu".
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger("gpu_launcher")

# Redis key: the instance id the EIP currently rides / the orchestrator manages.
REDIS_ACTIVE_KEY = "gpu:active_instance_id"
# Redis key: the most recently known spot instance id (survives relaunch, since a
# relaunched spot gets a brand-new id that the env var no longer matches).
REDIS_SPOT_KEY = "gpu:spot_instance_id"

_TERMINAL = ("terminated", "shutting-down")
_SPOT_CAPACITY_ERRORS = (
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "MaxSpotInstanceCountExceeded",
    "InsufficientSpotInstanceCapacity",
)


# ── Feature flag ──────────────────────────────────────────────────────────────

def _enabled() -> bool:
    return os.getenv("GPU_AUTO_LAUNCH", "0") == "1"


# ── boto3 / redis helpers (imported lazily so module loads without them) ───────

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


def _ondemand_id() -> str:
    return os.getenv("AWS_GPU_INSTANCE_ID", "").strip()


def _describe(ec2, iid: str) -> Optional[dict]:
    """Return the full Instance dict for iid, or None if not found."""
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


def _wait_for(ec2, iid: str, target: str, timeout: int, poll: int = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = _describe(ec2, iid)
        if inst and inst["State"]["Name"] == target:
            return True
        time.sleep(poll)
    return False


# ── Core API ──────────────────────────────────────────────────────────────────

def ensure_gpu_ready(timeout: Optional[int] = None) -> tuple[bool, str]:
    """
    Bring a GPU online via the spot-first / on-demand-fallback ladder.

    Returns (ready, reason). ``ready`` is True only when a GPU is fully running
    with the EIP attached. Reason is a short status word:
      "disabled", "active-running", "spot-running", "spot-started",
      "ondemand-running", "ondemand-started", or a failure word.
    """
    if not _enabled():
        return True, "disabled"

    timeout = int(timeout if timeout is not None else os.getenv("GPU_BOOT_TIMEOUT_S", "180"))

    try:
        ec2 = _ec2_client()
    except Exception as e:
        logger.warning("ensure_gpu_ready: boto3 unavailable: %s", e)
        return False, f"boto3-unavailable: {e}"

    # 0) STICK to a box that is already serving — never flip the EIP mid-job.
    active = get_active_instance_id()
    if active and _is_running(ec2, active):
        logger.debug("ensure_gpu_ready: active %s already running — sticking", active)
        return True, "active-running"

    # 1) SPOT preferred.
    ok, reason = _try_spot(ec2, timeout)
    if ok:
        return True, reason
    logger.warning("ensure_gpu_ready: spot unavailable (%s) — falling back to on-demand", reason)

    # 2) ON-DEMAND fallback.
    ok, reason = _try_ondemand(ec2, timeout)
    if ok:
        return True, reason
    logger.error("ensure_gpu_ready: on-demand fallback also failed (%s)", reason)
    return False, reason


def _is_running(ec2, iid: str) -> bool:
    inst = _describe(ec2, iid)
    return bool(inst) and inst["State"]["Name"] == "running"


def _alive(ec2, iid: str) -> bool:
    """True if the instance exists and is not terminated/shutting-down."""
    if not iid:
        return False
    inst = _describe(ec2, iid)
    return bool(inst) and inst["State"]["Name"] not in _TERMINAL


def _resolve_spot_id(ec2) -> Optional[str]:
    """
    Find the current spot instance id. A relaunched spot gets a NEW id, so we
    resolve in order: Redis (last launched) → env var → tag lookup. Returns the
    env id even if terminated (so the caller's terminated-branch relaunches),
    or None if nothing is configured.
    """
    r = _redis()
    if r is not None:
        try:
            v = r.get(REDIS_SPOT_KEY)
            if v and _alive(ec2, v):
                return v
        except Exception:
            pass

    env = os.getenv("AWS_GPU_SPOT_INSTANCE_ID", "").strip()
    if env and _alive(ec2, env):
        return env

    # Tag lookup — catches a spot launched out-of-band or after a relaunch whose
    # id we failed to persist.
    try:
        resp = ec2.describe_instances(Filters=[
            {"Name": "tag:Name", "Values": ["spark_gpu_spot_high", "spark_gpu_spot"]},
            {"Name": "instance-state-name", "Values": ["running", "pending", "stopped", "stopping"]},
        ])
        for res in resp.get("Reservations", []):
            for inst in res.get("Instances", []):
                if inst.get("InstanceLifecycle") == "spot":
                    return inst["InstanceId"]
    except Exception as e:
        logger.debug("spot tag lookup failed: %s", e)

    return env or None


def _try_spot(ec2, timeout: int) -> tuple[bool, str]:
    """Start/launch the spot box. Returns (ok, reason); ok=False ⇒ fall back."""
    iid = _resolve_spot_id(ec2)
    inst = _describe(ec2, iid) if iid else None

    # Missing or terminated → spot was DELETED. Launch a fresh persistent spot in
    # the BACKGROUND (don't wait — that's ~15-20 min), persist its new id, and
    # return False so the caller serves the current work on on-demand instead.
    if inst is None or inst["State"]["Name"] in _TERMINAL:
        new_id, reason = _launch_spot_from_ami(ec2)
        if new_id:
            _set_spot_id(new_id)
            logger.info("spot deleted — new spot %s launching in background; serving on-demand now", new_id)
            return False, f"spot-relaunching:{new_id}"
        # Couldn't even request a new spot (no AMI / capacity) → on-demand.
        return False, reason

    state = inst["State"]["Name"]
    logger.info("ensure_gpu_ready: spot %s state=%s", iid, state)

    if state == "running":
        _finalize(ec2, iid)
        return True, "spot-running"

    if state == "stopping":
        if not _wait_for(ec2, iid, "stopped", timeout):
            return False, "timeout-spot-stopping"
        state = "stopped"

    if state == "stopped":
        try:
            ec2.start_instances(InstanceIds=[iid])
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
            if code in _SPOT_CAPACITY_ERRORS:
                logger.warning("spot start hit capacity error %s — on-demand fallback", code)
                return False, f"spot-capacity:{code}"
            logger.warning("spot start_instances failed: %s — on-demand fallback", e)
            return False, f"spot-start-failed: {e}"

    # pending or just-started
    if _wait_for(ec2, iid, "running", timeout):
        _finalize(ec2, iid)
        return True, "spot-started"
    return False, "timeout-spot-pending"


def _try_ondemand(ec2, timeout: int) -> tuple[bool, str]:
    """Start the on-demand box. Returns (ok, reason)."""
    iid = _ondemand_id()
    if not iid:
        return False, "no-ondemand-configured"

    inst = _describe(ec2, iid)
    if inst is None or inst["State"]["Name"] == "terminated":
        return False, "ondemand-missing"

    state = inst["State"]["Name"]
    logger.info("ensure_gpu_ready: on-demand %s state=%s", iid, state)

    if state == "running":
        _finalize(ec2, iid)
        return True, "ondemand-running"

    if state == "stopping":
        if not _wait_for(ec2, iid, "stopped", timeout):
            return False, "timeout-ondemand-stopping"
        state = "stopped"

    if state == "stopped":
        try:
            ec2.start_instances(InstanceIds=[iid])
        except Exception as e:
            logger.warning("on-demand start_instances failed: %s", e)
            return False, f"ondemand-start-failed: {e}"

    if _wait_for(ec2, iid, "running", timeout):
        _finalize(ec2, iid)
        return True, "ondemand-started"
    return False, "timeout-ondemand-pending"


def _finalize(ec2, iid: str, prewarm: bool = False) -> None:
    """Attach the EIP to the winning box and record it as active in Redis."""
    _maybe_attach_eip(ec2, iid)
    _set_active(iid)
    if prewarm:
        r = _redis()
        if r is not None:
            try:
                r.setex(f"prewarm:pending:{iid}", 7200, "1")  # 2h TTL fallback
            except Exception as e:
                logger.warning("Could not set prewarm:pending: %s", e)


def _set_active(iid: str) -> None:
    r = _redis()
    if r is None:
        return
    try:
        r.set(REDIS_ACTIVE_KEY, iid)
        logger.info("recorded active GPU = %s", iid)
    except Exception as e:
        logger.warning("could not record active GPU: %s", e)


def _set_spot_id(iid: str) -> None:
    """Persist the current spot id so a relaunched spot (new id) is found next time."""
    r = _redis()
    if r is None:
        return
    try:
        r.set(REDIS_SPOT_KEY, iid)
    except Exception as e:
        logger.warning("could not record spot id: %s", e)


def get_active_instance_id() -> str:
    """Return the instance id the EIP currently rides, or the on-demand default."""
    r = _redis()
    if r is not None:
        try:
            val = r.get(REDIS_ACTIVE_KEY)
            if val:
                return val
        except Exception:
            pass
    return _ondemand_id()


def _launch_spot_from_ami(ec2) -> tuple[Optional[str], str]:
    """RunInstances → persistent spot from AMI. Returns (instance_id, reason)."""
    ami = os.getenv("AWS_GPU_AMI_ID", "").strip()
    if not ami:
        return None, "no-ami-configured"

    instance_type    = os.getenv("AWS_GPU_INSTANCE_TYPE",   "g7e.2xlarge")
    subnet_id        = os.getenv("AWS_GPU_SUBNET_ID",       "subnet-0c5b465f9ede9e6ce")
    sg_id            = os.getenv("AWS_GPU_SG_ID",           "sg-0d2c95823f90b5b25")
    key_name         = os.getenv("AWS_GPU_KEY_NAME",        "us_cpu_key")
    instance_profile = os.getenv("AWS_GPU_INSTANCE_PROFILE","ec2_s3")
    project_tag      = os.getenv("AWS_GPU_PROJECT_TAG",     "spark-gpu")

    logger.info("launching spot from AMI %s (%s in %s)", ami, instance_type, subnet_id)
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
                    {"Key": "Name",    "Value": "spark_gpu_spot_high"},
                    {"Key": "Project", "Value": project_tag},
                ]},
            ],
        )
        iid = resp["Instances"][0]["InstanceId"]
        return iid, "launched"
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
        if code in _SPOT_CAPACITY_ERRORS:
            logger.warning("spot launch hit capacity error %s", code)
            return None, f"spot-capacity:{code}"
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
        "instance_type": "g7e.2xlarge" or "",
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
            # Only show "Prewarming" if CPU explicitly flagged this instance
            # as needing a prewarm (i.e. freshly launched from AMI).
            # On stop/start the key is missing but prewarm was never triggered
            # by CPU — treat as ready.
            prewarm_pending = False
            if r is not None:
                try:
                    prewarm_pending = bool(r.get(f"prewarm:pending:{instance_id}"))
                except Exception:
                    pass
            if prewarm_pending:
                out["phase"]       = "prewarming"
                out["phase_label"] = "🟠 Prewarming model weights (~15-20 min)"
            else:
                out["phase"]       = "ready"
                out["phase_label"] = "🟢 Ready for inference"
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
    """Stop the active GPU instance. Returns (ok, reason)."""
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
