import boto3
import subprocess
import time
from app.config import settings
from app import infra

# GPU lifecycle (spot-first / on-demand-fallback + EIP) is owned by
# worker/lib/gpu_launcher.py. This module provides the lower-level start/stop/SSH
# primitives the orchestrator and routes call.


def _boto3_kwargs() -> dict:
    """
    Build boto3 credential kwargs from settings.
    When AWS_ACCESS_KEY_ID is unset (EC2 with IAM role attached), returns {}
    so boto3 uses instance metadata credentials automatically — no expiry, no tokens.
    """
    if settings.AWS_ACCESS_KEY_ID:
        return {
            "aws_access_key_id":     settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            "aws_session_token":     settings.AWS_SESSION_TOKEN,
        }
    return {}


# AWS EC2 client
ec2 = boto3.client("ec2", region_name=settings.AWS_REGION, **_boto3_kwargs())

# ── Instance ID map (alias → EC2 instance ID) ────────────────────────────────
# All GPU aliases map to the single active GPU instance.
# To swap instances: update infra.py only.
INSTANCE_MAP = {
    "cpu":           settings.AWS_CPU_INSTANCE_ID or infra.CPU_INSTANCE_ID,
    "gpu_a10":       settings.AWS_GPU_INSTANCE_ID or infra.GPU_INSTANCE_ID,
    "gpu_a10_image": settings.AWS_GPU_INSTANCE_ID or infra.GPU_INSTANCE_ID,
    "gpu_t4":        settings.AWS_GPU_INSTANCE_ID or infra.GPU_INSTANCE_ID,  # legacy alias
}

# GPU_CONFIG — SSH details per alias (all point to same GPU host for now)
GPU_CONFIG = infra.GPU_CONFIG


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_instance_id(instance_name: str) -> str:
    """Resolve alias to actual EC2 instance ID."""
    return INSTANCE_MAP.get(instance_name.lower(), instance_name)


_iam_describe_warned = False

def get_instance_state(instance_name: str) -> str:
    """Return current EC2 state of an instance, or 'unknown' on any error."""
    global _iam_describe_warned
    try:
        instance_id = get_instance_id(instance_name.lower())
        response = ec2.describe_instances(InstanceIds=[instance_id])
        return response["Reservations"][0]["Instances"][0]["State"]["Name"]
    except Exception as e:
        err = str(e)
        if "UnauthorizedOperation" in err or "AccessDenied" in err:
            if not _iam_describe_warned:
                print(f"[AWS] ec2:DescribeInstances not permitted by IAM role — instance state checks disabled. Manage GPU manually.")
                _iam_describe_warned = True
        else:
            print(f"[AWS ERROR] Failed to get instance state for {instance_name}: {e}")
        return "unknown"


def wait_for_instance_state(instance_id: str, target_state: str, timeout: int = 600) -> bool:
    """Poll until instance reaches target_state or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            state = get_instance_state(instance_id)
            print(f"[AWS] Instance {instance_id} state: {state}")
            if state == target_state:
                if target_state == "running":
                    time.sleep(30)  # extra wait for SSH readiness
                return True
            time.sleep(10)
        except Exception as e:
            print(f"[AWS ERROR] While waiting for instance: {e}")
            time.sleep(10)
    print(f"[AWS TIMEOUT] {instance_id} didn't reach {target_state} in {timeout}s")
    return False


def start_instance(instance_name: str) -> bool:
    """Start an EC2 instance by alias or raw instance id."""
    try:
        instance_id = get_instance_id(instance_name.lower())
        print(f"[AWS] Starting instance: {instance_name} ({instance_id})")
        response = ec2.start_instances(InstanceIds=[instance_id])
        state = response["StartingInstances"][0]["CurrentState"]["Name"]
        if state == "pending":
            return wait_for_instance_state(instance_id, "running")
        return True
    except Exception as e:
        print(f"[AWS ERROR] Failed to start instance {instance_name}: {e}")
        return False


def stop_instance(instance_name: str) -> bool:
    """Stop an EC2 instance by alias or raw instance id."""
    try:
        instance_id = get_instance_id(instance_name.lower())
        print(f"[AWS] Stopping instance: {instance_name} ({instance_id})")
        response = ec2.stop_instances(InstanceIds=[instance_id])
        state = response["StoppingInstances"][0]["CurrentState"]["Name"]
        if state == "stopping":
            return wait_for_instance_state(instance_id, "stopped")
        return True
    except Exception as e:
        print(f"[AWS ERROR] Failed to stop instance {instance_name}: {e}")
        return False


def ssh_to_gpu(gpu_type: str, command: str, timeout: int = 60) -> tuple[bool, str]:
    """
    Run an SSH command on the GPU instance.
    GPU type is resolved via GPU_CONFIG → fixed public IP.
    """
    gpu_type = gpu_type.lower()
    cfg = GPU_CONFIG.get(gpu_type)
    if not cfg:
        print(f"[SSH ERROR] Unknown GPU type: {gpu_type}")
        return False, ""

    public_ip   = cfg["public_ip"]
    ssh_user    = cfg["ssh_user"]
    key_path    = settings.GPU_SSH_KEY_PATH or infra.SSH_KEY_PATH

    full_cmd = [
        "ssh", "-i", key_path,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=30",
        "-o", "BatchMode=yes",
        f"{ssh_user}@{public_ip}",
        command,
    ]

    try:
        print(f"[SSH → {gpu_type}] {command}")
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            print(f"[SSH ERROR] rc={result.returncode}: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        print("[SSH ERROR] Command timed out")
        return False, "timeout"
    except Exception as e:
        print(f"[SSH ERROR] {e}")
        return False, str(e)


def start_gpu_worker(gpu_type: str) -> bool:
    """Start the systemd worker service on the GPU instance."""
    cfg = GPU_CONFIG.get(gpu_type.lower())
    if not cfg:
        print(f"[AWS] Unknown GPU type: {gpu_type}")
        return False
    service = cfg["worker_service"]
    print(f"[AWS] Starting {service} on GPU instance")
    ok, _ = ssh_to_gpu(gpu_type, f"sudo systemctl start {service}")
    return ok


def stop_gpu_worker(gpu_type: str) -> bool:
    """Stop the systemd worker service on the GPU instance."""
    cfg = GPU_CONFIG.get(gpu_type.lower())
    if not cfg:
        return False
    service = cfg["worker_service"]
    print(f"[AWS] Stopping {service} on GPU instance")
    ok, _ = ssh_to_gpu(gpu_type, f"sudo systemctl stop {service}")
    return ok


def restart_gpu_worker(gpu_type: str) -> bool:
    """Restart the systemd worker service on the GPU instance."""
    cfg = GPU_CONFIG.get(gpu_type.lower())
    if not cfg:
        return False
    service = cfg["worker_service"]
    print(f"[AWS] Restarting {service} on GPU instance")
    ok, _ = ssh_to_gpu(gpu_type, f"sudo systemctl restart {service}")
    return ok


def is_gpu_worker_running(gpu_type: str) -> bool:
    """
    Check if a GPU worker is actively processing.

    image-worker  → check systemctl is-active
    model-worker  → check for running subprocess OR VRAM usage
    """
    gpu_type = gpu_type.lower()
    cfg = GPU_CONFIG.get(gpu_type)
    if not cfg:
        return False

    if get_instance_state(gpu_type) != "running":
        return False

    service = cfg["worker_service"]

    # image-worker: direct systemd check
    if gpu_type == "gpu_a10_image":
        ok, output = ssh_to_gpu(gpu_type, f"systemctl is-active {service}", timeout=15)
        result = ok and output.strip() == "active"
        print(f"[AWS] {service} active={result}")
        return result

    # model-worker: check for running Python subprocess
    ok, output = ssh_to_gpu(
        gpu_type,
        "pgrep -f 'python.*model_worker_simple\\.py\\|python.*model_worker_trellis\\.py\\|python.*run_trellis' | wc -l",
        timeout=15,
    )
    if ok:
        try:
            if int(output.strip()) > 0:
                print("[AWS] Active processing worker found")
                return True
        except (ValueError, IndexError):
            pass

    # Fallback: VRAM usage check
    ok, output = ssh_to_gpu(
        gpu_type,
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0",
        timeout=15,
    )
    if ok:
        try:
            vram_mb = int(output.strip().splitlines()[0])
            if vram_mb > 1000:
                print(f"[AWS] GPU VRAM in use ({vram_mb}MB) — treating as busy")
                return True
        except (ValueError, IndexError):
            pass

    print("[AWS] No active processing — worker is idle")
    return False


def ensure_gpu_worker_running(gpu_type: str) -> bool:
    """Start the worker if it's not already running."""
    if is_gpu_worker_running(gpu_type):
        print(f"[AWS] {gpu_type} worker is already running")
        return True
    print(f"[AWS] {gpu_type} worker is not running — starting...")
    return start_gpu_worker(gpu_type)


def get_gpu_instance_status(gpu_type: str) -> dict:
    """Return a status dict for the GPU instance and its worker."""
    gpu_type = gpu_type.lower()
    cfg = GPU_CONFIG.get(gpu_type, {})
    instance_state = get_instance_state(gpu_type)
    worker_running = is_gpu_worker_running(gpu_type) if instance_state == "running" else False
    return {
        "instance_id":    cfg.get("instance_id"),
        "instance_state": instance_state,
        "worker_running": worker_running,
        "worker_service": cfg.get("worker_service"),
        "public_ip":      cfg.get("public_ip"),
    }


def _s3_client():
    return boto3.client("s3", region_name=settings.AWS_REGION, **_boto3_kwargs())


def download_from_s3(bucket: str, key: str, download_path: str):
    """Download a file from S3."""
    import os
    s3 = _s3_client()
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    s3.download_file(bucket, key, download_path)
    print(f"[S3] Downloaded: {bucket}/{key} → {download_path}")


def upload_to_s3(bucket: str, key: str, file_path: str):
    """Upload a file to S3."""
    s3 = _s3_client()
    s3.upload_file(file_path, bucket, key)
    print(f"[S3] Uploaded: {file_path} → {bucket}/{key}")
