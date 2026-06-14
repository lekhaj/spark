"""
Infrastructure Configuration
============================
Single source of truth for all EC2 instance IDs, public IPs, SSH config,
Redis, MongoDB endpoints, and worker service names.

Update this file when instances are replaced or IPs change.
All services (orchestrator, aws_service, workers) import from here.
"""

# ── EC2 Instance IDs ─────────────────────────────────────────────────────────
CPU_INSTANCE_ID  = "i-0f5a6edd3ce343281"             # new instance (AMI launch 2026-05-09)

# GPU fleet (all g7e.2xlarge in us-east-1d, 2026-06-13). The orchestrator brings
# ONE online at a time via the spot-first / on-demand-fallback ladder
# (worker/lib/gpu_launcher.py). The Elastic IP rides whichever is active, so the
# CPU always talks to GPU_PUBLIC_IP regardless of which box is up.
SPOT_GPU_INSTANCE_ID     = "i-09fca0acb4cc429f7"     # spark_gpu_spot_high (preferred, relaunched 2026-06-14)
ONDEMAND_GPU_INSTANCE_ID = "i-05e8570023728c112"     # spark_gpu_high (fallback)
# Back-compat default for callers that need a single static GPU id. env
# AWS_GPU_INSTANCE_ID (= on-demand) overrides in aws_service.
GPU_INSTANCE_ID  = ONDEMAND_GPU_INSTANCE_ID

# ── Placement (all GPU in us-east-1d) ────────────────────────────────────────
GPU_SUBNET_ID    = "subnet-0c5b465f9ede9e6ce"        # us-east-1d
GPU_AZ           = "us-east-1d"

# ── Two persistent stack root volumes (one per box, no migration) ─────────────
# Each GPU box keeps its OWN 256GB root in us-east-1d: conda envs + code +
# ~150GB model cache. They are independent duplicates — there is NO volume
# migration and NO AMI/snapshot backup (per 2026-06-14 directive). Each is
# protected from AWS deletion: the spot request is persistent with
# interruption=stop (reclaim STOPS, never terminates → volume survives), and both
# volumes have DeleteOnTermination=false (survive an explicit terminate → go
# 'available' in GPU_AZ, reattachable). Failover = just start the other box.
SPOT_STACK_VOLUME_ID     = "vol-0507d56e4766e06f6"   # root of spot box
ONDEMAND_STACK_VOLUME_ID = "vol-06a8484234613a987"   # root of on-demand box (dup of spot's)
STACK_VOLUME_ID  = SPOT_STACK_VOLUME_ID              # back-compat alias
GPU_ROOT_DEVICE  = "/dev/xvda"

# ── Two-pinned Elastic IPs (one per box) ─────────────────────────────────────
# Each box keeps its OWN EIP permanently. The CPU resolves the GPU host IP from
# whichever box is currently active (see active_gpu_ip()). A box NEVER steals the
# other's EIP. On a relaunch-from-AMI the box's own EIP is (re)attached.
SPOT_EIP_ALLOC_ID     = "eipalloc-0d50a0d05c513666a"  # 54.162.11.161 → spot
ONDEMAND_EIP_ALLOC_ID = "eipalloc-0db12aa4d8be94e92"  # 52.91.128.47  → on-demand

# Per-box descriptor: instance id → its pinned EIP + public IP + lifecycle.
GPU_BOXES: dict[str, dict] = {
    SPOT_GPU_INSTANCE_ID: {
        "lifecycle": "spot",
        "eip_alloc": SPOT_EIP_ALLOC_ID,
        "public_ip": "54.162.11.161",
    },
    ONDEMAND_GPU_INSTANCE_ID: {
        "lifecycle": "ondemand",
        "eip_alloc": ONDEMAND_EIP_ALLOC_ID,
        "public_ip": "52.91.128.47",
    },
}

# ── Public IPs ───────────────────────────────────────────────────────────────
CPU_PUBLIC_IP = "18.207.13.85"
# Back-compat constant: the spot is the steady-state active box, so callers that
# still read a single GPU_PUBLIC_IP get the spot. Prefer active_gpu_ip() which
# follows the *currently active* box (spot OR on-demand) at runtime.
GPU_PUBLIC_IP = GPU_BOXES[SPOT_GPU_INSTANCE_ID]["public_ip"]
GPU_PUBLIC_DNS = "ec2-54-162-11-161.compute-1.amazonaws.com"
# GPU private IP is AZ/instance-specific and resolved at runtime via IMDS on the
# GPU side — not pinned here.


def active_gpu_ip() -> str:
    """Public IP of the GPU box the orchestrator currently considers active.

    Reads gpu:active_instance_id from Redis (set by gpu_launcher) and maps it to
    that box's pinned EIP IP. Falls back to the spot's IP (steady-state primary)
    if Redis is unavailable or the id is unknown. This is what ssh_to_gpu must
    use under the two-pinned-EIP model — a single static IP would SSH the wrong
    box whenever the on-demand is the active one.
    """
    try:
        import redis as _r
        c = _r.Redis.from_url(REDIS_URL, decode_responses=True)
        iid = c.get("gpu:active_instance_id")
        if iid and iid in GPU_BOXES:
            return GPU_BOXES[iid]["public_ip"]
    except Exception:
        pass
    return GPU_BOXES[SPOT_GPU_INSTANCE_ID]["public_ip"]

# ── Private VPC IP (CPU side) ────────────────────────────────────────────────
# GPU→CPU Redis/MongoDB traffic uses this. GPU's own private IP is not
# pinned here; the GPU side resolves it via metadata at start-time.
CPU_PRIVATE_IP = "172.31.26.6"

# ── SSH Configuration ────────────────────────────────────────────────────────
SSH_USER     = "ubuntu"       # CPU instance user
GPU_SSH_USER = "ec2-user"     # GPU instance user (Amazon Linux 2023)
# Path to the private key on the CPU instance (orchestrator host)
SSH_KEY_PATH  = "/home/ubuntu/.ssh/us_cpu_key.pem"

# ── AWS Region ───────────────────────────────────────────────────────────────
AWS_REGION = "us-east-1"

# ── Redis (running on CPU instance) ──────────────────────────────────────────
REDIS_URL = f"redis://{CPU_PUBLIC_IP}:6379/0"

# ── MongoDB (running on CPU instance) ────────────────────────────────────────
MONGODB_HOST = CPU_PUBLIC_IP
MONGODB_PORT = 27017
MONGODB_DB   = "World_builder"

# ── GPU Instance Aliases ──────────────────────────────────────────────────────
# Maps logical GPU type names → actual instance ID
GPU_ALIAS_INSTANCE_MAP: dict[str, str] = {
    "gpu_a10":        GPU_INSTANCE_ID,
    "gpu_a10_image":  GPU_INSTANCE_ID,
    "gpu_t4":         GPU_INSTANCE_ID,  # legacy alias — same physical GPU
    "gpu_l4":         GPU_INSTANCE_ID,
}

# ── GPU Instance SSH Config (per alias) ──────────────────────────────────────
GPU_CONFIG: dict[str, dict] = {
    "gpu_a10": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    GPU_SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "manual_gen_worker",
    },
    "gpu_a10_image": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    GPU_SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "manual_gen_worker",
    },
    "gpu_t4": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    GPU_SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "manual_gen_worker",
    },
    "gpu_l4": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    GPU_SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "manual_gen_worker",
    },
}

# ── Redis Queue → Worker Service mapping ──────────────────────────────────────
QUEUE_WORKER_MAP: dict[str, str] = {
    "sd15_tasks":            "manual_gen_worker",
    "image_tasks":           "manual_gen_worker",
    "model_tasks":           "manual_gen_worker",
    "rig_model":             "manual_gen_worker",
    "manual_gen_tasks":      "manual_gen_worker",
    "manual_gen_tasks_spot": "manual_gen_worker",
}

# ── All GPU Redis Queues ──────────────────────────────────────────────────────
# Orchestrator polls every queue listed here when deciding whether the GPU
# has work or is idle. Adding a queue here is required for the GPU to be
# auto-started when work arrives and auto-stopped when idle.
# NOTE: manual_gen_tasks_spot is the ACTIVE queue (see GPU_INSTANCE_MAP in
# .env.cpu) — omitting it made the orchestrator blind to real work.
GPU_QUEUES = (
    "sd15_tasks",
    "image_tasks",
    "model_tasks",
    "rig_model",
    "manual_gen_tasks",
    "manual_gen_tasks_spot",
)

# ── Orchestrator Tuning ───────────────────────────────────────────────────────
TASK_TTL_SECONDS      = 14400   # 4 hours — expire stale queued tasks
# 15 min idle → stop GPU instance. Matches the GPU-side auto_shutdown default and
# the Gradio panel default. Overridable at runtime via Redis autoshutdown:idle_minutes.
IDLE_SHUTDOWN_SECONDS = 900
POLL_INTERVAL_SECONDS = 30      # how often orchestrator checks queues

# ─────────────────────────────────────────────────────────────────────────────
# SPOT INSTANCE CONFIG — FUTURE (disabled until custom AMI is ready)
# ─────────────────────────────────────────────────────────────────────────────
# SPOT_INSTANCE_TYPES   = ["g6.2xlarge", "g5.2xlarge"]
# SPOT_PROJECT_TAG      = "spark-gpu-worker"
# SPOT_AMI_ID           = None
# SPOT_KEY_NAME         = None
# SPOT_SECURITY_GROUP_IDS = ""
# SPOT_SUBNET_ID        = None
# SPOT_IAM_INSTANCE_PROFILE = None
