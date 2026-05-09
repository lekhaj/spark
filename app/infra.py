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
GPU_INSTANCE_ID  = "i-0d6b9d6d34ccc053d"   # g6.2xlarge — L4 GPU workers (image + 3D + rig)

# ── Public IPs ───────────────────────────────────────────────────────────────
CPU_PUBLIC_IP = "18.207.13.85"
GPU_PUBLIC_IP = "3.215.211.192"
GPU_PUBLIC_DNS = "ec2-3-215-211-192.compute-1.amazonaws.com"

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
        "worker_service": "model-worker",
    },
    "gpu_a10_image": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    GPU_SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "image-worker",
    },
    "gpu_t4": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    GPU_SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "image-worker",
    },
    "gpu_l4": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    GPU_SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "gpu-worker",
    },
}

# ── Redis Queue → Worker Service mapping ──────────────────────────────────────
QUEUE_WORKER_MAP: dict[str, str] = {
    "sd15_tasks":       "gpu-worker",
    "image_tasks":      "gpu-worker",
    "model_tasks":      "gpu-worker",
    "rig_model":        "gpu-worker",
    "manual_gen_tasks": "gpu-worker",
}

# ── All GPU Redis Queues ──────────────────────────────────────────────────────
# Orchestrator polls every queue listed here when deciding whether the GPU
# has work or is idle. Adding a queue here is required for the GPU to be
# auto-started when work arrives and auto-stopped when idle.
GPU_QUEUES = (
    "sd15_tasks",
    "image_tasks",
    "model_tasks",
    "rig_model",
    "manual_gen_tasks",
)

# ── Orchestrator Tuning ───────────────────────────────────────────────────────
TASK_TTL_SECONDS      = 14400   # 4 hours — expire stale queued tasks
IDLE_SHUTDOWN_SECONDS = 900     # 15 min idle → stop GPU instance
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
