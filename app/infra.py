"""
Infrastructure Configuration
============================
Single source of truth for all EC2 instance IDs, public IPs, SSH config,
Redis, MongoDB endpoints, and worker service names.

Update this file when instances are replaced or IPs change.
All services (orchestrator, aws_service, workers) import from here.
"""

# ── EC2 Instance IDs ─────────────────────────────────────────────────────────
CPU_INSTANCE_ID  = "i-0f53b275935e3ea6b"   # FastAPI / orchestrator / Redis / MongoDB
GPU_INSTANCE_ID  = "i-0e446ff2933c012cb"   # g5.2xlarge — A10G GPU workers (image + 3D + rig)

# ── Public IPs ───────────────────────────────────────────────────────────────
CPU_PUBLIC_IP = "15.206.99.66"
GPU_PUBLIC_IP = "43.205.175.32"
GPU_PUBLIC_DNS = "ec2-43-205-175-32.ap-south-1.compute.amazonaws.com"

# ── SSH Configuration ────────────────────────────────────────────────────────
SSH_USER      = "ubuntu"
# Path to the private key on the CPU instance (orchestrator host)
SSH_KEY_PATH  = "/home/ubuntu/.ssh/s_spu_key.pem"

# ── AWS Region ───────────────────────────────────────────────────────────────
AWS_REGION = "ap-south-1"

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
}

# ── GPU Instance SSH Config (per alias) ──────────────────────────────────────
GPU_CONFIG: dict[str, dict] = {
    "gpu_a10": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "model-worker",
    },
    "gpu_a10_image": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "image-worker",
    },
    "gpu_t4": {
        "instance_id": GPU_INSTANCE_ID,
        "ssh_user":    SSH_USER,
        "public_ip":   GPU_PUBLIC_IP,
        "worker_service": "image-worker",
    },
}

# ── Redis Queue → Worker Service mapping ──────────────────────────────────────
QUEUE_WORKER_MAP: dict[str, str] = {
    "image_tasks": "image-worker",
    "model_tasks": "model-worker",
    "rig_model":   "rig-worker",
}

# ── All GPU Redis Queues ──────────────────────────────────────────────────────
GPU_QUEUES = ("image_tasks", "model_tasks", "rig_model")

# ── Orchestrator Tuning ───────────────────────────────────────────────────────
TASK_TTL_SECONDS      = 14400   # 4 hours — expire stale queued tasks
IDLE_SHUTDOWN_SECONDS = 300     # 5 min idle → stop GPU instance
POLL_INTERVAL_SECONDS = 30      # how often orchestrator checks queues

# ─────────────────────────────────────────────────────────────────────────────
# SPOT INSTANCE CONFIG — FUTURE (disabled until custom AMI is ready)
# ─────────────────────────────────────────────────────────────────────────────
# When ready, set these values and switch orchestrator_service.py to use
# spot_gpu_service.SpotGPUManager instead of fixed instance IDs.
#
# SPOT_INSTANCE_TYPES   = ["g6.2xlarge", "g5.2xlarge"]   # try in order
# SPOT_PROJECT_TAG      = "spark-gpu-worker"
# SPOT_AMI_ID           = None    # pre-baked AMI with CUDA + workers + weights
# SPOT_KEY_NAME         = None    # EC2 key pair name
# SPOT_SECURITY_GROUP_IDS = ""    # comma-separated SG IDs
# SPOT_SUBNET_ID        = None    # public subnet for SSH access
# SPOT_IAM_INSTANCE_PROFILE = None
