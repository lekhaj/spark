
import asyncio, time, subprocess, os
import redis
from app.services.aws_service import start_instance, stop_instance
from app.config import settings

# ----- CONFIG -----
REDIS_URL      = settings.CELERY_BROKER_URL
GPU_SSH_USER   = settings.GPU_SSH_USER
GPU_PUBLIC_IP  = settings.GPU_PUBLIC_IP
POLL_INTERVAL  = 30                       # seconds
IDLE_SHUTDOWN  = 300                      # 5 minutes
# ------------------

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def ssh_cmd(*args: str) -> None:
    """Run a command on GPU instance via SSH."""
    print(f"[SSH] -> {' '.join(args)}")
    subprocess.run(
        [
            "ssh",
            "-i", "C:/Users/Hp/Downloads/s_spu_key.pem",   # full path to your key
            "-o", "StrictHostKeyChecking=accept-new",
            f"{GPU_SSH_USER}@{GPU_PUBLIC_IP}",
            *args
        ],
        check=False
    )

def active_service() -> str | None:
    """Return currently running worker service name or None."""
    print("[INFO] Checking active worker service on GPU …")
    result = subprocess.run(
        ["ssh", f"{GPU_SSH_USER}@{GPU_PUBLIC_IP}",
         "systemctl list-units --type=service --state=running"],
        capture_output=True, text=True
    )
    if "image-worker.service" in result.stdout:
        print("[INFO] image-worker is active")
        return "image-worker"
    if "model-worker.service" in result.stdout:
        print("[INFO] model-worker is active")
        return "model-worker"
    print("[INFO] No worker service running")
    return None

def wait_for_gpu():
    import boto3
    ec2 = boto3.client("ec2", region_name=settings.AWS_REGION)
    print("[AWS] Waiting for GPU instance to reach 'running' state …")
    while True:
        st = ec2.describe_instances(InstanceIds=[settings.AWS_GPU_INSTANCE_ID]) \
                 ["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"[AWS] GPU instance state: {st}")
        if st == "running":
            print("[AWS] GPU instance is running.")
            break
        time.sleep(5)

async def orchestrator_main():
    gpu_up     = False
    idle_start = None

    while True:
        img_q   = r.llen("image_tasks")
        model_q = r.llen("model_queue")#model_tasks
        print(f"[INFO] Queues -> image:{img_q}  model:{model_q}")

        current = active_service() if gpu_up else None

        if img_q > 0:
            if not gpu_up:
                print("[ACTION] Starting GPU for image worker …")
                if start_instance("gpu"):
                    wait_for_gpu()
                    gpu_up = True
            if current != "image-worker":
                if current:
                    print(f"[ACTION] Stopping {current} to start image-worker")
                    ssh_cmd("sudo","systemctl","stop",current)
                print("[ACTION] Starting image-worker")
                ssh_cmd("sudo","systemctl","start","image-worker")
            idle_start = None

        elif model_q > 0:
            if not gpu_up:
                print("[ACTION] Starting GPU for model worker …")
                if start_instance("gpu"):
                    wait_for_gpu()
                    gpu_up = True
            if current != "model-worker":
                if current:
                    print(f"[ACTION] Stopping {current} to start model-worker")
                    ssh_cmd("sudo","systemctl","stop",current)
                print("[ACTION] Starting model-worker")
                ssh_cmd("sudo","systemctl","start","model-worker")
            idle_start = None

        else:
            if current:
                print(f"[ACTION] No tasks → stopping {current}")
                ssh_cmd("sudo","systemctl","stop",current)
            if gpu_up and idle_start is None:
                idle_start = time.time()
                print("[INFO] Idle timer started")
            if gpu_up and idle_start and (time.time()-idle_start) > IDLE_SHUTDOWN:
                print("[ACTION] 5 min idle → stopping GPU instance")
                stop_instance("gpu")
                gpu_up = False
                idle_start = None

        await asyncio.sleep(POLL_INTERVAL)
