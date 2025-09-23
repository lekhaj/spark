import asyncio, time, subprocess
import redis
from app.services.aws_service import start_instance, stop_instance
from app.config import settings

# ----- CONFIG -----
REDIS_URL      = settings.CELERY_BROKER_URL
GPU_SSH_USER   = settings.GPU_SSH_USER
GPU_PUBLIC_IP  = settings.GPU_PUBLIC_IP
POLL_INTERVAL  = 30          # seconds
IDLE_SHUTDOWN  = 300         # 5 minutes
# -------------------

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def ssh_cmd(*args: str):
    print(f"[SSH] -> {' '.join(args)}")
    subprocess.run(
        ["ssh","-i","C:/Users/Harsh Thakur/.ssh/s_spu_key.pem",
         "-o","StrictHostKeyChecking=accept-new",
         f"{GPU_SSH_USER}@{GPU_PUBLIC_IP}", *args],
        check=False
    )

def wait_for_gpu():
    import boto3
    ec2 = boto3.client(
        "ec2",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )
    while True:
        st = ec2.describe_instances(InstanceIds=[settings.AWS_GPU_INSTANCE_ID]) \
                 ["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"[AWS] GPU state: {st}")
        if st == "running":
            break
        time.sleep(5)

def service_running(name: str) -> bool:
    """Check if a given systemd service is active on the GPU instance."""
    result = subprocess.run(
        ["ssh", f"{GPU_SSH_USER}@{GPU_PUBLIC_IP}",
         "systemctl", "is-active", name],
        capture_output=True, text=True
    )
    return result.stdout.strip() == "active"

async def orchestrator_main():
    gpu_up = False
    idle_start = None
    current_worker = None   # "image-worker" or "model-worker"

    while True:
        img_q   = r.llen("image_tasks")
        model_q = r.llen("model_queue")
        print(f"[INFO] Queues -> image:{img_q}  model:{model_q}")

        # decide which worker to start if none is running
        if current_worker is None:
            if img_q > 0:
                current_worker = "image-worker"
            elif model_q > 0:
                current_worker = "model-worker"

            if current_worker:
                if not gpu_up:
                    print("[ACTION] Starting GPU …")
                    if start_instance("gpu"):
                        wait_for_gpu()
                        gpu_up = True
                print(f"[ACTION] Starting {current_worker}")
                ssh_cmd("sudo","systemctl","start",current_worker)
                idle_start = None

        else:
            # monitor running worker
            if not service_running(current_worker):
                print(f"[INFO] {current_worker} has exited")
                current_worker = None

        # shut down GPU only when no queues and no worker
        if current_worker is None and img_q == 0 and model_q == 0:
            if idle_start is None:
                idle_start = time.time()
                print("[INFO] Idle timer started")
            elif time.time() - (idle_start+1) > IDLE_SHUTDOWN and gpu_up:
                print("[ACTION] Idle → stopping GPU instance")
                stop_instance("gpu")
                gpu_up = False
                idle_start = None
        else:
            idle_start = None

        await asyncio.sleep(POLL_INTERVAL)
