import asyncio
import time
import redis
from app.config import settings
from app.services.aws_service import (
    start_instance, stop_instance, is_gpu_worker_running,
    get_instance_state, ensure_gpu_worker_running
)

# Redis connection
r = redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

class DualGPUOrchestrator:
    def __init__(self):
        self.poll_interval = 30
        self.idle_shutdown = 300  # 5 minutes
        self.idle_start_t4 = None
        self.idle_start_a10 = None
        self.auto_mode = False

    def get_queue_lengths(self):
        """Get lengths of both queues"""
        return {
            "image_tasks": r.llen("image_tasks"),
            "model_tasks": r.llen("model_tasks")
        }

    def is_gpu_active(self, gpu_type: str) -> bool:
        """Check if GPU instance is actually running"""
        try:
            state = get_instance_state(gpu_type)
            return state in ["running", "pending"]
        except:
            return False

    async def manage_gpu_t4(self):
        """Manage GPU T4 (Image tasks)"""
        if not self.auto_mode:
            return

        queues = self.get_queue_lengths()
        image_queue = queues["image_tasks"]
        gpu_active = self.is_gpu_active("gpu_t4")

        print(f"[T4] Queue: {image_queue}, GPU Active: {gpu_active}")

        # CASE 1: There are tasks in queue
        if image_queue > 0:
            # Cancel any pending shutdown
            self.idle_start_t4 = None

            # If GPU not running, start it
            if not gpu_active:
                print("[T4] Starting GPU T4 instance...")
                if start_instance("gpu_t4"):
                    # Wait for instance to be ready
                    await asyncio.sleep(60)
                    print("[T4] Instance started, ensuring worker is running...")
                    ensure_gpu_worker_running("gpu_t4")
                return

            # If GPU is running, ensure worker is running
            # (Worker might have auto-stopped if it had no tasks previously)
            elif gpu_active:
                # Check if worker is actually running
                if not is_gpu_worker_running("gpu_t4"):
                    print("[T4] GPU running but worker not active. Starting worker...")
                    ensure_gpu_worker_running("gpu_t4")

        # CASE 2: No tasks in queue but GPU is active
        elif gpu_active and image_queue == 0:
            if self.idle_start_t4 is None:
                self.idle_start_t4 = time.time()
                print("[T4] Idle timer started")
            elif time.time() - self.idle_start_t4 > self.idle_shutdown:
                print("[T4] Stopping GPU instance (idle for 5 min)")
                if stop_instance("gpu_t4"):
                    print("[T4] GPU instance stopped")
                    self.idle_start_t4 = None

        # CASE 3: No tasks and GPU not active
        else:
            self.idle_start_t4 = None

    async def manage_gpu_a10(self):
        """Manage GPU A10 (Model tasks)"""
        if not self.auto_mode:
            return

        queues = self.get_queue_lengths()
        model_queue = queues["model_tasks"]
        gpu_active = self.is_gpu_active("gpu_a10")

        print(f"[A10] Queue: {model_queue}, GPU Active: {gpu_active}")

        # CASE 1: There are tasks in queue
        if model_queue > 0:
            # Cancel any pending shutdown
            self.idle_start_a10 = None

            # If GPU not running, start it
            if not gpu_active:
                print("[A10] Starting GPU A10 instance...")
                if start_instance("gpu_a10"):
                    # Wait for instance to be ready
                    await asyncio.sleep(60)
                    print("[A10] Instance started, ensuring worker is running...")
                    ensure_gpu_worker_running("gpu_a10")
                return

            # If GPU is running, ensure worker is running
            elif gpu_active:
                if not is_gpu_worker_running("gpu_a10"):
                    print("[A10] GPU running but worker not active. Starting worker...")
                    ensure_gpu_worker_running("gpu_a10")

        # CASE 2: No tasks in queue but GPU is active
        elif gpu_active and model_queue == 0:
            if self.idle_start_a10 is None:
                self.idle_start_a10 = time.time()
                print("[A10] Idle timer started")
            elif time.time() - self.idle_start_a10 > self.idle_shutdown:
                print("[A10] Stopping GPU instance (idle for 5 min)")
                if stop_instance("gpu_a10"):
                    print("[A10] GPU instance stopped")
                    self.idle_start_a10 = None

        # CASE 3: No tasks and GPU not active
        else:
            self.idle_start_a10 = None

    def get_status(self):
        """Get current orchestrator status"""
        queues = self.get_queue_lengths()
        gpu_t4_active = self.is_gpu_active("gpu_t4")
        gpu_a10_active = self.is_gpu_active("gpu_a10")

        return {
            "auto_mode": self.auto_mode,
            "gpu_t4": {
                "active": gpu_t4_active,
                "queue_length": queues["image_tasks"],
                "worker_running": is_gpu_worker_running("gpu_t4") if gpu_t4_active else False
            },
            "gpu_a10": {
                "active": gpu_a10_active,
                "queue_length": queues["model_tasks"],
                "worker_running": is_gpu_worker_running("gpu_a10") if gpu_a10_active else False
            }
        }

    async def run(self):
        """Main orchestrator loop"""
        print("[ORCHESTRATOR] Starting Dual GPU Orchestrator...")
        print("[ORCHESTRATOR] Workers auto-stop when no tasks. Only managing GPU instances.")

        while True:
            try:
                # Manage both GPUs
                await self.manage_gpu_t4()
                await self.manage_gpu_a10()

                # Log status
                status = self.get_status()
                print(f"[STATUS] T4: Active={status['gpu_t4']['active']}, Worker={status['gpu_t4']['worker_running']}")
                print(f"[STATUS] A10: Active={status['gpu_a10']['active']}, Worker={status['gpu_a10']['worker_running']}")
                print(f"[QUEUES] Image: {status['gpu_t4']['queue_length']}, Model: {status['gpu_a10']['queue_length']}")

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                print(f"[ERROR] {e}")
                await asyncio.sleep(self.poll_interval)

# Global orchestrator instance
orchestrator = DualGPUOrchestrator()

async def orchestrator_main():
    await orchestrator.run()
