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
        self.idle_start_a10 = None
        # A10 handles ALL tasks (image + 3D). T4 is disabled.
        self.auto_mode = True

    def get_queue_lengths(self):
        """Get lengths of both queues"""
        return {
            "image_tasks": r.llen("image_tasks"),
            "model_tasks": r.llen("model_tasks")
        }

    def total_pending_tasks(self) -> int:
        """Total tasks across both queues - A10 handles both"""
        q = self.get_queue_lengths()
        return q["image_tasks"] + q["model_tasks"]

    def is_gpu_active(self, gpu_type: str) -> bool:
        """Check if GPU instance is actually running"""
        try:
            state = get_instance_state(gpu_type)
            return state in ["running", "pending"]
        except:
            return False

    async def manage_gpu_a10(self):
        """
        Manage A10 GPU — handles BOTH image_tasks and model_tasks.
        T4 is not used; A10 is the primary (and only) GPU instance.
        """
        if not self.auto_mode:
            return

        total_tasks = self.total_pending_tasks()
        queues = self.get_queue_lengths()
        gpu_active = self.is_gpu_active("gpu_a10")

        print(f"[A10] Image Queue: {queues['image_tasks']}, Model Queue: {queues['model_tasks']}, GPU Active: {gpu_active}")

        # CASE 1: Tasks pending — start A10 and ensure both workers running
        if total_tasks > 0:
            self.idle_start_a10 = None  # reset idle timer

            if not gpu_active:
                print("[A10] Tasks queued — starting A10 instance...")
                if start_instance("gpu_a10"):
                    await asyncio.sleep(60)  # wait for boot + SSH readiness
                    print("[A10] Instance ready — starting workers...")
                    ensure_gpu_worker_running("gpu_a10")
                return

            # GPU already running — ensure workers are up
            if not is_gpu_worker_running("gpu_a10"):
                print("[A10] GPU active but worker not running — starting worker...")
                ensure_gpu_worker_running("gpu_a10")

        # CASE 2: No tasks but GPU is running — idle shutdown countdown
        elif gpu_active and total_tasks == 0:
            if self.idle_start_a10 is None:
                self.idle_start_a10 = time.time()
                print("[A10] No tasks — idle timer started (5 min to shutdown)")
            elif time.time() - self.idle_start_a10 > self.idle_shutdown:
                print("[A10] Idle for 5 min — stopping instance to save cost")
                if stop_instance("gpu_a10"):
                    print("[A10] Instance stopped")
                    self.idle_start_a10 = None

        # CASE 3: No tasks, GPU already stopped
        else:
            self.idle_start_a10 = None

    def get_status(self):
        """Get current orchestrator status"""
        queues = self.get_queue_lengths()
        gpu_a10_active = self.is_gpu_active("gpu_a10")

        return {
            "auto_mode": self.auto_mode,
            "primary_gpu": "A10",
            "gpu_a10": {
                "active": gpu_a10_active,
                "image_queue": queues["image_tasks"],
                "model_queue": queues["model_tasks"],
                "total_queue": queues["image_tasks"] + queues["model_tasks"],
                "worker_running": is_gpu_worker_running("gpu_a10") if gpu_a10_active else False
            },
            "gpu_t4": {
                "active": False,
                "note": "Disabled — A10 handles all tasks"
            }
        }

    async def run(self):
        """Main orchestrator loop"""
        print("[ORCHESTRATOR] Starting — A10 is primary GPU (handles image + 3D tasks)")

        while True:
            try:
                await self.manage_gpu_a10()

                status = self.get_status()
                a10 = status["gpu_a10"]
                print(f"[STATUS] A10: Active={a10['active']}, Worker={a10['worker_running']}, "
                      f"Image Queue={a10['image_queue']}, Model Queue={a10['model_queue']}")

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                print(f"[ORCHESTRATOR ERROR] {e}")
                await asyncio.sleep(self.poll_interval)

# Global orchestrator instance
orchestrator = DualGPUOrchestrator()

async def orchestrator_main():
    await orchestrator.run()
