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
    """
    A10-only orchestrator — handles both image and 3D generation.

    Pipeline:
      image_tasks → A10 (image-worker / SDXL)      → image generated → MongoDB updated
                                                     → pushes to model_tasks
      model_tasks → A10 (model-worker / Hunyuan3D)  → 3D mesh generated → MongoDB updated

    Both workers run on the single A10 instance. T4 is not used.
    Workers start when tasks arrive and stop when idle (30s for image, 3min for 3D).
    The A10 instance itself shuts down after 5 min of both queues being empty.
    """
    def __init__(self):
        self.poll_interval = 30
        self.idle_shutdown = 300  # 5 min with no tasks → stop A10 instance
        self.idle_start_a10 = None
        self.auto_mode = True

    def get_queue_lengths(self):
        return {
            "image_tasks": r.llen("image_tasks"),
            "model_tasks": r.llen("model_tasks")
        }

    def total_pending(self) -> int:
        q = self.get_queue_lengths()
        return q["image_tasks"] + q["model_tasks"]

    def is_gpu_active(self, gpu_type: str) -> bool:
        try:
            state = get_instance_state(gpu_type)
            return state in ["running", "pending"]
        except:
            return False

    async def manage_gpu_a10(self):
        """
        A10 handles both image_tasks and model_tasks.
        Starts instance when any task is pending.
        Ensures both image-worker and model-worker are running.
        Stops instance after idle_shutdown with no tasks.
        """
        if not self.auto_mode:
            return

        queues     = self.get_queue_lengths()
        total      = queues["image_tasks"] + queues["model_tasks"]
        gpu_active = self.is_gpu_active("gpu_a10")

        print(f"[A10] image_tasks={queues['image_tasks']} model_tasks={queues['model_tasks']} active={gpu_active}")

        if total > 0:
            self.idle_start_a10 = None

            if not gpu_active:
                print(f"[A10] {total} task(s) queued — starting A10 instance...")
                if start_instance("gpu_a10"):
                    await asyncio.sleep(60)  # boot + SSH ready
                    print("[A10] Instance ready — starting workers...")
                    ensure_gpu_worker_running("gpu_a10")         # model-worker (Hunyuan3D)
                    ensure_gpu_worker_running("gpu_a10_image")   # image-worker (SDXL)
                return

            # Instance running — ensure both workers are up
            if not is_gpu_worker_running("gpu_a10"):
                print("[A10] model-worker not running — starting...")
                ensure_gpu_worker_running("gpu_a10")
            if not is_gpu_worker_running("gpu_a10_image"):
                print("[A10] image-worker not running — starting...")
                ensure_gpu_worker_running("gpu_a10_image")

        elif gpu_active:
            if self.idle_start_a10 is None:
                self.idle_start_a10 = time.time()
                print(f"[A10] Queues empty — stopping in {self.idle_shutdown//60} min if no new tasks")
            elif time.time() - self.idle_start_a10 > self.idle_shutdown:
                print("[A10] Idle timeout — stopping instance to save cost")
                if stop_instance("gpu_a10"):
                    self.idle_start_a10 = None
        else:
            self.idle_start_a10 = None

    def get_status(self):
        queues     = self.get_queue_lengths()
        a10_active = self.is_gpu_active("gpu_a10")
        return {
            "auto_mode":  self.auto_mode,
            "pipeline":   "image_tasks + model_tasks → A10 (SDXL + Hunyuan3D)",
            "gpu_t4":     {"active": False, "note": "Disabled — A10 handles all tasks"},
            "gpu_a10": {
                "active":               a10_active,
                "image_queue":          queues["image_tasks"],
                "model_queue":          queues["model_tasks"],
                "image_worker_running": is_gpu_worker_running("gpu_a10_image") if a10_active else False,
                "model_worker_running": is_gpu_worker_running("gpu_a10")       if a10_active else False,
            }
        }

    async def run(self):
        print("[ORCHESTRATOR] Started — A10 only | image-worker + model-worker | auto_mode=True")
        while True:
            try:
                await self.manage_gpu_a10()
                s   = self.get_status()
                a10 = s["gpu_a10"]
                print(f"[A10] active={a10['active']} "
                      f"img_worker={a10['image_worker_running']} img_q={a10['image_queue']} "
                      f"model_worker={a10['model_worker_running']} model_q={a10['model_queue']}")
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                print(f"[ORCHESTRATOR ERROR] {e}")
                await asyncio.sleep(self.poll_interval)

# Global orchestrator instance
orchestrator = DualGPUOrchestrator()

async def orchestrator_main():
    await orchestrator.run()
