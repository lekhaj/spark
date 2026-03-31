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
    Manages two GPU instances:
      T4  (gpu_t4)  → image_tasks   → SDXL image generation  → service: image-worker
      A10 (gpu_a10) → model_tasks   → Hunyuan3D 3D generation → service: model-worker

    Pipeline flow:
      Submit image_tasks → T4 generates image → updates MongoDB → pushes to model_tasks
                                                                → A10 generates 3D mesh → updates MongoDB
    """
    def __init__(self):
        self.poll_interval = 30
        self.idle_shutdown = 300  # 5 minutes idle before stopping instance
        self.idle_start_t4  = None
        self.idle_start_a10 = None
        self.auto_mode = True

    def get_queue_lengths(self):
        return {
            "image_tasks": r.llen("image_tasks"),
            "model_tasks": r.llen("model_tasks")
        }

    def is_gpu_active(self, gpu_type: str) -> bool:
        try:
            state = get_instance_state(gpu_type)
            return state in ["running", "pending"]
        except:
            return False

    async def _manage_gpu(self, gpu_type: str, queue_name: str, idle_attr: str):
        """Generic GPU manager — start on tasks, stop after idle_shutdown seconds idle."""
        if not self.auto_mode:
            return

        queue_len  = r.llen(queue_name)
        gpu_active = self.is_gpu_active(gpu_type)
        label      = gpu_type.upper()

        print(f"[{label}] Queue={queue_name}:{queue_len}, Active={gpu_active}")

        if queue_len > 0:
            setattr(self, idle_attr, None)  # reset idle timer

            if not gpu_active:
                print(f"[{label}] Starting instance for {queue_len} pending task(s)...")
                if start_instance(gpu_type):
                    await asyncio.sleep(60)   # wait for boot + SSH readiness
                    ensure_gpu_worker_running(gpu_type)
                return

            if not is_gpu_worker_running(gpu_type):
                print(f"[{label}] Instance up but worker not running — starting...")
                ensure_gpu_worker_running(gpu_type)

        elif gpu_active:
            idle_start = getattr(self, idle_attr)
            if idle_start is None:
                setattr(self, idle_attr, time.time())
                print(f"[{label}] Idle — will stop in {self.idle_shutdown//60} min if no tasks")
            elif time.time() - idle_start > self.idle_shutdown:
                print(f"[{label}] Idle timeout — stopping instance")
                if stop_instance(gpu_type):
                    setattr(self, idle_attr, None)
        else:
            setattr(self, idle_attr, None)

    async def manage_gpu_t4(self):
        """T4 handles image_tasks (SDXL image generation)"""
        await self._manage_gpu("gpu_t4", "image_tasks", "idle_start_t4")

    async def manage_gpu_a10(self):
        """A10 handles model_tasks (Hunyuan3D 3D generation)"""
        await self._manage_gpu("gpu_a10", "model_tasks", "idle_start_a10")

    def get_status(self):
        queues        = self.get_queue_lengths()
        t4_active     = self.is_gpu_active("gpu_t4")
        a10_active    = self.is_gpu_active("gpu_a10")
        return {
            "auto_mode": self.auto_mode,
            "pipeline": "image_tasks → T4(SDXL) → model_tasks → A10(Hunyuan3D)",
            "gpu_t4": {
                "active":         t4_active,
                "queue":          queues["image_tasks"],
                "worker_running": is_gpu_worker_running("gpu_t4") if t4_active else False
            },
            "gpu_a10": {
                "active":         a10_active,
                "queue":          queues["model_tasks"],
                "worker_running": is_gpu_worker_running("gpu_a10") if a10_active else False
            }
        }

    async def run(self):
        print("[ORCHESTRATOR] Started — T4=image_tasks | A10=model_tasks | auto_mode=True")
        while True:
            try:
                await self.manage_gpu_t4()
                await self.manage_gpu_a10()

                s = self.get_status()
                print(f"[T4]  active={s['gpu_t4']['active']}  worker={s['gpu_t4']['worker_running']}  queue={s['gpu_t4']['queue']}")
                print(f"[A10] active={s['gpu_a10']['active']} worker={s['gpu_a10']['worker_running']} queue={s['gpu_a10']['queue']}")

                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                print(f"[ORCHESTRATOR ERROR] {e}")
                await asyncio.sleep(self.poll_interval)

# Global orchestrator instance
orchestrator = DualGPUOrchestrator()

async def orchestrator_main():
    await orchestrator.run()
