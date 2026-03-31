import asyncio
import json
import time
import redis
from app.config import settings
from app.services.aws_service import (
    start_instance, stop_instance, is_gpu_worker_running,
    get_instance_state, ensure_gpu_worker_running
)

# Redis connection
r = redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# ── Configurable limits ──────────────────────────────────────────────────────
TASK_TTL_SECONDS = 3600        # 1 hour — expire stale tasks
IDLE_SHUTDOWN_SECONDS = 300    # 5 min idle → stop GPU instance
POLL_INTERVAL_SECONDS = 30    # how often the orchestrator checks queues
BOOT_WAIT_SECONDS = 60        # wait for instance to boot + SSH ready


class GPUOrchestrator:
    """
    GPU orchestrator — auto-starts/stops a single GPU instance (A10G / g5.2xlarge)
    for both image generation (Z-Image-Turbo) and 3D generation (Hunyuan3D-2).

    Features:
      - auto_mode: when True, starts GPU when tasks arrive, stops when idle
      - Task expiration: removes tasks older than TASK_TTL_SECONDS from queues
      - Idle shutdown: stops GPU instance after IDLE_SHUTDOWN_SECONDS with empty queues
      - Runs on the CPU server, manages GPU via SSH + AWS API

    Pipeline:
      image_tasks → GPU (image-worker / Z-Image-Turbo) → image → MongoDB → model_tasks
      model_tasks → GPU (model-worker / Hunyuan3D-2)   → 3D mesh → MongoDB + S3
    """

    def __init__(self):
        self.poll_interval = POLL_INTERVAL_SECONDS
        self.idle_shutdown = IDLE_SHUTDOWN_SECONDS
        self.task_ttl = TASK_TTL_SECONDS
        self.idle_since = None
        self.auto_mode = True

    # ── Queue helpers ─────────────────────────────────────────────────────

    def get_queue_lengths(self):
        return {
            "image_tasks": r.llen("image_tasks"),
            "model_tasks": r.llen("model_tasks"),
        }

    def total_pending(self) -> int:
        q = self.get_queue_lengths()
        return q["image_tasks"] + q["model_tasks"]

    def expire_stale_tasks(self):
        """Remove tasks older than task_ttl from all queues."""
        now = time.time()
        expired_count = 0
        for queue_name in ("image_tasks", "model_tasks"):
            length = r.llen(queue_name)
            if length == 0:
                continue

            keep = []
            for i in range(length):
                raw = r.lindex(queue_name, i)
                if raw is None:
                    continue
                try:
                    task = json.loads(raw)
                    ts = task.get("timestamp", 0)
                    # Parse ISO timestamp or unix timestamp
                    if isinstance(ts, str):
                        from datetime import datetime
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            ts = dt.timestamp()
                        except (ValueError, TypeError):
                            ts = 0
                    age = now - ts
                    if age < self.task_ttl:
                        keep.append(raw)
                    else:
                        expired_count += 1
                        print(f"[EXPIRE] Removing stale task from {queue_name} "
                              f"(age={age/60:.0f}min, job_id={task.get('job_id','?')})")
                except (json.JSONDecodeError, TypeError):
                    keep.append(raw)  # keep unparseable tasks

            if expired_count > 0:
                # Replace queue atomically
                pipe = r.pipeline()
                pipe.delete(queue_name)
                if keep:
                    pipe.rpush(queue_name, *keep)
                pipe.execute()

        if expired_count > 0:
            print(f"[EXPIRE] Removed {expired_count} stale task(s) total")
        return expired_count

    # ── Instance helpers ──────────────────────────────────────────────────

    def is_gpu_active(self, gpu_type: str) -> bool:
        try:
            state = get_instance_state(gpu_type)
            return state in ("running", "pending")
        except Exception:
            return False

    # ── Main orchestration loop ───────────────────────────────────────────

    async def manage_gpu(self):
        """
        Core logic:
          1. Expire stale tasks (>1 hour)
          2. If tasks pending → start GPU + ensure workers
          3. If no tasks + GPU idle >5 min → stop GPU
        """
        if not self.auto_mode:
            return

        # Step 1: expire stale tasks
        self.expire_stale_tasks()

        # Step 2: check queues
        queues = self.get_queue_lengths()
        total = queues["image_tasks"] + queues["model_tasks"]
        gpu_active = self.is_gpu_active("gpu_a10")

        print(f"[GPU] image_tasks={queues['image_tasks']} model_tasks={queues['model_tasks']} "
              f"active={gpu_active} auto_mode={self.auto_mode}")

        if total > 0:
            # Work to do — reset idle timer
            self.idle_since = None

            if not gpu_active:
                print(f"[GPU] {total} task(s) queued — starting GPU instance...")
                if start_instance("gpu_a10"):
                    await asyncio.sleep(BOOT_WAIT_SECONDS)
                    print("[GPU] Instance booted — starting workers...")
                    ensure_gpu_worker_running("gpu_a10")         # model-worker
                    ensure_gpu_worker_running("gpu_a10_image")   # image-worker
                return

            # GPU running — ensure both workers are up
            if not is_gpu_worker_running("gpu_a10"):
                print("[GPU] model-worker not running — starting...")
                ensure_gpu_worker_running("gpu_a10")
            if not is_gpu_worker_running("gpu_a10_image"):
                print("[GPU] image-worker not running — starting...")
                ensure_gpu_worker_running("gpu_a10_image")

        elif gpu_active:
            # No tasks in queue — but check if workers are still busy
            # (model_runner pops tasks from queue and runs model_worker_simple
            #  as a subprocess, so queue can be empty while work is in progress)
            img_worker_up = is_gpu_worker_running("gpu_a10_image")
            model_worker_up = is_gpu_worker_running("gpu_a10")
            workers_busy = img_worker_up or model_worker_up

            if workers_busy:
                # Workers still running — don't start idle timer
                self.idle_since = None
                print(f"[GPU] Queues empty but workers still active "
                      f"(img={img_worker_up} model={model_worker_up}) — keeping alive")
            elif self.idle_since is None:
                self.idle_since = time.time()
                print(f"[GPU] Queues empty, no workers — will shut down in "
                      f"{self.idle_shutdown // 60}min if no new tasks")
            else:
                idle_elapsed = time.time() - self.idle_since
                remaining = max(0, self.idle_shutdown - idle_elapsed)
                if idle_elapsed >= self.idle_shutdown:
                    print("[GPU] Idle timeout reached — stopping instance to save cost")
                    if stop_instance("gpu_a10"):
                        self.idle_since = None
                else:
                    print(f"[GPU] Idle for {idle_elapsed:.0f}s, "
                          f"shutdown in {remaining:.0f}s")
        else:
            # GPU off, no tasks — nothing to do
            self.idle_since = None

    # ── Status ────────────────────────────────────────────────────────────

    def get_status(self):
        queues = self.get_queue_lengths()
        gpu_active = self.is_gpu_active("gpu_a10")
        idle_elapsed = None
        if self.idle_since:
            idle_elapsed = round(time.time() - self.idle_since)
        return {
            "auto_mode": self.auto_mode,
            "task_ttl_minutes": self.task_ttl // 60,
            "idle_shutdown_minutes": self.idle_shutdown // 60,
            "pipeline": "image_tasks + model_tasks → GPU (Z-Image-Turbo + Hunyuan3D-2)",
            "gpu": {
                "active": gpu_active,
                "image_queue": queues["image_tasks"],
                "model_queue": queues["model_tasks"],
                "image_worker_running": is_gpu_worker_running("gpu_a10_image") if gpu_active else False,
                "model_worker_running": is_gpu_worker_running("gpu_a10") if gpu_active else False,
                "idle_seconds": idle_elapsed,
            }
        }

    # ── Run ───────────────────────────────────────────────────────────────

    async def run(self):
        print(f"[ORCHESTRATOR] Started — auto_mode={self.auto_mode} | "
              f"task_ttl={self.task_ttl // 60}min | "
              f"idle_shutdown={self.idle_shutdown // 60}min | "
              f"poll={self.poll_interval}s")
        while True:
            try:
                await self.manage_gpu()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                print(f"[ORCHESTRATOR ERROR] {e}")
                await asyncio.sleep(self.poll_interval)


# Global orchestrator instance
orchestrator = GPUOrchestrator()


async def orchestrator_main():
    await orchestrator.run()
