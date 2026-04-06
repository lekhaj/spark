import asyncio
import json
import time
import redis
from app.config import settings
from app.services.aws_service import (
    start_instance, stop_instance, is_gpu_worker_running,
    get_instance_state, ensure_gpu_worker_running, ssh_to_gpu
)

# Redis connection
r = redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# ── Configurable limits ──────────────────────────────────────────────────────
TASK_TTL_SECONDS = 14400       # 4 hours — expire stale tasks
IDLE_SHUTDOWN_SECONDS = 300    # 5 min idle → stop GPU instance
POLL_INTERVAL_SECONDS = 30     # how often the orchestrator checks queues
BOOT_WAIT_SECONDS = 60         # wait for instance to boot + SSH ready

# All queues that require GPU — orchestrator keeps GPU alive while any has work
GPU_QUEUES = ("image_tasks", "model_tasks", "rig_model")


class GPUOrchestrator:
    """
    GPU orchestrator — auto-starts/stops the GPU instance (A10G / g5.2xlarge).

    Watches all GPU queues:
      image_tasks  → image-worker  (Z-Image-Turbo)
      model_tasks  → model-worker  (Hunyuan3D-2 / TRELLIS)
      rig_model    → rig-worker    (UniRig)

    Lifecycle:
      - Any queue has tasks   → start GPU, ensure relevant workers running
      - All queues empty AND no workers active → start 5-min idle timer → stop GPU
    """

    def __init__(self):
        self.poll_interval = POLL_INTERVAL_SECONDS
        self.idle_shutdown = IDLE_SHUTDOWN_SECONDS
        self.task_ttl = TASK_TTL_SECONDS
        self.idle_since = None
        self.auto_mode = True

    # ── Queue helpers ─────────────────────────────────────────────────────

    def get_queue_lengths(self):
        return {q: r.llen(q) for q in GPU_QUEUES}

    def total_pending(self) -> int:
        return sum(self.get_queue_lengths().values())

    def expire_stale_tasks(self):
        """Remove tasks older than task_ttl from image/model queues (not rig_model)."""
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
                    keep.append(raw)

            if expired_count > 0:
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

    def is_rig_worker_running(self) -> bool:
        """Check if rig-worker service is actively processing (not just idle)."""
        try:
            success, output = ssh_to_gpu("gpu_a10", "systemctl is-active rig-worker")
            if not (success and output == "active"):
                return False
            # rig-worker is active — check if it's actually processing (unirig subprocess running)
            success2, out2 = ssh_to_gpu(
                "gpu_a10",
                "pgrep -f 'generate_skeleton|generate_skin|merge.sh|unirig_batch_rig' | wc -l"
            )
            if success2:
                try:
                    if int(out2.strip()) > 0:
                        return True
                except (ValueError, IndexError):
                    pass
            # Also check rig_model queue — if items remain, worker is still busy
            return r.llen("rig_model") > 0 and success
        except Exception:
            return False

    def ensure_rig_worker(self):
        """Start rig-worker on GPU if not running."""
        try:
            success, output = ssh_to_gpu("gpu_a10", "systemctl is-active rig-worker")
            if success and output == "active":
                print("[GPU] rig-worker already active")
                return True
            print("[GPU] Starting rig-worker...")
            ok, _ = ssh_to_gpu("gpu_a10", "sudo systemctl start rig-worker")
            return ok
        except Exception as e:
            print(f"[GPU] Failed to start rig-worker: {e}")
            return False

    # ── Main orchestration loop ───────────────────────────────────────────

    async def manage_gpu(self):
        """
        Core logic:
          1. Expire stale image/model tasks
          2. Check all GPU queues (image_tasks, model_tasks, rig_model)
          3. If any queue has work → start GPU + ensure appropriate workers
          4. If all queues empty + no workers active → 5-min idle → stop GPU
        """
        if not self.auto_mode:
            return

        # Step 1: expire stale tasks
        self.expire_stale_tasks()

        # Step 2: check all queues
        queues = self.get_queue_lengths()
        total = sum(queues.values())
        gpu_active = self.is_gpu_active("gpu_a10")

        print(f"[GPU] image_tasks={queues['image_tasks']} model_tasks={queues['model_tasks']} "
              f"rig_model={queues['rig_model']} active={gpu_active} auto_mode={self.auto_mode}")

        if total > 0:
            # Work to do — reset idle timer
            self.idle_since = None

            if not gpu_active:
                print(f"[GPU] {total} task(s) queued — starting GPU instance...")
                if start_instance("gpu_a10"):
                    await asyncio.sleep(BOOT_WAIT_SECONDS)
                    print("[GPU] Instance booted — starting workers...")
                    ensure_gpu_worker_running("gpu_a10")        # model-worker
                    ensure_gpu_worker_running("gpu_a10_image")  # image-worker
                    if queues["rig_model"] > 0:
                        self.ensure_rig_worker()
                return

            # GPU running — ensure correct workers are up for pending queues
            if queues["image_tasks"] > 0 or queues["model_tasks"] > 0:
                if not is_gpu_worker_running("gpu_a10"):
                    print("[GPU] model-worker not running — starting...")
                    ensure_gpu_worker_running("gpu_a10")
                if not is_gpu_worker_running("gpu_a10_image"):
                    print("[GPU] image-worker not running — starting...")
                    ensure_gpu_worker_running("gpu_a10_image")

            if queues["rig_model"] > 0:
                self.ensure_rig_worker()

        elif gpu_active:
            # All queues empty — check if any worker is still processing
            img_busy   = is_gpu_worker_running("gpu_a10_image")
            model_busy = is_gpu_worker_running("gpu_a10")
            rig_busy   = self.is_rig_worker_running()
            workers_busy = img_busy or model_busy or rig_busy

            if workers_busy:
                self.idle_since = None
                print(f"[GPU] Queues empty but workers still active "
                      f"(img={img_busy} model={model_busy} rig={rig_busy}) — keeping alive")
            elif self.idle_since is None:
                self.idle_since = time.time()
                print(f"[GPU] All queues empty, no workers active — "
                      f"shutting down in {self.idle_shutdown // 60}min if no new tasks")
            else:
                idle_elapsed = time.time() - self.idle_since
                remaining = max(0, self.idle_shutdown - idle_elapsed)
                if idle_elapsed >= self.idle_shutdown:
                    print("[GPU] Idle timeout reached — stopping GPU instance to save cost")
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
            "pipeline": "image_tasks + model_tasks + rig_model → GPU",
            "gpu": {
                "active": gpu_active,
                "image_queue": queues["image_tasks"],
                "model_queue": queues["model_tasks"],
                "rig_queue": queues["rig_model"],
                "image_worker_running": is_gpu_worker_running("gpu_a10_image") if gpu_active else False,
                "model_worker_running": is_gpu_worker_running("gpu_a10") if gpu_active else False,
                "rig_worker_running": self.is_rig_worker_running() if gpu_active else False,
                "idle_seconds": idle_elapsed,
            }
        }

    # ── Run ───────────────────────────────────────────────────────────────

    async def run(self):
        print(f"[ORCHESTRATOR] Started — auto_mode={self.auto_mode} | "
              f"queues={GPU_QUEUES} | "
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
