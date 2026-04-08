"""
GPU Orchestrator — Fixed On-Demand Instance Edition
=====================================================
Watches Redis queues and manages the fixed g5.2xlarge GPU instance.

Flow:
  1. Poll all GPU queues (image_tasks, model_tasks, rig_model)
  2. Expire stale tasks (older than TASK_TTL)
  3. If tasks pending → ensure GPU instance is running → start workers
  4. If all queues empty + no workers active → idle timer → stop GPU instance

NOTE: Spot-instance orchestration is implemented in spot_gpu_service.py but
      is DISABLED until a custom AMI is ready. Switch by replacing the
      `aws_service` calls below with `spot_gpu` calls (see commented blocks).
"""

import asyncio
import json
import time
import logging
import redis as _redis

from app.config import settings
from app import infra
from app.services import aws_service

# ── Spot import — DISABLED (re-enable after custom AMI is ready) ──────────────
# from app.services.spot_gpu_service import spot_gpu

logger = logging.getLogger("orchestrator")

r = _redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# ── Tuning constants (sourced from infra.py) ──────────────────────────────────
TASK_TTL_SECONDS      = infra.TASK_TTL_SECONDS
IDLE_SHUTDOWN_SECONDS = infra.IDLE_SHUTDOWN_SECONDS
POLL_INTERVAL_SECONDS = infra.POLL_INTERVAL_SECONDS
GPU_QUEUES            = infra.GPU_QUEUES


class GPUOrchestrator:
    """
    Fixed on-demand GPU orchestrator.

    Manages the single GPU instance defined in infra.py:
      image_tasks  → image-worker  (SD 1.5 + ControlNet)
      model_tasks  → model-worker  (TRELLIS 3D)
      rig_model    → rig-worker    (UniRig)

    Lifecycle:
      - Any queue has tasks        → ensure GPU instance is running → start workers
      - All queues empty + idle    → idle timer → stop GPU instance (saves cost)
    """

    def __init__(self):
        self.poll_interval  = POLL_INTERVAL_SECONDS
        self.idle_shutdown  = IDLE_SHUTDOWN_SECONDS
        self.task_ttl       = TASK_TTL_SECONDS
        self.idle_since     = None
        self.auto_mode      = True
        # The logical GPU type used for all calls to aws_service
        self._gpu_alias     = "gpu_a10"

    # ── Queue helpers ─────────────────────────────────────────────────────────

    def get_queue_lengths(self) -> dict:
        return {q: r.llen(q) for q in GPU_QUEUES}

    def total_pending(self) -> int:
        return sum(self.get_queue_lengths().values())

    def expire_stale_tasks(self):
        """Remove tasks older than task_ttl from image/model queues."""
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
                        logger.info(
                            f"[EXPIRE] Removing stale task from {queue_name} "
                            f"(age={age / 60:.0f}min, job_id={task.get('job_id', '?')})"
                        )
                except (json.JSONDecodeError, TypeError):
                    keep.append(raw)

            if expired_count > 0:
                pipe = r.pipeline()
                pipe.delete(queue_name)
                if keep:
                    pipe.rpush(queue_name, *keep)
                pipe.execute()

        if expired_count > 0:
            logger.info(f"[EXPIRE] Removed {expired_count} stale task(s) total")
        return expired_count

    # ── Worker management ─────────────────────────────────────────────────────

    def _ensure_workers_for_queues(self, queues: dict):
        """Start worker services for queues that have pending tasks."""
        for queue_name, count in queues.items():
            if count <= 0:
                continue
            service = infra.QUEUE_WORKER_MAP.get(queue_name)
            if not service:
                continue
            ok, out = aws_service.ssh_to_gpu(
                self._gpu_alias,
                f"systemctl is-active {service}",
                timeout=15,
            )
            if not (ok and out.strip() == "active"):
                logger.info(f"[GPU] Starting {service} for {queue_name} ({count} task(s))...")
                aws_service.ssh_to_gpu(self._gpu_alias, f"sudo systemctl start {service}")

    def _any_worker_active(self) -> bool:
        """Return True if any worker service is currently active on the GPU."""
        for service in infra.QUEUE_WORKER_MAP.values():
            ok, out = aws_service.ssh_to_gpu(
                self._gpu_alias,
                f"systemctl is-active {service}",
                timeout=15,
            )
            if ok and out.strip() == "active":
                return True
        # Fallback: check VRAM
        return aws_service.is_gpu_worker_running(self._gpu_alias)

    # ── Main orchestration loop ───────────────────────────────────────────────

    async def manage_gpu(self):
        """
        Core orchestration logic (runs every poll_interval seconds):
          1. Expire stale tasks
          2. Check all GPU queues
          3. If any queue has work → ensure instance running → start workers
          4. If all empty + no workers → idle timer → stop instance
        """
        if not self.auto_mode:
            return

        self.expire_stale_tasks()

        queues      = self.get_queue_lengths()
        total       = sum(queues.values())
        gpu_state   = aws_service.get_instance_state(self._gpu_alias)
        gpu_running = gpu_state == "running"

        logger.info(
            f"[GPU] image={queues['image_tasks']} model={queues['model_tasks']} "
            f"rig={queues['rig_model']} instance={gpu_state} "
            f"ip={infra.GPU_PUBLIC_IP}"
        )

        if total > 0:
            # Work to do — reset idle timer
            self.idle_since = None

            if not gpu_running:
                logger.info(f"[GPU] {total} task(s) queued — starting GPU instance...")
                started = aws_service.start_instance(self._gpu_alias)
                if not started:
                    logger.error("[GPU] Failed to start GPU instance — will retry next cycle")
                    return
                logger.info("[GPU] GPU instance running — starting workers...")

            # Instance is running — start any missing workers
            self._ensure_workers_for_queues(queues)

            # ── SPOT INSTANCE PATH (disabled) ─────────────────────────────
            # launched = spot_gpu.ensure_gpu_available()
            # if launched:
            #     self._ensure_workers_for_queues(queues)

        elif gpu_running:
            # Queues empty — check if workers are still processing
            workers_busy = self._any_worker_active()

            if workers_busy:
                self.idle_since = None
                logger.info("[GPU] Queues empty but workers still active — keeping instance alive")
            elif self.idle_since is None:
                self.idle_since = time.time()
                logger.info(
                    f"[GPU] All queues empty, no workers active — "
                    f"stopping instance in {self.idle_shutdown // 60}min if no new tasks"
                )
            else:
                idle_elapsed = time.time() - self.idle_since
                remaining    = max(0, self.idle_shutdown - idle_elapsed)
                if idle_elapsed >= self.idle_shutdown:
                    logger.info("[GPU] Idle timeout reached — stopping GPU instance")
                    aws_service.stop_instance(self._gpu_alias)
                    self.idle_since = None

                    # ── SPOT INSTANCE PATH (disabled) ──────────────────────
                    # spot_gpu.terminate()
                else:
                    logger.info(
                        f"[GPU] Idle for {idle_elapsed:.0f}s, "
                        f"shutdown in {remaining:.0f}s"
                    )
        else:
            # No GPU running, no tasks — nothing to do
            self.idle_since = None

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        queues      = self.get_queue_lengths()
        gpu_state   = aws_service.get_instance_state(self._gpu_alias)

        idle_elapsed = None
        if self.idle_since:
            idle_elapsed = round(time.time() - self.idle_since)

        return {
            "auto_mode":            self.auto_mode,
            "task_ttl_minutes":     self.task_ttl // 60,
            "idle_shutdown_minutes": self.idle_shutdown // 60,
            "pipeline":             "image_tasks + model_tasks + rig_model → GPU (fixed on-demand)",
            "gpu_instance": {
                "instance_id":   infra.GPU_INSTANCE_ID,
                "instance_type": "g5.2xlarge",
                "public_ip":     infra.GPU_PUBLIC_IP,
                "state":         gpu_state,
            },
            "queues": {
                "image_tasks": queues["image_tasks"],
                "model_tasks": queues["model_tasks"],
                "rig_model":   queues["rig_model"],
            },
            "idle_seconds": idle_elapsed,
        }

    # ── Run loop ──────────────────────────────────────────────────────────────

    async def run(self):
        logger.info(
            f"[ORCHESTRATOR] Started — auto_mode={self.auto_mode} | "
            f"queues={GPU_QUEUES} | "
            f"task_ttl={self.task_ttl // 60}min | "
            f"idle_shutdown={self.idle_shutdown // 60}min | "
            f"poll={self.poll_interval}s | "
            f"gpu_instance={infra.GPU_INSTANCE_ID} ({infra.GPU_PUBLIC_IP})"
        )
        while True:
            try:
                await self.manage_gpu()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"[ORCHESTRATOR ERROR] {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)


# Global singleton
orchestrator = GPUOrchestrator()


async def orchestrator_main():
    await orchestrator.run()
