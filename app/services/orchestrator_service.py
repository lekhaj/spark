"""
GPU Orchestrator
================
Polls Redis queues every POLL_INTERVAL seconds.
When tasks arrive: ensures GPU instance is running and workers are active.
Shutdown is handled GPU-side (auto_shutdown.py) — orchestrator only starts.
"""

import asyncio
import json
import time
import logging
import redis as _redis

from app.config import settings
from app import infra
from app.services import aws_service

logger = logging.getLogger("orchestrator")

r = _redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# ── Tuning constants (sourced from infra.py) ──────────────────────────────────
TASK_TTL_SECONDS      = infra.TASK_TTL_SECONDS
IDLE_SHUTDOWN_SECONDS = infra.IDLE_SHUTDOWN_SECONDS
POLL_INTERVAL_SECONDS = infra.POLL_INTERVAL_SECONDS
GPU_QUEUES            = infra.GPU_QUEUES


# Redis keys shared with GPU-side auto_shutdown.py
_REDIS_AUTOSHUTDOWN_ENABLED = "autoshutdown:enabled"
_REDIS_AUTOSHUTDOWN_MINUTES = "autoshutdown:idle_minutes"


class GPUOrchestrator:
    def __init__(self):
        self.poll_interval  = POLL_INTERVAL_SECONDS
        self.idle_shutdown  = IDLE_SHUTDOWN_SECONDS
        self.task_ttl       = TASK_TTL_SECONDS
        self.idle_since     = None
        self.auto_mode      = True
        self._gpu_alias     = "gpu_a10"

    # ── AutoShutdown control ──────────────────────────────────────────────────

    def set_autoshutdown(self, enabled: bool, idle_minutes: int | None = None):
        """Toggle autoshutdown on/off and optionally update idle threshold."""
        r.set(_REDIS_AUTOSHUTDOWN_ENABLED, "1" if enabled else "0")
        if idle_minutes is not None and idle_minutes > 0:
            r.set(_REDIS_AUTOSHUTDOWN_MINUTES, str(idle_minutes))
            self.idle_shutdown = idle_minutes * 60
        logger.info(
            f"[AutoShutdown] {'enabled' if enabled else 'disabled'}"
            + (f", threshold={idle_minutes}min" if idle_minutes else "")
        )

    def get_autoshutdown_state(self) -> dict:
        enabled_raw = r.get(_REDIS_AUTOSHUTDOWN_ENABLED)
        minutes_raw = r.get(_REDIS_AUTOSHUTDOWN_MINUTES)
        enabled = True if enabled_raw is None else (enabled_raw == "1")
        minutes = self.idle_shutdown // 60
        if minutes_raw is not None:
            try:
                minutes = int(minutes_raw)
            except ValueError:
                pass
        return {"enabled": enabled, "idle_minutes": minutes}

    def _autoshutdown_enabled(self) -> bool:
        val = r.get(_REDIS_AUTOSHUTDOWN_ENABLED)
        return True if val is None else (val == "1")

    # ── Queue helpers ─────────────────────────────────────────────────────────

    def get_queue_lengths(self) -> dict:
        return {q: r.llen(q) for q in GPU_QUEUES}

    def total_pending(self) -> int:
        return sum(self.get_queue_lengths().values())

    def expire_stale_tasks(self):
        """Remove tasks older than task_ttl from image/model queues."""
        now = time.time()
        expired_count = 0
        for queue_name in GPU_QUEUES:
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
        if not self.auto_mode:
            return

        self.expire_stale_tasks()

        queues      = self.get_queue_lengths()
        total       = sum(queues.values())
        gpu_state   = aws_service.get_instance_state(self._gpu_alias)
        gpu_running = gpu_state == "running"

        logger.info(f"[GPU] {' '.join(f'{q}={n}' for q,n in queues.items())} instance={gpu_state} ip={infra.GPU_PUBLIC_IP}")

        if total > 0:
            self.idle_since = None          # tasks arrived — reset idle clock
            if not gpu_running:
                logger.info(f"[GPU] {total} task(s) queued — starting instance...")
                if not aws_service.start_instance(self._gpu_alias):
                    logger.error("[GPU] Failed to start instance — will retry next cycle")
                    return
            self._ensure_workers_for_queues(queues)
        else:
            # No work in any queue — track idle time and stop when threshold hit
            if not gpu_running:
                self.idle_since = None      # already stopped, nothing to do
            else:
                if self.idle_since is None:
                    self.idle_since = time.time()
                    logger.info("[GPU] All queues empty — idle timer started")
                else:
                    idle_elapsed = time.time() - self.idle_since
                    idle_min     = idle_elapsed / 60
                    logger.info(
                        f"[GPU] Idle {idle_min:.1f}/{self.idle_shutdown // 60} min "
                        f"(threshold={self.idle_shutdown // 60} min)"
                    )
                    if idle_elapsed >= self.idle_shutdown:
                        if not self._autoshutdown_enabled():
                            logger.info("[GPU] Idle threshold reached but autoshutdown is disabled — skipping stop")
                            self.idle_since = None
                        elif self._any_worker_active():
                            # Primary shutdown is GPU-side (auto_shutdown.py),
                            # which tracks in-flight tasks. The CPU stop is only
                            # a backstop for a crashed worker — never kill a box
                            # whose worker service is alive (it may be mid-job
                            # on a long stage with an empty queue).
                            logger.info(
                                "[GPU] Idle threshold reached but worker service is "
                                "active — deferring to GPU-side autoshutdown"
                            )
                            self.idle_since = None
                        else:
                            logger.warning(
                                f"[GPU] Idle threshold reached ({self.idle_shutdown // 60} min), "
                                f"worker not active — backstop-stopping {infra.GPU_INSTANCE_ID}"
                            )
                            aws_service.stop_instance(self._gpu_alias)
                            self.idle_since = None

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        queues      = self.get_queue_lengths()
        gpu_state   = aws_service.get_instance_state(self._gpu_alias)

        idle_elapsed = None
        if self.idle_since:
            idle_elapsed = round(time.time() - self.idle_since)

        return {
            "auto_mode":             self.auto_mode,
            "task_ttl_minutes":      self.task_ttl // 60,
            "idle_shutdown_minutes": self.idle_shutdown // 60,
            "autoshutdown":          self.get_autoshutdown_state(),
            "gpu_instance": {
                "instance_id": infra.GPU_INSTANCE_ID,
                "public_ip":   infra.GPU_PUBLIC_IP,
                "state":       gpu_state,
            },
            "queues":       queues,
            "idle_seconds": idle_elapsed,
        }

    async def run(self):
        logger.info(f"[ORCHESTRATOR] Started — poll={self.poll_interval}s gpu={infra.GPU_INSTANCE_ID} ({infra.GPU_PUBLIC_IP})")
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
