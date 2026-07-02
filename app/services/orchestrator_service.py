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
from typing import Optional

import redis as _redis

from app.config import settings
from app import infra
from app.services import aws_service
from worker.lib import gpu_heartbeat
from worker.lib.gpu_launcher import ensure_gpu_ready, REDIS_ACTIVE_KEY

logger = logging.getLogger("orchestrator")

r = _redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# ── Tuning constants (sourced from infra.py) ──────────────────────────────────
TASK_TTL_SECONDS      = infra.TASK_TTL_SECONDS
# Primary idle-stop window (pipeline-aware). Falls back to the legacy value if a
# build predates IDLE_STOP_SECONDS.
IDLE_SHUTDOWN_SECONDS = getattr(infra, "IDLE_STOP_SECONDS", infra.IDLE_SHUTDOWN_SECONDS)
HEARTBEAT_FRESH_SECONDS = getattr(infra, "HEARTBEAT_FRESH_SECONDS", 120)
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
        self._gpu_alias     = infra.GPU_ALIAS   # "gpu" — ssh_to_gpu resolves the live box IP

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
                    # manual_gen payloads timestamp as 'queued_at'; older/other
                    # producers use 'timestamp' or 'created_at'. Default to None
                    # (undateable) — NEVER treat a missing timestamp as age 0, or
                    # every fresh task gets expired within one poll cycle while the
                    # GPU is still cold-starting (the goblin-task loss bug).
                    ts = task.get("queued_at",
                         task.get("timestamp",
                         task.get("created_at", None)))
                    if isinstance(ts, str):
                        from datetime import datetime
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            ts = dt.timestamp()
                        except (ValueError, TypeError):
                            ts = None
                    if ts is None:
                        keep.append(raw)          # can't date it → keep (fail-safe)
                        continue
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

    # ── Active GPU resolution ─────────────────────────────────────────────────

    def _active_iid(self) -> str:
        """The instance the EIP currently rides (set by gpu_launcher), or the
        on-demand default if nothing has been recorded yet."""
        try:
            val = r.get(REDIS_ACTIVE_KEY)
            if val:
                return val
        except Exception:
            pass
        return infra.SPOT_GPU_INSTANCE_ID   # spot is the steady-state primary

    # ── Pipeline-aware work detection (the single stop authority) ─────────────

    def _has_inflight_asset_run(self) -> Optional[bool]:
        """True if any cyclezero asset_run still owes GPU work — using the SAME
        ``_run_has_gpu_work`` predicate the state machine uses, so the stop
        decision can't drift from the pipeline. A run that hard-failed and
        lingers ``generating`` returns no work (won't hold the GPU forever); a
        healthy run between stages does (won't be stopped in the gap).

        Returns None if Mongo is unreachable so the caller fails safe (treat as
        work present, never stop blind)."""
        try:
            from worker.lib import manual_gen_schema as mgs
            from app.routes.asset_run_routes import _run_has_gpu_work
            db = mgs.get_db()
            for doc in db["asset_runs"].find({"status": "generating"}):
                if _run_has_gpu_work(doc):
                    return True
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[GPU] asset_run inflight check failed: {e}")
            return None

    def _pipeline_has_work(self, total: int) -> bool:
        """
        Authoritative 'is there GPU work outstanding?' used to gate idle-stop.

        True if ANY of:
          (a) queued GPU tasks (total > 0),
          (b) a fresh GPU heartbeat — a task is being processed right now, even
              with an empty queue (long pixal3d/hunyuan3d subprocess),
          (c) an in-flight asset_run with a non-terminal GPU stage (the CPU is
              about to enqueue the next stage — never stop in that gap).

        FAIL SAFE: a Redis/Mongo error on (b)/(c) reads as work-present so we
        never stop a box we can't fully assess.
        """
        if total > 0:
            return True
        active_iid = self._active_iid()
        if gpu_heartbeat.is_busy(r, active_iid, HEARTBEAT_FRESH_SECONDS):
            return True
        inflight = self._has_inflight_asset_run()
        if inflight or inflight is None:  # True → work; None → unknown → fail safe
            return True
        return False

    def _honor_stop_requests(self, total: int):
        """Honour GPU-delegated stop requests from ANY box.

        The GPU-side autoshutdown publishes ``autoshutdown:stop_requested:<iid>``
        when it wants to stop but lacks ec2:StopInstances (S3-only profile). It
        already verified its own queues are empty + no in-flight task. With two
        boxes (on-demand + a parallel spot) either may publish, so we scan all
        keys and stop each requester — unless new work arrived meanwhile.
        """
        try:
            keys = list(r.scan_iter(match="autoshutdown:stop_requested:*"))
        except Exception as e:
            logger.warning(f"[GPU] could not scan stop requests: {e}")
            return
        has_work = self._pipeline_has_work(total)
        for key in keys:
            iid = key.rsplit(":", 1)[-1]
            if not has_work:
                logger.warning(f"[GPU] Stop requested by GPU-side autoshutdown — stopping {iid}")
                r.delete(key)
                aws_service.stop_instance(iid)
            else:
                logger.info(f"[GPU] Stop request for {iid} cleared — pipeline still has work")
                r.delete(key)

    # ── Main orchestration loop ───────────────────────────────────────────────

    def manage_gpu(self):
        # NOTE: this is intentionally SYNChronous. Its body calls blocking
        # boto3 / Redis primitives; the orchestrator loop runs it via
        # asyncio.to_thread so a slow AWS call can never freeze the FastAPI
        # event loop (the root cause of "site unusable every few minutes").
        if not self.auto_mode:
            return

        self.expire_stale_tasks()

        queues      = self.get_queue_lengths()
        total       = sum(queues.values())
        active_iid  = self._active_iid()
        gpu_state   = aws_service.get_instance_state(active_iid)
        gpu_running = gpu_state == "running"

        logger.info(f"[GPU] {' '.join(f'{q}={n}' for q,n in queues.items())} active={active_iid} state={gpu_state} ip={infra.active_gpu_ip()}")

        # Honour GPU-delegated stop requests (any box) before anything else.
        self._honor_stop_requests(total)

        # The orchestrator is the PRIMARY, pipeline-aware stop authority. It
        # judges idleness from real work (queued tasks OR a fresh GPU heartbeat
        # OR an in-flight asset_run with a pending GPU stage) — never from bare
        # queue depth. This is what stops the box mid-pipeline bug.
        has_work = self._pipeline_has_work(total)

        if total > 0:
            self.idle_since = None          # tasks arrived — reset idle clock
            if not gpu_running:
                logger.info(f"[GPU] {total} task(s) queued — bringing a GPU online (spot-first)...")
                ok, reason = ensure_gpu_ready()
                logger.info(f"[GPU] ensure_gpu_ready → ok={ok} reason={reason}")
                if not ok:
                    logger.error("[GPU] Could not bring a GPU online — will retry next cycle")
                    return
                active_iid = self._active_iid()   # may now be spot or on-demand
            self._ensure_workers_for_queues(queues)
        elif has_work:
            # Queue is empty but the pipeline is still working — a long 3D
            # subprocess (fresh heartbeat) or an asset_run between stages. Hold
            # the box up and reset the idle clock; never stop here.
            self.idle_since = None
            logger.info("[GPU] Queue empty but pipeline still active (heartbeat/in-flight run) — holding GPU up")
        else:
            # Genuinely idle — no queued tasks, no heartbeat, no in-flight run.
            # Track idle time and stop the active box once the window elapses.
            if not gpu_running:
                self.idle_since = None      # already stopped, nothing to do
            else:
                if self.idle_since is None:
                    self.idle_since = time.time()
                    logger.info("[GPU] No pipeline work — idle timer started")
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
                        else:
                            logger.warning(
                                f"[GPU] Idle {self.idle_shutdown // 60} min with no pipeline "
                                f"work — stopping {active_iid}"
                            )
                            aws_service.stop_instance(active_iid)
                            self.idle_since = None

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        queues      = self.get_queue_lengths()
        active_iid  = self._active_iid()
        gpu_state   = aws_service.get_instance_state(active_iid)
        lifecycle   = ("spot" if active_iid == infra.SPOT_GPU_INSTANCE_ID
                       else "on-demand" if active_iid == infra.ONDEMAND_GPU_INSTANCE_ID
                       else "unknown")

        idle_elapsed = None
        if self.idle_since:
            idle_elapsed = round(time.time() - self.idle_since)

        return {
            "auto_mode":             self.auto_mode,
            "task_ttl_minutes":      self.task_ttl // 60,
            "idle_shutdown_minutes": self.idle_shutdown // 60,
            "autoshutdown":          self.get_autoshutdown_state(),
            "gpu_instance": {
                "instance_id": active_iid,
                "lifecycle":   lifecycle,
                "public_ip":   infra.active_gpu_ip(),
                "state":       gpu_state,
            },
            "queues":       queues,
            "idle_seconds": idle_elapsed,
        }

    async def run(self):
        logger.info(f"[ORCHESTRATOR] Started — poll={self.poll_interval}s spot={infra.SPOT_GPU_INSTANCE_ID} ondemand={infra.ONDEMAND_GPU_INSTANCE_ID} eip={infra.GPU_PUBLIC_IP}")
        while True:
            try:
                # Run the blocking GPU-management cycle in a worker thread so a
                # slow boto3/Redis call never blocks the event loop serving HTTP.
                await asyncio.to_thread(self.manage_gpu)
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"[ORCHESTRATOR ERROR] {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)


# Global singleton
orchestrator = GPUOrchestrator()


async def orchestrator_main():
    await orchestrator.run()
