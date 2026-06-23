#!/usr/bin/env python3
"""
Auto-Shutdown Module
====================
Monitors all GPU queues. If every queue has been empty for IDLE_THRESHOLD
minutes and no worker reports active processing, stops this EC2 instance
via boto3 so the user is not charged for idle GPU time.

Usage (from gpu_main.py):
    from workers.auto_shutdown import AutoShutdown
    shutdown = AutoShutdown(queues=[
        "sd15_tasks", "model_tasks", "rig_model", "manual_gen_tasks"
    ])
    shutdown.start()          # background thread
    # in worker idle callback:
    shutdown.notify_idle(seconds_idle)
    # in worker task-start:
    shutdown.notify_active()
"""

import logging
import os
import threading
import time
import urllib.request

import boto3
import redis

logger = logging.getLogger("AutoShutdown")

# ── Config ────────────────────────────────────────────────────────────────────
# REDIS_HOST defaults to the CPU's private VPC IP — the public IP is firewalled
# off from GPU egress and AWS-throttled during abuse-case mitigations.
# Safety-net threshold. The CPU orchestrator is now the PRIMARY stop authority
# (pipeline-aware: queues + heartbeat + in-flight asset_runs). This GPU-side
# clock only catches a *dead* CPU orchestrator, so it is deliberately long —
# never the thing that ends a normal run. Overridable via env / Redis.
IDLE_THRESHOLD_MIN = int(os.getenv("IDLE_SHUTDOWN_MINUTES", "45"))
CHECK_INTERVAL_SEC = int(os.getenv("IDLE_CHECK_INTERVAL_SEC", "60"))
# A GPU heartbeat fresher than this means a task is being processed right now —
# even if the Redis queue is momentarily empty (long subprocess stage). Treated
# as "active" so this safety net never fires mid-task.
HEARTBEAT_FRESH_SEC = int(os.getenv("GPU_HEARTBEAT_FRESH_SEC", "120"))
# NOTE: do NOT default AWS_GPU_INSTANCE_ID — it's host-specific (different
# per GPU box). .env.gpu is shared across spark_l4 and spark_gpu_spot, so the
# value must come from IMDS at runtime. A hardcoded default here previously
# caused the spot to send a stop-instance for spark_l4 instead of itself.
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
REDIS_HOST         = os.getenv("REDIS_HOST", "172.31.26.6")   # CPU private IP
REDIS_PORT         = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD     = os.getenv("REDIS_PASSWORD") or None
# Path to a file that, when missing, suppresses idle counting. spark-prewarm.service
# touches it once the page cache is warm. Lets a fresh spot finish loading
# weights without auto-stopping itself mid-warmup.
PREWARM_SENTINEL   = os.getenv("PREWARM_SENTINEL", "/home/ec2-user/spark-prewarm.done")
# Redis key used by the CPU orchestrator to block on prewarm completion.
# Filled in lazily once IMDS resolves the instance-id.
PREWARM_READY_KEY  = "prewarm:ready:{instance_id}"


def _imds_instance_id() -> str | None:
    """Fetch this instance's id via IMDSv2. Returns None if not on EC2."""
    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode()
        iid_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(iid_req, timeout=2).read().decode().strip()
    except Exception:
        return None


class AutoShutdown:
    """
    Thread that watches Redis queues and stops the EC2 instance when idle.

    Logic:
      - Every CHECK_INTERVAL_SEC seconds, check all watched queues.
      - If ALL queues are empty AND no worker has called notify_active()
        in the last IDLE_THRESHOLD_MIN minutes → stop instance.
      - notify_active() resets the idle timer (called when task starts).
      - notify_idle() is called by the worker loop with seconds idle;
        used only for logging, the real check is queue depth.
    """

    def __init__(self, queues: list[str]):
        self.queues          = queues
        self._active         = False        # True while a task is running
        self._last_active_ts = time.time()  # last time any task was processed
        self._thread         = threading.Thread(
            target=self._monitor_loop, daemon=True, name="AutoShutdown"
        )

    def start(self):
        logger.info(
            f"AutoShutdown started — watching queues: {self.queues}, "
            f"threshold: {IDLE_THRESHOLD_MIN} min"
        )
        # Clear any stale prewarm:ready:<iid> key in Redis. The sentinel file
        # /var/run/spark-prewarm.done lives on tmpfs and is wiped on every
        # reboot, but the Redis key has a 24h TTL — so without this DEL, a
        # rebooted GPU would falsely advertise "ready" to the CPU orchestrator
        # before spark-prewarm.service had a chance to re-warm the page cache.
        # We only republish the key once the actual sentinel file exists
        # (handled in _monitor_loop → _publish_prewarm_ready).
        try:
            r = self._get_redis()
            if r is not None:
                iid = os.getenv("AWS_GPU_INSTANCE_ID", "").strip() or _imds_instance_id()
                if iid:
                    key = PREWARM_READY_KEY.format(instance_id=iid)
                    deleted = r.delete(key)
                    if deleted:
                        logger.info(f"Cleared stale {key} on startup (will republish after sentinel)")
        except Exception as e:
            logger.warning(f"Could not clear stale prewarm:ready on startup: {e}")

        self._thread.start()

    def notify_active(self):
        """Call when a worker starts processing a task."""
        self._active         = True
        self._last_active_ts = time.time()

    def notify_done(self):
        """Call when a worker finishes processing a task."""
        self._active         = False
        self._last_active_ts = time.time()

    def notify_idle(self, seconds_idle: float):
        """Optional: called by worker when a poll returns nothing."""
        if seconds_idle > 60:
            logger.debug(f"Worker idle for {seconds_idle:.0f}s")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_redis(self):
        try:
            r = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0,
                password=REDIS_PASSWORD,
                socket_connect_timeout=5, socket_timeout=5,
            )
            r.ping()
            return r
        except Exception as e:
            logger.warning(f"Redis unreachable for queue check: {e}")
            return None

    def _all_queues_empty(self, r) -> bool:
        try:
            for q in self.queues:
                if r.llen(q) > 0:
                    return False
            return True
        except Exception as e:
            logger.warning(f"Queue check failed: {e}")
            return False   # fail-safe: don't shutdown if we can't check

    def _stop_instance(self):
        # Always resolve THIS instance's id at call time (env var if set,
        # else IMDS). Never use a module-level constant — it'd be wrong on
        # any host that doesn't match the constant's default.
        iid = self._this_iid()
        if not iid:
            logger.error("Cannot stop — instance-id unknown (env unset, IMDS unreachable)")
            return
        logger.warning(
            f"All queues empty for >{IDLE_THRESHOLD_MIN} min — "
            f"stopping EC2 instance {iid}"
        )
        try:
            ec2 = boto3.client(
                "ec2",
                region_name=AWS_REGION,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            ec2.stop_instances(InstanceIds=[iid])
            logger.warning(f"Stop request sent for {iid}")
        except Exception as e:
            logger.error(f"Failed to stop instance: {e}")
            self._delegate_stop_to_cpu(iid)

    def _delegate_stop_to_cpu(self, iid: str):
        """
        Fallback when this box can't call ec2:StopInstances itself (e.g. the
        instance profile is S3-only). Publish a stop request in Redis; the
        CPU orchestrator honours it on its next poll — it has the EC2 perms.
        TTL keeps a stale request from stopping the box hours later.
        """
        r = self._get_redis()
        if r is None:
            logger.error("Cannot delegate stop — Redis unreachable")
            return
        try:
            r.set(f"autoshutdown:stop_requested:{iid}", str(int(time.time())), ex=900)
            logger.warning(f"Published autoshutdown:stop_requested:{iid} — CPU orchestrator will stop us")
        except Exception as e:
            logger.error(f"Failed to publish stop request: {e}")

    def _this_iid(self) -> str | None:
        """Cached lookup of this instance's id (for per-instance Redis keys)."""
        if not hasattr(self, "_cached_iid"):
            self._cached_iid = (
                os.getenv("AWS_GPU_INSTANCE_ID", "").strip()
                or _imds_instance_id()
            )
        return self._cached_iid

    def _is_enabled(self, r) -> bool:
        """
        Check Redis flag — per-instance key takes precedence, then global,
        then default (True).
        Keys read:
          autoshutdown:enabled:<iid>   (preferred)
          autoshutdown:enabled         (fallback)
        """
        iid = self._this_iid()
        keys = ([f"autoshutdown:enabled:{iid}"] if iid else []) + ["autoshutdown:enabled"]
        for k in keys:
            try:
                val = r.get(k)
            except Exception:
                continue
            if val is None:
                continue
            s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
            return s == "1"
        return True   # no flag set anywhere → default on

    def _get_idle_threshold(self, r) -> int:
        """
        Read idle threshold from Redis (per-instance preferred), falling back
        to global key, falling back to env var default.
        """
        iid = self._this_iid()
        keys = ([f"autoshutdown:idle_minutes:{iid}"] if iid else []) + ["autoshutdown:idle_minutes"]
        for k in keys:
            try:
                val = r.get(k)
            except Exception:
                continue
            if val is None:
                continue
            v = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
        return IDLE_THRESHOLD_MIN

    def _publish_prewarm_ready(self, r) -> None:
        """Set Redis flag so the CPU orchestrator can unblock pending tasks."""
        iid = os.getenv("AWS_GPU_INSTANCE_ID", "").strip() or _imds_instance_id()
        if not iid:
            logger.warning("Cannot publish prewarm:ready — instance-id unknown")
            return
        try:
            key = PREWARM_READY_KEY.format(instance_id=iid)
            # 24h TTL — sentinel survives reboot of this thread but expires
            # if the spot is stopped/started (page cache is cold again).
            r.set(key, "1", ex=86400)
            logger.info(f"Published {key}=1 — CPU orchestrator can release queued tasks")
        except Exception as e:
            logger.warning(f"Failed to publish prewarm:ready: {e}")

    def _monitor_loop(self):
        idle_since: float | None = None
        # Log "waiting for prewarm" at most once per N ticks so we don't spam.
        _last_prewarm_log: float = 0.0
        # Latch: only publish prewarm:ready to Redis once per thread lifetime
        # (the sentinel file persists, so once True we don't keep republishing).
        _prewarm_published: bool = False

        while True:
            time.sleep(CHECK_INTERVAL_SEC)

            # Don't count idle time while spark-prewarm.service is still
            # priming the page cache — the instance would otherwise stop
            # itself mid-warmup. The service touches PREWARM_SENTINEL
            # when it's safe to start counting idle.
            if PREWARM_SENTINEL and not os.path.exists(PREWARM_SENTINEL):
                idle_since = None
                now = time.time()
                if now - _last_prewarm_log > 300:   # log every 5 min
                    logger.info(
                        f"Pre-warm sentinel {PREWARM_SENTINEL} not present — "
                        f"deferring idle counting"
                    )
                    _last_prewarm_log = now
                continue

            r = self._get_redis()
            if r is None:
                idle_since = None   # can't determine — reset timer
                continue

            # Sentinel exists AND Redis is reachable → publish prewarm:ready
            # exactly once so the CPU orchestrator's _wait_for_prewarm() returns.
            if not _prewarm_published:
                self._publish_prewarm_ready(r)
                _prewarm_published = True

            if not self._is_enabled(r):
                if idle_since is not None:
                    logger.info("AutoShutdown disabled via Redis flag — idle timer cleared")
                idle_since = None
                continue

            if self._active:
                idle_since = None
                continue

            # Heartbeat guard: a fresh heartbeat means a task is in flight right
            # now (e.g. a long pixal3d/hunyuan3d subprocess with an empty queue).
            # Treat as active so the safety net never kills a working box. Fail
            # safe: gpu_heartbeat.is_busy returns True on any Redis error.
            try:
                from lib import gpu_heartbeat
                if gpu_heartbeat.is_busy(r, self._this_iid(), HEARTBEAT_FRESH_SEC):
                    if idle_since is not None:
                        logger.info("GPU heartbeat fresh — idle timer cleared")
                    idle_since = None
                    continue
            except Exception:
                pass

            threshold = self._get_idle_threshold(r)

            if self._all_queues_empty(r):
                if idle_since is None:
                    idle_since = time.time()
                    logger.info("All queues empty — idle timer started")
                else:
                    idle_seconds = time.time() - idle_since
                    idle_minutes = idle_seconds / 60
                    logger.info(
                        f"Queues still empty — idle {idle_minutes:.1f}/{threshold} min"
                    )
                    if idle_minutes >= threshold:
                        self._stop_instance()
                        break   # instance is stopping; exit thread
            else:
                if idle_since is not None:
                    logger.info("Queue activity detected — idle timer reset")
                idle_since = None
