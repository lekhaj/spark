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

import boto3
import redis

logger = logging.getLogger("AutoShutdown")

# ── Config ────────────────────────────────────────────────────────────────────
# REDIS_HOST defaults to the CPU's private VPC IP — the public IP is firewalled
# off from GPU egress and AWS-throttled during abuse-case mitigations.
IDLE_THRESHOLD_MIN = int(os.getenv("IDLE_SHUTDOWN_MINUTES", "15"))
CHECK_INTERVAL_SEC = int(os.getenv("IDLE_CHECK_INTERVAL_SEC", "60"))
INSTANCE_ID        = os.getenv("AWS_GPU_INSTANCE_ID", "i-0d6b9d6d34ccc053d")
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
REDIS_HOST         = os.getenv("REDIS_HOST", "172.31.26.6")   # CPU private IP
REDIS_PORT         = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD     = os.getenv("REDIS_PASSWORD") or None


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
        logger.warning(
            f"All queues empty for >{IDLE_THRESHOLD_MIN} min — "
            f"stopping EC2 instance {INSTANCE_ID}"
        )
        try:
            ec2 = boto3.client(
                "ec2",
                region_name=AWS_REGION,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            ec2.stop_instances(InstanceIds=[INSTANCE_ID])
            logger.warning(f"Stop request sent for {INSTANCE_ID}")
        except Exception as e:
            logger.error(f"Failed to stop instance: {e}")

    def _monitor_loop(self):
        idle_since: float | None = None

        while True:
            time.sleep(CHECK_INTERVAL_SEC)

            r = self._get_redis()
            if r is None:
                idle_since = None   # can't determine — reset timer
                continue

            if self._active:
                idle_since = None
                continue

            if self._all_queues_empty(r):
                if idle_since is None:
                    idle_since = time.time()
                    logger.info("All queues empty — idle timer started")
                else:
                    idle_seconds = time.time() - idle_since
                    idle_minutes = idle_seconds / 60
                    logger.info(
                        f"Queues still empty — idle {idle_minutes:.1f}/{IDLE_THRESHOLD_MIN} min"
                    )
                    if idle_minutes >= IDLE_THRESHOLD_MIN:
                        self._stop_instance()
                        break   # instance is stopping; exit thread
            else:
                if idle_since is not None:
                    logger.info("Queue activity detected — idle timer reset")
                idle_since = None
