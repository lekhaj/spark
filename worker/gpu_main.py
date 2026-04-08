#!/usr/bin/env python3
"""
GPU Main — Spark Pipeline Entry Point
======================================
Starts all GPU workers in separate threads and runs AutoShutdown monitoring.

Workers started:
  1. SD15Worker   — reads sd15_tasks  → generates character images
  2. TRELLISWorker— reads model_tasks → generates 3D GLB models

AutoShutdown:
  Monitors both queues; stops this EC2 instance after IDLE_SHUTDOWN_MINUTES
  of inactivity to avoid idle GPU charges.

Usage:
  python gpu_main.py [--workers sd15,trellis]  (default: all)

Screen session example:
  screen -dmS gpu_workers python /home/ec2-user/spark/worker/gpu_main.py
  screen -r gpu_workers
"""

import argparse
import fcntl
import logging
import os
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/gpu_workers.log", mode="a"),
    ],
)
logger = logging.getLogger("GPUMain")

# ── Single-instance lock ──────────────────────────────────────────────────────
LOCKFILE = "/tmp/gpu_main.lock"


def acquire_lock():
    fd = open(LOCKFILE, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        logger.error("Another gpu_main.py is already running. Exiting.")
        sys.exit(1)


# ── Worker thread wrapper ─────────────────────────────────────────────────────

def run_worker_thread(worker_cls, shutdown_monitor):
    """Run a worker in a thread; hook idle/active into AutoShutdown."""
    worker = worker_cls()

    def idle_cb(seconds_idle):
        shutdown_monitor.notify_idle(seconds_idle)

    # Patch notify_active / notify_done into process_task
    original_process = worker.process_task

    def patched_process(task, r, db):
        shutdown_monitor.notify_active()
        try:
            original_process(task, r, db)
        finally:
            shutdown_monitor.notify_done()

    worker.process_task = patched_process
    worker.run(idle_notify_callback=idle_cb)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spark GPU Worker Manager")
    parser.add_argument(
        "--workers",
        default="sd15,trellis",
        help="Comma-separated list of workers to start: sd15, trellis (default: all)",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="Disable auto-shutdown (useful for development)",
    )
    args = parser.parse_args()

    acquire_lock()

    requested = {w.strip().lower() for w in args.workers.split(",")}
    logger.info(f"Starting GPU workers: {requested}")

    # ── Import workers lazily (avoid loading torch before needed) ────────────
    worker_map = {}
    if "sd15" in requested:
        # Refactored SD15 worker that extends BaseWorker
        sys.path.insert(0, os.path.dirname(__file__))
        from workers.sd15_image_worker import SD15Worker
        worker_map["sd15"] = SD15Worker

    if "trellis" in requested:
        from workers.trellis_worker import TRELLISWorker
        worker_map["trellis"] = TRELLISWorker

    if not worker_map:
        logger.error(f"No valid workers in: {requested}")
        sys.exit(1)

    # ── AutoShutdown ──────────────────────────────────────────────────────────
    from workers.auto_shutdown import AutoShutdown
    all_queues = []
    if "sd15"    in requested: all_queues.append("sd15_tasks")
    if "trellis" in requested: all_queues.append("model_tasks")

    shutdown = AutoShutdown(queues=all_queues)
    if not args.no_shutdown:
        shutdown.start()
    else:
        logger.info("Auto-shutdown DISABLED (--no-shutdown flag)")

    # ── Start worker threads ──────────────────────────────────────────────────
    threads = []
    for name, cls in worker_map.items():
        t = threading.Thread(
            target=run_worker_thread,
            args=(cls, shutdown),
            name=f"Worker-{name}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        logger.info(f"Thread started: {t.name}")
        # Stagger starts by 2s so model loading doesn't OOM simultaneously
        time.sleep(2)

    logger.info("All workers running. Press Ctrl+C to stop.")

    try:
        while True:
            alive = [t for t in threads if t.is_alive()]
            if not alive:
                logger.warning("All worker threads have exited.")
                break
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down gracefully.")


if __name__ == "__main__":
    main()
