#!/usr/bin/env python3
"""
GPU Main — Spark Pipeline Entry Point
======================================
Starts all GPU workers in separate threads and runs AutoShutdown monitoring.

Workers:
  sd15    — reads sd15_tasks  → SD1.5+ControlNet character image generation
  trellis — reads model_tasks → TRELLIS.2-4B image-to-3D GLB generation
  rig     — reads rig_model   → Auto-Rig Pro headless rigging via Blender

AutoShutdown:
  Monitors all queues; stops this EC2 instance after IDLE_SHUTDOWN_MINUTES
  of inactivity to avoid idle GPU charges.

Usage:
  python gpu_main.py                          # all workers (default)
  python gpu_main.py --workers sd15,trellis   # skip rig
  python gpu_main.py --workers rig            # rig only
  python gpu_main.py --no-shutdown            # keep alive (dev mode)

Screen session:
  screen -dmS workers bash -c 'cd ~/spark && \
    PYTHONPATH=~/trellis python worker/gpu_main.py 2>&1 | tee /tmp/gpu_workers.log'
  screen -r workers
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

# ── Ensure worker/lib is importable (result_channel, manual_gen_schema …) ─────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir  = os.path.join(_this_dir, "lib")
for _p in (_this_dir, _lib_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
        default="manual",
        help="Comma-separated list of workers: manual, sd15, trellis, rig (default: manual)",
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
        sys.path.insert(0, os.path.dirname(__file__))
        from workers.sd15_image_worker import SD15Worker
        worker_map["sd15"] = SD15Worker

    if "trellis" in requested:
        # Ensure TRELLIS repo is on sys.path (installed by install_trellis.sh)
        trellis_repo = os.getenv("TRELLIS_REPO_PATH", os.path.expanduser("~/trellis"))
        if os.path.isdir(trellis_repo):
            if trellis_repo not in sys.path:
                sys.path.insert(0, trellis_repo)
                logger.info(f"Added TRELLIS repo to path: {trellis_repo}")
            else:
                logger.info(f"TRELLIS repo already in path: {trellis_repo}")
        else:
            logger.warning(
                f"TRELLIS repo not found at {trellis_repo}. "
                "Run: bash worker/gpu_setup/install_trellis.sh"
            )
        from workers.trellis_worker import TRELLISWorker
        worker_map["trellis"] = TRELLISWorker

    if "rig" in requested:
        from workers.rig_worker import RigWorker
        worker_map["rig"] = RigWorker

    if "manual" in requested:
        from workers.manual_gen_worker import ManualGenWorker
        worker_map["manual"] = ManualGenWorker

    if not worker_map:
        logger.error(f"No valid workers in: {requested}")
        sys.exit(1)

    # ── AutoShutdown ──────────────────────────────────────────────────────────
    from workers.auto_shutdown import AutoShutdown
    all_queues = []
    if "sd15"    in requested: all_queues.append("sd15_tasks")
    if "trellis" in requested: all_queues.append("model_tasks")
    if "rig"     in requested: all_queues.append("rig_model")
    if "manual"  in requested: all_queues.append("manual_gen_tasks")

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
