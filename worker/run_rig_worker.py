#!/usr/bin/env python3
"""
CPU Rig Worker — entry point
=============================
Run on the CPU instance to process rig_tasks from Redis.
GPU manual_gen_worker forwards rig tasks here; this worker
runs Blender + Auto-Rig Pro and uploads the rigged GLB.

Usage:
  cd /home/ubuntu/spark/worker
  python run_rig_worker.py
"""

import logging
import os
import sys

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/rig_worker.log", mode="a"),
    ],
)
logger = logging.getLogger("RigWorkerMain")

_worker_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir    = os.path.join(_worker_dir, "lib")
for _p in (_worker_dir, _lib_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workers.rig_cpu_worker import RigCPUWorker  # noqa: E402


def main():
    logger.info("=" * 55)
    logger.info("  Spark — CPU Rig Worker (Auto-Rig Pro via Blender)")
    logger.info("=" * 55)
    logger.info(f"  Queue   : rig_tasks")
    logger.info(f"  Redis   : {os.getenv('REDIS_HOST','localhost')}:{os.getenv('REDIS_PORT','6379')}")
    logger.info(f"  Blender : {os.getenv('BLENDER_PATH','~/blender/blender')}")
    logger.info(f"  Log     : /tmp/rig_worker.log")
    logger.info("=" * 55)

    worker = RigCPUWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        sys.exit(0)


if __name__ == "__main__":
    main()
