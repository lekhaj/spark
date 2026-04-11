#!/usr/bin/env python3
"""
Manual Generation Worker — GPU entry point
==========================================
Processes manual_gen_tasks from Redis queue.
Completely independent from the automated biome pipeline.

Usage (on GPU instance):
  cd /home/ec2-user/worker
  python run_manual_worker.py

Queue: manual_gen_tasks
MongoDB: manual_gen_sessions collection
"""

import logging
import os
import sys

from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
# Resolve .env relative to this file so the script works regardless of cwd.
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/manual_gen_worker.log", mode="a"),
    ],
)
logger = logging.getLogger("ManualGenMain")

# ── Worker import ─────────────────────────────────────────────────────────────
# Ensure the worker package is importable regardless of how this script is
# invoked (e.g. `python run_manual_worker.py` from the worker/ directory).
_worker_dir = os.path.dirname(os.path.abspath(__file__))
if _worker_dir not in sys.path:
    sys.path.insert(0, _worker_dir)

from workers.manual_gen_worker import ManualGenWorker  # noqa: E402


# ── GPU info helper ───────────────────────────────────────────────────────────

def _gpu_info() -> str:
    """Return a one-line GPU summary, or a friendly message if CUDA is absent."""
    try:
        import torch
        if torch.cuda.is_available():
            idx  = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            return f"{name}  ({vram:.1f} GB VRAM)"
        return "CUDA not available — running on CPU"
    except ImportError:
        return "torch not installed — GPU info unavailable"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Instantiate ManualGenWorker and run its blocking poll loop."""

    # Print startup banner so ops can confirm the right process is running.
    redis_host = os.getenv("REDIS_HOST", "18.207.13.85")
    redis_port = os.getenv("REDIS_PORT", "6380")
    mongo_db   = os.getenv("MONGO_DB",   "World_builder")

    logger.info("=" * 60)
    logger.info("  Spark — Manual Generation Worker")
    logger.info("=" * 60)
    logger.info(f"  Queue   : manual_gen_tasks")
    logger.info(f"  Redis   : {redis_host}:{redis_port}")
    logger.info(f"  MongoDB : {mongo_db}  (collection: manual_gen_sessions)")
    logger.info(f"  GPU     : {_gpu_info()}")
    logger.info(f"  Log     : /tmp/manual_gen_worker.log")
    logger.info("=" * 60)

    worker = ManualGenWorker()

    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
