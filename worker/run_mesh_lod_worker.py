#!/usr/bin/env python3
"""
Manual Generation — Mesh LOD Worker entry point
================================================
Runs on the CPU instance. Pulls tasks from Redis queue ``mesh_lod_tasks``
(or whatever MESH_LOD_QUEUE is set to) and generates LOD GLBs using
Blender + gltfpack.

Usage:
  cd ~/spark/worker
  python run_mesh_lod_worker.py
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
        logging.FileHandler("/tmp/mesh_lod_worker.log", mode="a"),
    ],
)
logger = logging.getLogger("MeshLodMain")

_worker_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir    = os.path.join(_worker_dir, "lib")
for _p in (_worker_dir, _lib_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workers.mesh_lod_worker import main  # noqa: E402


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Spark — Mesh LOD Worker (CPU)")
    logger.info("=" * 60)
    logger.info(f"  Queue   : {os.getenv('MESH_LOD_QUEUE', 'mesh_lod_tasks')}")
    logger.info(f"  Redis   : {os.getenv('REDIS_HOST','localhost')}:{os.getenv('REDIS_PORT','6379')}")
    logger.info(f"  Blender : {os.getenv('BLENDER_BIN','/usr/bin/blender')}")
    logger.info(f"  gltfpack: {os.getenv('GLTFPACK_BIN','/usr/local/bin/gltfpack')}")
    logger.info("=" * 60)

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
