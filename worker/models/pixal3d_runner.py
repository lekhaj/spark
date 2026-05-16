"""
pixal3d_runner.py — subprocess wrapper for TencentARC/Pixal3D
==============================================================

Pixal3D lives in its own conda env (TRELLIS.2 stack: torch 2.4+cu124,
flash-attn 2, nvdiffrast, kaolin, utils3d 0.0.2). It conflicts with our
TRELLIS-v1 `spark` env, so we shell out to its env's python.

Public API
----------
    run_pixal3d(front_image: PIL.Image, params: dict) -> bytes  (GLB binary)

Env vars (read at call time)
----------------------------
    PIXAL3D_PYTHON     — interpreter inside the pixal3d conda env
                         default: ~/miniconda3/envs/pixal3d/bin/python
    PIXAL3D_REPO       — path to cloned Pixal3D repo
                         default: ~/Pixal3D
    PIXAL3D_ATTN       — ATTN_BACKEND value (L4 needs flash_attn, not flash_attn_3)
                         default: flash_attn
    PIXAL3D_LOW_VRAM   — "1" to enable pipeline.low_vram (required on L4 23GB)
                         default: 1
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile

from PIL import Image

logger = logging.getLogger("models.pixal3d")

DEFAULT_PYTHON = os.path.expanduser("~/miniconda3/envs/pixal3d/bin/python")
DEFAULT_REPO   = os.path.expanduser("~/Pixal3D")


def run_pixal3d(front_image: Image.Image, params: dict | None = None) -> bytes:
    """
    Run Pixal3D on a single front-view image and return GLB binary.

    Args:
        front_image: PIL.Image — required.
        params: optional overrides:
            seed (int)              42
            texture_size (int)      2048
            decimation_target (int) 200000

    Returns:
        bytes — raw GLB binary.
    """
    p = params or {}
    python_bin = os.getenv("PIXAL3D_PYTHON", DEFAULT_PYTHON)
    repo_path  = os.getenv("PIXAL3D_REPO",   DEFAULT_REPO)
    attn       = os.getenv("PIXAL3D_ATTN",   "flash_attn")
    low_vram   = os.getenv("PIXAL3D_LOW_VRAM", "1")

    script = os.path.join(repo_path, "inference.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Pixal3D inference script not found: {script}")
    if not os.path.isfile(python_bin):
        raise FileNotFoundError(f"Pixal3D python not found: {python_bin}")

    with tempfile.TemporaryDirectory(prefix="pixal3d_") as tmp:
        in_path  = os.path.join(tmp, "input.png")
        out_path = os.path.join(tmp, "output.glb")
        front_image.convert("RGBA").save(in_path)

        cmd = [
            python_bin, script,
            "--image",  in_path,
            "--output", out_path,
            "--seed",   str(int(p.get("seed", 42))),
        ]
        env = {
            **os.environ,
            "ATTN_BACKEND":     attn,
            "PIXAL3D_LOW_VRAM": low_vram,
        }

        logger.info(f"[pixal3d] running subprocess: {' '.join(cmd)}")
        # First cold run on a fresh spot can take 30-50 min for EBS lazy-load
        # of all model weights. Subsequent warm calls are 1-3 min.
        timeout_s = int(os.getenv("PIXAL3D_TIMEOUT_S", "3600"))
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(
                f"pixal3d inference failed (rc={proc.returncode})\n"
                f"--- stderr tail ---\n{tail}"
            )

        if not os.path.isfile(out_path):
            raise RuntimeError(f"pixal3d produced no output GLB at {out_path}")

        with open(out_path, "rb") as f:
            glb_bytes = f.read()

    logger.info(f"[pixal3d] done  size={len(glb_bytes) // 1024} KB")
    return glb_bytes
