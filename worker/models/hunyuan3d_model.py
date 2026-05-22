"""
hunyuan3d_model.py — subprocess wrapper for Hunyuan3D-2.0
==========================================================

Hunyuan3D lives in its own conda env (hunyuan3d: torch 2.5.1+cu124,
hy3dgen, custom_rasterizer, DifferentiableRenderer). It ships compiled
CUDA extensions that conflict with the spark env's trellis and sd stacks,
so we shell out exactly like pixal3d does.

Public API
----------
    run_hunyuan3d(front_image: PIL.Image, params: dict) -> bytes  (GLB binary)

Env vars (read at call time)
----------------------------
    HUNYUAN3D_PYTHON     — interpreter inside the hunyuan3d conda env
                           default: ~/miniconda3/envs/hunyuan3d/bin/python
    HUNYUAN3D_REPO       — path to cloned Hunyuan3D-2.0 source tree
                           default: ~/Hunyuan3D-2
    HUNYUAN3D_TIMEOUT_S  — subprocess timeout in seconds (default 3600)
    HUNYUAN3D_MODEL_ID   — HuggingFace model id (default: tencent/Hunyuan3D-2)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from PIL import Image

logger = logging.getLogger("models.hunyuan3d")

DEFAULT_PYTHON = os.path.expanduser("~/miniconda3/envs/hunyuan3d/bin/python")
DEFAULT_REPO   = os.path.expanduser("~/Hunyuan3D-2")

# This script ships inside the spark repo and is called by the subprocess.
_INFER_SCRIPT = os.path.join(os.path.dirname(__file__), "hunyuan3d_infer.py")


def run_hunyuan3d(front_image: Image.Image, params: dict | None = None) -> bytes:
    """
    Run Hunyuan3D-2.0 on a single front-view image and return GLB binary.

    Args:
        front_image: PIL.Image — the character image (background removal is
                     done inside the subprocess via rembg if installed).
        params: optional overrides:
            seed (int)                42
            steps (int)               50      shape diffusion steps
            guidance_scale (float)    7.5
            texture_resolution (int)  2048

    Returns:
        bytes — raw GLB binary.
    """
    p          = params or {}
    python_bin = os.getenv("HUNYUAN3D_PYTHON", DEFAULT_PYTHON)
    timeout_s  = int(os.getenv("HUNYUAN3D_TIMEOUT_S", "3600"))

    if not os.path.isfile(python_bin):
        raise FileNotFoundError(f"Hunyuan3D python not found: {python_bin}")
    if not os.path.isfile(_INFER_SCRIPT):
        raise FileNotFoundError(f"Hunyuan3D infer script not found: {_INFER_SCRIPT}")

    with tempfile.TemporaryDirectory(prefix="hunyuan3d_") as tmp:
        in_path  = os.path.join(tmp, "input.png")
        out_path = os.path.join(tmp, "output.glb")
        front_image.convert("RGBA").save(in_path)

        cmd = [
            python_bin, _INFER_SCRIPT,
            "--image",               in_path,
            "--output",              out_path,
            "--seed",                str(int(p.get("seed", 42))),
            "--steps",               str(int(p.get("steps", 50))),
            "--guidance-scale",      str(float(p.get("guidance_scale", 7.5))),
            "--texture-resolution",  str(int(p.get("texture_resolution", 2048))),
            "--remove-bg",           # always strip background in the subprocess
        ]

        logger.info(f"[hunyuan3d] launching subprocess: {' '.join(cmd[:4])} ...")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(
                f"hunyuan3d inference failed (rc={proc.returncode})\n"
                f"--- stderr tail ---\n{tail}"
            )

        if not os.path.isfile(out_path):
            raise RuntimeError(f"hunyuan3d produced no output GLB at {out_path}")

        with open(out_path, "rb") as f:
            glb_bytes = f.read()

    # Relay subprocess stderr (our infer script writes progress there)
    for line in (proc.stderr or "").splitlines():
        logger.debug(f"  [hunyuan3d-sub] {line}")

    logger.info(f"[hunyuan3d] done  size={len(glb_bytes) // 1024} KB")
    return glb_bytes
