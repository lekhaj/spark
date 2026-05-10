"""
trellis_model.py — TRELLIS.2 image-to-3D reconstruction
=========================================================

Public API
----------
    load_trellis(model_id, repo_path, remove_bg)  -> TrellisPipes
    run_trellis(pipes, front_image, params)        -> bytes  (GLB binary)

TrellisPipes (dataclass)
------------------------
    pipe      — Trellis2ImageTo3DPipeline on CUDA
    rembg_session — rembg session for background removal (or None)

Param defaults
--------------
    texture_size  (int)  1024   — baked texture atlas resolution
    decimation    (int)  1000000 — target face count for mesh simplification
    remesh        (bool) True   — run remeshing before decimation

Notes
-----
- Input image is converted to RGB, background optionally removed, then
  composited on white and resized to 512×512 before passing to TRELLIS.
- Returns raw GLB bytes — caller writes them to disk or uploads directly.
"""

from __future__ import annotations

import io
import logging
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional

import requests
import torch
from PIL import Image

logger = logging.getLogger("models.trellis")

# ── Param defaults ────────────────────────────────────────────────────────────

PARAM_DEFAULTS: dict = {
    "texture_size": 1024,
    "decimation":   1_000_000,
    "remesh":       True,
}


# ── TrellisPipes ──────────────────────────────────────────────────────────────

@dataclass
class TrellisPipes:
    """Container for TRELLIS pipeline and optional rembg session."""
    pipe:          object
    rembg_session: Optional[object] = field(default=None)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_trellis(
    model_id:   str,
    repo_path:  str = "",
    remove_bg:  bool = True,
) -> TrellisPipes:
    """
    Load TRELLIS.2 pipeline onto CUDA.

    Args:
        model_id:  HuggingFace model ID, e.g. "microsoft/TRELLIS.2-4B"
        repo_path: Path to cloned TRELLIS repo (added to sys.path if given).
        remove_bg: Whether to load the rembg background-removal session.

    Returns:
        TrellisPipes with .pipe on CUDA and optional .rembg_session.

    Raises:
        ImportError: If trellis2 package is not importable.
    """
    import sys
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        logger.info(f"[trellis] Added repo to path: {repo_path}")

    logger.info(f"[trellis] Loading pipeline: {model_id}")
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except ImportError:
        raise ImportError(
            "trellis2 package not found. "
            "Run: bash worker/gpu_setup/install_trellis.sh"
        )

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipe.cuda()
    logger.info("[trellis] Pipeline loaded on CUDA.")

    rembg_session = None
    if remove_bg:
        try:
            import rembg
            rembg_session = rembg.new_session("u2net")
            logger.info("[trellis] rembg loaded (u2net).")
        except ImportError:
            logger.warning("[trellis] rembg not installed — background will not be removed.")

    return TrellisPipes(pipe=pipe, rembg_session=rembg_session)


# ── Image prep ────────────────────────────────────────────────────────────────

def _prepare_image(image: Image.Image, rembg_session) -> Image.Image:
    """Remove background (if session present), composite on white, resize to 512×512."""
    image = image.convert("RGBA")

    if rembg_session is not None:
        try:
            import rembg
            image = rembg.remove(image, session=rembg_session)
        except Exception as exc:
            logger.warning(f"[trellis] rembg failed: {exc} — using original")

    bg = Image.new("RGB", image.size, (255, 255, 255))
    if image.mode == "RGBA":
        bg.paste(image, mask=image.split()[3])
    else:
        bg.paste(image)

    return bg.resize((512, 512))


# ── Run ───────────────────────────────────────────────────────────────────────

def run_trellis(
    pipes:       TrellisPipes,
    front_image: Image.Image,
    params:      dict | None = None,
) -> bytes:
    """
    Run TRELLIS.2 on a single front-view image and return a GLB binary.

    Args:
        pipes:       TrellisPipes from load_trellis().
        front_image: PIL.Image — the front-view character image (any size/mode).
        params:      Override dict — any subset of PARAM_DEFAULTS keys:
                       texture_size (int)  — texture atlas resolution
                       decimation   (int)  — target polygon count
                       remesh       (bool) — remesh before decimation

    Returns:
        bytes — raw GLB binary (write to .glb file or upload directly).
    """
    p = {**PARAM_DEFAULTS, **(params or {})}

    texture_size = int(p["texture_size"])
    decimation   = int(p["decimation"])
    remesh       = bool(p["remesh"])

    img = _prepare_image(front_image, pipes.rembg_session)

    logger.info(
        f"[trellis] run  texture={texture_size}  "
        f"decimation={decimation}  remesh={remesh}"
    )

    import o_voxel
    with torch.no_grad():
        mesh = pipes.pipe.run(img)[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation,
        texture_size=texture_size,
        remesh=remesh,
    )

    # Export to an in-memory bytes buffer
    buf = io.BytesIO()
    glb.export(buf, extension_webp=True)
    glb_bytes = buf.getvalue()

    torch.cuda.empty_cache()
    logger.info(f"[trellis] done  size={len(glb_bytes) // 1024} KB")
    return glb_bytes
