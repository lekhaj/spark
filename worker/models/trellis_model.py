"""
trellis_model.py — TRELLIS.2 image-to-3D reconstruction
=========================================================

Public API
----------
    load_trellis(model_id, repo_path, remove_bg)  -> TrellisPipes
    run_trellis(pipes, front_image, params)        -> bytes  (GLB binary)

TrellisPipes (dataclass)
------------------------
    pipe          — Trellis2ImageTo3DPipeline on CUDA
    rembg_session — rembg session for background removal (or None)

Param defaults
--------------
    texture_size     (int)   1024    — baked texture atlas resolution
    decimation_target(int)   1000000 — target face count
    remesh           (bool)  True    — remesh before decimation

Notes
-----
- Uses microsoft/TRELLIS.2 repo (package: trellis2, class: Trellis2ImageTo3DPipeline).
- Repo must be cloned to TRELLIS_REPO_PATH (/home/ec2-user/trellis).
  Run: bash worker/gpu_setup/install_trellis.sh  to install.
- GLB export uses o_voxel.postprocess.to_glb() — requires o_voxel extension built.
- Returns raw GLB bytes — caller uploads directly.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger("models.trellis")

TRELLIS_REPO_PATH = os.getenv("TRELLIS_REPO_PATH", "/home/ec2-user/trellis")

PARAM_DEFAULTS: dict = {
    "texture_size":      1024,
    "decimation_target": 1_000_000,
    "remesh":            True,
}


@dataclass
class TrellisPipes:
    pipe:          object
    rembg_session: Optional[object] = field(default=None)


def load_trellis(
    model_id:   str  = "microsoft/TRELLIS.2-4B",
    repo_path:  str  = TRELLIS_REPO_PATH,
    remove_bg:  bool = True,
) -> TrellisPipes:
    """
    Load TRELLIS.2 pipeline onto CUDA.

    Args:
        model_id:  HuggingFace model ID (microsoft/TRELLIS.2-4B)
        repo_path: Path to cloned TRELLIS.2 repo — added to sys.path.
        remove_bg: Whether to load rembg background-removal session.

    Raises:
        ImportError: If trellis2 package is not importable.
                     Run: bash worker/gpu_setup/install_trellis.sh
    """
    import sys
    if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        logger.info(f"[trellis] Added repo to sys.path: {repo_path}")

    logger.info(f"[trellis] Loading pipeline: {model_id}")
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except ImportError as exc:
        raise ImportError(
            "trellis2 package not found. "
            "Run: bash worker/gpu_setup/install_trellis.sh\n"
            f"(repo_path={repo_path!r}  error={exc})"
        ) from exc

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


def _prepare_image(image: Image.Image, rembg_session) -> Image.Image:
    """Remove background (if session present), composite on white, resize to 512×512."""
    image = image.convert("RGBA")

    if rembg_session is not None:
        try:
            import rembg
            image = rembg.remove(image, session=rembg_session)
        except Exception as exc:
            logger.warning(f"[trellis] rembg failed: {exc} — using original image")

    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB").resize((512, 512))


def run_trellis(
    pipes:       TrellisPipes,
    front_image: Image.Image,
    params:      dict | None = None,
) -> bytes:
    """
    Run TRELLIS.2 on a single front-view image and return a GLB binary.

    Args:
        pipes:       TrellisPipes from load_trellis().
        front_image: PIL.Image — the front-view character image.
        params:      Override dict — any subset of PARAM_DEFAULTS keys:
                       texture_size      (int)  — texture atlas resolution
                       decimation_target (int)  — target face count
                       remesh            (bool) — remesh before decimation

    Returns:
        bytes — raw GLB binary.
    """
    import sys
    if TRELLIS_REPO_PATH not in sys.path:
        sys.path.insert(0, TRELLIS_REPO_PATH)

    import o_voxel

    p                = {**PARAM_DEFAULTS, **(params or {})}
    texture_size     = int(p["texture_size"])
    decimation_target = int(p["decimation_target"])
    remesh           = bool(p["remesh"])

    img = _prepare_image(front_image, pipes.rembg_session)

    logger.info(
        f"[trellis] run  texture={texture_size}  "
        f"decimation={decimation_target}  remesh={remesh}"
    )

    with torch.no_grad():
        mesh = pipes.pipe.run(img)[0]

    mesh.simplify(16_777_216)  # nvdiffrast vertex limit

    glb = o_voxel.postprocess.to_glb(
        vertices         = mesh.vertices,
        faces            = mesh.faces,
        attr_volume      = mesh.attrs,
        coords           = mesh.coords,
        attr_layout      = mesh.layout,
        voxel_size       = mesh.voxel_size,
        aabb             = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target = decimation_target,
        texture_size     = texture_size,
        remesh           = remesh,
        remesh_band      = 1,
        remesh_project   = 0,
        verbose          = False,
    )

    buf = io.BytesIO()
    glb.export(buf, extension_webp=True)
    glb_bytes = buf.getvalue()

    torch.cuda.empty_cache()
    logger.info(f"[trellis] done  size={len(glb_bytes) // 1024} KB")
    return glb_bytes
