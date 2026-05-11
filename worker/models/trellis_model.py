"""
trellis_model.py — TRELLIS image-to-3D reconstruction
======================================================

Public API
----------
    load_trellis(model_id, repo_path, remove_bg)  -> TrellisPipes
    run_trellis(pipes, front_image, params)        -> bytes  (GLB binary)

TrellisPipes (dataclass)
------------------------
    pipe          — TrellisImageTo3DPipeline on CUDA
    rembg_session — rembg session for background removal (or None)

Param defaults
--------------
    texture_size  (int)  1024   — baked texture atlas resolution
    simplify      (float) 0.95  — fraction of triangles to remove (0=none, 1=all)

Notes
-----
- Uses the original microsoft/TRELLIS repo (package name: trellis).
- Repo must be at TRELLIS_REPO_PATH (/home/ec2-user/trellis by default)
  OR installed into the Python env.
- Input image is converted to RGBA, background optionally removed,
  composited on white, then resized to 512×512 before passing to TRELLIS.
- Returns raw GLB bytes — caller writes them to disk or uploads directly.
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
    "texture_size": 1024,
    "simplify":     0.95,
}


@dataclass
class TrellisPipes:
    pipe:          object
    rembg_session: Optional[object] = field(default=None)


def load_trellis(
    model_id:   str  = "JeffreyXiang/TRELLIS-image-large",
    repo_path:  str  = TRELLIS_REPO_PATH,
    remove_bg:  bool = True,
) -> TrellisPipes:
    """
    Load TRELLIS pipeline onto CUDA.

    Args:
        model_id:  HuggingFace model ID (default: JeffreyXiang/TRELLIS-image-large)
        repo_path: Path to cloned TRELLIS repo — added to sys.path so the
                   'trellis' package is importable without pip install.
        remove_bg: Whether to load the rembg background-removal session.
    """
    import sys
    if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        logger.info(f"[trellis] Added repo to sys.path: {repo_path}")

    logger.info(f"[trellis] Loading pipeline: {model_id}")
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except ImportError as exc:
        raise ImportError(
            f"trellis package not found (tried repo_path={repo_path!r}). "
            "Clone https://github.com/microsoft/TRELLIS.git to /home/ec2-user/trellis "
            "and run bash worker/gpu_setup/install_trellis.sh"
        ) from exc

    pipe = TrellisImageTo3DPipeline.from_pretrained(model_id)
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
            logger.warning(f"[trellis] rembg failed: {exc} — using original")

    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB").resize((512, 512))


def run_trellis(
    pipes:       TrellisPipes,
    front_image: Image.Image,
    params:      dict | None = None,
) -> bytes:
    """
    Run TRELLIS on a single front-view image and return a GLB binary.

    Args:
        pipes:       TrellisPipes from load_trellis().
        front_image: PIL.Image — the front-view character image.
        params:      Override dict — any subset of PARAM_DEFAULTS keys:
                       texture_size (int)   — texture atlas resolution
                       simplify     (float) — triangle reduction ratio (0–1)

    Returns:
        bytes — raw GLB binary.
    """
    import sys
    if TRELLIS_REPO_PATH not in sys.path:
        sys.path.insert(0, TRELLIS_REPO_PATH)

    from trellis.utils import postprocessing_utils

    p            = {**PARAM_DEFAULTS, **(params or {})}
    texture_size = int(p["texture_size"])
    simplify     = float(p["simplify"])

    img = _prepare_image(front_image, pipes.rembg_session)

    logger.info(f"[trellis] run  texture={texture_size}  simplify={simplify}")

    with torch.no_grad():
        outputs = pipes.pipe.run(
            img,
            formats=["mesh", "gaussian"],
            preprocess_image=False,  # already preprocessed above
        )

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=simplify,
        texture_size=texture_size,
    )

    buf = io.BytesIO()
    glb.export(buf)
    glb_bytes = buf.getvalue()

    torch.cuda.empty_cache()
    logger.info(f"[trellis] done  size={len(glb_bytes) // 1024} KB")
    return glb_bytes
