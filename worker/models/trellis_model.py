"""
trellis_model.py — TRELLIS.2 (microsoft/TRELLIS.2-4B) image-to-3D
===========================================================================

Public API
----------
    load_trellis(model_id, repo_path, remove_bg)  -> TrellisPipes
    run_trellis(pipes, front_image, params,
                side_image=None, back_image=None) -> bytes  (GLB binary)

Single-view
-----------
    TRELLIS.2's `pipeline.run(image)` is single-image (it preprocesses the
    image itself, including background removal, when preprocess_image=True).
    side_image / back_image are accepted for call-site compatibility but are
    ignored with a warning — TRELLIS.2 has no multi-image entry point.

TrellisPipes (dataclass)
------------------------
    pipe          — Trellis2ImageTo3DPipeline on CUDA
    rembg_session — unused (kept for signature compatibility; None)

Param defaults
--------------
    texture_size       (int)  1024    — baked texture atlas resolution
    decimation_target  (int)  200000  — target face count for the exported mesh
    seed               (int)  42      — sampler seed

Notes
-----
- Uses microsoft/TRELLIS.2 (package: trellis2, class: Trellis2ImageTo3DPipeline).
- Repo must be cloned to TRELLIS_REPO_PATH (~/trellis) and on PYTHONPATH.
- GLB export uses o_voxel.postprocess.to_glb(...) on the MeshWithVoxel result,
  with WebP-embedded textures.
- Returns raw GLB bytes — caller uploads directly.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger("models.trellis")

# expandable_segments reduces fragmentation OOMs on the big 4B sparse model.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TRELLIS_REPO_PATH = os.getenv("TRELLIS_REPO_PATH", os.path.expanduser("~/trellis"))

PARAM_DEFAULTS: dict = {
    "texture_size":      1024,
    "decimation_target": 200000,
    "seed":              42,
}

# TRELLIS.2 voxel grid is normalized to the unit cube centered at origin.
_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
# nvdiffrast hard triangle limit — simplify the raw mesh below this before export.
_NVDIFFRAST_LIMIT = 16_777_216


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
    Load the TRELLIS.2 pipeline onto CUDA.

    Args:
        model_id:  HuggingFace model ID (microsoft/TRELLIS.2-4B).
        repo_path: Path to cloned TRELLIS.2 repo — added to sys.path.
        remove_bg: Unused — TRELLIS.2 removes the background internally during
                   run(preprocess_image=True). Kept for call-site compatibility.

    Raises:
        ImportError: If the trellis2 package is not importable.
                     Set PYTHONPATH=/home/ec2-user/trellis
    """
    import sys
    if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        logger.info(f"[trellis] Added repo to sys.path: {repo_path}")

    logger.info(f"[trellis] Loading TRELLIS.2 pipeline: {model_id}")
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except ImportError as exc:
        raise ImportError(
            "trellis2 package not found. "
            f"Set PYTHONPATH={repo_path}\n"
            f"(repo_path={repo_path!r}  error={exc})"
        ) from exc

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipe.cuda()
    logger.info("[trellis] TRELLIS.2 pipeline loaded on CUDA.")

    return TrellisPipes(pipe=pipe, rembg_session=None)


def run_trellis(
    pipes:       TrellisPipes,
    front_image: Image.Image,
    params:      dict | None = None,
    side_image:  Image.Image | None = None,
    back_image:  Image.Image | None = None,
) -> bytes:
    """
    Run TRELLIS.2 on the front view and return a GLB binary.

    Args:
        pipes:       TrellisPipes from load_trellis().
        front_image: PIL.Image — required front-view character image.
        params:      Override dict — any subset of PARAM_DEFAULTS keys:
                       texture_size      (int) — texture atlas resolution
                       decimation_target (int) — target exported face count
                       seed              (int) — sampler seed
        side_image:  Ignored (TRELLIS.2 is single-image) — warns if supplied.
        back_image:  Ignored (TRELLIS.2 is single-image) — warns if supplied.

    Returns:
        bytes — raw GLB binary (textures embedded as WebP).
    """
    import sys
    if TRELLIS_REPO_PATH not in sys.path:
        sys.path.insert(0, TRELLIS_REPO_PATH)

    import o_voxel

    if side_image is not None or back_image is not None:
        logger.warning(
            "[trellis] side/back views supplied but TRELLIS.2 is single-image — "
            "using front view only."
        )

    p                 = {**PARAM_DEFAULTS, **(params or {})}
    texture_size      = int(p["texture_size"])
    decimation_target = int(p["decimation_target"])
    seed              = int(p["seed"])

    image = front_image.convert("RGB")

    logger.info(
        f"[trellis] run  texture={texture_size}  "
        f"decimation={decimation_target}  seed={seed}"
    )

    with torch.no_grad():
        mesh = pipes.pipe.run(image, seed=seed, preprocess_image=True)[0]
        # Cap raw triangle count to nvdiffrast's limit before texturing/export.
        mesh.simplify(_NVDIFFRAST_LIMIT)

        glb = o_voxel.postprocess.to_glb(
            vertices          = mesh.vertices,
            faces             = mesh.faces,
            attr_volume       = mesh.attrs,
            coords            = mesh.coords,
            attr_layout       = mesh.layout,
            voxel_size        = mesh.voxel_size,
            aabb              = _AABB,
            decimation_target = decimation_target,
            texture_size      = texture_size,
            remesh            = True,
            remesh_band       = 1,
            remesh_project    = 0,
            verbose           = False,
        )

    # o_voxel's GLB object writes via .export(path, extension_webp=...); it has no
    # in-memory writer, so round-trip through a temp file to get the bytes.
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        glb.export(tmp_path, extension_webp=True)
        with open(tmp_path, "rb") as fh:
            glb_bytes = fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    torch.cuda.empty_cache()
    logger.info(f"[trellis] done  size={len(glb_bytes) // 1024} KB")
    return glb_bytes
