"""
flux_model_bhavesh_v1.py — EXACT COPY of worker/models/flux_model.py
=====================================================================
Branch  : bhavesh-dev
Purpose : Sandbox Flux model for experiments — NO changes whatsoever.
          Copied verbatim from production so Stage 0 is 100% identical.

CHANGES vs production  : NONE
DO NOT MERGE TO MAIN   : Not needed — same as production.
"""

from __future__ import annotations

import logging
import torch
from PIL import Image

logger = logging.getLogger("models.flux.bhavesh_v1")

# ── Default params (same as production) ──────────────────────────────────────
PARAM_DEFAULTS: dict = {
    "width":          512,
    "height":         512,
    "steps":          4,
    "guidance_scale": 0.0,
}

# Hard ceiling imposed by L4 22 GB VRAM under sequential CPU offload
# NOTE: Production Flux CAPS at 512. NOT 768 or 1024.
MAX_SIZE = 512


def load_flux(model_id: str) -> object:
    """
    Load Flux.1-schnell pipeline — EXACT same as production flux_model.load_flux().

    Uses sequential CPU offload so it never holds the full transformer in VRAM.
    Safe on any GPU with >= 8 GB.
    """
    from diffusers import FluxPipeline

    logger.info(f"[flux] Loading model: {model_id}")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    logger.info("[flux] Model loaded (sequential CPU offload).")
    return pipe


def run_flux(pipe, prompt: str, params: dict | None = None) -> Image.Image:
    """
    Generate one image from prompt — EXACT same as production flux_model.run_flux().

    Args:
        pipe:   FluxPipeline returned by load_flux().
        prompt: Positive text prompt.
        params: Override dict for width / height / steps / guidance_scale.
                Width and height are CLAMPED to MAX_SIZE (512).

    Returns:
        PIL.Image (RGB).
    """
    p = {**PARAM_DEFAULTS, **(params or {})}

    width          = min(int(p["width"]),          MAX_SIZE)
    height         = min(int(p["height"]),         MAX_SIZE)
    steps          = int(p["steps"])
    guidance_scale = float(p["guidance_scale"])

    logger.info(
        f"[flux] run  size={width}x{height}  steps={steps}  "
        f"guidance={guidance_scale}"
    )

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )

    torch.cuda.empty_cache()
    img = result.images[0]
    logger.info(f"[flux] done  size={img.size}")
    return img
