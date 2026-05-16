"""
flux_model.py — Flux.1-schnell text-to-image
=============================================

Public API
----------
    load_flux(model_id)                    -> FluxPipeline
    run_flux(pipe, prompt, params)         -> PIL.Image

Param defaults (override any key via the params dict)
------------------------------------------------------
    width          (int)   512   — output width in pixels
    height         (int)   512   — output height in pixels
    steps          (int)   4     — number of denoising steps
    guidance_scale (float) 0.0   — classifier-free guidance scale

Notes
-----
- L4 GPU has 22 GB usable VRAM. Flux transformer is ~24 GB in bfloat16 so
  enable_model_cpu_offload() OOMs.  enable_sequential_cpu_offload() loads one
  layer at a time and always fits — at the cost of ~3–4× slower inference.
- MAX_SIZE = 512.  Larger activations also OOM under sequential offload.
- Returns a single PIL.Image (mode RGB).
"""

from __future__ import annotations

import logging
import torch
from PIL import Image

logger = logging.getLogger("models.flux")

# ── Default params ────────────────────────────────────────────────────────────
PARAM_DEFAULTS: dict = {
    "width":          512,
    "height":         512,
    "steps":          4,
    "guidance_scale": 0.0,
}

# Hard ceiling imposed by L4 22 GB VRAM under sequential CPU offload
MAX_SIZE = 512


# ── Load ──────────────────────────────────────────────────────────────────────

def load_flux(model_id: str) -> object:
    """
    Download (or load from cache) and return a Flux pipeline ready for inference.

    The pipeline uses sequential CPU offload so it never holds the full
    transformer in VRAM — safe on any GPU with ≥8 GB.

    Args:
        model_id: HuggingFace model ID, e.g. "black-forest-labs/FLUX.1-schnell"

    Returns:
        diffusers.FluxPipeline with sequential CPU offload enabled.
    """
    from diffusers import FluxPipeline
    import os

    logger.info(f"[flux] Loading model: {model_id}")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # FLUX_OFFLOAD controls how aggressively to spill weights to CPU RAM.
    #   "sequential" — every layer cycles CPU<->GPU per step (default; safest on
    #                  small VRAM like L4 23 GB, but I/O-bound and slow)
    #   "model"      — whole model moved to GPU per step (~16 GB VRAM)
    #   "none"       — full pipeline pinned on GPU (~24 GB VRAM; fastest on L40S)
    offload_mode = os.getenv("FLUX_OFFLOAD", "sequential").lower()
    if offload_mode == "none":
        pipe.to("cuda")
        logger.info("[flux] Model loaded (no offload — full GPU residency).")
    elif offload_mode == "model":
        pipe.enable_model_cpu_offload()
        logger.info("[flux] Model loaded (model-level CPU offload).")
    else:
        pipe.enable_sequential_cpu_offload()
        logger.info("[flux] Model loaded (sequential CPU offload).")
    pipe.vae.enable_slicing()
    return pipe


# ── Run ───────────────────────────────────────────────────────────────────────

def run_flux(pipe, prompt: str, params: dict | None = None) -> Image.Image:
    """
    Generate one image from *prompt* using the Flux pipeline.

    Args:
        pipe:   FluxPipeline returned by load_flux().
        prompt: Positive text prompt.
        params: Override dict — any subset of PARAM_DEFAULTS keys:
                  width          (int)   — clamped to MAX_SIZE (512)
                  height         (int)   — clamped to MAX_SIZE (512)
                  steps          (int)
                  guidance_scale (float)

    Returns:
        PIL.Image (RGB, size = width × height).
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
