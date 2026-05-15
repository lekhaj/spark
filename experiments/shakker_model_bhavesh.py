#!/usr/bin/env python3
"""
shakker_model_bhavesh.py — SHAKKER LABS FLUX CONTROLNET PIPELINE
=================================================================
Branch : bhavesh-dev  |  GPU : spark_l4

Replaces the old SD1.5 (DreamShaper) T-pose stage with a native
Flux ControlNet pipeline using Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro.

WHY THIS IS BETTER THAN SD1.5:
  - SD1.5 is 2022 architecture trying to re-draw a 2024 Flux image.
    The quality mismatch causes ghost hands, wrong anatomy, and floating legs.
  - Shakker Labs is built natively FOR Flux. The same AI that designed
    the character is now posing it. Zero quality loss.

MODEL ARCHITECTURE:
  Base     : black-forest-labs/FLUX.1-dev  (12B param diffusion transformer)
  ControlNet: Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro
             Supports 7 control modes. We use mode 0 = Canny edges.

MEMORY STRATEGY (L4 24GB GPU):
  Both the base model and the ControlNet together exceed 24GB of VRAM.
  We use enable_sequential_cpu_offload() to keep them in system RAM
  and stream layers to the GPU one at a time. This is slower but safe
  and eliminates all OOM (Out of Memory) crashes.

CONTROL MODES (for reference):
  0 = Canny   ← we use this (edge/structure control)
  1 = Tile
  2 = Depth
  3 = Blur
  4 = Pose
  5 = Gray
  6 = Low Quality
"""

import logging
import torch
import numpy as np
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger("shakker_model")

# ── Model identifiers ─────────────────────────────────────────────────────────
SHAKKER_REPO  = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"

# ── ControlNet Union Pro control mode ─────────────────────────────────────────
CANNY_MODE = 0   # 0 = Canny edge detection mode

# ── Default inference parameters (tuned for L4 GPU) ──────────────────────────
DEFAULT_STEPS     = 20       # 20 steps = good quality. SD1.5 used 30.
DEFAULT_CFG       = 3.5      # Flux guidance scale (lower is better here)
DEFAULT_CANNY_LO  = 50       # Canny low threshold — lower = more edges
DEFAULT_CANNY_HI  = 150      # Canny high threshold
DEFAULT_CTRL_SCALE = 0.7     # How strongly to follow Canny. 1.0 = rigid. 0.5 = loose.


@dataclass
class ShakkerPipes:
    """Container for all loaded Shakker pipeline objects."""
    pipe: object   # FluxControlNetPipeline


# ── Canny edge extractor ──────────────────────────────────────────────────────
def _extract_canny(img: Image.Image, low: int = DEFAULT_CANNY_LO, high: int = DEFAULT_CANNY_HI) -> Image.Image:
    """
    Convert a PIL image into a black-and-white Canny edge map.
    This is the structural reference we feed to the ControlNet.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not found. Run: pip install opencv-python-headless")

    arr  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    # ControlNet expects a 3-channel (RGB) image, not grayscale
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_shakker() -> ShakkerPipes:
    """
    Load the Shakker Labs ControlNet and FLUX.1-dev base pipeline.

    Memory usage:
      - FLUX.1-dev alone  ≈ 24 GB in bfloat16
      - ControlNet        ≈  3 GB in bfloat16
      - Total            ≈ 27 GB  (exceeds L4 24GB VRAM)
      → We MUST use sequential_cpu_offload to handle this safely.

    Returns:
        ShakkerPipes with a ready-to-use pipeline.
    """
    from diffusers import FluxControlNetModel, FluxControlNetPipeline

    logger.info(f"[shakker] Loading ControlNet: {SHAKKER_REPO}")
    controlnet = FluxControlNetModel.from_pretrained(
        SHAKKER_REPO,
        torch_dtype=torch.bfloat16,
    )
    logger.info("[shakker] ControlNet loaded.")

    logger.info(f"[shakker] Loading base model: {FLUX_DEV_REPO}")
    pipe = FluxControlNetPipeline.from_pretrained(
        FLUX_DEV_REPO,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )

    logger.info("[shakker] Enabling sequential_cpu_offload (required for L4 24GB)...")
    pipe.enable_sequential_cpu_offload()
    logger.info("[shakker] All Shakker models loaded and offloaded safely.")

    return ShakkerPipes(pipe=pipe)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_shakker(
    pipes:               ShakkerPipes,
    concept_img:         Image.Image,
    prompt:              str,
    controlnet_scale:    float = DEFAULT_CTRL_SCALE,
    num_inference_steps: int   = DEFAULT_STEPS,
    guidance_scale:      float = DEFAULT_CFG,
    width:               int   = 512,
    height:              int   = 512,
    seed:                int   = 42,
) -> Image.Image:
    """
    Generate a posed character image using Shakker Labs Flux ControlNet.

    NOTE: FLUX.1-dev does NOT support negative prompts.
          Do NOT pass a negative_prompt parameter. The negative in our
          old SD1.5 pipeline was a workaround for SD1.5's weaknesses.
          Shakker doesn't need it — the model is smart enough.

    Args:
        pipes              : Loaded ShakkerPipes object.
        concept_img        : The character concept image from Flux Stage 0.
        prompt             : Text describing the desired pose.
                             Keep UNDER 77 tokens (SD1.5 limit no longer applies
                             to Flux, but keep it clean and simple).
        controlnet_scale   : 0.7 is the sweet spot. Increase to 1.0 if
                             anatomy drifts too far from the Canny reference.
        num_inference_steps: 20 gives good quality. 28 gives excellent.
        guidance_scale     : 3.5 is Flux optimal. Do not increase above 7.0.
        width / height     : Output image size.
        seed               : Fixed seed for reproducibility.

    Returns:
        PIL Image of the posed character.
    """
    logger.info("[shakker] Extracting Canny edges from concept image...")
    canny_img = _extract_canny(concept_img)

    logger.info(
        f"[shakker] Running inference | "
        f"steps={num_inference_steps} cfg={guidance_scale} "
        f"canny_scale={controlnet_scale} size={width}x{height}"
    )

    generator = torch.Generator().manual_seed(seed)

    result = pipes.pipe(
        prompt=prompt,
        control_image=canny_img,
        control_mode=CANNY_MODE,
        width=width,
        height=height,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    torch.cuda.empty_cache()
    img = result.images[0]
    logger.info(f"[shakker] Done. Output size: {img.size}")
    return img
