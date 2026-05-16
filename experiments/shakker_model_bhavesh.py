#!/usr/bin/env python3
"""
shakker_model_bhavesh.py — SHAKKER LABS FLUX CONTROLNET ENGINE
===============================================================
Branch : bhavesh-dev  |  GPU : spark_l4

Supports BOTH Strategy A and Strategy B via the control_mode parameter.

STRATEGY A (mode=4, POSE):
  Input:  OpenPose skeleton image (programmatically generated T-pose)
  Flow:   Text Prompt + Skeleton → Shakker → T-pose character (one step)
  Use:    Production path. Best quality. No SD1.5 needed.

STRATEGY B (mode=0, CANNY):
  Input:  Concept image from Flux Stage 0
  Flow:   Stage0 → Canny edges → Shakker → posed character
  Use:    When you need to preserve a specific concept image's identity.

MODEL ARCHITECTURE:
  Base      : black-forest-labs/FLUX.1-dev  (12B param transformer)
  ControlNet: Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro

CONTROL MODES (ControlNet-Union-Pro):
  0 = Canny   ← Strategy B
  1 = Tile
  2 = Depth
  3 = Blur
  4 = Pose    ← Strategy A  ★ ACTIVE
  5 = Gray
  6 = Low Quality

MEMORY STRATEGY (L4 24GB GPU):
  Base + ControlNet together ≈ 27GB (exceeds L4 24GB).
  enable_sequential_cpu_offload() streams layers from system RAM
  to GPU one at a time. Safe. No OOM crashes. Slightly slower.
"""

import logging
import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger("shakker_model")

# ── Model identifiers ─────────────────────────────────────────────────────────
SHAKKER_REPO  = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"

# ── Control modes ─────────────────────────────────────────────────────────────
CANNY_MODE = 0   # Strategy B — edge/structure control from concept image
POSE_MODE  = 4   # Strategy A — OpenPose skeleton control (ACTIVE)

# ── Default inference parameters (tuned for L4 GPU) ──────────────────────────
DEFAULT_STEPS      = 20      # 20 = good quality, ~15-20 min on L4 with model_cpu_offload
                             # raise to 28 for higher quality (adds ~10 min)
DEFAULT_CFG        = 3.5     # Flux optimal guidance scale
DEFAULT_CTRL_SCALE = 0.65   # Pose conditioning: 0.65 gives creativity WITH structure

# Canny thresholds (Strategy B only — not used in Strategy A)
DEFAULT_CANNY_LO   = 50     # lower = more edges detected
DEFAULT_CANNY_HI   = 150    # higher = only strong edges detected

# Offload strategy — auto-selected based on available VRAM
# model_cpu_offload  : moves whole submodels GPU ↔ CPU between phases (~15-25 min)
# sequential_cpu_offload: moves layer-by-layer (~2 hours — AVOID on 24GB GPU)


@dataclass
class ShakkerPipes:
    """Container for the loaded Shakker pipeline."""
    pipe: object   # FluxControlNetPipeline


# ── Canny extractor (Strategy B only) ─────────────────────────────────────────
def _extract_canny(img: Image.Image, low: int = DEFAULT_CANNY_LO, high: int = DEFAULT_CANNY_HI) -> Image.Image:
    """
    Convert a PIL image to a black-and-white Canny edge map.
    Used only in Strategy B (CANNY_MODE).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required for Canny. Run: pip install opencv-python-headless")

    arr   = np.array(img.convert("RGB"))
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


# ── Model loader ──────────────────────────────────────────────────────────────
def load_shakker() -> ShakkerPipes:
    """
    Load Shakker Labs ControlNet + FLUX.1-dev.

    Memory breakdown:
      FLUX.1-dev alone  ≈ 24 GB  (bfloat16)
      ControlNet        ≈  3 GB  (bfloat16)
      Total             ≈ 27 GB
      → model_cpu_offload moves whole submodels GPU↔CPU (~15-20 min on L4)
      → sequential_cpu_offload is the fallback (~2 hrs) — only if model_cpu_offload OOMs

    Returns:
        ShakkerPipes ready for inference.
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

    # ── Memory management: model_cpu_offload is 3-5x faster than sequential
    # on a 24GB GPU like the L4. It moves whole submodels (text encoder, transformer,
    # VAE) to CPU between phases instead of shuffling thousands of tiny layers.
    # If this OOMs (transformer alone is ~24GB), it falls back to sequential.
    logger.info("[shakker] Setting up memory offload strategy for L4 24GB GPU...")
    try:
        pipe.enable_model_cpu_offload()
        logger.info("[shakker] ✅ model_cpu_offload ACTIVE — expected ~15-25 min generation")
    except Exception as e:
        logger.warning(f"[shakker] model_cpu_offload failed ({e}), falling back to sequential...")
        pipe.enable_sequential_cpu_offload()
        logger.info("[shakker] ⚠️ sequential_cpu_offload active — generation will be ~2 hours")

    logger.info("[shakker] All models loaded safely.")

    return ShakkerPipes(pipe=pipe)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_shakker(
    pipes:               ShakkerPipes,
    control_image:       Image.Image,
    prompt:              str,
    control_mode:        int   = POSE_MODE,        # ★ Default is now POSE (Strategy A)
    controlnet_scale:    float = DEFAULT_CTRL_SCALE,
    num_inference_steps: int   = DEFAULT_STEPS,
    guidance_scale:      float = DEFAULT_CFG,
    width:               int   = 512,
    height:              int   = 512,
    seed:                int   = 42,
) -> Image.Image:
    """
    Generate a posed character image using Shakker Labs Flux ControlNet.

    Strategy A (recommended):
        control_image = generate_tpose_skeleton(512, 512)  # from openpose_humanoid.py
        control_mode  = POSE_MODE  (4)

    Strategy B (legacy concept transfer):
        control_image = concept_img_from_flux_stage0
        control_mode  = CANNY_MODE (0)
        NOTE: run_shakker will auto-extract Canny edges from the concept image.

    Args:
        pipes           : Loaded ShakkerPipes.
        control_image   : For Strategy A → OpenPose skeleton PIL image.
                          For Strategy B → Flux concept PIL image (Canny extracted auto).
        prompt          : Text prompt. FLUX does NOT use negative prompts.
        control_mode    : 4=Pose (Strategy A), 0=Canny (Strategy B).
        controlnet_scale: Pose adherence strength. 0.65 gives anatomy + style freedom.
                          Increase to 0.9 if the pose drifts too much.
        num_inference_steps: 28 = excellent. 20 = faster but slightly lower quality.
        guidance_scale  : 3.5 is Flux optimal. Do not raise above 7.0.
        width / height  : Output size. 512x512 recommended for Trellis input.
        seed            : Fixed seed for reproducibility.

    Returns:
        PIL Image of the posed character.
    """
    # Strategy B: auto-extract canny if using CANNY_MODE
    if control_mode == CANNY_MODE:
        logger.info("[shakker] Strategy B — extracting Canny edges from concept image...")
        control_image = _extract_canny(control_image)
    else:
        logger.info(f"[shakker] Strategy A — using provided skeleton (mode={control_mode})")

    logger.info(
        f"[shakker] Inference | mode={control_mode} steps={num_inference_steps} "
        f"cfg={guidance_scale} ctrl_scale={controlnet_scale} size={width}x{height}"
    )

    generator = torch.Generator().manual_seed(seed)

    result = pipes.pipe(
        prompt=prompt,
        control_image=control_image,
        control_mode=control_mode,
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
