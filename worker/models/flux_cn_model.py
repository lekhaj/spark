"""
flux_cn_model.py — FLUX.1-dev + Shakker-Labs ControlNet-Union-Pro-2.0
=====================================================================

Pose-aware Flux image generation. Conditions FLUX.1-dev on a control image
(OpenPose skeleton, Canny edges, depth, soft-edge, blur, gray, low-quality).

Public API
----------
    load_flux_cn(base_model_id, controlnet_id)  -> FluxControlNetPipeline
    run_flux_cn(pipe, prompt, params, ...)      -> PIL.Image
    extract_control_image(src_img, mode)        -> PIL.Image (utility)

Control modes (Union Pro 2.0 — auto-detects mode if not passed)
----------------------------------------------------------------
    canny         — Canny edges via OpenCV
    soft_edge     — HED / PidiNet soft-edges via controlnet-aux
    depth         — MiDaS depth map via controlnet-aux
    blur          — Gaussian blur of the source
    pose          — OpenPose skeleton via controlnet-aux
    gray          — grayscale of the source
    low_quality   — downscale+upscale (intentional degradation)

VRAM
----
    L40S 48GB → set FLUX_CN_OFFLOAD=none (default) → full GPU residency, ~30GB
    L4   23GB → set FLUX_CN_OFFLOAD=sequential   → fits, slower
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter

logger = logging.getLogger("models.flux_cn")

DEFAULT_BASE_MODEL   = "black-forest-labs/FLUX.1-dev"
DEFAULT_CONTROLNET   = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

PARAM_DEFAULTS: dict = {
    "width":                          1024,
    "height":                         1024,
    "steps":                          28,
    "guidance_scale":                 3.5,    # FLUX.1-dev distilled guidance
    "true_cfg_scale":                 1.0,    # >1 enables real CFG with negative prompt
    "controlnet_conditioning_scale":  0.7,
    "control_guidance_start":         0.0,
    "control_guidance_end":           0.8,
    "control_mode":                   None,   # None = auto (Union Pro 2.0)
    "seed":                           -1,     # -1 = random
}

# Modes accepted by Union Pro 2.0; ID mapping (for the optional control_mode arg)
CONTROL_MODE_IDS = {
    "canny":       0,
    "soft_edge":   1,
    "depth":       2,
    "blur":        3,
    "pose":        4,
    "gray":        5,
    "low_quality": 6,
}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_flux_cn(
    base_model_id: str = DEFAULT_BASE_MODEL,
    controlnet_id: str = DEFAULT_CONTROLNET,
):
    """Load FluxControlNetPipeline with the Union Pro 2.0 ControlNet."""
    from diffusers import FluxControlNetPipeline
    from diffusers.models import FluxControlNetModel

    logger.info(f"[flux_cn] Loading controlnet: {controlnet_id}")
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.bfloat16
    )

    logger.info(f"[flux_cn] Loading base model: {base_model_id}")
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )

    offload = os.getenv("FLUX_CN_OFFLOAD", "none").lower()
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        logger.info("[flux_cn] sequential CPU offload enabled")
    elif offload == "model":
        pipe.enable_model_cpu_offload()
        logger.info("[flux_cn] model-level CPU offload enabled")
    else:
        pipe.to("cuda")
        logger.info("[flux_cn] full GPU residency (no offload)")

    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    return pipe


# ── Control-image extraction ──────────────────────────────────────────────────

def extract_control_image(src_img: Image.Image, mode: str,
                          target_size: int = 1024) -> Image.Image:
    """Convert *src_img* into a control image for the requested ControlNet mode."""
    mode = (mode or "canny").lower()
    img  = src_img.convert("RGB")
    # Keep aspect by resizing the long side
    img.thumbnail((target_size, target_size), Image.LANCZOS)

    if mode == "canny":
        import cv2
        arr   = np.array(img)
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        edges = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges)

    if mode == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=12))

    if mode == "gray":
        return img.convert("L").convert("RGB")

    if mode == "low_quality":
        w, h = img.size
        small = img.resize((max(1, w // 8), max(1, h // 8)), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)

    if mode == "pose":
        from controlnet_aux import OpenposeDetector
        det = _get_processor("openpose",
                             lambda: OpenposeDetector.from_pretrained("lllyasviel/Annotators"))
        return det(img, hand_and_face=False).resize(img.size, Image.LANCZOS)

    if mode == "soft_edge":
        # Try PidiNet first (recommended for Union Pro 2.0), fall back to HED
        try:
            from controlnet_aux import PidiNetDetector
            det = _get_processor("pidi",
                                 lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"))
        except Exception:
            from controlnet_aux import HEDdetector
            det = _get_processor("hed",
                                 lambda: HEDdetector.from_pretrained("lllyasviel/Annotators"))
        return det(img).resize(img.size, Image.LANCZOS)

    if mode == "depth":
        from controlnet_aux import MidasDetector
        det = _get_processor("midas",
                             lambda: MidasDetector.from_pretrained("lllyasviel/Annotators"))
        out = det(img)
        if isinstance(out, tuple):  # (depth, normal)
            out = out[0]
        return out.convert("RGB").resize(img.size, Image.LANCZOS)

    raise ValueError(f"Unknown control mode: {mode!r}. "
                     f"Valid: {list(CONTROL_MODE_IDS.keys())}")


# Lazy processor cache so OpenPose etc. only loads once per worker
_PROCESSOR_CACHE: dict = {}

def _get_processor(key: str, factory):
    if key not in _PROCESSOR_CACHE:
        logger.info(f"[flux_cn] loading control processor: {key}")
        _PROCESSOR_CACHE[key] = factory()
    return _PROCESSOR_CACHE[key]


# ── Run ───────────────────────────────────────────────────────────────────────

def run_flux_cn(
    pipe,
    prompt: str,
    params: Optional[dict] = None,
    *,
    source_image: Optional[Image.Image] = None,
    control_image: Optional[Image.Image] = None,
    negative_prompt: str = "",
) -> Image.Image:
    """
    Generate one image with FLUX.1-dev + ControlNet-Union-Pro-2.0.

    Either pass a pre-built *control_image*, or pass *source_image* + a
    `control_mode` in *params* and the control image will be extracted.
    """
    p = {**PARAM_DEFAULTS, **(params or {})}

    width  = int(p["width"])
    height = int(p["height"])
    steps  = int(p["steps"])
    cfg    = float(p["guidance_scale"])
    true_cfg = float(p["true_cfg_scale"])
    cn_scale = float(p["controlnet_conditioning_scale"])
    cn_start = float(p["control_guidance_start"])
    cn_end   = float(p["control_guidance_end"])
    mode     = p.get("control_mode")
    seed     = int(p["seed"])

    if control_image is None and source_image is None:
        # Pure text-to-image mode — synthesize a neutral gray control image and
        # force ControlNet conditioning to zero so it doesn't influence the
        # output. The pipeline still requires a control_image tensor.
        logger.info("[flux_cn] no source/control image → text-to-image mode "
                    "(cn_scale forced to 0)")
        control_image = Image.new("RGB", (width, height), (128, 128, 128))
        cn_scale = 0.0
    elif control_image is None:
        if not mode:
            raise ValueError("run_flux_cn: control_mode is required when extracting "
                             "control_image from source_image")
        logger.info(f"[flux_cn] extracting control image (mode={mode})")
        control_image = extract_control_image(source_image, mode,
                                              target_size=max(width, height))

    # Resize control image to output dims
    if control_image.size != (width, height):
        control_image = control_image.resize((width, height), Image.LANCZOS)

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Optional control_mode index (Union Pro 2.0 auto-detects if omitted)
    extra = {}
    if mode and mode in CONTROL_MODE_IDS:
        extra["control_mode"] = CONTROL_MODE_IDS[mode]

    logger.info(
        f"[flux_cn] run  size={width}x{height} steps={steps} cfg={cfg} "
        f"true_cfg={true_cfg} cn={cn_scale} start={cn_start} end={cn_end} "
        f"mode={mode} seed={seed}"
    )

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            control_image=control_image,
            controlnet_conditioning_scale=cn_scale,
            control_guidance_start=cn_start,
            control_guidance_end=cn_end,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            true_cfg_scale=true_cfg,
            num_images_per_prompt=1,
            generator=generator,
            **extra,
        )

    torch.cuda.empty_cache()
    img = result.images[0]
    logger.info(f"[flux_cn] done  size={img.size}")
    return img
