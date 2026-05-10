"""
sd_model.py — SD1.5 + ControlNet image-to-image
================================================

Public API
----------
    load_sd(model_id, openpose_ref_path)       -> SDPipes
    run_stage1(pipes, init_img, prompt, negative, params)   -> PIL.Image
    run_stage2(pipes, init_img, prompt, negative, params)   -> PIL.Image
    run_multiview(pipes, init_img, prompt, negative, params) -> PIL.Image

SDPipes (dataclass)
-------------------
    pipe_biped    — ControlNetImg2Img [openpose + canny]  — humanoid characters
    pipe_quad     — ControlNetImg2Img [canny only]        — quadruped / non-humanoid
    pipe_i2i      — plain StableDiffusionImg2ImgPipeline  — polish / multiview
    openpose_ref  — PIL.Image T-pose skeleton (512×512)   — used by pipe_biped

Param defaults (per stage)
--------------------------
Stage 1 (stage1_defaults):
    denoise         (float) 0.20
    cfg             (float) 5.5
    steps           (int)   20
    openpose_weight (float) 0.85  — humanoid only
    canny_weight    (float) 0.55
    category        (str)   "humanoid"  — "humanoid" | "quadruped"

Stage 2 (stage2_defaults):
    denoise         (float) 0.35
    cfg             (float) 7.0
    steps           (int)   20

Multiview (multiview_defaults):
    denoise         (float) 0.45
    cfg             (float) 7.0
    steps           (int)   20

Notes
-----
- All inputs are resized to 512×512 before inference.
- Returns a single PIL.Image (mode RGB).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("models.sd")

IMG_SIZE = 512

# ── Param defaults ────────────────────────────────────────────────────────────

STAGE1_DEFAULTS: dict = {
    "denoise":         0.20,
    "cfg":             5.5,
    "steps":           20,
    "openpose_weight": 0.85,
    "canny_weight":    0.55,
    "category":        "humanoid",
}

STAGE2_DEFAULTS: dict = {
    "denoise": 0.35,
    "cfg":     7.0,
    "steps":   20,
}

MULTIVIEW_DEFAULTS: dict = {
    "denoise": 0.45,
    "cfg":     7.0,
    "steps":   20,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_canny(img: Image.Image) -> Image.Image:
    """Return a 3-channel Canny edge map at IMG_SIZE."""
    img_r = img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    gray  = cv2.cvtColor(np.array(img_r), cv2.COLOR_RGB2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 180)
    return Image.fromarray(np.stack([edges] * 3, axis=-1).astype(np.uint8))


def _blank_image(w: int = IMG_SIZE, h: int = IMG_SIZE) -> Image.Image:
    return Image.new("RGB", (w, h), color=(0, 0, 0))


def _load_openpose_ref(path: str) -> Image.Image:
    """Load T-pose reference image or return black fallback."""
    if path and os.path.isfile(path):
        try:
            return Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        except Exception as exc:
            logger.warning(f"[sd] Failed to load openpose ref ({path}): {exc}")
    logger.warning("[sd] openpose_ref_path not set — using blank reference")
    return _blank_image()


# ── SDPipes ───────────────────────────────────────────────────────────────────

@dataclass
class SDPipes:
    """Container for all loaded SD1.5 + ControlNet pipelines."""
    pipe_biped:   object       # StableDiffusionControlNetImg2ImgPipeline [openpose, canny]
    pipe_quad:    object       # StableDiffusionControlNetImg2ImgPipeline [canny]
    pipe_i2i:     object       # StableDiffusionImg2ImgPipeline
    openpose_ref: Image.Image  # T-pose skeleton (512×512 RGB)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_sd(model_id: str, openpose_ref_path: str = "") -> SDPipes:
    """
    Load SD1.5 + ControlNet pipelines onto CUDA.

    Builds three shared-backbone pipelines from a single checkpoint:
      - pipe_biped  : dual ControlNet [openpose, canny] for humanoid characters
      - pipe_quad   : single ControlNet [canny] for quadrupeds
      - pipe_i2i    : plain img2img for stage 2 polish and multiview

    Args:
        model_id:          HuggingFace model ID, e.g. "Lykon/DreamShaper"
        openpose_ref_path: Absolute path to a T-pose openpose skeleton image.
                           Pass "" to use a black fallback (ControlNet disabled).

    Returns:
        SDPipes dataclass with all three pipelines on CUDA.
    """
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionImg2ImgPipeline,
        UniPCMultistepScheduler,
    )

    logger.info(f"[sd] Loading ControlNet: openpose")
    cn_openpose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16,
    )

    logger.info(f"[sd] Loading ControlNet: canny")
    cn_canny = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
    )

    logger.info(f"[sd] Loading base pipeline: {model_id}")
    pipe_cn = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=cn_openpose,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe_cn.scheduler = UniPCMultistepScheduler.from_config(pipe_cn.scheduler.config)
    pipe_cn.enable_xformers_memory_efficient_attention()
    pipe_cn.to("cuda")
    cn_canny.to("cuda")

    base = dict(pipe_cn.components)

    # Humanoid: openpose + canny
    pipe_biped = StableDiffusionControlNetImg2ImgPipeline(
        **{**base, "controlnet": [cn_openpose, cn_canny]}
    )

    # Quadruped: canny only
    pipe_quad = StableDiffusionControlNetImg2ImgPipeline(
        **{**base, "controlnet": cn_canny}
    )

    # Plain img2img (stage 2, multiview)
    pipe_i2i = StableDiffusionImg2ImgPipeline(
        **{k: v for k, v in base.items() if k != "controlnet"}
    )

    openpose_ref = _load_openpose_ref(openpose_ref_path)

    logger.info("[sd] All SD1.5 pipelines loaded on CUDA.")
    return SDPipes(
        pipe_biped=pipe_biped,
        pipe_quad=pipe_quad,
        pipe_i2i=pipe_i2i,
        openpose_ref=openpose_ref,
    )


# ── Run — Stage 1 ─────────────────────────────────────────────────────────────

def run_stage1(
    pipes:    SDPipes,
    init_img: Image.Image,
    prompt:   str,
    negative: str = "",
    params:   dict | None = None,
) -> Image.Image:
    """
    SD1.5 + ControlNet img2img — Stage 1 character refinement.

    Selects the correct pipeline based on params["category"]:
      "humanoid" → pipe_biped  (dual ControlNet: openpose + canny)
      anything else → pipe_quad (canny only)

    Args:
        pipes:    SDPipes from load_sd().
        init_img: Input PIL.Image (any size — resized internally to 512×512).
        prompt:   Positive prompt.
        negative: Negative prompt.
        params:   Override dict — any subset of STAGE1_DEFAULTS keys:
                    denoise         (float) — img2img strength
                    cfg             (float) — classifier-free guidance
                    steps           (int)
                    openpose_weight (float) — humanoid only
                    canny_weight    (float)
                    category        (str)   — "humanoid" | "quadruped"

    Returns:
        PIL.Image (RGB, 512×512).
    """
    p = {**STAGE1_DEFAULTS, **(params or {})}

    denoise         = float(p["denoise"])
    cfg             = float(p["cfg"])
    steps           = int(p["steps"])
    openpose_weight = float(p["openpose_weight"])
    canny_weight    = float(p["canny_weight"])
    category        = str(p["category"])

    init_img  = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    canny_img = _extract_canny(init_img)

    logger.info(
        f"[sd] stage1  category={category}  denoise={denoise}  "
        f"cfg={cfg}  steps={steps}"
    )

    with torch.no_grad():
        if category == "humanoid":
            result = pipes.pipe_biped(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                control_image=[pipes.openpose_ref, canny_img],
                controlnet_conditioning_scale=[openpose_weight, canny_weight],
                strength=denoise,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=1,
            )
        else:
            result = pipes.pipe_quad(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                control_image=canny_img,
                controlnet_conditioning_scale=canny_weight,
                strength=denoise,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=1,
            )

    torch.cuda.empty_cache()
    img = result.images[0]
    logger.info(f"[sd] stage1 done  size={img.size}")
    return img


# ── Run — Stage 2 ─────────────────────────────────────────────────────────────

def run_stage2(
    pipes:    SDPipes,
    init_img: Image.Image,
    prompt:   str,
    negative: str = "",
    params:   dict | None = None,
) -> Image.Image:
    """
    SD1.5 plain img2img — Stage 2 polish pass.  No ControlNet.

    Args:
        pipes:    SDPipes from load_sd().
        init_img: Input PIL.Image (any size — resized to 512×512).
        prompt:   Positive prompt.
        negative: Negative prompt.
        params:   Override dict — any subset of STAGE2_DEFAULTS keys:
                    denoise (float)
                    cfg     (float)
                    steps   (int)

    Returns:
        PIL.Image (RGB, 512×512).
    """
    p = {**STAGE2_DEFAULTS, **(params or {})}

    denoise = float(p["denoise"])
    cfg     = float(p["cfg"])
    steps   = int(p["steps"])

    init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

    logger.info(f"[sd] stage2  denoise={denoise}  cfg={cfg}  steps={steps}")

    with torch.no_grad():
        result = pipes.pipe_i2i(
            prompt=prompt,
            negative_prompt=negative,
            image=init_img,
            strength=denoise,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=1,
        )

    torch.cuda.empty_cache()
    img = result.images[0]
    logger.info(f"[sd] stage2 done  size={img.size}")
    return img


# ── Run — Multiview ────────────────────────────────────────────────────────────

def run_multiview(
    pipes:    SDPipes,
    init_img: Image.Image,
    prompt:   str,
    negative: str = "",
    params:   dict | None = None,
) -> Image.Image:
    """
    SD1.5 plain img2img — multiview generation (side or back view).

    Uses the same pipe_i2i as stage 2 but with different default denoise/cfg
    tuned for view-change tasks.

    Args:
        pipes:    SDPipes from load_sd().
        init_img: Input PIL.Image (any size — resized to 512×512).
        prompt:   Positive prompt (should describe the desired view angle).
        negative: Negative prompt.
        params:   Override dict — any subset of MULTIVIEW_DEFAULTS keys:
                    denoise (float)
                    cfg     (float)
                    steps   (int)

    Returns:
        PIL.Image (RGB, 512×512).
    """
    p = {**MULTIVIEW_DEFAULTS, **(params or {})}

    denoise = float(p["denoise"])
    cfg     = float(p["cfg"])
    steps   = int(p["steps"])

    init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

    logger.info(f"[sd] multiview  denoise={denoise}  cfg={cfg}  steps={steps}")

    with torch.no_grad():
        result = pipes.pipe_i2i(
            prompt=prompt,
            negative_prompt=negative,
            image=init_img,
            strength=denoise,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=1,
        )

    torch.cuda.empty_cache()
    img = result.images[0]
    logger.info(f"[sd] multiview done  size={img.size}")
    return img
