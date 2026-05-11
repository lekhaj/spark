"""
sd_model.py — SD1.5 + ControlNet + IP-Adapter image-to-image
=============================================================

Public API
----------
    load_sd(model_id, openpose_ref_path)       -> SDPipes
    run_stage1(pipes, init_img, prompt, negative, params)   -> PIL.Image
    run_stage2(pipes, init_img, prompt, negative, params)   -> PIL.Image
    run_multiview(pipes, init_img, prompt, negative, params) -> PIL.Image

SDPipes (dataclass)
-------------------
    pipe_biped       — ControlNetImg2Img [openpose + canny]  — humanoid characters
    pipe_quad        — ControlNetImg2Img [canny only]        — quadruped / non-humanoid
    pipe_i2i         — plain StableDiffusionImg2ImgPipeline  — polish / multiview
    openpose_ref     — PIL.Image T-pose skeleton (512×512)   — used by pipe_biped
    ip_adapter_ready — bool, True if IP-Adapter was loaded successfully

Stage 1 — T-pose lock with IP-Adapter identity preservation
------------------------------------------------------------
    Input:  Flux image (character in any pose)
    Output: Same character in T-pose, Flux details preserved via IP-Adapter

    Pipeline:
      - IP-Adapter Plus (h94/IP-Adapter) encodes Flux image → identity vector
      - OpenPose ControlNet: T-pose skeleton template → locks pose
      - Canny ControlNet:    extracted from T-pose template (not Flux)
                             → structural edges aligned with T-pose
      - High denoise (0.65): enough freedom to fully change pose

    For quadruped:
      - IP-Adapter weight slightly higher (0.75) — more identity
      - Canny only (no OpenPose — animal skeletons differ too much)
      - Canny from Flux (best structural info available)

Param defaults (per stage)
--------------------------
Stage 1 (STAGE1_DEFAULTS):
    denoise          (float) 0.65   — high: needed to actually change pose
    cfg              (float) 7.0
    steps            (int)   25
    openpose_weight  (float) 1.00   — strong T-pose lock
    canny_weight     (float) 0.25   — light structural hint
    ip_adapter_weight(float) 0.65   — identity from Flux
    category         (str)   "humanoid"

Stage 2 (STAGE2_DEFAULTS):
    denoise (float) 0.35
    cfg     (float) 7.0
    steps   (int)   20

Multiview (MULTIVIEW_DEFAULTS):
    denoise (float) 0.45
    cfg     (float) 7.0
    steps   (int)   20

Notes
-----
- All inputs are resized to 512×512 before inference.
- Returns a single PIL.Image (mode RGB).
- IP-Adapter scale is set to 0.0 before stage2 / multiview calls to avoid
  bleeding IP features into those passes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("models.sd")

# Public alias — preferred name for the T-pose stage
# run_tpose is defined at the bottom of this file as: run_tpose = run_stage1

IMG_SIZE = 512

# IP-Adapter model (SD1.5 Plus — better identity fidelity than base)
IP_ADAPTER_REPO    = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = "ip-adapter-plus_sd15.bin"

# ── Param defaults ────────────────────────────────────────────────────────────

STAGE1_DEFAULTS: dict = {
    "denoise":           0.82,   # high — must overcome Flux pose fully
    "cfg":               7.5,
    "steps":             30,
    "openpose_weight":   1.30,   # >1.0 — forces T-pose skeleton to win
    "canny_weight":      0.20,   # light structural hint only
    "ip_adapter_weight": 0.45,   # identity from Flux, lower to reduce pose bleed
    "category":          "humanoid",
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


_S3_TPOSE_URL = (
    "https://sparkassets-us.s3.us-east-1.amazonaws.com"
    "/controlnet_refs/tpose_openpose.png"
)


def _load_openpose_ref(path: str) -> Image.Image:
    """Load T-pose reference image.
    Priority: local file → S3 fallback → blank (logs warning).
    """
    # 1. Try local file
    if path and os.path.isfile(path):
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            logger.info(f"[sd] T-pose reference loaded from disk: {path}")
            return img
        except Exception as exc:
            logger.warning(f"[sd] Failed to load openpose ref ({path}): {exc}")

    # 2. Download from S3 and save locally so subsequent loads are instant
    try:
        import urllib.request, tempfile
        logger.info(f"[sd] Downloading T-pose reference from S3: {_S3_TPOSE_URL}")
        dest = path if path else "/tmp/tpose_openpose.png"
        os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)
        urllib.request.urlretrieve(_S3_TPOSE_URL, dest)
        img = Image.open(dest).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        logger.info(f"[sd] T-pose reference downloaded and loaded from {dest}")
        return img
    except Exception as exc:
        logger.warning(f"[sd] S3 download failed ({exc}) — OpenPose will use blank reference")

    logger.warning(
        "[sd] TPOSE_OPENPOSE_PATH not set or file missing — "
        "OpenPose ControlNet will use blank reference (pose lock disabled)"
    )
    return _blank_image()


# ── SDPipes ───────────────────────────────────────────────────────────────────

@dataclass
class SDPipes:
    """Container for all loaded SD1.5 + ControlNet pipelines."""
    pipe_biped:       object        # ControlNetImg2Img [openpose, canny] — humanoid
    pipe_quad:        object        # ControlNetImg2Img [canny only]      — quadruped
    pipe_i2i:         object        # StableDiffusionImg2ImgPipeline      — stage2/multiview
    openpose_ref:     Image.Image   # T-pose skeleton (512×512 RGB)
    ip_adapter_ready: bool = False  # True if IP-Adapter loaded successfully


# ── Load ──────────────────────────────────────────────────────────────────────

def load_sd(model_id: str, openpose_ref_path: str = "") -> SDPipes:
    """
    Load SD1.5 + ControlNet + IP-Adapter pipelines onto CUDA.

    Builds three shared-backbone pipelines from a single checkpoint:
      - pipe_biped  : dual ControlNet [openpose, canny] + IP-Adapter — humanoid
      - pipe_quad   : single ControlNet [canny] + IP-Adapter — quadruped
      - pipe_i2i    : plain img2img (no ControlNet, no IP-Adapter active) — stage2/multiview

    IP-Adapter is loaded once onto pipe_biped; since all three pipes share the
    same UNet object, ip_adapter_scale can be toggled per call.

    Args:
        model_id:          HuggingFace model ID, e.g. "Lykon/DreamShaper"
        openpose_ref_path: Absolute path to T-pose openpose skeleton image.

    Returns:
        SDPipes dataclass.
    """
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionImg2ImgPipeline,
        UniPCMultistepScheduler,
    )

    logger.info("[sd] Loading ControlNet: openpose")
    cn_openpose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16,
    )

    logger.info("[sd] Loading ControlNet: canny")
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

    # Plain img2img (stage 2 polish, multiview)
    pipe_i2i = StableDiffusionImg2ImgPipeline(
        **{k: v for k, v in base.items() if k != "controlnet"}
    )

    openpose_ref = _load_openpose_ref(openpose_ref_path)

    # ── IP-Adapter ────────────────────────────────────────────────────────────
    # Load onto pipe_biped; all three pipes share the same UNet so this enables
    # IP-Adapter for all of them. We set scale=0.0 before stage2/multiview calls.
    ip_ready = False
    try:
        logger.info(f"[sd] Loading IP-Adapter Plus: {IP_ADAPTER_REPO}")
        pipe_biped.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder="models",
            weight_name=IP_ADAPTER_WEIGHTS,
        )
        pipe_biped.set_ip_adapter_scale(0.0)   # off by default; run_stage1 sets it
        ip_ready = True
        logger.info("[sd] IP-Adapter Plus loaded successfully.")
    except Exception as exc:
        logger.warning(
            f"[sd] IP-Adapter load failed — identity preservation disabled: {exc}"
        )

    logger.info("[sd] All SD1.5 pipelines ready on CUDA.")
    return SDPipes(
        pipe_biped=pipe_biped,
        pipe_quad=pipe_quad,
        pipe_i2i=pipe_i2i,
        openpose_ref=openpose_ref,
        ip_adapter_ready=ip_ready,
    )


# ── Run — Stage 1 (T-pose + IP-Adapter identity) ─────────────────────────────

def run_stage1(
    pipes:            SDPipes,
    init_img:         Image.Image,
    prompt:           str,
    negative:         str = "",
    params:           dict | None = None,
    ip_adapter_image: Image.Image | None = None,
) -> Image.Image:
    """
    SD1.5 + ControlNet + IP-Adapter — T-pose character generation.

    Takes a Flux character image and outputs the same character in T-pose:
      - IP-Adapter:  encodes Flux image → preserves character identity
      - OpenPose CN: T-pose skeleton template → locks pose (humanoid only)
      - Canny CN:    extracted from T-pose template (not Flux) → structural
                     edges aligned with T-pose, avoids fighting OpenPose

    For quadruped:
      - Canny from Flux init_img (no OpenPose for animal skeletons)
      - IP-Adapter weight slightly higher

    Args:
        pipes:            SDPipes from load_sd().
        init_img:         Flux image (init point for img2img denoising).
        prompt:           Positive prompt (describe character).
        negative:         Negative prompt.
        params:           Override any STAGE1_DEFAULTS keys.
        ip_adapter_image: Image for IP-Adapter identity (usually same as init_img).
                          If None, IP-Adapter is disabled for this call.

    Returns:
        PIL.Image (RGB, 512×512).
    """
    p = {**STAGE1_DEFAULTS, **(params or {})}

    denoise           = float(p["denoise"])
    cfg               = float(p["cfg"])
    steps             = int(p["steps"])
    openpose_weight   = float(p["openpose_weight"])
    canny_weight      = float(p["canny_weight"])
    ip_adapter_weight = float(p["ip_adapter_weight"])
    category          = str(p["category"])

    # ── Per-call skeleton override (for A/B testing different T-pose refs) ────
    # Pass openpose_ref_url in params to use a different skeleton for this call
    # without reloading the model. Falls back to pipes.openpose_ref if not set.
    openpose_ref_url = p.get("openpose_ref_url", "")
    if openpose_ref_url:
        try:
            import urllib.request, tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            urllib.request.urlretrieve(openpose_ref_url, tmp.name)
            openpose_ref = Image.open(tmp.name).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            logger.info(f"[sd] Using per-call openpose ref: {openpose_ref_url}")
        except Exception as exc:
            logger.warning(f"[sd] Failed to load openpose_ref_url ({exc}), using default")
            openpose_ref = pipes.openpose_ref
    else:
        openpose_ref = pipes.openpose_ref

    init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

    # ── IP-Adapter scale ──────────────────────────────────────────────────────
    use_ip = pipes.ip_adapter_ready and ip_adapter_image is not None
    if pipes.ip_adapter_ready:
        scale = ip_adapter_weight if use_ip else 0.0
        pipes.pipe_biped.set_ip_adapter_scale(scale)
        pipes.pipe_quad.set_ip_adapter_scale(scale)

    logger.info(
        f"[sd] stage1  category={category}  denoise={denoise}  cfg={cfg}  "
        f"steps={steps}  openpose_w={openpose_weight}  canny_w={canny_weight}  "
        f"ip_adapter={'ON @' + str(ip_adapter_weight) if use_ip else 'OFF'}"
    )

    extra = {}
    if use_ip:
        extra["ip_adapter_image"] = ip_adapter_image.resize(
            (IMG_SIZE, IMG_SIZE)
        ).convert("RGB")

    with torch.no_grad():
        if category == "humanoid":
            # Canny from T-pose template — consistent direction, avoids fighting OpenPose
            canny_img = _extract_canny(openpose_ref)
            result = pipes.pipe_biped(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                control_image=[openpose_ref, canny_img],
                controlnet_conditioning_scale=[openpose_weight, canny_weight],
                strength=denoise,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                **extra,
            )
        else:
            # Quadruped: Canny from Flux (best structural info; no OpenPose)
            canny_img = _extract_canny(init_img)
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
                **extra,
            )

    torch.cuda.empty_cache()
    img = result.images[0]
    logger.info(f"[sd] stage1 done  size={img.size}")
    return img


# ── Run — Stage 2 (detail polish, no pose constraint) ────────────────────────

def run_stage2(
    pipes:    SDPipes,
    init_img: Image.Image,
    prompt:   str,
    negative: str = "",
    params:   dict | None = None,
) -> Image.Image:
    """
    SD1.5 plain img2img — Stage 2 polish pass.  No ControlNet, no IP-Adapter.

    Args:
        pipes:    SDPipes from load_sd().
        init_img: Input PIL.Image (any size — resized to 512×512).
        prompt:   Positive prompt.
        negative: Negative prompt.
        params:   Override dict — any subset of STAGE2_DEFAULTS keys.

    Returns:
        PIL.Image (RGB, 512×512).
    """
    p = {**STAGE2_DEFAULTS, **(params or {})}

    denoise = float(p["denoise"])
    cfg     = float(p["cfg"])
    steps   = int(p["steps"])

    init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

    # Disable IP-Adapter for plain polish pass
    if pipes.ip_adapter_ready:
        pipes.pipe_i2i.set_ip_adapter_scale(0.0)

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

    Args:
        pipes:    SDPipes from load_sd().
        init_img: Input PIL.Image (any size — resized to 512×512).
        prompt:   Positive prompt (should describe the desired view angle).
        negative: Negative prompt.
        params:   Override dict — any subset of MULTIVIEW_DEFAULTS keys.

    Returns:
        PIL.Image (RGB, 512×512).
    """
    p = {**MULTIVIEW_DEFAULTS, **(params or {})}

    denoise = float(p["denoise"])
    cfg     = float(p["cfg"])
    steps   = int(p["steps"])

    init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

    # Disable IP-Adapter for multiview pass
    if pipes.ip_adapter_ready:
        pipes.pipe_i2i.set_ip_adapter_scale(0.0)

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


# ── Public alias ──────────────────────────────────────────────────────────────
# run_tpose is the canonical name for the active T-pose generation stage.
# run_stage1 is kept for backward compatibility.
run_tpose = run_stage1

# run_stage2 and run_multiview are kept as legacy functions but no longer
# registered in STAGE_REGISTRY. They remain importable for experiments.
