#!/usr/bin/env python3
"""
ManualGenWorker
===============
GPU worker for the manual character-image generation pipeline.

Handles all stages driven by the Pipeline Dashboard:
    flux          — Flux.1-schnell text-to-image concept art
    sd_stage1     — SD1.5 + ControlNet (openpose+canny for humanoid,
                    canny-only for quadruped) img2img refinement
    sd_stage2     — SD1.5 plain img2img polish pass
    multiview_side/back — SD1.5 plain img2img for alternate views
    trellis       — forwards task to the TRELLIS 3D worker queue

Model loading is lazy (first use) to avoid spending GPU memory on a backend
that may never be needed in a given session.

Queue:  manual_gen_tasks
Schema: lib/manual_gen_schema.py  (COLLECTION = "manual_gen_sessions")
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from workers.base_worker import BaseWorker
from lib.manual_gen_schema import (
    COLLECTION,
    mark_running,
    mark_done,
    mark_error,
    mark_queued,
    get_stage_image_url,
)

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_SIZE       = 512
FLUX_MODEL     = os.getenv("FLUX_MODEL_ID",  "black-forest-labs/FLUX.1-schnell")
SD_MODEL       = os.getenv("SD_MODEL_ID",    "Lykon/DreamShaper")
TPOSE_OPENPOSE_PATH = os.getenv("TPOSE_OPENPOSE_PATH", "")

# Downstream TRELLIS worker queue name (must match trellis_worker.py)
TRELLIS_QUEUE  = "model_tasks"

logger = logging.getLogger("ManualGenWorker")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _extract_canny(img: Image.Image) -> Image.Image:
    """Return a 3-channel Canny edge map (512×512) for *img*."""
    img_r = img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr   = np.array(img_r)
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 180)
    return Image.fromarray(np.stack([edges] * 3, axis=-1).astype(np.uint8))


def _blank_image(w: int = IMG_SIZE, h: int = IMG_SIZE) -> Image.Image:
    """Return a solid-black RGB image (used as fallback openpose reference)."""
    return Image.new("RGB", (w, h), color=(0, 0, 0))


# ── Worker ────────────────────────────────────────────────────────────────────

class ManualGenWorker(BaseWorker):
    """GPU worker for the manual character generation pipeline."""

    worker_name = "ManualGenWorker"
    input_queue = "manual_gen_tasks"

    def __init__(self):
        super().__init__()

        # Lazy-loaded model handles — None until first use.
        self._flux_pipe    = None   # FluxPipeline

        self._pipe_cn      = None   # StableDiffusionControlNetPipeline (base, openpose)
        self._pipe_biped   = None   # StableDiffusionControlNetImg2ImgPipeline (openpose+canny)
        self._pipe_quad    = None   # StableDiffusionControlNetImg2ImgPipeline (canny only)
        self._pipe_i2i     = None   # StableDiffusionImg2ImgPipeline (no controlnet)
        self._cn_canny     = None   # ControlNetModel (canny)
        self._cn_openpose  = None   # ControlNetModel (openpose)

    # ── BaseWorker abstract ───────────────────────────────────────────────────

    def load_models(self):
        """No-op: all models are loaded lazily on first use."""
        self.logger.info("ManualGenWorker ready — models will load lazily on first use.")

    def process_task(self, task: dict, r, db) -> None:
        """Route an incoming task to the correct stage handler."""
        session_id = task.get("session_id", "")
        stage      = task.get("stage", "")
        task_id    = task.get("task_id", "")

        self.logger.info(
            f"[{task_id[:8]}] session={session_id[:8]} stage={stage} "
            f"char={task.get('char_label', '?')}"
        )

        # Mark stage as running in MongoDB
        try:
            mark_running(db, session_id, stage)
        except Exception as exc:
            self.logger.warning(f"mark_running failed: {exc}")

        try:
            if stage == "flux":
                self._run_flux(task, db)
            elif stage == "sd_stage1":
                self._run_sd_stage1(task, db)
            elif stage == "sd_stage2":
                self._run_sd_stage2(task, db)
            elif stage in ("multiview_side", "multiview_back"):
                self._run_multiview(task, db)
            elif stage == "trellis":
                self._run_trellis(task, r, db)
            else:
                raise ValueError(f"Unknown stage: '{stage}'")

        except Exception as exc:
            self.logger.exception(
                f"[{task_id[:8]}] stage={stage} FAILED: {exc}"
            )
            try:
                mark_error(db, session_id, stage, str(exc))
            except Exception as db_exc:
                self.logger.error(f"mark_error also failed: {db_exc}")

    # ── Lazy model loaders ────────────────────────────────────────────────────

    def _ensure_flux(self):
        """Load Flux pipeline into GPU memory if not already loaded."""
        if self._flux_pipe is not None:
            return

        self.logger.info(f"Loading Flux model: {FLUX_MODEL}")
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        self._flux_pipe = pipe
        self.logger.info("Flux model loaded.")

    def _ensure_sd(self):
        """Load SD1.5 + ControlNet pipelines into GPU memory if not already loaded."""
        if self._pipe_i2i is not None:
            return

        self.logger.info(f"Loading SD1.5 model: {SD_MODEL}")
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionControlNetPipeline,
            StableDiffusionImg2ImgPipeline,
            UniPCMultistepScheduler,
        )

        # ControlNet models
        self.logger.info("Loading ControlNet: openpose")
        cn_openpose = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16,
        )
        self.logger.info("Loading ControlNet: canny")
        cn_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16,
        )

        # Base pipeline (openpose controlnet)
        self.logger.info("Loading SD1.5 base (openpose controlnet) pipeline")
        pipe_cn = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL,
            controlnet=cn_openpose,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe_cn.scheduler = UniPCMultistepScheduler.from_config(
            pipe_cn.scheduler.config
        )
        pipe_cn.enable_xformers_memory_efficient_attention()
        pipe_cn.to("cuda")

        # Move standalone canny controlnet to CUDA
        cn_canny.to("cuda")

        base_components = dict(pipe_cn.components)

        # Bipedal (humanoid): dual controlnet [openpose, canny]
        biped_components = {**base_components, "controlnet": [cn_openpose, cn_canny]}
        pipe_biped_i2i = StableDiffusionControlNetImg2ImgPipeline(**biped_components)

        # Quadruped: canny only
        quad_components = {**base_components, "controlnet": cn_canny}
        pipe_quad_i2i = StableDiffusionControlNetImg2ImgPipeline(**quad_components)

        # Plain img2img (no controlnet)
        i2i_components = {k: v for k, v in base_components.items() if k != "controlnet"}
        pipe_i2i = StableDiffusionImg2ImgPipeline(**i2i_components)

        # Store all references
        self._cn_openpose  = cn_openpose
        self._cn_canny     = cn_canny
        self._pipe_cn      = pipe_cn
        self._pipe_biped   = pipe_biped_i2i
        self._pipe_quad    = pipe_quad_i2i
        self._pipe_i2i     = pipe_i2i

        self.logger.info("SD1.5 + ControlNet pipelines loaded.")

    # ── Stage handlers ─────────────────────────────────────────────────────────

    def _run_flux(self, task: dict, db) -> None:
        """
        Generate a concept image with Flux.1-schnell (text → image).

        Task params: width, height, steps, guidance_scale
        """
        session_id = task["session_id"]
        stage      = task["stage"]   # "flux"
        prompt     = task.get("prompt", "")
        params     = task.get("params") or {}

        width          = int(params.get("width",          768))
        height         = int(params.get("height",         1024))
        steps          = int(params.get("steps",          4))
        guidance_scale = float(params.get("guidance_scale", 0.0))

        self._ensure_flux()

        self.logger.info(
            f"[flux] session={session_id[:8]} "
            f"size={width}x{height} steps={steps} guidance={guidance_scale}"
        )

        with torch.no_grad():
            result = self._flux_pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
            )

        img     = result.images[0]
        s3_key  = f"manual_gen/{session_id}/{stage}.png"
        self.upload_image(img, s3_key)
        url     = self.s3_public_url(s3_key)

        torch.cuda.empty_cache()

        mark_done(db, session_id, stage, url, s3_key)
        self.logger.info(f"[flux] done → {url}")

    def _run_sd_stage1(self, task: dict, db) -> None:
        """
        SD1.5 + ControlNet img2img refinement (Stage 1).

        - humanoid: StableDiffusionControlNetImg2ImgPipeline with [openpose, canny]
        - quadruped: StableDiffusionControlNetImg2ImgPipeline with canny only

        Task params: denoise, cfg, steps, openpose_weight, canny_weight, category
        """
        session_id       = task["session_id"]
        stage            = task["stage"]   # "sd_stage1"
        prompt           = task.get("prompt", "")
        negative         = task.get("negative", "")
        params           = task.get("params") or {}
        input_image_url  = task.get("input_image_url", "")

        denoise          = float(params.get("denoise",          0.20))
        cfg              = float(params.get("cfg",              5.5))
        steps            = int(params.get("steps",              20))
        openpose_weight  = float(params.get("openpose_weight",  0.85))
        canny_weight     = float(params.get("canny_weight",     0.55))
        category         = params.get("category",               "humanoid")

        self._ensure_sd()

        # Download init image
        init_img = self._download_image(input_image_url)
        init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

        # Canny control image (always needed)
        canny_img = _extract_canny(init_img)

        self.logger.info(
            f"[sd_stage1] session={session_id[:8]} category={category} "
            f"denoise={denoise} cfg={cfg} steps={steps}"
        )

        with torch.no_grad():
            if category == "humanoid":
                # Load openpose reference (T-pose skeleton image)
                openpose_ref = self._load_openpose_ref()
                result = self._pipe_biped(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=init_img,
                    control_image=[openpose_ref, canny_img],
                    controlnet_conditioning_scale=[openpose_weight, canny_weight],
                    strength=denoise,
                    guidance_scale=cfg,
                    num_inference_steps=steps,
                    num_images_per_prompt=1,
                )
            else:
                # Quadruped: canny only
                result = self._pipe_quad(
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

        img    = result.images[0]
        s3_key = f"manual_gen/{session_id}/{stage}.png"
        self.upload_image(img, s3_key)
        url    = self.s3_public_url(s3_key)

        torch.cuda.empty_cache()

        mark_done(db, session_id, stage, url, s3_key)
        self.logger.info(f"[sd_stage1] done → {url}")

    def _run_sd_stage2(self, task: dict, db) -> None:
        """
        SD1.5 plain img2img polish pass (Stage 2). No ControlNet.

        Task params: denoise, cfg, steps
        """
        session_id      = task["session_id"]
        stage           = task["stage"]   # "sd_stage2"
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_image_url", "")

        denoise = float(params.get("denoise", 0.35))
        cfg     = float(params.get("cfg",     7.0))
        steps   = int(params.get("steps",     20))

        self._ensure_sd()

        init_img = self._download_image(input_image_url)
        init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

        self.logger.info(
            f"[sd_stage2] session={session_id[:8]} "
            f"denoise={denoise} cfg={cfg} steps={steps}"
        )

        with torch.no_grad():
            result = self._pipe_i2i(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                strength=denoise,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=1,
            )

        img    = result.images[0]
        s3_key = f"manual_gen/{session_id}/{stage}.png"
        self.upload_image(img, s3_key)
        url    = self.s3_public_url(s3_key)

        torch.cuda.empty_cache()

        mark_done(db, session_id, stage, url, s3_key)
        self.logger.info(f"[sd_stage2] done → {url}")

    def _run_multiview(self, task: dict, db) -> None:
        """
        Generate a multiview image (side or back) via plain SD1.5 img2img.

        Stage is "multiview_side" or "multiview_back".
        Task params: denoise, cfg, steps
        """
        session_id      = task["session_id"]
        stage           = task["stage"]   # "multiview_side" | "multiview_back"
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_image_url", "")

        denoise = float(params.get("denoise", 0.45))
        cfg     = float(params.get("cfg",     7.0))
        steps   = int(params.get("steps",     20))

        self._ensure_sd()

        init_img = self._download_image(input_image_url)
        init_img = init_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")

        self.logger.info(
            f"[{stage}] session={session_id[:8]} "
            f"denoise={denoise} cfg={cfg} steps={steps}"
        )

        with torch.no_grad():
            result = self._pipe_i2i(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                strength=denoise,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=1,
            )

        img    = result.images[0]
        s3_key = f"manual_gen/{session_id}/{stage}.png"
        self.upload_image(img, s3_key)
        url    = self.s3_public_url(s3_key)

        torch.cuda.empty_cache()

        mark_done(db, session_id, stage, url, s3_key)
        self.logger.info(f"[{stage}] done → {url}")

    def _run_trellis(self, task: dict, r, db) -> None:
        """
        Forward a TRELLIS 3D reconstruction task to the TRELLIS worker queue.

        Does NOT run any GPU inference itself — just translates the manual_gen
        task payload into the format expected by trellis_worker.py and pushes
        it onto the model_tasks Redis queue.

        Marks the trellis stage as "queued" in MongoDB (waiting for TRELLIS worker).

        Task params (expected inside task["params"]):
            front_url  — public S3 URL of front image
            side_url   — public S3 URL of side image
            back_url   — public S3 URL of back image
        """
        session_id = task["session_id"]
        task_id    = task["task_id"]
        stage      = task["stage"]   # "trellis"
        params     = task.get("params") or {}

        front_url  = params.get("front_url",  "")
        side_url   = params.get("side_url",   "")
        back_url   = params.get("back_url",   "")
        output_key = f"manual_gen/{session_id}/trellis.glb"

        if not front_url:
            raise ValueError(
                "trellis task missing required param: front_url. "
                "Ensure sd_stage2 has completed and front_url is populated in params."
            )

        trellis_payload = {
            "task_id":    task_id,
            "session_id": session_id,
            "stage":      "trellis",
            "front_url":  front_url,
            "side_url":   side_url,
            "back_url":   back_url,
            "output_key": output_key,
            "timestamp":  time.time(),
        }

        r.rpush(TRELLIS_QUEUE, json.dumps(trellis_payload))
        self.logger.info(
            f"[trellis] session={session_id[:8]} pushed to {TRELLIS_QUEUE} "
            f"front={front_url!r}"
        )

        # Mark the trellis stage as queued (not done — TRELLIS worker owns done/error)
        mark_queued(db, session_id, stage, task_id)
        self.logger.info(f"[trellis] session={session_id[:8]} stage marked queued.")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _download_image(self, url: str) -> Image.Image:
        """
        Download an image from an S3 public URL (or any HTTP URL).

        Returns a PIL Image in RGB mode.
        Raises requests.HTTPError on non-2xx status.
        """
        if not url:
            raise ValueError("_download_image: url is empty")

        self.logger.debug(f"Downloading image: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        self.logger.debug(f"Downloaded image size: {img.size}")
        return img

    def _load_openpose_ref(self) -> Image.Image:
        """
        Load the T-pose openpose skeleton reference image.

        Priority:
          1. TPOSE_OPENPOSE_PATH env var — local file path to a pre-rendered
             openpose skeleton image (512×512 RGB).
          2. Blank black image fallback (ControlNet will receive no guidance).
        """
        if TPOSE_OPENPOSE_PATH and os.path.isfile(TPOSE_OPENPOSE_PATH):
            try:
                img = Image.open(TPOSE_OPENPOSE_PATH).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                self.logger.debug(f"Loaded T-pose openpose ref: {TPOSE_OPENPOSE_PATH}")
                return img
            except Exception as exc:
                self.logger.warning(
                    f"Failed to load TPOSE_OPENPOSE_PATH={TPOSE_OPENPOSE_PATH!r}: {exc}. "
                    "Using blank fallback."
                )

        self.logger.warning(
            "TPOSE_OPENPOSE_PATH not set or file not found — "
            "using blank black image as openpose reference. "
            "Set TPOSE_OPENPOSE_PATH to a real T-pose skeleton image for better results."
        )
        return _blank_image(IMG_SIZE, IMG_SIZE)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    worker = ManualGenWorker()
    worker.run()
