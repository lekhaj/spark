#!/usr/bin/env python3
"""GPU worker — handles all manual_gen pipeline stages: Flux, SD1.5, TRELLIS, Rig."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from workers.base_worker import BaseWorker
from result_channel import (
    push_running,
    push_done,
    push_error,
    push_queued,
    push_glb_done,
)

# ── Flux ──────────────────────────────────────────────────────────────────────
IMG_SIZE            = 512
FLUX_MODEL          = os.getenv("FLUX_MODEL_ID",     "black-forest-labs/FLUX.1-schnell")
SD_MODEL            = os.getenv("SD_MODEL_ID",       "Lykon/DreamShaper")
TPOSE_OPENPOSE_PATH = os.getenv("TPOSE_OPENPOSE_PATH", "")

# ── TRELLIS ───────────────────────────────────────────────────────────────────
TRELLIS_REPO_PATH  = os.getenv("TRELLIS_REPO_PATH", os.path.expanduser("~/trellis"))
TRELLIS_MODEL_ID   = os.getenv("TRELLIS_MODEL_ID",  "microsoft/TRELLIS.2-4B")
TRELLIS_TEXTURE_SZ = int(os.getenv("TRELLIS_TEXTURE_SIZE", "1024"))
TRELLIS_DECIMATION = int(os.getenv("TRELLIS_DECIMATION",   "1000000"))
TRELLIS_REMESH     = os.getenv("TRELLIS_REMESH", "true").lower() == "true"
REMOVE_BG          = os.getenv("REMOVE_BG",     "true").lower() == "true"

def _ensure_trellis_in_path():
    if TRELLIS_REPO_PATH not in sys.path:
        sys.path.insert(0, TRELLIS_REPO_PATH)

# ── Rig (Blender ARP) ─────────────────────────────────────────────────────────
BLENDER_PATH        = os.getenv("BLENDER_PATH", os.path.expanduser("~/blender/blender"))
_THIS_DIR           = os.path.dirname(os.path.abspath(__file__))
ARP_SCRIPT          = os.path.join(os.path.dirname(_THIS_DIR), "blender_scripts", "auto_rig_smart.py")
BLENDER_TIMEOUT_SEC = int(os.getenv("BLENDER_TIMEOUT_SEC", str(10 * 60)))

_CHAR_TYPE_MAP = {
    "humanoid": "humanoid", "bipedal": "humanoid", "human": "humanoid",
    "quadruped": "quadruped", "quad": "quadruped", "animal": "quadruped",
    "bird": "bird", "fish": "fish",
}

def _map_char_type(raw: str) -> str:
    return _CHAR_TYPE_MAP.get((raw or "humanoid").lower(), "humanoid")

logger = logging.getLogger("ManualGenWorker")


def _extract_canny(img: Image.Image) -> Image.Image:
    """Return a 3-channel Canny edge map resized to IMG_SIZE."""
    img_r = img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    gray  = cv2.cvtColor(np.array(img_r), cv2.COLOR_RGB2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 180)
    return Image.fromarray(np.stack([edges] * 3, axis=-1).astype(np.uint8))


def _blank_image(w: int = IMG_SIZE, h: int = IMG_SIZE) -> Image.Image:
    return Image.new("RGB", (w, h), color=(0, 0, 0))


class ManualGenWorker(BaseWorker):
    worker_name = "ManualGenWorker"
    input_queue = "manual_gen_tasks"

    def __init__(self):
        super().__init__()
        # All models are lazy-loaded on first use
        self._flux_pipe    = None
        self._pipe_cn      = None
        self._pipe_biped   = None
        self._pipe_quad    = None
        self._pipe_i2i     = None
        self._cn_canny     = None
        self._cn_openpose  = None
        self._trellis_pipe = None
        self._trellis_rembg = None

    def load_models(self):
        self.logger.info("ManualGenWorker ready — models will load lazily on first use.")

    def run(self, idle_notify_callback=None):
        """
        Override base run() to avoid MongoDB connection.
        ManualGenWorker uses result_channel (Redis) for all status updates;
        MongoDB is written only by the CPU result_consumer.
        """
        import json as _json

        self.logger.info(f"{self.worker_name} starting — queue: {self.input_queue}")
        self.load_models()

        r          = self.get_redis()
        idle_since = time.time()

        while True:
            try:
                raw = r.blpop(self.input_queue, timeout=30)
            except Exception as e:
                self.logger.error(f"Redis error: {e}; reconnecting in 5s")
                time.sleep(5)
                self._redis = None
                continue

            if raw is None:
                if idle_notify_callback:
                    idle_notify_callback(time.time() - idle_since)
                continue

            idle_since = time.time()
            _, payload = raw

            try:
                task = _json.loads(payload)
            except _json.JSONDecodeError:
                self.logger.error(f"Bad JSON in queue: {payload[:120]}")
                continue

            if self.is_expired(task):
                continue

            self.logger.info(
                f"Task → session={task.get('session_id','?')[:8]} "
                f"stage={task.get('stage','?')} char={task.get('char_label','?')}"
            )
            try:
                self.process_task(task, r, None)  # db=None — no MongoDB on GPU
            except Exception as exc:
                self.logger.exception(f"Task failed: {exc}")
                try:
                    push_error(r, task.get("session_id", ""), task.get("stage", ""), str(exc))
                except Exception:
                    pass

    def process_task(self, task: dict, r, db) -> None:
        """Route an incoming task to the correct stage handler."""
        session_id = task.get("session_id", "")
        stage      = task.get("stage", "")
        task_id    = task.get("task_id", "")

        self.logger.info(
            f"[{task_id[:8]}] session={session_id[:8]} stage={stage} "
            f"char={task.get('char_label', '?')}"
        )

        # Signal running via Redis → CPU result_consumer → MongoDB
        push_running(r, session_id, stage)

        try:
            if stage == "flux":
                self._run_flux(task, r)
            elif stage == "sd_stage1":
                self._run_sd_stage1(task, r)
            elif stage == "sd_stage2":
                self._run_sd_stage2(task, r)
            elif stage in ("multiview_side", "multiview_back"):
                self._run_multiview(task, r)
            elif stage == "trellis":
                self._run_trellis(task, r)
            elif stage == "rig":
                self._run_rig(task, r)
            else:
                raise ValueError(f"Unknown stage: '{stage}'")

        except Exception as exc:
            self.logger.exception(
                f"[{task_id[:8]}] stage={stage} FAILED: {exc}"
            )
            push_error(r, session_id, stage, str(exc))

    # ── Lazy model loaders ────────────────────────────────────────────────────

    def _evict_sd(self):
        """Move SD/ControlNet models off VRAM."""
        if self._pipe_cn is None:
            return
        for obj in (self._pipe_cn, self._cn_openpose, self._cn_canny,
                    self._pipe_biped, self._pipe_quad, self._pipe_i2i):
            if obj is not None:
                try:
                    obj.to("cpu")
                except Exception:
                    pass
        torch.cuda.empty_cache()
        self.logger.info("[evict_sd] SD models moved to CPU")

    def _evict_flux(self):
        """Flux uses sequential CPU offload — no VRAM state after inference. Clear cache."""
        torch.cuda.empty_cache()

    def _evict_trellis(self):
        """Move TRELLIS pipeline off VRAM."""
        if self._trellis_pipe is None:
            return
        try:
            self._trellis_pipe.to("cpu")
        except Exception:
            try:
                self._trellis_pipe.cpu()
            except Exception:
                pass
        torch.cuda.empty_cache()
        self.logger.info("[evict_trellis] TRELLIS pipeline moved to CPU")

    def _ensure_flux(self):
        """Load Flux pipeline into GPU memory if not already loaded."""
        if self._flux_pipe is not None:
            return

        self.logger.info(f"Loading Flux model: {FLUX_MODEL}")
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
        # L4 has 22 GB usable VRAM; Flux transformer is ~24 GB in bfloat16 —
        # enable_model_cpu_offload loads the full transformer at once and OOMs.
        # sequential_cpu_offload loads one layer at a time (~few hundred MB peak)
        # which always fits. Slower but the only option without quantization.
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        self._flux_pipe = pipe
        self.logger.info("Flux model loaded (sequential CPU offload).")

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

    def _ensure_trellis(self):
        """Load TRELLIS.2 pipeline onto CUDA if not already loaded."""
        if self._trellis_pipe is not None:
            # Already loaded — ensure it's on GPU
            try:
                self._trellis_pipe.to("cuda")
            except Exception:
                pass
            return

        _ensure_trellis_in_path()
        self.logger.info(f"Loading TRELLIS.2 pipeline: {TRELLIS_MODEL_ID} ...")
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except ImportError:
            raise ImportError(
                "trellis2 package not found. "
                "Run: bash worker/gpu_setup/install_trellis.sh"
            )
        self._trellis_pipe = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_ID)
        self._trellis_pipe.cuda()
        self.logger.info("TRELLIS.2 pipeline loaded on CUDA.")

        if REMOVE_BG:
            try:
                import rembg
                self._trellis_rembg = rembg.new_session("u2net")
                self.logger.info("rembg loaded.")
            except ImportError:
                self.logger.warning("rembg not installed — background will not be removed.")

    # ── Stage handlers ─────────────────────────────────────────────────────────

    def _run_flux(self, task: dict, r) -> None:
        """
        Generate a concept image with Flux.1-schnell (text → image).

        Task params: width, height, steps, guidance_scale
        """
        session_id = task["session_id"]
        stage      = task["stage"]   # "flux"
        prompt     = task.get("prompt", "")
        params     = task.get("params") or {}

        # Clamp to 512×512 max — downstream stages resize to 512 anyway,
        # and 768×1024 activations OOM on L4 24 GB with the Flux transformer.
        width          = min(int(params.get("width",          512)), 512)
        height         = min(int(params.get("height",         512)), 512)
        steps          = int(params.get("steps",          4))
        guidance_scale = float(params.get("guidance_scale", 0.0))

        self._evict_sd()
        self._evict_trellis()
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

        push_done(r, session_id, stage, url, s3_key)
        self.logger.info(f"[flux] done → {url}")

    def _run_sd_stage1(self, task: dict, r) -> None:
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
        input_image_url  = task.get("input_url") or task.get("input_image_url", "")

        denoise          = float(params.get("denoise",          0.20))
        cfg              = float(params.get("cfg",              5.5))
        steps            = int(params.get("steps",              20))
        openpose_weight  = float(params.get("openpose_weight",  0.85))
        canny_weight     = float(params.get("canny_weight",     0.55))
        category         = params.get("category",               "humanoid")

        self._evict_flux()
        self._evict_trellis()
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

        push_done(r, session_id, stage, url, s3_key)
        self.logger.info(f"[sd_stage1] done → {url}")

    def _run_sd_stage2(self, task: dict, r) -> None:
        """
        SD1.5 plain img2img polish pass (Stage 2). No ControlNet.

        Task params: denoise, cfg, steps
        """
        session_id      = task["session_id"]
        stage           = task["stage"]   # "sd_stage2"
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        denoise = float(params.get("denoise", 0.35))
        cfg     = float(params.get("cfg",     7.0))
        steps   = int(params.get("steps",     20))

        self._evict_flux()
        self._evict_trellis()
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

        push_done(r, session_id, stage, url, s3_key)
        self.logger.info(f"[sd_stage2] done → {url}")

    def _run_multiview(self, task: dict, r) -> None:
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
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        denoise = float(params.get("denoise", 0.45))
        cfg     = float(params.get("cfg",     7.0))
        steps   = int(params.get("steps",     20))

        self._evict_flux()
        self._evict_trellis()
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

        push_done(r, session_id, stage, url, s3_key)
        self.logger.info(f"[{stage}] done → {url}")

    def _run_trellis(self, task: dict, r) -> None:
        """Inline TRELLIS.2 3D reconstruction — no external queue forwarding."""
        session_id = task["session_id"]
        stage      = task.get("stage", "trellis")
        params     = task.get("params") or {}

        front_url = task.get("input_front") or params.get("front_url", "")
        if not front_url:
            raise ValueError(
                "trellis stage missing input_front URL. "
                "Ensure sd_stage2 is done and its image URL is passed."
            )

        output_key = f"manual_gen/{session_id}/trellis.glb"

        # Evict SD before loading TRELLIS (~20-22 GB on CUDA)
        self._evict_sd()
        self._evict_flux()
        self._ensure_trellis()

        # Download and prep front image
        resp = requests.get(front_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGBA")

        if REMOVE_BG and self._trellis_rembg is not None:
            import rembg
            image = rembg.remove(image, session=self._trellis_rembg)

        # Composite transparent areas to white, resize to 512×512
        bg = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "RGBA":
            bg.paste(image, mask=image.split()[3])
        else:
            bg.paste(image)
        image_rgb = bg.resize((512, 512))

        self.logger.info(
            f"[trellis] session={session_id[:8]} running TRELLIS.2 "
            f"(texture={TRELLIS_TEXTURE_SZ} decimation={TRELLIS_DECIMATION})"
        )

        import o_voxel
        with torch.no_grad():
            mesh = self._trellis_pipe.run(image_rgb)[0]

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=TRELLIS_DECIMATION,
            texture_size=TRELLIS_TEXTURE_SZ,
            remesh=TRELLIS_REMESH,
        )

        tmp_dir  = tempfile.mkdtemp(prefix="trellis2_")
        glb_path = os.path.join(tmp_dir, "output.glb")
        try:
            glb.export(glb_path, extension_webp=True)
            self.logger.info(f"[trellis] GLB exported ({os.path.getsize(glb_path) // 1024} KB)")
            self.upload_file(glb_path, output_key, "model/gltf-binary")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        glb_url = self.s3_public_url(output_key)
        torch.cuda.empty_cache()

        push_glb_done(r, session_id, stage, glb_url, output_key)
        self.logger.info(f"[trellis] done → {glb_url}")

    def _run_rig(self, task: dict, r) -> None:
        """Blender + Auto-Rig Pro headless rigging. CPU-only — no VRAM needed."""
        session_id    = task["session_id"]
        stage         = task.get("stage", "rig")
        char_type     = task.get("char_type", "humanoid")
        input_glb_url = task.get("input_glb_url") or task.get("glb_url", "")

        if not input_glb_url:
            raise ValueError(
                "rig stage missing input_glb_url. "
                "Ensure trellis stage is done and its GLB URL is passed."
            )

        arp_type   = _map_char_type(char_type)
        output_key = f"manual_gen/{session_id}/rig_{char_type}.glb"

        with tempfile.TemporaryDirectory() as tmp:
            input_glb  = os.path.join(tmp, "input.glb")
            output_glb = os.path.join(tmp, "output_rigged.glb")

            # Download GLB (binary — cannot use PIL)
            resp = requests.get(input_glb_url, timeout=60)
            resp.raise_for_status()
            with open(input_glb, "wb") as f:
                f.write(resp.content)

            self.logger.info(
                f"[rig] session={session_id[:8]} type={arp_type} "
                f"input={os.path.getsize(input_glb)/1e6:.2f} MB"
            )

            cmd = [
                BLENDER_PATH, "--background",
                "--python", ARP_SCRIPT,
                "--",
                "--input",          input_glb,
                "--output",         output_glb,
                "--character_type", arp_type,
            ]

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT_SEC,
            )

            blender_out = proc.stdout + proc.stderr
            for line in blender_out.splitlines():
                if any(tag in line for tag in ("[ARP]", "Error", "error", "WARNING", "RIGGING")):
                    self.logger.info(f"  blender: {line}")

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Blender exited {proc.returncode}.\n"
                    "Last 20 lines:\n" + "\n".join(blender_out.splitlines()[-20:])
                )

            if not os.path.exists(output_glb):
                raise RuntimeError(f"Blender exited OK but output not found: {output_glb}")

            self.logger.info(f"[rig] rigged GLB {os.path.getsize(output_glb)/1e6:.2f} MB")
            self.upload_file(output_glb, output_key, "model/gltf-binary")

        glb_url = self.s3_public_url(output_key)
        push_glb_done(r, session_id, stage, glb_url, output_key)
        self.logger.info(f"[rig] done → {glb_url}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _download_image(self, url: str) -> Image.Image:
        """Download image from URL and return as RGB PIL Image."""
        if not url:
            raise ValueError("_download_image: url is empty")

        self.logger.debug(f"Downloading image: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        self.logger.debug(f"Downloaded image size: {img.size}")
        return img

    def _load_openpose_ref(self) -> Image.Image:
        """Load T-pose openpose reference from TPOSE_OPENPOSE_PATH, or return blank fallback."""
        if TPOSE_OPENPOSE_PATH and os.path.isfile(TPOSE_OPENPOSE_PATH):
            try:
                return Image.open(TPOSE_OPENPOSE_PATH).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            except Exception as exc:
                self.logger.warning(f"Failed to load openpose ref ({TPOSE_OPENPOSE_PATH}): {exc}")
        self.logger.warning("TPOSE_OPENPOSE_PATH not set — using blank openpose reference")
        return _blank_image(IMG_SIZE, IMG_SIZE)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    worker = ManualGenWorker()
    worker.run()
