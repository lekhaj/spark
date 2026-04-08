#!/usr/bin/env python3
"""
TRELLIS 3D Worker
=================
Reads from queue : model_tasks  (output of SD1.5 image worker)
Writes to queue  : rig_model    (input of UniRig rigging worker)

3D backend: microsoft/TRELLIS-image-large  (default, 16 GB VRAM)
            microsoft/TRELLIS.2-4B          (set TRELLIS_MODEL_ID env var, 24 GB)

TRELLIS.2-4B is preferred for game character generation — it produces
cleaner topology and supports arbitrary non-manifold geometry that is
required for proper character rigging.

Pipeline per character:
  1. Download refined PNG from S3
  2. Remove background (rembg)
  3. TRELLIS image-to-3D → Gaussian + Mesh outputs
  4. Export fused GLB (geometry + baked texture)
  5. Upload GLB to S3
  6. Update MongoDB  status = model_complete
  7. Push task to rig_model queue

VRAM budget on L4 (24 GB):
  TRELLIS-image-large  : ~16 GB fp16
  TRELLIS.2-4B         : ~22 GB fp16
  Both run safely on the L4.

Installation (run install_trellis.sh before starting this worker):
  bash worker/gpu_setup/install_trellis.sh
"""

import os
import sys
import time
import uuid
import tempfile
import logging

import torch
from PIL import Image

from .base_worker import BaseWorker

logger = logging.getLogger("TRELLISWorker")

# ── Queue names ───────────────────────────────────────────────────────────────
INPUT_QUEUE  = "model_tasks"
OUTPUT_QUEUE = "rig_model"

# ── Model selection ───────────────────────────────────────────────────────────
# Override via env: TRELLIS_MODEL_ID=microsoft/TRELLIS.2-4B
TRELLIS_MODEL_ID = os.getenv(
    "TRELLIS_MODEL_ID",
    "microsoft/TRELLIS-image-large",   # default — 16 GB, well-tested
)

# ── Generation config ─────────────────────────────────────────────────────────
TRELLIS_SEED        = int(os.getenv("TRELLIS_SEED",        "42"))
TRELLIS_STEPS_SPARSE = int(os.getenv("TRELLIS_STEPS_SPARSE", "12"))
TRELLIS_STEPS_DENSE  = int(os.getenv("TRELLIS_STEPS_DENSE",  "12"))
TRELLIS_CFG_STRENGTH = float(os.getenv("TRELLIS_CFG_STRENGTH", "7.5"))
# Mesh simplification ratio (0.95 = keep 95% triangles — good for game use)
TRELLIS_SIMPLIFY    = float(os.getenv("TRELLIS_SIMPLIFY",   "0.95"))
TRELLIS_TEXTURE_SZ  = int(os.getenv("TRELLIS_TEXTURE_SIZE", "1024"))

REMOVE_BG           = os.getenv("REMOVE_BG", "true").lower() == "true"

# ── TRELLIS repo path (cloned by install_trellis.sh) ─────────────────────────
TRELLIS_REPO_PATH = os.getenv(
    "TRELLIS_REPO_PATH",
    os.path.expanduser("~/trellis"),
)


def _ensure_trellis_in_path():
    """Add TRELLIS repo to sys.path so `trellis` package can be imported."""
    if TRELLIS_REPO_PATH not in sys.path:
        sys.path.insert(0, TRELLIS_REPO_PATH)


class TRELLISWorker(BaseWorker):
    """TRELLIS image-to-3D worker — produces rig-ready GLB for game characters."""

    worker_name = "TRELLISWorker"
    input_queue = INPUT_QUEUE

    def __init__(self):
        super().__init__()
        self.pipeline    = None   # TrellisImageTo3DPipeline
        self.bg_remover  = None   # rembg session

    # ── Model Loading ─────────────────────────────────────────────────────────

    def load_models(self):
        _ensure_trellis_in_path()

        logger.info(f"Loading TRELLIS pipeline: {TRELLIS_MODEL_ID} ...")
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
        except ImportError:
            raise ImportError(
                "TRELLIS not found. Run: bash worker/gpu_setup/install_trellis.sh"
            )

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_ID)
        self.pipeline.cuda()
        logger.info(f"TRELLIS pipeline loaded on CUDA. Model: {TRELLIS_MODEL_ID}")

        if REMOVE_BG:
            logger.info("Loading rembg background remover...")
            try:
                import rembg
                self.bg_remover = rembg.new_session("u2net")
                logger.info("rembg loaded.")
            except ImportError:
                logger.warning("rembg not installed — background will NOT be removed.")

    # ── Background removal ────────────────────────────────────────────────────

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Return RGBA image with background removed. Falls back to original if unavailable."""
        if self.bg_remover is None:
            return image.convert("RGBA")
        import rembg
        return rembg.remove(image, session=self.bg_remover)

    # ── TRELLIS inference ──────────────────────────────────────────────────────

    def _run_trellis(self, image_rgba: Image.Image) -> str:
        """
        Run TRELLIS on a single RGBA PIL image.
        Returns path to exported GLB in a temp directory.

        The caller must delete the temp dir after uploading.
        """
        from trellis.utils import postprocessing_utils

        # TRELLIS expects RGB PIL (white-composited for transparent areas)
        bg = Image.new("RGB", image_rgba.size, (255, 255, 255))
        if image_rgba.mode == "RGBA":
            bg.paste(image_rgba, mask=image_rgba.split()[3])
        else:
            bg.paste(image_rgba)
        image_rgb = bg.resize((512, 512))

        with torch.no_grad():
            outputs = self.pipeline.run(
                image_rgb,
                seed=TRELLIS_SEED,
                formats=["gaussian", "mesh"],
                preprocess_image=False,   # we handle bg removal ourselves
                sparse_structure_sampler_params={
                    "steps": TRELLIS_STEPS_SPARSE,
                    "cfg_strength": TRELLIS_CFG_STRENGTH,
                },
                slat_sampler_params={
                    "steps": TRELLIS_STEPS_DENSE,
                    "cfg_strength": TRELLIS_CFG_STRENGTH,
                },
            )

        # Fuse Gaussian splatting + mesh → textured GLB
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=TRELLIS_SIMPLIFY,
            texture_size=TRELLIS_TEXTURE_SZ,
            verbose=False,
        )

        # Export to temp file
        tmp_dir  = tempfile.mkdtemp(prefix="trellis_")
        glb_path = os.path.join(tmp_dir, "output.glb")
        glb.export(glb_path)
        return glb_path, tmp_dir

    # ── Task Processing ───────────────────────────────────────────────────────

    def process_task(self, task: dict, r, db) -> None:
        biome_id  = task["biome_id"]
        char_name = task["character_name"]
        char_type = task.get("character_type", "humanoid")
        image_key = task["image_path"]        # S3 key from SD1.5 worker

        self.mongo_update(biome_id, char_name, {
            "status":           "model_generating",
            "generation_stage": "trellis_3d",
        })

        with tempfile.TemporaryDirectory() as img_tmp:
            # ── Step 1: Download image from S3 ───────────────────────────────
            img_local = os.path.join(img_tmp, "input.png")
            self.download_file(image_key, img_local)
            image = Image.open(img_local).convert("RGBA")
            logger.info(f"[{char_name}] Image downloaded: {image_key}")

            # ── Step 2: Remove background ─────────────────────────────────────
            if REMOVE_BG:
                image = self._remove_background(image)
                logger.info(f"[{char_name}] Background removed.")

        # ── Step 3: TRELLIS 3D generation ────────────────────────────────────
        logger.info(f"[{char_name}] Running TRELLIS ({TRELLIS_MODEL_ID})...")
        glb_path, glb_tmp = self._run_trellis(image)
        logger.info(f"[{char_name}] GLB exported: {glb_path}")

        # ── Step 4: Upload GLB to S3 ──────────────────────────────────────────
        glb_key = f"models/{biome_id}/{char_name}.glb"
        self.upload_file(glb_path, glb_key, "model/gltf-binary")
        logger.info(f"[{char_name}] GLB → s3://{self.cfg.S3_BUCKET}/{glb_key}")

        # Cleanup temp dir
        import shutil
        shutil.rmtree(glb_tmp, ignore_errors=True)

        # ── Step 5: Update MongoDB ────────────────────────────────────────────
        self.mongo_update(biome_id, char_name, {
            "status":           "model_complete",
            "generation_stage": "rig_pending",
            "model_path":       glb_key,
            "trellis_model":    TRELLIS_MODEL_ID,
        })

        # ── Step 6: Push to rig queue ─────────────────────────────────────────
        self.push_task(OUTPUT_QUEUE, {
            "task_id":        str(uuid.uuid4()),
            "biome_id":       biome_id,
            "character_name": char_name,
            "character_type": char_type,
            "model_path":     glb_key,
            "timestamp":      time.time(),
        })
        logger.info(f"[{char_name}] Pushed to {OUTPUT_QUEUE} for rigging.")

        torch.cuda.empty_cache()
