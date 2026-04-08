#!/usr/bin/env python3
"""
TRELLIS 3D Worker  (Hunyuan3D-2.1 backend)
==========================================
Reads from queue: model_tasks
Input : S3 key of a refined PNG character image (output of SD1.5 worker)
Output: GLB 3D model uploaded to S3 → pushes to rig_model queue

Pipeline per character:
  1. Download image from S3
  2. Remove background (rembg)
  3. Run Hunyuan3D-2.1 shape generation → .glb
  4. (Optional) PBR texture pass
  5. Upload GLB to S3
  6. Update MongoDB  status = model_complete
  7. Push to rig_model queue for UniRig

VRAM budget on L4 (24 GB):
  Hunyuan3D shape model : ~12 GB fp16
  VAE / decoder         :  ~4 GB
  Peak activations      :  ~4 GB
  Total                 : ~20 GB  → safe with flashVDM + low_vram_mode
"""

import os
import time
import uuid
import tempfile
import logging

import torch
from PIL import Image

from .base_worker import BaseWorker, WorkerConfig

logger = logging.getLogger("TRELLISWorker")

# ── Queue names ───────────────────────────────────────────────────────────────
INPUT_QUEUE  = "model_tasks"   # SD1.5 worker pushes here
OUTPUT_QUEUE = "rig_model"     # UniRig worker reads from here

# ── Hunyuan3D Config ──────────────────────────────────────────────────────────
HY3D_MODEL_PATH = os.getenv(
    "HY3D_MODEL_PATH",
    os.path.expanduser("~/.cache/hy3dgen/tencent/Hunyuan3D-2.1"),
)
HY3D_STEPS       = int(os.getenv("HY3D_STEPS",       "30"))
HY3D_GUIDANCE    = float(os.getenv("HY3D_GUIDANCE",  "7.5"))
HY3D_OCTREE_RES  = int(os.getenv("HY3D_OCTREE_RES",  "256"))
HY3D_NUM_CHUNKS  = int(os.getenv("HY3D_NUM_CHUNKS",  "200000"))
HY3D_WITH_TEX    = os.getenv("HY3D_WITH_TEXTURE", "false").lower() == "true"
HY3D_LOW_VRAM    = os.getenv("HY3D_LOW_VRAM", "true").lower()  == "true"
REMOVE_BG        = os.getenv("REMOVE_BG", "true").lower() == "true"


class TRELLISWorker(BaseWorker):
    """Hunyuan3D-2.1 3D generation worker."""

    worker_name = "TRELLISWorker"
    input_queue = INPUT_QUEUE

    def __init__(self):
        super().__init__()
        self.pipeline    = None   # Hunyuan3D shape pipeline
        self.tex_pipeline = None  # PBR texture pipeline (optional)
        self.bg_remover   = None  # rembg session

    # ── Model Loading ─────────────────────────────────────────────────────────

    def load_models(self):
        logger.info("Loading Hunyuan3D-2.1 shape generation model...")
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HY3D_MODEL_PATH,
                torch_dtype=torch.float16,
                low_vram_mode=HY3D_LOW_VRAM,
                enable_flashvdm=True,
            )
            self.pipeline.to("cuda")
            logger.info("Hunyuan3D shape model loaded on CUDA.")
        except ImportError:
            logger.error(
                "hy3dgen not installed. Run: pip install git+https://github.com/tencent/Hunyuan3D-2"
            )
            raise

        if HY3D_WITH_TEX:
            logger.info("Loading PBR texture pipeline...")
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                self.tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(HY3D_MODEL_PATH)
                self.tex_pipeline.to("cuda")
                logger.info("Texture pipeline loaded.")
            except Exception as e:
                logger.warning(f"Texture pipeline failed to load (skipping): {e}")

        if REMOVE_BG:
            logger.info("Loading background remover (rembg)...")
            try:
                import rembg
                self.bg_remover = rembg.new_session("u2net")
                logger.info("rembg loaded.")
            except ImportError:
                logger.warning("rembg not installed — background will NOT be removed.")

    # ── Task Processing ───────────────────────────────────────────────────────

    def process_task(self, task: dict, r, db) -> None:
        biome_id   = task["biome_id"]
        char_name  = task["character_name"]
        char_type  = task.get("character_type", "humanoid")
        image_key  = task["image_path"]   # S3 key of refined PNG

        self.mongo_update(biome_id, char_name, {
            "status": "model_generating",
            "generation_stage": "trellis_3d",
        })

        # ── Step 1: Download image from S3 ───────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            img_local = os.path.join(tmp, "input.png")
            self.download_file(image_key, img_local)
            image = Image.open(img_local).convert("RGBA")

            # ── Step 2: Remove background ─────────────────────────────────────
            if REMOVE_BG and self.bg_remover is not None:
                import rembg
                image = rembg.remove(image, session=self.bg_remover)
                logger.info(f"[{char_name}] Background removed.")

            # Convert to RGB for shape gen (white bg for transparent areas)
            bg = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                bg.paste(image, mask=image.split()[3])
            else:
                bg.paste(image)
            image_rgb = bg

            # ── Step 3: Run Hunyuan3D shape generation ────────────────────────
            logger.info(f"[{char_name}] Running Hunyuan3D shape gen...")
            with torch.no_grad():
                mesh_output = self.pipeline(
                    image=image_rgb,
                    num_inference_steps=HY3D_STEPS,
                    guidance_scale=HY3D_GUIDANCE,
                    octree_resolution=HY3D_OCTREE_RES,
                    num_chunks=HY3D_NUM_CHUNKS,
                )

            # ── Step 4: Export GLB ────────────────────────────────────────────
            glb_local = os.path.join(tmp, f"{char_name}.glb")
            mesh_output.export(glb_local)
            logger.info(f"[{char_name}] GLB exported: {glb_local}")

            # ── Step 5: Optional PBR texture ──────────────────────────────────
            tex_key = None
            if HY3D_WITH_TEX and self.tex_pipeline is not None:
                logger.info(f"[{char_name}] Generating PBR textures...")
                try:
                    textured_local = os.path.join(tmp, f"{char_name}_textured.glb")
                    self.tex_pipeline(
                        mesh=mesh_output,
                        image=image_rgb,
                    ).export(textured_local)
                    tex_key = f"models/{biome_id}/{char_name}_textured.glb"
                    self.upload_file(textured_local, tex_key, "model/gltf-binary")
                    logger.info(f"[{char_name}] Textured GLB → s3://{self.cfg.S3_BUCKET}/{tex_key}")
                    glb_local = textured_local  # use textured as final
                except Exception as e:
                    logger.warning(f"[{char_name}] Texture pass failed (using untextured): {e}")

            # ── Step 6: Upload base GLB ───────────────────────────────────────
            glb_key = f"models/{biome_id}/{char_name}.glb"
            self.upload_file(glb_local, glb_key, "model/gltf-binary")
            logger.info(f"[{char_name}] GLB → s3://{self.cfg.S3_BUCKET}/{glb_key}")

        # ── Step 7: Update MongoDB ────────────────────────────────────────────
        mongo_fields = {
            "status":           "model_complete",
            "generation_stage": "rig_pending",
            "model_path":       glb_key,
        }
        if tex_key:
            mongo_fields["model_path_textured"] = tex_key
        self.mongo_update(biome_id, char_name, mongo_fields)

        # ── Step 8: Push to rig queue ─────────────────────────────────────────
        self.push_task(OUTPUT_QUEUE, {
            "task_id":        str(uuid.uuid4()),
            "biome_id":       biome_id,
            "character_name": char_name,
            "character_type": char_type,
            "model_path":     glb_key,
            "timestamp":      time.time(),
        })
        logger.info(f"[{char_name}] Pushed to {OUTPUT_QUEUE} for rigging.")

        # Free GPU memory
        torch.cuda.empty_cache()
