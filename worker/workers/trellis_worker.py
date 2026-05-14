#!/usr/bin/env python3
"""
TRELLIS.2 3D Worker
===================
Reads from queue : model_tasks  (output of SD1.5 image worker)
Writes to queue  : rig_model    (input of Auto-Rig Pro rigging worker)

3D backend: microsoft/TRELLIS.2-4B
  - 4B parameters, O-Voxel representation
  - Full PBR materials (base color, roughness, metallic)
  - Cleaner topology → better Auto-Rig Pro results
  - 24 GB VRAM (L4 exact fit at 512³ resolution)

Pipeline per character:
  1. Download refined PNG from S3
  2. Remove background (rembg)
  3. TRELLIS.2 image-to-3D → O-Voxel mesh output
  4. Export GLB with PBR materials (WebP textures)
  5. Upload GLB to S3
  6. Update MongoDB  status = model_complete
  7. Push task to rig_model queue

VRAM budget on L4 (24 GB):
  TRELLIS.2-4B at 512³  : ~20-22 GB  ✓ safe
  TRELLIS.2-4B at 1024³ : ~24+ GB    ⚠ may OOM — use 512³ by default

Installation (run install_trellis.sh before starting this worker):
  bash worker/gpu_setup/install_trellis.sh
"""

import os
import sys
import time
import uuid
import tempfile
import logging
import shutil

import torch
from PIL import Image

from .base_worker import BaseWorker
from result_channel import push_running, push_glb_done, push_error

logger = logging.getLogger("TRELLISWorker")

# ── Queue names ───────────────────────────────────────────────────────────────
INPUT_QUEUE  = "model_tasks"
OUTPUT_QUEUE = "rig_model"

# ── Model selection ───────────────────────────────────────────────────────────
TRELLIS_MODEL_ID = os.getenv("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")

# ── Generation config ─────────────────────────────────────────────────────────
# Texture size: 1024 is safe for game assets; 2048 for higher detail if VRAM allows
TRELLIS_TEXTURE_SZ   = int(os.getenv("TRELLIS_TEXTURE_SIZE",   "1024"))
# Decimation target: max triangles in final mesh (1M is plenty for a game character)
TRELLIS_DECIMATION   = int(os.getenv("TRELLIS_DECIMATION",     "1000000"))
# Remesh: True = cleaner quad-dominant topology → better Auto-Rig Pro binding
TRELLIS_REMESH       = os.getenv("TRELLIS_REMESH", "true").lower() == "true"

REMOVE_BG = os.getenv("REMOVE_BG", "true").lower() == "true"

# ── TRELLIS.2 repo path (cloned by install_trellis.sh) ───────────────────────
TRELLIS_REPO_PATH = os.getenv(
    "TRELLIS_REPO_PATH",
    os.path.expanduser("~/trellis"),
)


def _ensure_trellis_in_path():
    """Add TRELLIS.2 repo to sys.path so `trellis2` package can be imported."""
    if TRELLIS_REPO_PATH not in sys.path:
        sys.path.insert(0, TRELLIS_REPO_PATH)


class TRELLISWorker(BaseWorker):
    """TRELLIS.2 image-to-3D worker — produces PBR GLB for game characters."""

    worker_name = "TRELLISWorker"
    input_queue = INPUT_QUEUE

    def __init__(self):
        super().__init__()
        self.pipeline   = None   # Trellis2ImageTo3DPipeline
        self.bg_remover = None   # rembg session

    # ── Model Loading ─────────────────────────────────────────────────────────

    def load_models(self):
        _ensure_trellis_in_path()

        logger.info(f"Loading TRELLIS.2 pipeline: {TRELLIS_MODEL_ID} ...")
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except ImportError:
            raise ImportError(
                "trellis2 package not found. Run: bash worker/gpu_setup/install_trellis.sh"
            )

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_ID)
        self.pipeline.cuda()
        logger.info(f"TRELLIS.2 pipeline loaded on CUDA. Model: {TRELLIS_MODEL_ID}")

        if REMOVE_BG:
            logger.info("Loading rembg background remover...")
            try:
                import rembg
                self.bg_remover = rembg.new_session("u2net")
                logger.info("rembg loaded.")
            except ImportError:
                logger.warning("rembg not installed — background will NOT be removed.")

    # ── Background removal ────────────────────────────────────────────────────

    def _process_manual_gen(self, task: dict, r) -> None:
        """Handle a TRELLIS task originating from the manual-gen pipeline.

        Uses result_channel (Redis) for status updates — no MongoDB writes.
        Input image URL is task["front_url"] (S3 public URL).
        """
        import requests, io as _io
        session_id = task["session_id"]
        stage      = task.get("stage", "trellis")
        front_url  = task.get("front_url", "")
        output_key = task.get("output_key", f"manual_gen/{session_id}/trellis.glb")

        push_running(r, session_id, stage)

        # Download front image
        resp = requests.get(front_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(_io.BytesIO(resp.content)).convert("RGBA")

        if REMOVE_BG:
            image = self._remove_background(image)

        glb_path, glb_tmp = self._run_trellis(image)

        try:
            self.upload_file(glb_path, output_key, "model/gltf-binary")
        finally:
            shutil.rmtree(glb_tmp, ignore_errors=True)

        glb_url = self.s3_public_url(output_key)
        push_glb_done(r, session_id, stage, glb_url, output_key)
        logger.info(f"[manual_gen trellis] session={session_id[:8]} done → {glb_url}")

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Return RGBA image with background removed. Falls back to original if unavailable."""
        if self.bg_remover is None:
            return image.convert("RGBA")
        import rembg
        return rembg.remove(image, session=self.bg_remover)

    # ── TRELLIS.2 inference ───────────────────────────────────────────────────

    def _run_trellis(self, image_rgba: Image.Image) -> tuple[str, str]:
        """
        Run TRELLIS.2 on a single RGBA PIL image.
        Returns (glb_path, tmp_dir). Caller must delete tmp_dir after uploading.

        Output: PBR-ready GLB with WebP-compressed textures.
        """
        import o_voxel

        logger.info(f"Running TRELLIS.2 inference (texture={TRELLIS_TEXTURE_SZ}, "
                    f"decimation={TRELLIS_DECIMATION}, remesh={TRELLIS_REMESH})...")

        with torch.no_grad():
            # TRELLIS.2 run() returns a list — take the first (and only) mesh
            mesh = self.pipeline.run(image_rgba)[0]

        # Export to PBR GLB using o_voxel postprocessor
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
        glb.export(glb_path, extension_webp=True)

        logger.info(f"GLB exported: {glb_path} ({os.path.getsize(glb_path) // 1024} KB)")
        return glb_path, tmp_dir

    # ── Task Processing ───────────────────────────────────────────────────────

    def _process_job_task(self, task: dict, r, db) -> None:
        """Handle tasks from image_worker ({job_id, image_s3_url, biome_id, update_key})
        and the Gradio UI ({job_id, image_path}).
        Both share the same inference + upload + MongoDB flow.
        """
        import requests, io as _io

        job_id     = task.get("job_id", str(uuid.uuid4()))
        biome_id   = task.get("biome_id", job_id)
        update_key = task.get("update_key", "3d_generation_details.model_url")

        # ── Step 1: Resolve image URL / S3 key ───────────────────────────────
        image_url = task.get("image_s3_url") or task.get("image_path", "")
        logger.info(f"[job={job_id[:8]}] Fetching image from: {image_url}")

        with tempfile.TemporaryDirectory() as img_tmp:
            img_local = os.path.join(img_tmp, "input.png")
            if image_url.startswith("http"):
                resp = requests.get(image_url, timeout=60)
                resp.raise_for_status()
                with open(img_local, "wb") as f:
                    f.write(resp.content)
            else:
                # treat as an S3 key
                self.download_file(image_url, img_local)

            image = Image.open(img_local).convert("RGBA")

            # ── Step 2: Remove background ─────────────────────────────────────
            if REMOVE_BG:
                image = self._remove_background(image)
                logger.info(f"[job={job_id[:8]}] Background removed.")

        # ── Step 3: Update status → model_generating ─────────────────────────
        try:
            if db is not None:
                db["biomes"].update_one(
                    {"_id": biome_id},
                    {"$set": {
                        "status": "model_generating",
                        "generation_stage": "trellis2_3d",
                        "trellis_model": TRELLIS_MODEL_ID,
                    }},
                    upsert=True,
                )
        except Exception as e:
            logger.warning(f"[job={job_id[:8]}] MongoDB update skipped: {e}")

        # ── Step 4: TRELLIS.2 3D generation ──────────────────────────────────
        logger.info(f"[job={job_id[:8]}] Running TRELLIS.2 ({TRELLIS_MODEL_ID})...")
        t0 = time.time()
        glb_path, glb_tmp = self._run_trellis(image)
        elapsed = time.time() - t0
        logger.info(f"[job={job_id[:8]}] TRELLIS.2 done in {elapsed:.1f}s → {glb_path}")

        # ── Step 5: Upload GLB to S3 ──────────────────────────────────────────
        glb_key = task.get("output_key") or f"models/{biome_id}/{job_id}.glb"
        self.upload_file(glb_path, glb_key, "model/gltf-binary")
        logger.info(f"[job={job_id[:8]}] Uploaded → s3://{self.cfg.S3_BUCKET}/{glb_key}")
        shutil.rmtree(glb_tmp, ignore_errors=True)

        glb_url = f"https://{self.cfg.S3_BUCKET}.s3.{self.cfg.S3_REGION}.amazonaws.com/{glb_key}"

        # ── Step 6: Update MongoDB → model_complete ───────────────────────────
        try:
            if db is not None:
                db["biomes"].update_one(
                    {"_id": biome_id},
                    {"$set": {
                        "status": "model_complete",
                        "generation_stage": "done",
                        update_key: glb_url,
                        "model_path": glb_key,
                        "model_url": glb_url,
                        "trellis_model": TRELLIS_MODEL_ID,
                        "trellis_elapsed_s": round(elapsed, 1),
                    }},
                    upsert=True,
                )
                logger.info(f"[job={job_id[:8]}] MongoDB updated: {update_key} → {glb_url}")
        except Exception as e:
            logger.warning(f"[job={job_id[:8]}] MongoDB final update skipped: {e}")

        # ── Step 7: Push to rig queue ─────────────────────────────────────────
        self.push_task(OUTPUT_QUEUE, {
            "task_id":    str(uuid.uuid4()),
            "job_id":     job_id,
            "biome_id":   biome_id,
            "model_path": glb_key,
            "model_url":  glb_url,
            "timestamp":  time.time(),
        })
        logger.info(f"[job={job_id[:8]}] Pushed to {OUTPUT_QUEUE} for rigging.")

        torch.cuda.empty_cache()

    def process_task(self, task: dict, r, db) -> None:
        """Route incoming tasks to the correct handler based on task schema.

        Schema 1 — manual_gen:   {session_id, stage, front_url, output_key}
        Schema 2 — image_worker: {job_id, image_s3_url, biome_id, update_key, output_key}
        Schema 3 — Gradio UI:    {job_id, image_path}
        """
        # Manual-gen pipeline (session-based, Redis result channel)
        if "session_id" in task and "biome_id" not in task:
            self._process_manual_gen(task, r)
            return

        # Legacy biome pipeline with character_name (old SD1.5 → TRELLIS V1)
        if "character_name" in task and "biome_id" in task:
            biome_id  = task["biome_id"]
            char_name = task["character_name"]
            char_type = task.get("character_type", "humanoid")
            image_key = task["image_path"]

            self.mongo_update(biome_id, char_name, {
                "status":           "model_generating",
                "generation_stage": "trellis2_3d",
                "trellis_model":    TRELLIS_MODEL_ID,
            })

            with tempfile.TemporaryDirectory() as img_tmp:
                img_local = os.path.join(img_tmp, "input.png")
                self.download_file(image_key, img_local)
                image = Image.open(img_local).convert("RGBA")
                if REMOVE_BG:
                    image = self._remove_background(image)

            t0 = time.time()
            glb_path, glb_tmp = self._run_trellis(image)
            elapsed = time.time() - t0

            glb_key = f"models/{biome_id}/{char_name}.glb"
            self.upload_file(glb_path, glb_key, "model/gltf-binary")
            shutil.rmtree(glb_tmp, ignore_errors=True)

            glb_url = f"https://{self.cfg.S3_BUCKET}.s3.{self.cfg.S3_REGION}.amazonaws.com/{glb_key}"
            self.mongo_update(biome_id, char_name, {
                "status":            "model_complete",
                "generation_stage":  "rig_pending",
                "model_path":        glb_key,
                "model_url":         glb_url,
                "trellis_model":     TRELLIS_MODEL_ID,
                "trellis_elapsed_s": round(elapsed, 1),
            })
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
            return

        # image_worker schema ({job_id, image_s3_url}) OR Gradio schema ({job_id, image_path})
        if "job_id" in task:
            self._process_job_task(task, r, db)
            return

        logger.error(f"Unrecognised task schema — dropping: {list(task.keys())}")
