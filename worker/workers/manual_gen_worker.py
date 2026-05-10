#!/usr/bin/env python3
"""
manual_gen_worker.py — Slim pipeline router
============================================

Receives tasks from the manual_gen_tasks Redis queue and routes each stage
to the appropriate model function.  All model loading and VRAM eviction is
handled by ModelManager.  All inference logic lives in worker/models/.

Stage → model family mapping
-----------------------------
    flux           → flux
    sd_stage1      → sd
    sd_stage2      → sd
    multiview_side → sd
    multiview_back → sd
    trellis        → trellis
    rig            → rig (CPU-only, no VRAM)

This file contains *no* model loading code.  To change inference behaviour,
edit the relevant file in worker/models/.
"""

from __future__ import annotations

import io
import json as _json
import logging
import os
import tempfile
import time

import requests
from PIL import Image

from workers.base_worker import BaseWorker
from result_channel import push_running, push_done, push_error, push_glb_done

# ── Model layer ───────────────────────────────────────────────────────────────
import sys
_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from model_manager import ModelManager
from models.flux_model    import run_flux
from models.sd_model      import run_stage1, run_stage2, run_multiview
from models.trellis_model import run_trellis
from models.rig_model     import run_rig

logger = logging.getLogger("ManualGenWorker")


class ManualGenWorker(BaseWorker):
    worker_name = "ManualGenWorker"
    input_queue = "manual_gen_tasks"

    def __init__(self):
        super().__init__()
        self._mgr = ModelManager()

    def load_models(self):
        logger.info("ManualGenWorker ready — models load lazily on first use.")
        logger.info(self._mgr.vram_summary())

    # ── Queue loop ────────────────────────────────────────────────────────────

    def run(self, idle_notify_callback=None):
        """Poll manual_gen_tasks; route each task to process_task()."""
        logger.info(f"{self.worker_name} starting — queue: {self.input_queue}")
        self.load_models()

        r          = self.get_redis()
        idle_since = time.time()

        while True:
            try:
                raw = r.blpop(self.input_queue, timeout=30)
            except Exception as exc:
                logger.error(f"Redis error: {exc}; reconnecting in 5s")
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
                logger.error(f"Bad JSON in queue: {payload[:120]}")
                continue

            if self.is_expired(task):
                continue

            logger.info(
                f"Task → session={task.get('session_id','?')[:8]}  "
                f"stage={task.get('stage','?')}  char={task.get('char_label','?')}"
            )
            try:
                self.process_task(task, r, None)
            except Exception as exc:
                logger.exception(f"Task failed: {exc}")
                try:
                    push_error(r, task.get("session_id", ""), task.get("stage", ""), str(exc))
                except Exception:
                    pass

    # ── Router ────────────────────────────────────────────────────────────────

    def process_task(self, task: dict, r, db) -> None:
        """Route task → correct model fn.  All VRAM logic is in ModelManager."""
        session_id = task.get("session_id", "")
        stage      = task.get("stage", "")
        task_id    = task.get("task_id", "")

        logger.info(
            f"[{task_id[:8]}] session={session_id[:8]}  stage={stage}  "
            f"char={task.get('char_label','?')}"
        )
        push_running(r, session_id, stage)

        try:
            if stage == "flux":
                self._handle_flux(task, r)

            elif stage == "sd_stage1":
                self._handle_sd_stage1(task, r)

            elif stage == "sd_stage2":
                self._handle_sd_stage2(task, r)

            elif stage in ("multiview_side", "multiview_back"):
                self._handle_multiview(task, r)

            elif stage == "trellis":
                self._handle_trellis(task, r)

            elif stage == "rig":
                self._handle_rig(task, r)

            else:
                raise ValueError(f"Unknown stage: {stage!r}")

        except Exception as exc:
            logger.exception(f"[{task_id[:8]}] stage={stage} FAILED: {exc}")
            push_error(r, session_id, stage, str(exc))

    # ── Stage handlers ────────────────────────────────────────────────────────
    # Each handler: validate inputs → ensure model family → call model fn →
    #               upload result → push done.

    def _handle_flux(self, task: dict, r) -> None:
        session_id = task["session_id"]
        stage      = task["stage"]
        prompt     = task.get("prompt", "")
        params     = task.get("params") or {}

        self._mgr.ensure("flux")
        logger.info(f"[flux] {self._mgr.vram_summary()}")

        img = run_flux(self._mgr.flux_pipe, prompt, params)

        s3_key = f"manual_gen/{session_id}/{stage}.png"
        url    = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[flux] done → {url}")

    def _handle_sd_stage1(self, task: dict, r) -> None:
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)

        self._mgr.ensure("sd")
        logger.info(f"[sd_stage1] {self._mgr.vram_summary()}")

        img = run_stage1(self._mgr.sd_pipes, init_img, prompt, negative, params)

        s3_key = f"manual_gen/{session_id}/{stage}.png"
        url    = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[sd_stage1] done → {url}")

    def _handle_sd_stage2(self, task: dict, r) -> None:
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)

        self._mgr.ensure("sd")
        logger.info(f"[sd_stage2] {self._mgr.vram_summary()}")

        img = run_stage2(self._mgr.sd_pipes, init_img, prompt, negative, params)

        s3_key = f"manual_gen/{session_id}/{stage}.png"
        url    = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[sd_stage2] done → {url}")

    def _handle_multiview(self, task: dict, r) -> None:
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)

        self._mgr.ensure("sd")
        logger.info(f"[{stage}] {self._mgr.vram_summary()}")

        img = run_multiview(self._mgr.sd_pipes, init_img, prompt, negative, params)

        s3_key = f"manual_gen/{session_id}/{stage}.png"
        url    = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[{stage}] done → {url}")

    def _handle_trellis(self, task: dict, r) -> None:
        session_id = task["session_id"]
        stage      = task.get("stage", "trellis")
        params     = task.get("params") or {}

        front_url = task.get("input_front") or params.get("front_url", "")
        if not front_url:
            raise ValueError(
                "trellis stage missing input_front URL. "
                "Ensure sd_stage2 is done and its image URL is passed."
            )

        front_img = self._download_image(front_url)

        self._mgr.ensure("trellis")
        logger.info(f"[trellis] {self._mgr.vram_summary()}")

        glb_bytes = run_trellis(self._mgr.trellis_pipes, front_img, params)

        s3_key = f"manual_gen/{session_id}/trellis.glb"
        url    = self._upload_bytes(glb_bytes, s3_key, content_type="model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[trellis] done → {url}")

    def _handle_rig(self, task: dict, r) -> None:
        session_id    = task["session_id"]
        stage         = task.get("stage", "rig")
        params        = task.get("params") or {}
        input_glb_url = task.get("input_glb_url") or task.get("glb_url", "")

        if not input_glb_url:
            raise ValueError(
                "rig stage missing input_glb_url. "
                "Ensure trellis stage is done and its GLB URL is passed."
            )

        char_type = task.get("char_type") or params.get("char_type", "humanoid")
        params    = {**params, "char_type": char_type}

        with tempfile.TemporaryDirectory() as tmp:
            input_glb  = os.path.join(tmp, "input.glb")
            output_glb = os.path.join(tmp, "output_rigged.glb")

            # Download GLB binary
            resp = requests.get(input_glb_url, timeout=60)
            resp.raise_for_status()
            with open(input_glb, "wb") as f:
                f.write(resp.content)

            logger.info(
                f"[rig] session={session_id[:8]}  char_type={char_type}  "
                f"input={os.path.getsize(input_glb)/1e6:.2f} MB"
            )

            # CPU-only — no ensure() needed
            run_rig(input_glb, output_glb, params)

            with open(output_glb, "rb") as f:
                glb_bytes = f.read()

        s3_key = f"manual_gen/{session_id}/rig_{char_type}.glb"
        url    = self._upload_bytes(glb_bytes, s3_key, content_type="model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[rig] done → {url}")

    # ── Upload helpers ────────────────────────────────────────────────────────

    def _download_image(self, url: str) -> Image.Image:
        if not url:
            raise ValueError("_download_image: url is empty")
        logger.debug(f"Downloading image: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _upload_image(self, img: Image.Image, s3_key: str) -> str:
        """Save PIL Image as PNG, upload to S3, return public URL."""
        self.upload_image(img, s3_key)
        return self.s3_public_url(s3_key)

    def _upload_bytes(self, data: bytes, s3_key: str, content_type: str) -> str:
        """Upload raw bytes to S3, return public URL."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(s3_key)[1]) as f:
            f.write(data)
            tmp_path = f.name
        try:
            self.upload_file(tmp_path, s3_key, content_type)
        finally:
            os.unlink(tmp_path)
        return self.s3_public_url(s3_key)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    worker = ManualGenWorker()
    worker.run()
