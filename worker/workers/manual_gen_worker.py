#!/usr/bin/env python3
"""
manual_gen_worker.py — Registry-based pipeline router
======================================================

Receives tasks from the ``manual_gen_tasks`` Redis queue and routes each task
to the correct handler via STAGE_REGISTRY.

How to add a new stage / experiment
-------------------------------------
1. Write ``worker/models/your_model.py`` with ``load_xxx()`` + ``run_xxx()``.
2. If it's a new model *family* (not reusing flux/sd/trellis):
     - Add it to ModelManager via ``mgr.register(...)`` in the
       ``_register_custom_families()`` method below.
3. Add a handler method ``_handle_xxx(self, task, r)`` to this class.
4. Add one line to ``STAGE_REGISTRY``:
     "your_stage_name": ("model_family", ManualGenWorker._handle_xxx),

That's it.  The queue loop, VRAM eviction, status tracking, and error
handling all work automatically.

STAGE_REGISTRY format
---------------------
    { stage_name: (model_family, handler_method) }

    stage_name    — matches task["stage"] from the Redis payload
    model_family  — passed to mgr.ensure(family) before handler is called
                    use None for CPU-only stages (no VRAM management)
    handler_method — unbound method on ManualGenWorker
"""

from __future__ import annotations

import io
import json as _json
import logging
import os
import sys
import tempfile
import time

import requests
from PIL import Image

from workers.base_worker import BaseWorker
from result_channel import push_running, push_done, push_error, push_glb_done

# ── Ensure worker root is importable ─────────────────────────────────────────
_WORKER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_ROOT not in sys.path:
    sys.path.insert(0, _WORKER_ROOT)

from model_manager import ModelManager
from models.flux_model    import run_flux
from models.sd_model      import run_stage1, run_stage2, run_multiview
from models.trellis_model import run_trellis
from models.rig_model     import run_rig

logger = logging.getLogger("ManualGenWorker")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# One entry per stage name.  Format:
#   "stage_name": ("model_family", ManualGenWorker._handle_method)
#
# To add a new stage add ONE line here and ONE handler method below.
# Nothing else in this file needs to change.
# ─────────────────────────────────────────────────────────────────────────────

STAGE_REGISTRY: dict[str, tuple[str | None, str]] = {
    # stage name          model family   handler method name
    "flux":               ("flux",       "_handle_flux"),
    "sd_stage1":          ("sd",         "_handle_sd_stage1"),
    "sd_stage2":          ("sd",         "_handle_sd_stage2"),
    "multiview_side":     ("sd",         "_handle_multiview"),
    "multiview_back":     ("sd",         "_handle_multiview"),
    "trellis":            ("trellis",    "_handle_trellis"),
    "rig":                (None,         "_handle_rig"),          # CPU-only
    # ── Add new experiments here ────────────────────────────────────────────
    # "controlnet_pre":  ("sd",         "_handle_controlnet_pre"),
    # "sdxl_stage1":     ("sdxl",       "_handle_sdxl_stage1"),
    # "animatediff":     ("animatediff","_handle_animatediff"),
}


class ManualGenWorker(BaseWorker):
    worker_name = "ManualGenWorker"
    input_queue = "manual_gen_tasks"

    def __init__(self):
        super().__init__()
        self._mgr = ModelManager()
        self._register_custom_families()

    # ── Custom family registration ────────────────────────────────────────────
    # When you add a new model family that isn't flux/sd/trellis/rig:
    #   self._mgr.register("sdxl", lambda: load_sdxl(...), evicts={"flux","trellis"})

    def _register_custom_families(self) -> None:
        pass  # nothing custom yet

    # ── Startup ───────────────────────────────────────────────────────────────

    def load_models(self):
        logger.info(
            f"ManualGenWorker ready — {len(STAGE_REGISTRY)} stages registered, "
            "models load lazily on first use."
        )
        logger.info(self._mgr.vram_summary())

    # ── Queue loop ────────────────────────────────────────────────────────────

    def run(self, idle_notify_callback=None, active_callback=None, done_callback=None):
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

            if active_callback:
                active_callback()

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
            finally:
                if done_callback:
                    done_callback()

    # ── Router ────────────────────────────────────────────────────────────────

    def process_task(self, task: dict, r, db) -> None:
        """
        Route task to the correct handler via STAGE_REGISTRY.

        Steps:
          1. Look up stage in STAGE_REGISTRY → (family, handler_name)
          2. If family is not None → mgr.ensure(family)   [loads + evicts]
          3. Call handler(task, r)
          4. Handler uploads result and pushes done/error via result_channel
        """
        session_id = task.get("session_id", "")
        stage      = task.get("stage", "")
        task_id    = task.get("task_id", "")

        logger.info(
            f"[{task_id[:8]}] session={session_id[:8]}  "
            f"stage={stage}  char={task.get('char_label','?')}"
        )
        push_running(r, session_id, stage)

        # ── Registry lookup ───────────────────────────────────────────────────
        entry = STAGE_REGISTRY.get(stage)
        if entry is None:
            err = (
                f"Unknown stage: {stage!r}. "
                f"Registered stages: {sorted(STAGE_REGISTRY)}"
            )
            logger.error(err)
            push_error(r, session_id, stage, err)
            return

        family, handler_name = entry

        # ── VRAM: load model, evict incompatible families ─────────────────────
        if family is not None:
            try:
                self._mgr.ensure(family)
                logger.info(f"  {self._mgr.vram_summary()}")
            except Exception as exc:
                push_error(r, session_id, stage, f"Model load failed: {exc}")
                raise

        # ── Dispatch to handler ───────────────────────────────────────────────
        handler = getattr(self, handler_name, None)
        if handler is None:
            err = f"Handler {handler_name!r} not found on {self.__class__.__name__}"
            logger.error(err)
            push_error(r, session_id, stage, err)
            return

        try:
            handler(task, r)
        except Exception as exc:
            logger.exception(f"[{task_id[:8]}] stage={stage} FAILED: {exc}")
            push_error(r, session_id, stage, str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE HANDLERS
    # Each handler:
    #   1. Reads what it needs from task (prompt, params, input URLs, …)
    #   2. Calls the model function (already loaded by process_task)
    #   3. Uploads result → push_done / push_glb_done
    #
    # Handlers do NOT call mgr.ensure() — process_task handles that.
    # To add a handler: add a method here + one line in STAGE_REGISTRY above.
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_flux(self, task: dict, r) -> None:
        session_id = task["session_id"]
        stage      = task["stage"]
        prompt     = task.get("prompt", "")
        params     = task.get("params") or {}

        img    = run_flux(self._mgr.get("flux"), prompt, params)
        s3_key = f"manual_gen/{session_id}/{stage}.png"
        url    = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[flux] → {url}")

    def _handle_sd_stage1(self, task: dict, r) -> None:
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)
        img      = run_stage1(self._mgr.get("sd"), init_img, prompt, negative, params)
        s3_key   = f"manual_gen/{session_id}/{stage}.png"
        url      = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[sd_stage1] → {url}")

    def _handle_sd_stage2(self, task: dict, r) -> None:
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)
        img      = run_stage2(self._mgr.get("sd"), init_img, prompt, negative, params)
        s3_key   = f"manual_gen/{session_id}/{stage}.png"
        url      = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[sd_stage2] → {url}")

    def _handle_multiview(self, task: dict, r) -> None:
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)
        img      = run_multiview(self._mgr.get("sd"), init_img, prompt, negative, params)
        s3_key   = f"manual_gen/{session_id}/{stage}.png"
        url      = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[{stage}] → {url}")

    def _handle_trellis(self, task: dict, r) -> None:
        session_id = task["session_id"]
        stage      = task.get("stage", "trellis")
        params     = task.get("params") or {}

        front_url = task.get("input_front") or params.get("front_url", "")
        if not front_url:
            raise ValueError(
                "trellis task missing input_front URL — "
                "ensure sd_stage2 is done and its URL is passed."
            )

        front_img = self._download_image(front_url)

        # Optional side/back views — use multi-image pipeline if provided
        side_img = back_img = None
        side_url = task.get("input_side") or params.get("side_url", "")
        back_url = task.get("input_back") or params.get("back_url", "")
        if side_url:
            try:
                side_img = self._download_image(side_url)
            except Exception as exc:
                logger.warning(f"[trellis] side image download failed ({exc}) — front-only mode")
        if back_url:
            try:
                back_img = self._download_image(back_url)
            except Exception as exc:
                logger.warning(f"[trellis] back image download failed ({exc}) — skipping back view")

        glb_bytes = run_trellis(
            self._mgr.get("trellis"), front_img, params,
            side_image=side_img, back_image=back_img,
        )
        s3_key = f"manual_gen/{session_id}/trellis.glb"
        url    = self._upload_bytes(glb_bytes, s3_key, "model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[trellis] → {url}")

    def _handle_rig(self, task: dict, r) -> None:
        session_id    = task["session_id"]
        stage         = task.get("stage", "rig")
        params        = task.get("params") or {}
        input_glb_url = task.get("input_glb_url") or task.get("glb_url", "")

        if not input_glb_url:
            raise ValueError(
                "rig task missing input_glb_url — "
                "ensure trellis stage is done and its GLB URL is passed."
            )

        char_type = task.get("char_type") or params.get("char_type", "humanoid")
        params    = {**params, "char_type": char_type}

        with tempfile.TemporaryDirectory() as tmp:
            input_glb  = os.path.join(tmp, "input.glb")
            output_glb = os.path.join(tmp, "output_rigged.glb")

            resp = requests.get(input_glb_url, timeout=60)
            resp.raise_for_status()
            with open(input_glb, "wb") as f:
                f.write(resp.content)

            logger.info(
                f"[rig] session={task['session_id'][:8]}  char_type={char_type}  "
                f"input={os.path.getsize(input_glb)/1e6:.2f} MB"
            )
            run_rig(input_glb, output_glb, params)

            with open(output_glb, "rb") as f:
                glb_bytes = f.read()

        s3_key = f"manual_gen/{session_id}/rig_{char_type}.glb"
        url    = self._upload_bytes(glb_bytes, s3_key, "model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[rig] → {url}")

    # ─────────────────────────────────────────────────────────────────────────
    # Upload helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _download_image(self, url: str) -> Image.Image:
        if not url:
            raise ValueError("_download_image: url is empty")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _upload_image(self, img: Image.Image, s3_key: str) -> str:
        self.upload_image(img, s3_key)
        return self.s3_public_url(s3_key)

    def _upload_bytes(self, data: bytes, s3_key: str, content_type: str) -> str:
        suffix = os.path.splitext(s3_key)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
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
