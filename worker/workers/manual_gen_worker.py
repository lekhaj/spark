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
from models.flux_model import run_flux
# NOTE: sd_model and trellis_model are imported lazily inside handlers.
# This prevents flash-attention version errors (diffusers.loaders.ip_adapter
# enforces flash-attn <=2.7.4) from crashing the worker at startup when
# only flux tasks are needed.

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

def _char_s3_key(task: dict, stage: str, ext: str) -> str:
    """Build a unique, human-readable S3 key.
    Path:  chars/{char}/v{major}.{minor}/{char}_{major}_{minor}_{stage}.{ext}
    """
    char = (task.get("char_name") or "unknown").replace(" ", "_").lower()
    major = task.get("major", 1)
    minor = task.get("minor", 0)
    filename = f"{char}_{major}_{minor}_{stage}.{ext}"
    return f"chars/{char}/v{major}.{minor}/{filename}"


STAGE_REGISTRY: dict[str, tuple[str | None, str]] = {
    # stage name          model family   handler method name
    "flux":               ("flux",       "_handle_flux"),
    "flux_pose":          ("flux_cn",    "_handle_flux_pose"),
    "sd_stage1":          ("sd",         "_handle_sd_stage1"),
    "sd_stage2":          ("sd",         "_handle_sd_stage2"),
    "multiview_side":     ("sd",         "_handle_multiview"),
    "multiview_back":     ("sd",         "_handle_multiview"),
    "trellis":            ("trellis",    "_handle_trellis"),
    "pixal3d":            (None,         "_handle_pixal3d"),    # subprocess — isolated conda env
    "hunyuan3d":          (None,         "_handle_hunyuan3d"),  # subprocess — isolated conda env
    # "rig" tasks go directly to rig_tasks queue from the frontend — not routed here
    # ── Add new experiments here ────────────────────────────────────────────
    # "controlnet_pre":  ("sd",         "_handle_controlnet_pre"),
    # "sdxl_stage1":     ("sdxl",       "_handle_sdxl_stage1"),
    # "animatediff":     ("animatediff","_handle_animatediff"),
}


class ManualGenWorker(BaseWorker):
    worker_name = "ManualGenWorker"
    # Queue name is env-configurable so a second GPU (e.g. spot test instance)
    # can listen on its own queue without competing with the prod worker.
    input_queue = os.getenv("MANUAL_GEN_QUEUE", "manual_gen_tasks")

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
        s3_key = _char_s3_key(task, stage, "png")
        url    = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[flux] → {url}")

    def _handle_flux_pose(self, task: dict, r) -> None:
        """
        FLUX.1-dev + ControlNet-Union-Pro-2.0 — pose-conditioned image gen.

        Task params (all optional except input_url OR control_image_url):
          input_url             — source image (used for auto-extraction)
          control_image_url     — pre-built control image (bypasses extraction)
          prompt, negative      — text conditioning
          params:
            control_mode        — canny | soft_edge | depth | blur | pose |
                                  gray | low_quality
            auto_extract        — if True (default), build control image from
                                  input_url using control_mode; else require
                                  control_image_url
            controlnet_conditioning_scale, control_guidance_start,
            control_guidance_end, steps, guidance_scale, true_cfg_scale,
            width, height, seed
        """
        from models.flux_cn_model import run_flux_cn

        session_id          = task["session_id"]
        stage               = task["stage"]
        prompt              = task.get("prompt", "")
        negative            = task.get("negative", "")
        params              = dict(task.get("params") or {})
        input_url           = task.get("input_url") or task.get("input_image_url", "")
        control_image_url   = task.get("control_image_url") or params.get("control_image_url", "")
        use_control         = bool(params.get("use_control", True))
        auto_extract        = bool(params.get("auto_extract", True))

        source_img  = None
        control_img = None

        if not use_control:
            # Pure text-to-image — force ControlNet off
            params["controlnet_conditioning_scale"] = 0.0
            # run_flux_cn will synthesize a blank control image
        elif control_image_url:
            control_img = self._download_image(control_image_url)
        elif auto_extract and input_url:
            source_img = self._download_image(input_url)
        else:
            # No URL, no source image → fall back to the bundled T-pose skeleton
            # that ships in the repo (worker/controlnet_refs/). Lets a user enable
            # "Use ControlNet conditioning" → mode=pose with no extra inputs and
            # get a guaranteed T-pose lock right away.
            mode = (params.get("control_mode") or "").lower()
            bundled = self._bundled_control_image(mode)
            if bundled is None:
                raise ValueError(
                    "flux_pose: use_control=True but no control_image_url, no source "
                    f"image, and no bundled fallback for mode={mode!r}. Provide one."
                )
            from PIL import Image
            control_img = Image.open(bundled).convert("RGB")
            logger.info(f"[flux_pose] using bundled control image: {bundled}")

        img = run_flux_cn(
            self._mgr.get("flux_cn"),
            prompt,
            params,
            source_image=source_img,
            control_image=control_img,
            negative_prompt=negative,
        )

        s3_key = _char_s3_key(task, stage, "png")
        url    = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[flux_pose] → {url}")

    def _handle_sd_stage1(self, task: dict, r) -> None:
        from models.sd_model import run_stage1  # lazy — avoids flash-attn check at startup
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)
        img      = run_stage1(self._mgr.get("sd"), init_img, prompt, negative, params)
        s3_key   = _char_s3_key(task, stage, "png")
        url      = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[sd_stage1] → {url}")

    def _handle_sd_stage2(self, task: dict, r) -> None:
        from models.sd_model import run_stage2  # lazy — avoids flash-attn check at startup
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)
        img      = run_stage2(self._mgr.get("sd"), init_img, prompt, negative, params)
        s3_key   = _char_s3_key(task, stage, "png")
        url      = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[sd_stage2] → {url}")

    def _handle_multiview(self, task: dict, r) -> None:
        from models.sd_model import run_multiview  # lazy — avoids flash-attn check at startup
        session_id      = task["session_id"]
        stage           = task["stage"]
        prompt          = task.get("prompt", "")
        negative        = task.get("negative", "")
        params          = task.get("params") or {}
        input_image_url = task.get("input_url") or task.get("input_image_url", "")

        init_img = self._download_image(input_image_url)
        img      = run_multiview(self._mgr.get("sd"), init_img, prompt, negative, params)
        s3_key   = _char_s3_key(task, stage, "png")
        url      = self._upload_image(img, s3_key)
        push_done(r, session_id, stage, url, s3_key)
        logger.info(f"[{stage}] → {url}")

    def _handle_trellis(self, task: dict, r) -> None:
        from models.trellis_model import run_trellis  # lazy — avoids flash-attn check at startup
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
        s3_key = _char_s3_key(task, "trellis", "glb")
        url    = self._upload_bytes(glb_bytes, s3_key, "model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[trellis] → {url}")

    def _handle_pixal3d(self, task: dict, r) -> None:
        """
        Pixal3D image→3D via subprocess in isolated conda env.

        Evicts ALL in-process VRAM models first (pixal3d subprocess needs
        most of the 23GB on L4 even with low_vram=True).
        """
        from models.pixal3d_runner import run_pixal3d

        session_id = task["session_id"]
        stage      = task.get("stage", "pixal3d")
        params     = task.get("params") or {}

        front_url = task.get("input_front") or params.get("front_url", "")
        if not front_url:
            raise ValueError("pixal3d task missing input_front URL")

        front_img = self._download_image(front_url)

        # Free VRAM for the subprocess
        for fam in ("flux", "sd", "trellis"):
            try:
                self._mgr.evict(fam)
            except Exception:
                pass
        logger.info(f"[pixal3d] pre-subprocess  {self._mgr.vram_summary()}")

        glb_bytes = run_pixal3d(front_img, params)
        s3_key    = _char_s3_key(task, "pixal3d", "glb")
        url       = self._upload_bytes(glb_bytes, s3_key, "model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[pixal3d] → {url}")

    def _handle_hunyuan3d(self, task: dict, r) -> None:
        """
        Hunyuan3D-2.0 image→3D via subprocess in isolated conda env.

        Task params (all optional):
          input_front             — source image URL
          params:
            seed (int)            42
            steps (int)           50     shape diffusion steps
            guidance_scale (float) 7.5
            texture_resolution (int) 2048
        """
        from models.hunyuan3d_model import run_hunyuan3d

        session_id = task["session_id"]
        stage      = task.get("stage", "hunyuan3d")
        params     = task.get("params") or {}

        front_url = task.get("input_front") or params.get("front_url", "")
        if not front_url:
            raise ValueError("hunyuan3d task missing input_front URL")

        front_img = self._download_image(front_url)

        # Evict in-process models — subprocess needs the full 48 GB on L40S
        for fam in ("flux", "flux_cn", "sd", "trellis"):
            try:
                self._mgr.evict(fam)
            except Exception:
                pass
        logger.info(f"[hunyuan3d] pre-subprocess  {self._mgr.vram_summary()}")

        glb_bytes = run_hunyuan3d(front_img, params)
        s3_key    = _char_s3_key(task, "hunyuan3d", "glb")
        url       = self._upload_bytes(glb_bytes, s3_key, "model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[hunyuan3d] → {url}")


    # ─────────────────────────────────────────────────────────────────────────
    # Upload helpers
    # ─────────────────────────────────────────────────────────────────────────

    # Bundled control images live in the repo so a freshly-launched spot
    # can do pose-locked generation without any S3 fetch. tpose_canonical.png
    # is generated programmatically from COCO-18 keypoint coordinates with the
    # standard OpenPose color scheme — anatomically exact (perfectly horizontal
    # arms, shoulder-width stance) and reproducible (see generate_tpose.py).
    # This is what an OpenPose detector would emit for a real photo of a person
    # in T-pose, so a Union Pro ControlNet trained on OpenPose-format
    # annotations should respond to it most reliably.
    _BUNDLED_CONTROL = {
        "pose": "tpose_canonical.png",
    }

    def _bundled_control_image(self, mode: str) -> str | None:
        fname = self._BUNDLED_CONTROL.get(mode)
        if not fname:
            return None
        path = os.path.join(_WORKER_ROOT, "controlnet_refs", fname)
        return path if os.path.exists(path) else None

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
