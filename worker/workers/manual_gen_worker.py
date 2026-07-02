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

        # ── GPU busy heartbeat ────────────────────────────────────────────────
        # Single source of truth for "this GPU is doing work right now", read by
        # both the GPU-side auto_shutdown and the CPU orchestrator so neither
        # stops a box mid-task — even during a long pixal3d/hunyuan3d subprocess
        # that runs with the Redis queue already empty. A daemon thread keeps the
        # heartbeat fresh while a task is in flight.
        import threading as _threading
        from lib import gpu_heartbeat
        try:
            from workers.auto_shutdown import _imds_instance_id as _iid_fn
        except Exception:  # pragma: no cover
            _iid_fn = lambda: None  # noqa: E731
        self._hb_iid = os.getenv("AWS_GPU_INSTANCE_ID", "").strip() or _iid_fn()
        self._hb_processing = False

        def _heartbeat_loop():
            while True:
                if getattr(self, "_hb_processing", False):
                    try:
                        gpu_heartbeat.touch(self.get_redis(), self._hb_iid)
                    except Exception:
                        pass
                time.sleep(30)

        _threading.Thread(target=_heartbeat_loop, name="GpuHeartbeat", daemon=True).start()

        # Spot-reclaim safety: blpop atomically removes the task, so if AWS
        # reclaims this (spot) box mid-job, systemd sends SIGTERM and the
        # in-flight task would be lost. Catch SIGTERM/SIGINT and push the
        # current task back to the FRONT of the queue so it's retried first
        # when a worker (spot-relaunched or the on-demand) comes back.
        import signal
        self._current_task = None
        # P2-2a lookahead flag: set per-task in the loop; default False = evict.
        self._keep_stage_warm = False

        def _on_term(signum, _frame):
            t = getattr(self, "_current_task", None)
            if t is not None:
                try:
                    self.get_redis().lpush(self.input_queue, _json.dumps(t))
                    logger.warning(
                        f"Signal {signum} — re-queued in-flight task "
                        f"(stage={t.get('stage','?')}) to front of {self.input_queue}"
                    )
                except Exception as exc:
                    logger.error(f"Failed to re-queue task on signal {signum}: {exc}")
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)

        # Stage affinity: after a task, prefer the next QUEUED task of the same
        # stage so that stage's model/server stays warm (one load per batch, not
        # per character). Reorders the pull only — every task still runs. Reset to
        # None on idle so a fresh batch starts FIFO.
        from lib.stage_affinity import pop_next_task, peek_has_stage
        last_stage: Optional[str] = None

        while True:
            try:
                payload = pop_next_task(r, self.input_queue, last_stage, timeout=30)
            except Exception as exc:
                logger.error(f"Redis error: {exc}; reconnecting in 5s")
                time.sleep(5)
                self._redis = None
                r = self.get_redis()
                continue

            if payload is None:
                last_stage = None  # queue drained — next batch starts FIFO
                if idle_notify_callback:
                    idle_notify_callback(time.time() - idle_since)
                continue

            idle_since = time.time()

            try:
                task = _json.loads(payload)
            except _json.JSONDecodeError:
                logger.error(f"Bad JSON in queue: {payload[:120]}")
                continue

            # Remember this stage so the next pull prefers the same one (warm model).
            last_stage = task.get("stage") or None

            # Lookahead (P2-2a): is another task of THIS stage still queued? If so,
            # handlers keep that stage's model/server resident (one load per batch);
            # if not, they evict at the group boundary. Defaults to False (evict) on
            # any error — the safe, pre-existing per-task behavior.
            self._keep_stage_warm = peek_has_stage(r, self.input_queue, last_stage)

            if self.is_expired(task):
                continue

            if active_callback:
                active_callback()

            # Mark busy + stamp the heartbeat immediately on pop so the stop
            # paths see this box as working before the (possibly long) handler
            # even starts; the heartbeat thread then keeps it fresh.
            self._hb_processing = True
            try:
                gpu_heartbeat.touch(r, self._hb_iid)
            except Exception:
                pass

            # Track the in-flight task so a SIGTERM (spot reclaim) can re-queue it.
            self._current_task = task

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
                self._current_task = None   # completed — no longer re-queueable
                self._hb_processing = False
                # Final stamp so the "just finished" moment is recorded; the idle
                # window is measured from here by the stop paths.
                try:
                    gpu_heartbeat.touch(r, self._hb_iid)
                except Exception:
                    pass
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
        morphology          = (params.get("morphology") or "B1_humanoid").strip()
        view                = (params.get("view") or "primary").strip()

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
            # No URL, no source image → fall back to the bundled proxy for the
            # requested morphology + view. Lets a user enable "Use ControlNet
            # conditioning" with no extra inputs and get a guaranteed pose lock.
            mode = (params.get("control_mode") or "").lower()
            bundled = self._bundled_control_image(mode, morphology=morphology, view=view)
            if bundled is None:
                raise ValueError(
                    "flux_pose: use_control=True but no control_image_url, no source "
                    f"image, and no bundled fallback for mode={mode!r} "
                    f"morphology={morphology!r} view={view!r}. Provide one."
                )
            from PIL import Image
            control_img = Image.open(bundled).convert("RGB")
            logger.info(
                f"[flux_pose] bundled control: {bundled}  "
                f"(morphology={morphology}, view={view}, mode={mode})"
            )

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
        # VRAM hygiene (P2-2a): only evict trellis at the GROUP boundary. With
        # stage-affinity grouping, consecutive characters' trellis tasks run
        # back-to-back; keeping trellis resident across them loads it once per
        # batch instead of once per character. If no more trellis tasks are
        # queued (or on any lookahead error → flag False), evict now so the next
        # stage (pixal3d/hunyuan3d subprocess) gets clean headroom.
        if getattr(self, "_keep_stage_warm", False):
            logger.info("[trellis] more trellis queued — keeping model warm")
        else:
            try:
                self._mgr.evict("trellis")
            except Exception:
                pass
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

        # Fully unload in-process models — the subprocess loads its own multi-GB
        # models in a separate process, so the in-process families must release
        # host RAM (not just VRAM) or the box OOM-kills the worker.
        for fam in ("flux", "flux_cn", "sd", "trellis"):
            try:
                self._mgr.release(fam)
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

        # queue_hunyuan3d sends the source image as 'input_url' (the other 3D
        # stages use 'input_front'); accept either so the contract is forgiving.
        front_url = (task.get("input_front") or task.get("input_url")
                     or params.get("front_url", ""))
        if not front_url:
            raise ValueError("hunyuan3d task missing input_front/input_url URL")

        front_img = self._download_image(front_url)

        # Fully unload in-process models — the subprocess loads its own multi-GB
        # models in a separate process, so the in-process families must release
        # host RAM (not just VRAM) or the box OOM-kills the worker.
        for fam in ("flux", "flux_cn", "sd", "trellis"):
            try:
                self._mgr.release(fam)
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
    # can do pose-locked generation without any S3 fetch.
    #
    # B1_humanoid → apose_canonical.png — COCO-18 OpenPose A-pose skeleton
    #   (see worker/controlnet_refs/generate_apose.py). Used with mode="pose".
    #
    # B2..B7 → <morphology>_<view>.png — white-line skeletons rendered by
    #   worker/controlnet_refs/proxy_generators.py. Designed for mode="soft_edge"
    #   since OpenPose's "pose" mode is humanoid-only. See bake_proxies.py.
    #
    # tpose_canonical.png is retained for callers that explicitly want it
    # (curated S3 preset or hand-supplied control_image_url).
    _BUNDLED_HUMANOID = {
        "pose": "apose_canonical.png",
    }

    def _bundled_control_image(
        self, mode: str,
        morphology: str = "B1_humanoid",
        view: str = "primary",
    ) -> str | None:
        """
        Resolve the bundled control image path for a (mode, morphology, view) tuple.

        B1_humanoid is the legacy humanoid path — keyed by ControlNet mode.
        B2..B7 use morphology-specific proxies regardless of ``mode`` (since
        only ``soft_edge`` makes sense for non-humanoid line skeletons).
        Returns None if nothing applicable is bundled.
        """
        refs_dir = os.path.join(_WORKER_ROOT, "controlnet_refs")

        if morphology == "B1_humanoid":
            fname = self._BUNDLED_HUMANOID.get((mode or "").lower())
            if not fname:
                return None
            path = os.path.join(refs_dir, fname)
            return path if os.path.exists(path) else None

        # Non-humanoid morphologies — look up <morphology>_<view>.png
        fname = f"{morphology}_{view or 'primary'}.png"
        path = os.path.join(refs_dir, fname)
        if os.path.exists(path):
            return path
        # Fallback: try the primary view if the requested view is missing.
        primary = os.path.join(refs_dir, f"{morphology}_primary.png")
        return primary if os.path.exists(primary) else None

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
