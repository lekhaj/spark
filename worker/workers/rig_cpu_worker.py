#!/usr/bin/env python3
"""
rig_cpu_worker.py — CPU-side Auto-Rig Pro worker
=================================================
Reads rig tasks from the ``rig_tasks`` Redis queue (queued directly by the
frontend via _q_rig in generation_studio_page.py), runs Blender + ARP
headlessly, and pushes the result back via the standard result_channel.

No GPU or ML dependencies — pure CPU subprocess worker.
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
import tempfile
import time

import requests

from workers.base_worker import BaseWorker
from result_channel import push_running, push_rig_done, push_error
from models.rig_model import run_rig

logger = logging.getLogger("RigCPUWorker")


class RigCPUWorker(BaseWorker):
    worker_name = "RigCPUWorker"
    input_queue = "rig_tasks"

    def load_models(self):
        from models.rig_model import BLENDER_PATH, ARP_SCRIPT_PATH
        if not os.path.isfile(BLENDER_PATH):
            raise FileNotFoundError(
                f"Blender not found at {BLENDER_PATH!r}. Set BLENDER_PATH in .env"
            )
        result = __import__("subprocess").run(
            [BLENDER_PATH, "--version"], capture_output=True, text=True, timeout=15
        )
        ver = (result.stdout + result.stderr).strip().splitlines()[0]
        logger.info(f"Blender OK: {ver}")
        logger.info(f"ARP script: {ARP_SCRIPT_PATH}")

    def run(self):
        logger.info(f"{self.worker_name} starting — queue: {self.input_queue}")
        self.load_models()
        r = self.get_redis()

        while True:
            try:
                raw = r.blpop(self.input_queue, timeout=30)
            except Exception as exc:
                logger.error(f"Redis error: {exc}; reconnecting in 5s")
                time.sleep(5)
                self._redis = None
                continue

            if raw is None:
                continue

            _, payload = raw
            try:
                task = _json.loads(payload)
            except _json.JSONDecodeError:
                logger.error(f"Bad JSON: {payload[:120]}")
                continue

            if self.is_expired(task):
                continue

            try:
                self.process_task(task, r, None)
            except Exception as exc:
                logger.exception(f"Rig task failed: {exc}")
                try:
                    push_error(r, task.get("session_id", ""), task.get("stage", "rig"), str(exc))
                except Exception:
                    pass

    def process_task(self, task: dict, r, db) -> None:
        session_id    = task["session_id"]
        stage         = task.get("stage", "rig")
        params        = task.get("params") or {}
        input_glb_url = task.get("input_glb_url") or task.get("glb_url", "")

        if not input_glb_url:
            raise ValueError("rig task missing input_glb_url")

        char_type  = task.get("char_type") or params.get("char_type", "humanoid")
        morphology = task.get("morphology") or params.get("morphology", "B1_humanoid")
        params     = {**params, "char_type": char_type, "morphology": morphology}

        push_running(r, session_id, stage)

        with tempfile.TemporaryDirectory() as tmp:
            input_glb  = os.path.join(tmp, "input.glb")
            output_glb = os.path.join(tmp, "output_rigged.glb")
            output_fbx = os.path.join(tmp, "output_rigged.fbx")

            resp = requests.get(input_glb_url, timeout=60)
            resp.raise_for_status()
            with open(input_glb, "wb") as f:
                f.write(resp.content)

            logger.info(
                f"[rig] session={session_id[:8]}  char_type={char_type}  "
                f"morphology={morphology}  input={os.path.getsize(input_glb)/1e6:.2f} MB"
            )
            result = run_rig(input_glb, output_glb, params, output_fbx_path=output_fbx)

            char_slug = (task.get("char_name") or "unknown").replace(" ", "_").lower()
            major     = task.get("major", 1)
            minor     = task.get("minor", 0)
            base      = f"chars/{char_slug}/v{major}.{minor}/{char_slug}_{major}_{minor}_rigged"

            glb_key = f"{base}.glb"
            self.upload_file(output_glb, glb_key, "model/gltf-binary")
            glb_url = self.s3_public_url(glb_key)

            fbx_url = fbx_key = ""
            if result.get("fbx"):
                fbx_key = f"{base}.fbx"
                self.upload_file(output_fbx, fbx_key, "application/octet-stream")
                fbx_url = self.s3_public_url(fbx_key)

        push_rig_done(r, session_id, stage, glb_url, glb_key,
                      fbx_url=fbx_url, fbx_s3_key=fbx_key,
                      rig_status=result.get("rig_status", "auto"))
        logger.info(f"[rig] → {glb_url}  rig_status={result.get('rig_status')}"
                    f"{'  + FBX ' + fbx_url if fbx_url else ''}")
