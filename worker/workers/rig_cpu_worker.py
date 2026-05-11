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
from result_channel import push_running, push_glb_done, push_error
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

        char_type = task.get("char_type") or params.get("char_type", "humanoid")
        params    = {**params, "char_type": char_type}

        push_running(r, session_id, stage)

        with tempfile.TemporaryDirectory() as tmp:
            input_glb  = os.path.join(tmp, "input.glb")
            output_glb = os.path.join(tmp, "output_rigged.glb")

            resp = requests.get(input_glb_url, timeout=60)
            resp.raise_for_status()
            with open(input_glb, "wb") as f:
                f.write(resp.content)

            logger.info(
                f"[rig] session={session_id[:8]}  char_type={char_type}  "
                f"input={os.path.getsize(input_glb)/1e6:.2f} MB"
            )
            run_rig(input_glb, output_glb, params)

            with open(output_glb, "rb") as f:
                glb_bytes = f.read()

        s3_key = f"manual_gen/{session_id}/rig_{char_type}.glb"
        url    = self._upload_bytes(glb_bytes, s3_key, "model/gltf-binary")
        push_glb_done(r, session_id, stage, url, s3_key)
        logger.info(f"[rig] → {url}")
