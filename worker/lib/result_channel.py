"""
result_channel.py — GPU-side Redis result push helpers
=======================================================
GPU workers call these instead of writing to MongoDB directly.
The CPU-side result_consumer.py reads and writes to MongoDB.

This keeps all MongoDB writes on the CPU, GPU only needs Redis + S3.

Queue: spark_results
Schema: {"session_id", "stage", "status", "ts", ...extra fields}
"""
from __future__ import annotations

import json
import time

RESULT_QUEUE = "spark_results"


def _push(r, session_id: str, stage: str, status: str, **kwargs) -> None:
    payload = {
        "session_id": session_id,
        "stage":      stage,
        "status":     status,
        "ts":         time.time(),
        **kwargs,
    }
    r.rpush(RESULT_QUEUE, json.dumps(payload))


def push_running(r, session_id: str, stage: str) -> None:
    """GPU worker started processing a task."""
    _push(r, session_id, stage, "running")


def push_done(r, session_id: str, stage: str, image_url: str, s3_key: str,
              view: str = "front") -> None:
    """GPU worker finished — image uploaded to S3.
    view: 'front' (default), 'side', or 'back'.
    For front view, image_url is stored as the primary image_url for backward compat.
    For side/back, the view field signals the consumer to store to side_url/back_url.
    """
    _push(r, session_id, stage, "done", image_url=image_url, s3_key=s3_key, view=view)


def push_glb_done(r, session_id: str, stage: str, glb_url: str, s3_key: str) -> None:
    """GPU worker finished a 3D stage — GLB uploaded to S3."""
    _push(r, session_id, stage, "done", glb_url=glb_url, s3_key=s3_key)


def push_rig_done(r, session_id: str, stage: str, glb_url: str, s3_key: str,
                  fbx_url: str = "", fbx_s3_key: str = "",
                  rig_status: str = "auto") -> None:
    """Rig worker finished — clean deform-only GLB + engine FBX uploaded.
    rig_status is 'auto' (skeleton bound) or 'manual' (auto-rig fell back; mesh
    shipped unrigged for an artist to finish)."""
    _push(r, session_id, stage, "done", glb_url=glb_url, s3_key=s3_key,
          fbx_url=fbx_url, fbx_s3_key=fbx_s3_key, rig_status=rig_status)


def push_error(r, session_id: str, stage: str, error: str) -> None:
    """GPU worker hit an error."""
    _push(r, session_id, stage, "error", error=error)


def push_queued(r, session_id: str, stage: str, task_id: str) -> None:
    """A worker forwarded this stage to another worker (e.g. manual→trellis)."""
    _push(r, session_id, stage, "queued", task_id=task_id)
