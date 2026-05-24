"""
manual_gen_routes.py — REST surface over manual generation pipeline
====================================================================

Exposes the existing ``manual_gen_stage_runs`` / ``manual_gen_tpose_runs`` /
``manual_gen_characters`` MongoDB collections (managed by Gradio today) as
HTTP endpoints, plus thin POST wrappers over the Redis queue.

All write/queue endpoints share the same underlying ``worker.lib.manual_gen_queue``
functions used by the Gradio UI — so a "queue from REST" and "click Queue in
Gradio" produce identical Mongo documents and identical Redis tasks.

Consumed by:
  - spark_studio (Netlify front-end) → reads versions, picks selections,
    queues new runs as the demo iterates
  - any other automation / CLI

Routes mounted at ``/manual-gen`` (set in app/main.py).

Status codes
------------
  200  read/queue success
  400  validation error (bad stage, malformed body)
  404  not found (no such run / character / version)
  409  stage already queued or running (StageBusyError)
  500  unexpected error
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from worker.lib import manual_gen_schema as mgs
from worker.lib import manual_gen_queue as mgq

log = logging.getLogger("manual_gen_routes")

router = APIRouter()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _db():
    """Open a Mongo connection. Cheap — pymongo pools internally."""
    return mgs.get_db()


def _serialize_run(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Strip Mongo-specific keys and return a JSON-safe shape."""
    if not doc:
        return {}
    out = {**doc, "run_id": str(doc.get("_id", ""))}
    out.pop("_id", None)
    return out


def _require_stage(stage: str) -> None:
    if stage not in mgs.STAGE_NAMES:
        raise HTTPException(
            400,
            detail=f"Unknown stage '{stage}'. Valid: {list(mgs.STAGE_NAMES)}",
        )


# ─── Read: characters ─────────────────────────────────────────────────────────

@router.get("/characters", summary="List character labels (filterable + paginated)")
def list_characters(
    prefix: Optional[str] = Query(None, description="Only return labels starting with this prefix (e.g. 'VIC_')"),
    contains: Optional[str] = Query(None, description="Only return labels containing this substring (case-insensitive)"),
    limit: int = Query(200, ge=1, le=2000, description="Max labels to return"),
    offset: int = Query(0, ge=0, description="Number of labels to skip"),
) -> Dict[str, Any]:
    all_labels = mgs.list_characters(_db())
    filtered = all_labels
    if prefix:
        filtered = [l for l in filtered if l.startswith(prefix)]
    if contains:
        needle = contains.lower()
        filtered = [l for l in filtered if needle in l.lower()]
    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {
        "characters": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(page)) < total,
    }


class CreateCharacterBody(BaseModel):
    label: str = Field(..., min_length=1, max_length=128)


@router.post("/characters", summary="Register a character label (idempotent)")
def create_character(body: CreateCharacterBody) -> Dict[str, Any]:
    db = _db()
    ok = mgs.create_character(db, body.label)
    if not ok:
        raise HTTPException(500, "failed to register character")
    return {"label": body.label, "created": True}


@router.get("/characters/{char}/stages",
            summary="List stages with at least one run for a character")
def list_stages_for_char(char: str) -> Dict[str, Any]:
    return {"char": char, "stages": mgs.list_stages_for_char(_db(), char)}


# ─── Read: runs / versions ────────────────────────────────────────────────────

@router.get("/characters/{char}/stages/{stage}/versions",
            summary="List versions for (char, stage) with status + URL (paginated)")
def list_stage_versions(
    char: str,
    stage: str,
    status: Optional[str] = Query(None, description="Filter by status: idle|queued|running|done|error"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    _require_stage(stage)
    db = _db()
    coll = mgs._coll_for_stage(stage)
    query: Dict[str, Any] = {"char_label": char, "stage": stage}
    if status:
        query["status"] = status
    total = db[coll].count_documents(query)
    cursor = (
        db[coll]
        .find(
            query,
            {
                "_id": 1, "major": 1, "minor": 1, "version": 1,
                "status": 1, "image_url": 1, "s3_key": 1,
                "prompt": 1, "created_at": 1, "completed_at": 1, "error": 1,
            },
        )
        .sort([("major", 1), ("minor", 1)])
        .skip(offset)
        .limit(limit)
    )
    versions = [
        {
            "run_id": str(d["_id"]),
            "major": int(d.get("major", 0)),
            "minor": int(d.get("minor", 0)),
            "version": d.get("version") or f"{d.get('major', 0)}.{d.get('minor', 0)}",
            "status": d.get("status", "idle"),
            "image_url": d.get("image_url"),
            "s3_key": d.get("s3_key"),
            "prompt": d.get("prompt", ""),
            "created_at": d.get("created_at"),
            "completed_at": d.get("completed_at"),
            "error": d.get("error"),
        }
        for d in cursor
    ]
    return {
        "char": char,
        "stage": stage,
        "versions": versions,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(versions)) < total,
    }


@router.get("/characters/{char}/stages/{stage}/latest-done",
            summary="URL of the most recent 'done' run for (char, stage)")
def latest_done(char: str, stage: str) -> Dict[str, Any]:
    _require_stage(stage)
    doc = mgs.get_latest_done_run(_db(), char, stage)
    if not doc:
        raise HTTPException(404, f"No 'done' run for {char}/{stage}")
    return _serialize_run(doc)


@router.get("/views/{char}/{stage}/{major}/{minor}",
            summary="Front/side/back view URLs for a specific run")
def get_views(char: str, stage: str, major: int, minor: int) -> Dict[str, Any]:
    _require_stage(stage)
    urls = mgs.get_view_urls(_db(), char, stage, major, minor)
    return {"char": char, "stage": stage, "version": f"{major}.{minor}", "urls": urls}


@router.get("/runs/{run_id}", summary="Get a single run document by ID")
def get_run(run_id: str) -> Dict[str, Any]:
    doc = mgs.get_run_any(_db(), run_id)
    if not doc:
        raise HTTPException(404, f"No run with id {run_id}")
    return _serialize_run(doc)


class UpdateRunBody(BaseModel):
    fields: Dict[str, Any] = Field(default_factory=dict)


@router.patch("/runs/{run_id}", summary="Partial update of a run document")
def patch_run(run_id: str, body: UpdateRunBody) -> Dict[str, Any]:
    db = _db()
    doc = mgs.get_run_any(db, run_id)
    if not doc:
        raise HTTPException(404, f"No run with id {run_id}")
    coll = mgs._coll_for_stage(doc.get("stage", ""))
    mgs.update_run(db, run_id, body.fields, coll=coll)
    return _serialize_run(mgs.get_run_any(db, run_id) or {})


@router.delete("/runs/{run_id}", summary="Delete a run document")
def delete_run(run_id: str) -> Dict[str, Any]:
    db = _db()
    doc = mgs.get_run_any(db, run_id)
    if not doc:
        raise HTTPException(404, f"No run with id {run_id}")
    coll = mgs._coll_for_stage(doc.get("stage", ""))
    db[coll].delete_one({"_id": run_id})
    return {"run_id": run_id, "deleted": True}


# ─── Queue: stage-specific POST endpoints ─────────────────────────────────────
# These wrap the pure functions in worker/lib/manual_gen_queue.py and produce
# the same Mongo + Redis state as the Gradio UI's _q_* handlers.

class QueueFluxBody(BaseModel):
    char: str
    major: int = 1
    minor: int = 0
    prompt: str
    negative: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 4.0
    queue: Optional[str] = None  # override active queue if needed


@router.post("/queue/flux", summary="Queue a FLUX text-to-image generation")
def queue_flux(body: QueueFluxBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_flux(
            _db(),
            char_label=body.char,
            major=body.major,
            minor=body.minor,
            prompt=body.prompt,
            negative=body.negative,
            width=body.width,
            height=body.height,
            steps=body.steps,
            guidance=body.guidance,
            queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_flux failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


class QueueHunyuanBody(BaseModel):
    char: str
    major: int = 1
    minor: int = 0
    char_type: str = "humanoid"
    src_stage: str = "flux"
    src_ver: str = ""
    queue: Optional[str] = None


@router.post("/queue/hunyuan3d", summary="Queue a Hunyuan3D image→3D job")
def queue_hunyuan3d(body: QueueHunyuanBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_hunyuan3d(
            _db(),
            char_label=body.char,
            major=body.major,
            minor=body.minor,
            char_type=body.char_type,
            src_stage=body.src_stage,
            src_ver=body.src_ver,
            queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_hunyuan3d failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


class QueueFluxPoseBody(BaseModel):
    char: str
    major: int = 1
    minor: int = 0
    prompt: str
    negative: str = ""
    use_control: bool = True
    control_mode: str = "pose"
    auto_extract: bool = True
    control_image_url: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 3.5
    true_cfg_scale: float = 1.0
    cn_scale: float = 0.85
    cn_start: float = 0.0
    cn_end: float = 0.85
    seed: int = 0
    src_stage: str = "flux"
    src_ver: str = ""
    morphology: str = "B1_humanoid"
    view: str = "primary"
    queue: Optional[str] = None


@router.post("/queue/flux_pose", summary="Queue FLUX.1-dev + ControlNet-Union-Pro pose")
def queue_flux_pose(body: QueueFluxPoseBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_flux_pose(
            _db(),
            char_label=body.char, major=body.major, minor=body.minor,
            prompt=body.prompt, negative=body.negative,
            use_control=body.use_control, control_mode=body.control_mode,
            auto_extract=body.auto_extract, control_image_url=body.control_image_url,
            width=body.width, height=body.height, steps=body.steps,
            guidance_scale=body.guidance_scale, true_cfg_scale=body.true_cfg_scale,
            cn_scale=body.cn_scale, cn_start=body.cn_start, cn_end=body.cn_end,
            seed=body.seed, src_stage=body.src_stage, src_ver=body.src_ver,
            morphology=body.morphology, view=body.view, queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_flux_pose failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


class QueueSDTposeBody(BaseModel):
    char: str
    major: int = 1
    minor: int = 0
    prompt: str
    negative: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    src_stage: str = "flux"
    src_ver: str = ""
    queue: Optional[str] = None


@router.post("/queue/sd_tpose", summary="Queue SD1.5 + ControlNet T-pose")
def queue_sd_tpose(body: QueueSDTposeBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_sd_tpose(
            _db(),
            char_label=body.char, major=body.major, minor=body.minor,
            prompt=body.prompt, negative=body.negative, params=body.params,
            src_stage=body.src_stage, src_ver=body.src_ver, queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_sd_tpose failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


class Queue3DBody(BaseModel):
    """Shared body for trellis / pixal3d (front-image-only image→3D)."""
    char: str
    major: int = 1
    minor: int = 0
    char_type: str = "humanoid"
    src_stage: str = "flux"
    src_ver: str = ""
    queue: Optional[str] = None


@router.post("/queue/trellis", summary="Queue TRELLIS image→3D")
def queue_trellis(body: Queue3DBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_trellis(
            _db(),
            char_label=body.char, major=body.major, minor=body.minor,
            char_type=body.char_type, src_stage=body.src_stage,
            src_ver=body.src_ver, queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_trellis failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


@router.post("/queue/pixal3d", summary="Queue Pixal3D image→3D")
def queue_pixal3d(body: Queue3DBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_pixal3d(
            _db(),
            char_label=body.char, major=body.major, minor=body.minor,
            char_type=body.char_type, src_stage=body.src_stage,
            src_ver=body.src_ver, queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_pixal3d failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


class QueueMeshLodBody(BaseModel):
    char: str
    major: int = 1
    minor: int = 0
    src_stage: str = "trellis"
    src_ver: str = ""
    profiles: list = Field(default_factory=lambda: ["a", "b", "c", "d"])
    ratio_b: float = 0.5
    ratio_c: float = 0.2
    voxel_size_d: float = 0.02
    quadriflow: bool = False
    quadriflow_target: int = 4000
    bake_resolution: int = 1024
    queue: Optional[str] = None


@router.post("/queue/mesh_lod", summary="Queue Mesh-LOD (CPU queue)")
def queue_mesh_lod(body: QueueMeshLodBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_mesh_lod(
            _db(),
            char_label=body.char, major=body.major, minor=body.minor,
            src_stage=body.src_stage, src_ver=body.src_ver,
            profiles=body.profiles, ratio_b=body.ratio_b, ratio_c=body.ratio_c,
            voxel_size_d=body.voxel_size_d, quadriflow=body.quadriflow,
            quadriflow_target=body.quadriflow_target,
            bake_resolution=body.bake_resolution, queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_mesh_lod failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


class QueueRigBody(BaseModel):
    char: str
    major: int = 1
    minor: int = 0
    char_type: str = "humanoid"
    trellis_src_ver: str = ""
    queue: Optional[str] = None


@router.post("/queue/rig", summary="Queue an Auto-Rig Pro rigging job")
def queue_rig(body: QueueRigBody) -> Dict[str, Any]:
    try:
        result = mgq.queue_rig(
            _db(),
            char_label=body.char,
            major=body.major,
            minor=body.minor,
            char_type=body.char_type,
            trellis_src_ver=body.trellis_src_ver,
            queue=body.queue,
        )
    except mgq.StageBusyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("queue_rig failed")
        raise HTTPException(500, str(e))
    return result.to_dict()


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", summary="Mongo + Redis reachability check")
def health() -> Dict[str, Any]:
    out: Dict[str, Any] = {"mongo": "unknown", "redis": "unknown"}
    try:
        _db().command("ping")
        out["mongo"] = "ok"
    except Exception as e:
        out["mongo"] = f"error: {e}"
    try:
        mgq._redis_client().ping()
        out["redis"] = "ok"
    except Exception as e:
        out["redis"] = f"error: {e}"
    out["queue"] = mgq.resolve_active_queue()
    out["stages"] = list(mgs.STAGE_NAMES)
    return out
