"""
asset_run_routes.py — asset_spec → GPU pipeline bridge (CycleZero U05/T13)
==========================================================================

An accepted ``asset_spec`` run becomes a generation job: image (flux) →
3D (hunyuan3d) → rigged GLB (rig), reusing the existing manual-gen pipeline.
The asset run tracks the three stages and, when the rig lands, writes the
``manifest_entry`` the engine's asset registry loads. Data-only by design —
a changed asset never requires engine code.

Routes mounted at ``/asset-runs`` (set in app/main.py).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pymongo
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from worker.lib import manual_gen_schema as mgs

log = logging.getLogger("asset_run_routes")

router = APIRouter()

COLLECTION = "asset_runs"
SPEC_RUNS = "spec_gen_runs"

# asset-run stage name → manual-gen pipeline stage
STAGE_MAP = {"image": "flux", "model3d": "hunyuan3d", "rigged": "rig"}


def _db():
    """Open a Mongo connection. Overridden in tests (mongomock)."""
    return mgs.get_db()


def _submit_image_job(db, asset_id: str, prompt: str, major: int, minor: int) -> str:
    """Queue the first pipeline stage (flux). Patched out in tests."""
    from worker.lib import manual_gen_queue as mgq

    result = mgq.queue_flux(db, char_label=asset_id, major=major, minor=minor, prompt=prompt)
    return result.task_id


class AssetRunCreate(BaseModel):
    spec_run_id: str = Field(min_length=1)


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {**doc}
    out.pop("_id", None)
    for k in ("created_at", "completed_at"):
        if isinstance(out.get(k), datetime):
            out[k] = out[k].isoformat()
    return out


@router.post("")
def create_asset_run(body: AssetRunCreate) -> Dict[str, Any]:
    db = _db()
    spec = db[SPEC_RUNS].find_one({"run_id": body.spec_run_id})
    if not spec:
        raise HTTPException(404, f"no such spec run: {body.spec_run_id}")
    if spec["stage"] != "asset_spec":
        raise HTTPException(409, f"asset runs need an asset_spec run (stage={spec['stage']})")
    if spec["status"] != "accepted":
        raise HTTPException(409, f"asset runs need an ACCEPTED asset_spec run (status={spec['status']})")

    output = spec["output"]
    asset_id = output["asset_id"]
    doc = {
        "asset_run_id": uuid.uuid4().hex,
        "project_id":   spec["project_id"],
        "asset_id":     asset_id,
        "spec_run_id":  body.spec_run_id,
        "spec_version": f"{spec['major']}.{spec['minor']}",
        "status":       "generating",
        "stages": {
            "image":   {"status": "queued",  "url": None},
            "model3d": {"status": "pending", "url": None},
            "rigged":  {"status": "pending", "url": None},
        },
        "manifest_entry": None,
        "created_at":   datetime.now(timezone.utc),
        "completed_at": None,
    }
    try:
        doc["task_id"] = _submit_image_job(
            db, asset_id, output["generation_prompt"], spec["major"], spec["minor"]
        )
    except Exception as e:
        log.exception("manual-gen submit failed for %s", asset_id)
        raise HTTPException(502, f"manual-gen submit failed: {e}")
    db[COLLECTION].insert_one(doc)
    return _serialize(doc)


def _refresh(db, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Fill stage statuses/urls from the manual-gen pipeline's run records."""
    if doc["status"] == "complete":
        return doc
    changed = False
    for slot, mg_stage in STAGE_MAP.items():
        run = db[mgs.COLLECTION].find_one(
            {"char_label": doc["asset_id"], "stage": mg_stage,
             "created_at": {"$gte": doc["created_at"]}},
            sort=[("created_at", pymongo.DESCENDING)],
        )
        if not run:
            continue
        status = run.get("status") or "pending"
        url = run.get("image_url")
        if doc["stages"][slot]["status"] != status or doc["stages"][slot]["url"] != url:
            doc["stages"][slot] = {"status": status, "url": url}
            changed = True
    if doc["stages"]["rigged"]["status"] == "done" and not doc.get("manifest_entry"):
        spec = db[SPEC_RUNS].find_one({"run_id": doc["spec_run_id"]}) or {}
        output = spec.get("output", {})
        doc["manifest_entry"] = {
            "asset_id":       doc["asset_id"],
            "glb_url":        doc["stages"]["rigged"]["url"],
            "kind":           output.get("kind"),
            "attach_scripts": output.get("attach_scripts", []),
            "supersedes":     output.get("supersedes"),
        }
        doc["status"] = "complete"
        doc["completed_at"] = datetime.now(timezone.utc)
        changed = True
    if changed:
        db[COLLECTION].update_one(
            {"asset_run_id": doc["asset_run_id"]},
            {"$set": {k: doc[k] for k in ("stages", "manifest_entry", "status", "completed_at")}},
        )
    return doc


@router.get("/{asset_run_id}")
def get_asset_run(asset_run_id: str) -> Dict[str, Any]:
    db = _db()
    doc = db[COLLECTION].find_one({"asset_run_id": asset_run_id})
    if not doc:
        raise HTTPException(404, f"no such asset run: {asset_run_id}")
    return _serialize(_refresh(db, doc))


@router.get("")
def list_asset_runs(project_id: str, asset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    db = _db()
    q: Dict[str, Any] = {"project_id": project_id}
    if asset_id:
        q["asset_id"] = asset_id
    docs = list(db[COLLECTION].find(q).sort("created_at", -1))
    return [_serialize(d) for d in docs]
