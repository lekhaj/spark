"""P5 — asset-generation bridge (now wired to the real pipeline).

A generation trigger on a CycleZero entity is turned into a run of the existing
asset-run orchestrator (``app/routes/asset_run_routes.py``): the same
``image → 3D fan-out → rig → manifest`` stage machine the studio uses. We do not
fork that pipeline — we synthesize the accepted ``asset_spec`` it expects and let
it drive the manual-gen workers.

**Game segmentation (the invariant that keeps parallel games safe).** Every
cyclezero-driven asset is identified by ``asset_id = f"{slug}__{entity_key}"``.
That value becomes the ``char_label`` the whole pipeline keys on, which segments
*both*:
  * Mongo — ``manual_gen_*`` / ``asset_runs`` / ``spec_gen_runs`` are keyed by
    char_label / project, and
  * S3 — keys are built as ``chars/{char_label}/v{M.N}/…`` (see
    ``worker/workers/manual_gen_worker._char_s3_key``).
So two games (different slugs) never share a Mongo doc or an S3 prefix.

Submission and reconciliation are best-effort and isolated: if the pipeline isn't
reachable (e.g. in tests, or with the GPU stopped) the job stays ``queued`` with a
recorded note rather than raising — the API call still succeeds and can be retried.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from . import models
from .mongo import get_mongo

log = logging.getLogger("cyclezero.generation")

# Durable cyclezero-side record of each generation request (cyclezero Mongo DB).
REQUESTS = "asset_requests"

# layer → (asset kind, rig_required, morphology). Characters/NPCs are rigged
# humanoids; props/environments are static meshes (no rig).
_LAYER_SPEC = {
    "character":   ("character", True, "B1_humanoid"),
    "npc":         ("character", True, "B1_humanoid"),
    "prop":        ("prop", False, "B1_humanoid"),
    "environment": ("environment", False, "B1_humanoid"),
}


def asset_id_for(slug: str, entity_key: str) -> str:
    """Game-scoped asset identity — the segmentation key (see module docstring)."""
    return f"{slug}__{entity_key}"


def _generation_prompt(entity: models.Entity) -> str:
    """Deterministic FLUX prompt from the entity's authored content. No LLM call —
    the fixed style suffix keeps every asset on-model and consistent."""
    data = entity.data or {}
    name = (entity.name or entity.key or "character").strip()
    desc = (data.get("description") or "").strip()
    extra = (data.get("appearance") or data.get("background") or "").strip()
    parts = [p for p in (name, desc, extra) if p]
    base = ". ".join(parts)
    return (
        f"{base}. Full-body, T-pose, game-ready character, stylized low-poly PBR, "
        f"clean silhouette, neutral plain background."
    )


def _spec_for(entity: models.Entity) -> Tuple[str, bool, str]:
    """(kind, rig_required, morphology) from the entity layer, with a static-prop
    default for unknown layers."""
    return _LAYER_SPEC.get(entity.layer, ("prop", False, "B1_humanoid"))


def submit(
    job: models.AssetJob,
    entity: Optional[models.Entity],
    game: Optional[models.Game] = None,
) -> Dict[str, Any]:
    """Record the request and hand off to the asset-run orchestrator.

    Returns a dict merged into the job ``result`` (asset_id + spec/run ids).
    Never raises — failures come back as ``{"submitted": False, "submit_error": …}``.
    """
    slug = game.slug if game is not None else None

    # 1) Durable cyclezero-side record (best-effort; never fail the API call).
    doc = {
        "job_id": str(job.id),
        "game_id": str(job.game_id),
        "game_slug": slug,
        "entity_id": str(job.entity_id) if job.entity_id else None,
        "entity_key": entity.key if entity else None,
        "entity_layer": entity.layer if entity else None,
        "kind": job.kind,
        "params": job.params,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
    }
    try:
        get_mongo()[REQUESTS].insert_one(doc)
    except Exception as exc:  # noqa: BLE001 — best-effort record only
        log.warning("asset request record deferred for job %s: %s", job.id, exc)

    # 2) Need a game + entity to drive the real pipeline.
    if entity is None or slug is None:
        return {"submitted": False, "submit_error": "missing game or entity"}

    asset_id = asset_id_for(slug, entity.key)
    kind, rig_required, morphology = _spec_for(entity)

    try:
        # asset_run_routes reads/writes the World_builder DB (mgs.get_db()).
        # Synthesize the accepted asset_spec it expects, then create the run.
        from app.routes import asset_run_routes as arr

        wb = arr._db()
        spec_run_id = uuid.uuid4().hex
        wb[arr.SPEC_RUNS].insert_one(
            {
                "run_id": spec_run_id,
                "stage": "asset_spec",
                "status": "accepted",
                "project_id": slug,
                "major": 1,
                "minor": 0,
                "output": {
                    "asset_id": asset_id,
                    "kind": kind,
                    "rig_required": rig_required,
                    "morphology": morphology,
                    "generation_prompt": _generation_prompt(entity),
                    "attach_scripts": [],
                },
                "created_at": datetime.now(timezone.utc),
                "source": "cyclezero",
            }
        )
        run = arr.create_asset_run(arr.AssetRunCreate(spec_run_id=spec_run_id))
        return {
            "submitted": True,
            "asset_id": asset_id,
            "spec_run_id": spec_run_id,
            "asset_run_id": run.get("asset_run_id"),
        }
    except Exception as exc:  # noqa: BLE001 — best-effort; keep the job queued
        log.warning("asset job %s submit deferred: %s", job.id, exc)
        return {"submitted": False, "submit_error": str(exc), "asset_id": asset_id}


def reconcile(asset_run_id: Optional[str]) -> Dict[str, Any]:
    """Advance the linked asset_run's stage machine and report the result.

    Returns ``{"status": …}`` and, once the run is ``complete``, the produced
    ``glb``/``fbx``/``lod`` so the caller can write them back onto the entity.
    Never raises.
    """
    out: Dict[str, Any] = {"status": "unknown"}
    if not asset_run_id:
        return out
    try:
        from app.routes import asset_run_routes as arr

        db = arr._db()
        doc = db[arr.COLLECTION].find_one({"asset_run_id": asset_run_id})
        if not doc:
            return out
        doc = arr._refresh(db, doc)
        out["status"] = doc.get("status")
        me = doc.get("manifest_entry") or {}
        if doc.get("status") == "complete" and me.get("glb_url"):
            out.update(
                glb=me["glb_url"],
                fbx=me.get("fbx_url"),
                lod="auto",
                generator=me.get("model3d_generator"),
            )
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.warning("reconcile asset_run %s failed: %s", asset_run_id, exc)
    return out
