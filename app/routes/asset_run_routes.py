"""
asset_run_routes.py — asset_spec → GPU pipeline bridge (CycleZero U05/T13)
==========================================================================

An accepted ``asset_spec`` run becomes a generation job that walks a small
stage machine, reusing the existing manual-gen pipeline:

    image → 3D (fan-out) → rig → manifest

* **image** — characters (``rig_required``) use FLUX-dev + ControlNet-Union-Pro
  (``flux_pose``) with the morphology's bundled control proxy (``pose`` for
  B1_humanoid, ``soft_edge`` for B2..B7). Props/environment use plain
  FLUX-schnell (``flux``).
* **3D** — fans out to ``trellis`` + ``pixal3d`` + ``hunyuan3d`` for quality
  comparison. These run **sequentially** on the single L40S worker (each handler
  evicts the others) — no parallel GPU execution, so a fan-out never OOMs.
* **rig** — once a generator is chosen (artist pick, else first to finish),
  Auto-Rig Pro rigs that mesh and exports BOTH a clean deform-only GLB and an
  engine-ready FBX. Rig failure is non-fatal: ``rig_status=manual`` and the
  unrigged mesh still ships.
* **manifest** — when the rig lands, writes the ``manifest_entry`` the engine's
  asset registry loads. Data-only by design — a changed asset never requires
  engine code.

Routes mounted at ``/asset-runs`` (set in app/main.py).
"""

from __future__ import annotations

import logging
import os
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

import time
import pymongo
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import infra
from worker.lib import gpu_heartbeat
from worker.lib import manual_gen_schema as mgs

log = logging.getLogger("asset_run_routes")

router = APIRouter()

COLLECTION = "asset_runs"
SPEC_RUNS = "spec_gen_runs"

# The three 3D generators we fan out to for quality comparison.
GENERATORS = ("trellis", "pixal3d", "hunyuan3d")

# Terminal stage-run statuses (a stage is "settled" once it reaches one of these).
_TERMINAL = {"done", "failed", "error"}

# Morphology → ControlNet mode for the Union-Pro image step.
# B1 humanoid uses real OpenPose; everything else uses soft_edge proxies.
_IMAGE_MODE = {
    "B1_humanoid": "pose",
    "B2_centaur": "soft_edge",
    "B3_naga": "soft_edge",
    "B4_quadruped": "soft_edge",
    "B5_winged_quad": "soft_edge",
    "B6_hydra": "soft_edge",
    "B7_arthropod": "soft_edge",
}

# Morphology → best-fit ARP char_type. B6/B7 have no clean preset — the rig
# script attempts the closest and falls back to manual.
_CHAR_TYPE = {
    "B1_humanoid": "humanoid",
    "B2_centaur": "humanoid",
    "B3_naga": "humanoid",
    "B4_quadruped": "quadruped",
    "B5_winged_quad": "quadruped",
    "B6_hydra": "quadruped",
    "B7_arthropod": "quadruped",
}


def _db():
    """Open a Mongo connection. Overridden in tests (mongomock)."""
    return mgs.get_db()


def _notify_complete(doc: Dict[str, Any]) -> None:
    """Free email notification via Gmail SMTP (stdlib). Best-effort: silently
    skips if SMTP_* env is unset and never raises into the request path.
    Config in .env.secrets: SMTP_HOST/PORT/USER/PASSWORD/SMTP_TO."""
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    pw   = os.environ.get("SMTP_PASSWORD")
    to   = os.environ.get("SMTP_TO") or user
    if not (host and user and pw and to):
        return
    try:
        m = doc.get("manifest_entry") or {}
        st = doc["stages"]
        rs = (st["rigged"].get("rig_status") or "auto")
        lines = [
            f"Asset: {doc['asset_id']}  v{doc['spec_version']}  ({doc.get('morphology')})",
            f"Rig: {rs}    Generator: {st.get('model3d_chosen')}",
            "",
            f"Image: {st['image'].get('url') or '-'}",
            f"GLB:   {st['rigged'].get('url') or '-'}",
            f"FBX:   {st['rigged'].get('fbx_url') or '-'}",
            "",
            "These are public S3 links — open in a browser / 3D viewer.",
        ]
        msg = EmailMessage()
        msg["Subject"] = f"✅ Asset ready: {doc['asset_id']} v{doc['spec_version']} (rig {rs})"
        msg["From"] = user
        msg["To"] = to
        msg.set_content("\n".join(lines))
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        log.info("asset-run %s completion email sent to %s", doc["asset_run_id"], to)
    except Exception:
        log.exception("completion email failed (non-fatal) for %s", doc.get("asset_run_id"))


# ── Stage submitters (patched out in tests) ───────────────────────────────────

def _submit_image_job(db, doc: Dict[str, Any], output: Dict[str, Any]) -> str:
    """Queue the image stage. Characters → flux_pose (Union-Pro); props → flux."""
    from worker.lib import manual_gen_queue as mgq

    asset_id = doc["asset_id"]
    major, minor = doc["_major"], doc["_minor"]
    prompt = output["generation_prompt"]

    if doc["rig_required"]:
        morphology = doc["morphology"]
        result = mgq.queue_flux_pose(
            db, char_label=asset_id, major=major, minor=minor, prompt=prompt,
            use_control=True, auto_extract=False,
            control_mode=_IMAGE_MODE.get(morphology, "soft_edge"),
            morphology=morphology, src_stage="",
        )
    else:
        result = mgq.queue_flux(
            db, char_label=asset_id, major=major, minor=minor, prompt=prompt,
        )
    return result.task_id


def _submit_3d_jobs(db, doc: Dict[str, Any]) -> None:
    """Fan out all three 3D generators over the image. Sequential on the GPU."""
    from worker.lib import manual_gen_queue as mgq

    asset_id = doc["asset_id"]
    major, minor = doc["_major"], doc["_minor"]
    char_type = _CHAR_TYPE.get(doc["morphology"], "humanoid")
    src_stage = doc["stages"]["image"]["stage"]
    submit = {
        "trellis": mgq.queue_trellis,
        "pixal3d": mgq.queue_pixal3d,
        "hunyuan3d": mgq.queue_hunyuan3d,
    }
    for gen in GENERATORS:
        submit[gen](
            db, char_label=asset_id, major=major, minor=minor,
            char_type=char_type, src_stage=src_stage,
        )


def _submit_rig_job(db, doc: Dict[str, Any], generator: str) -> None:
    """Queue the rig job over the chosen generator's mesh."""
    from worker.lib import manual_gen_queue as mgq

    mgq.queue_rig(
        db, char_label=doc["asset_id"], major=doc["_major"], minor=doc["_minor"],
        char_type=_CHAR_TYPE.get(doc["morphology"], "humanoid"),
        src_stage=generator, morphology=doc["morphology"],
    )


def _submit_one_3d(db, doc: Dict[str, Any], gen: str) -> None:
    """Re-queue a single 3D generator (used by stuck-stage recovery)."""
    from worker.lib import manual_gen_queue as mgq

    fn = {"trellis": mgq.queue_trellis,
          "pixal3d": mgq.queue_pixal3d,
          "hunyuan3d": mgq.queue_hunyuan3d}[gen]
    fn(db, char_label=doc["asset_id"], major=doc["_major"], minor=doc["_minor"],
       char_type=_CHAR_TYPE.get(doc["morphology"], "humanoid"),
       src_stage=doc["stages"]["image"]["stage"])


# ── Stuck-stage recovery helpers ───────────────────────────────────────────────

def _hb_redis():
    """Best-effort Redis client for heartbeat reads / re-enqueue. None if down."""
    try:
        from worker.lib.manual_gen_queue import _redis_client
        return _redis_client()
    except Exception:  # noqa: BLE001
        return None


def _stage_timeout(stage: str) -> int:
    return infra.STAGE_TIMEOUT_SECONDS.get(stage, infra.STAGE_TIMEOUT_DEFAULT)


def _run_age(run: Dict[str, Any]) -> float:
    """Seconds since the run entered its current in-flight state (epoch floats)."""
    ts = run.get("started_at") or run.get("queued_at") or run.get("created_at")
    try:
        return time.time() - float(ts)
    except (TypeError, ValueError):
        return 0.0


def _recover_stuck_stages(db, doc: Dict[str, Any]) -> bool:
    """
    Re-enqueue GPU stages whose underlying run doc is stuck (``queued``/
    ``running`` past its per-stage timeout) when the GPU is **not** actively
    working — i.e. the task was lost to a stopped/reclaimed box. Idempotent and
    retry-capped (``STAGE_MAX_RETRIES``); after the cap the stage is left
    ``error`` so the run can finish with another generator (3D) or fail cleanly.

    Returns True if it changed the asset_run doc.

    FAIL SAFE: if Redis is unreachable (can't read heartbeat or enqueue) it does
    nothing. If the GPU heartbeat is fresh (a task is running right now) it does
    nothing — a slow-but-alive stage is never clobbered.
    """
    if doc["status"] in ("complete", "failed"):
        return False
    r = _hb_redis()
    if r is None:
        return False
    fresh = getattr(infra, "HEARTBEAT_FRESH_SECONDS", 120)
    if gpu_heartbeat.is_busy(r, None, fresh):
        return False  # GPU is doing work — give it time

    asset_id = doc["asset_id"]
    retries: Dict[str, int] = doc.setdefault("_retries", {})
    changed = False

    def _exhausted(stage: str) -> bool:
        return retries.get(stage, 0) >= infra.STAGE_MAX_RETRIES

    def _stuck(stage: str) -> bool:
        run = _poll_run(db, asset_id, stage, doc["created_at"])
        if not run or run.get("status") not in ("queued", "running"):
            return False
        if _run_age(run) <= _stage_timeout(stage):
            return False
        # Mark the lost run errored so the queue_* retry can auto-bump a fresh
        # minor (prepare_run refuses to re-queue a queued/running run).
        mgs.mark_error(db, run["_id"], stage=stage, error="timeout/lost — auto-recovered")
        return True

    # ── image ──
    img_stage = doc["stages"]["image"]["stage"]
    if doc["stages"]["image"]["status"] in ("queued", "running") and _stuck(img_stage):
        if _exhausted(img_stage):
            doc["stages"]["image"]["status"] = "failed"
            doc["status"] = "failed"
            log.warning("asset-run %s: image exhausted retries — failing", asset_id)
        else:
            spec = db[SPEC_RUNS].find_one({"run_id": doc["spec_run_id"]}) or {}
            try:
                _submit_image_job(db, doc, spec.get("output", {}))
                retries[img_stage] = retries.get(img_stage, 0) + 1
                doc["stages"]["image"]["status"] = "queued"
                log.warning("asset-run %s: re-enqueued lost image (try %d)", asset_id, retries[img_stage])
            except Exception:
                log.exception("image re-enqueue failed for %s", asset_id)
        return True  # image is upstream of everything — settle it before the rest

    # ── 3D generators (only after image done + fan-out happened) ──
    if doc.get("_model3d_queued"):
        for gen in GENERATORS:
            if doc["stages"]["model3d"][gen]["status"] in ("queued", "running") and _stuck(gen):
                if _exhausted(gen):
                    doc["stages"]["model3d"][gen]["status"] = "error"
                    log.warning("asset-run %s: %s exhausted retries — leaving error", asset_id, gen)
                else:
                    try:
                        _submit_one_3d(db, doc, gen)
                        retries[gen] = retries.get(gen, 0) + 1
                        doc["stages"]["model3d"][gen]["status"] = "queued"
                        log.warning("asset-run %s: re-enqueued lost %s (try %d)", asset_id, gen, retries[gen])
                    except Exception:
                        log.exception("%s re-enqueue failed for %s", gen, asset_id)
                changed = True

    # ── rig ──
    if doc.get("_rig_queued") and doc["stages"]["rigged"]["status"] in ("queued", "running"):
        chosen = doc["stages"]["model3d_chosen"]
        if chosen and _stuck("rig"):
            if _exhausted("rig"):
                doc["stages"]["rigged"]["status"] = "failed"
                doc["status"] = "failed"
                log.warning("asset-run %s: rig exhausted retries — failing", asset_id)
            else:
                try:
                    _submit_rig_job(db, doc, chosen)
                    retries["rig"] = retries.get("rig", 0) + 1
                    doc["stages"]["rigged"]["status"] = "queued"
                    log.warning("asset-run %s: re-enqueued lost rig (try %d)", asset_id, retries["rig"])
                except Exception:
                    log.exception("rig re-enqueue failed for %s", asset_id)
            changed = True

    return changed


class AssetRunCreate(BaseModel):
    spec_run_id: str = Field(min_length=1)


class ChooseModel(BaseModel):
    generator: str = Field(min_length=1)


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {**doc}
    out.pop("_id", None)
    out.pop("_major", None)
    out.pop("_minor", None)
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
    rig_required = bool(output.get("rig_required", output.get("kind") == "character"))
    morphology = output.get("morphology") or "B1_humanoid"
    image_stage = "flux_pose" if rig_required else "flux"

    doc = {
        "asset_run_id": uuid.uuid4().hex,
        "project_id":   spec["project_id"],
        "asset_id":     asset_id,
        "spec_run_id":  body.spec_run_id,
        "spec_version": f"{spec['major']}.{spec['minor']}",
        "kind":         output.get("kind"),
        "morphology":   morphology,
        "rig_required": rig_required,
        "status":       "generating",
        "stages": {
            "image":   {"stage": image_stage, "status": "queued", "url": None},
            "model3d": {gen: {"status": "pending", "url": None} for gen in GENERATORS},
            "model3d_chosen": None,
            "rigged":  {"status": "pending", "url": None, "fbx_url": None,
                        "rig_status": None},
        },
        "manifest_entry": None,
        "created_at":   datetime.now(timezone.utc),
        "completed_at": None,
        "_major":       spec["major"],
        "_minor":       spec["minor"],
    }
    try:
        doc["task_id"] = _submit_image_job(db, doc, output)
    except Exception as e:
        log.exception("manual-gen submit failed for %s", asset_id)
        raise HTTPException(502, f"manual-gen submit failed: {e}")
    db[COLLECTION].insert_one(doc)
    return _serialize(doc)


def _poll_run(db, asset_id: str, stage: str, after: datetime) -> Optional[Dict[str, Any]]:
    """Latest manual-gen run for (char_label, stage). We intentionally do NOT
    filter on created_at: the pipeline stores it as a float epoch while the
    asset-run doc uses a datetime, so a ``$gte`` comparison silently matches
    nothing in production. The stage machine only polls a stage AFTER it has
    queued that stage, so the most-recent run for the stage is always ours."""
    return db[mgs.COLLECTION].find_one(
        {"char_label": asset_id, "stage": stage},
        sort=[("created_at", pymongo.DESCENDING)],
    )


def _refresh(db, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Advance the stage machine: image → 3D fan-out → chosen → rig → manifest."""
    if doc["status"] in ("complete", "failed"):
        return doc
    asset_id = doc["asset_id"]
    after = doc["created_at"]
    changed = False

    # ── 0. self-heal: re-enqueue any GPU stage lost to a stopped/reclaimed box ─
    if _recover_stuck_stages(db, doc):
        changed = True

    # ── 1. image ──────────────────────────────────────────────────────────────
    img = _poll_run(db, asset_id, doc["stages"]["image"]["stage"], after)
    if img:
        status, url = img.get("status") or "pending", img.get("image_url")
        if (doc["stages"]["image"]["status"], doc["stages"]["image"]["url"]) != (status, url):
            doc["stages"]["image"].update(status=status, url=url)
            changed = True

    # ── 2. fan out 3D once the image is done ──────────────────────────────────
    # Atomic claim: only one concurrent _refresh (reconciler vs. GET) wins the
    # flip from unset → True, so the fan-out runs exactly once (no duplicate
    # tasks). On submit failure the claim is released for a later retry.
    if doc["stages"]["image"]["status"] == "done" and not doc.get("_model3d_queued"):
        claimed = db[COLLECTION].update_one(
            {"asset_run_id": doc["asset_run_id"], "_model3d_queued": {"$ne": True}},
            {"$set": {"_model3d_queued": True}},
        ).modified_count == 1
        if claimed:
            doc["_model3d_queued"] = True
            try:
                _submit_3d_jobs(db, doc)
                for gen in GENERATORS:
                    doc["stages"]["model3d"][gen]["status"] = "queued"
                changed = True
            except Exception:
                log.exception("3D fan-out submit failed for %s", asset_id)
                db[COLLECTION].update_one(
                    {"asset_run_id": doc["asset_run_id"]},
                    {"$unset": {"_model3d_queued": ""}},
                )
                doc["_model3d_queued"] = False
        else:
            doc["_model3d_queued"] = True  # another pass already fanned out

    # ── 3. poll the 3D candidates ─────────────────────────────────────────────
    for gen in GENERATORS:
        if doc["stages"]["model3d"][gen]["status"] in ("pending",):
            continue
        run = _poll_run(db, asset_id, gen, after)
        if not run:
            continue
        status, url = run.get("status") or "pending", run.get("image_url")
        if (doc["stages"]["model3d"][gen]["status"], doc["stages"]["model3d"][gen]["url"]) != (status, url):
            doc["stages"]["model3d"][gen].update(status=status, url=url)
            changed = True

    # ── 4. auto-pick a generator only once ALL three are terminal, so the
    #       artist gets the full comparison window to choose manually first.
    #       Falls back to the first successful generator in order.
    if doc["stages"]["model3d_chosen"] is None:
        states = [doc["stages"]["model3d"][g]["status"] for g in GENERATORS]
        if all(s in _TERMINAL for s in states):
            picked = False
            for gen in GENERATORS:
                if doc["stages"]["model3d"][gen]["status"] == "done":
                    doc["stages"]["model3d_chosen"] = gen
                    changed = True
                    picked = True
                    break
            if not picked:
                # All three 3D generators failed (after recovery retries) — the
                # run cannot produce a mesh; fail cleanly instead of hanging.
                doc["status"] = "failed"
                changed = True
                log.warning("asset-run %s: all 3D generators failed — run failed", asset_id)

    # ── 5. queue rig over the chosen mesh (atomic claim → exactly once) ────────
    chosen = doc["stages"]["model3d_chosen"]
    if chosen and not doc.get("_rig_queued") and doc["status"] != "failed":
        claimed = db[COLLECTION].update_one(
            {"asset_run_id": doc["asset_run_id"], "_rig_queued": {"$ne": True}},
            {"$set": {"_rig_queued": True}},
        ).modified_count == 1
        if claimed:
            doc["_rig_queued"] = True
            try:
                _submit_rig_job(db, doc, chosen)
                doc["stages"]["rigged"]["status"] = "queued"
                changed = True
            except Exception:
                log.exception("rig submit failed for %s", asset_id)
                db[COLLECTION].update_one(
                    {"asset_run_id": doc["asset_run_id"]},
                    {"$unset": {"_rig_queued": ""}},
                )
                doc["_rig_queued"] = False
        else:
            doc["_rig_queued"] = True

    # ── 6. poll rig ───────────────────────────────────────────────────────────
    if doc.get("_rig_queued"):
        run = _poll_run(db, asset_id, "rig", after)
        if run:
            status = run.get("status") or "pending"
            url = run.get("image_url")
            fbx_url = run.get("fbx_url")
            rig_status = (run.get("params") or {}).get("rig_status") or run.get("rig_status")
            new = {"status": status, "url": url, "fbx_url": fbx_url, "rig_status": rig_status}
            if doc["stages"]["rigged"] != {**doc["stages"]["rigged"], **new}:
                doc["stages"]["rigged"].update(new)
                changed = True

    # ── 7. manifest + complete on rig done ────────────────────────────────────
    if doc["stages"]["rigged"]["status"] == "done" and not doc.get("manifest_entry"):
        spec = db[SPEC_RUNS].find_one({"run_id": doc["spec_run_id"]}) or {}
        output = spec.get("output", {})
        rigged = doc["stages"]["rigged"]
        doc["manifest_entry"] = {
            "asset_id":          asset_id,
            "kind":              output.get("kind"),
            "morphology":        doc["morphology"],
            "char_type":         _CHAR_TYPE.get(doc["morphology"], "humanoid"),
            "rig_status":        rigged.get("rig_status") or "auto",
            "glb_url":           rigged.get("url"),
            "fbx_url":           rigged.get("fbx_url"),
            "model3d_generator": chosen,
            "attach_scripts":    output.get("attach_scripts", []),
            "supersedes":        output.get("supersedes"),
        }
        doc["status"] = "complete"
        doc["completed_at"] = datetime.now(timezone.utc)
        changed = True
        _notify_complete(doc)

    if changed:
        db[COLLECTION].update_one(
            {"asset_run_id": doc["asset_run_id"]},
            {"$set": {k: doc[k] for k in (
                "stages", "manifest_entry", "status", "completed_at",
                "_model3d_queued", "_rig_queued", "_retries") if k in doc}},
        )
    return doc


@router.get("/{asset_run_id}")
def get_asset_run(asset_run_id: str) -> Dict[str, Any]:
    db = _db()
    doc = db[COLLECTION].find_one({"asset_run_id": asset_run_id})
    if not doc:
        raise HTTPException(404, f"no such asset run: {asset_run_id}")
    return _serialize(_refresh(db, doc))


@router.post("/{asset_run_id}/choose-model3d")
def choose_model3d(asset_run_id: str, body: ChooseModel) -> Dict[str, Any]:
    """Artist picks which generator's mesh feeds rigging."""
    db = _db()
    if body.generator not in GENERATORS:
        raise HTTPException(422, f"unknown generator: {body.generator}")
    doc = db[COLLECTION].find_one({"asset_run_id": asset_run_id})
    if not doc:
        raise HTTPException(404, f"no such asset run: {asset_run_id}")
    if doc.get("_rig_queued"):
        raise HTTPException(409, "rig already queued — generator can no longer change")
    # Refresh candidate statuses first so a just-finished mesh is visible.
    doc = _refresh(db, doc)
    if doc["stages"]["model3d"][body.generator]["status"] != "done":
        raise HTTPException(409, f"{body.generator} mesh is not done yet")
    db[COLLECTION].update_one(
        {"asset_run_id": asset_run_id},
        {"$set": {"stages.model3d_chosen": body.generator}},
    )
    doc["stages"]["model3d_chosen"] = body.generator
    return _serialize(_refresh(db, doc))


@router.get("")
def list_asset_runs(project_id: str, asset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    db = _db()
    q: Dict[str, Any] = {"project_id": project_id}
    if asset_id:
        q["asset_id"] = asset_id
    docs = list(db[COLLECTION].find(q).sort("created_at", -1))
    return [_serialize(_refresh(db, d)) for d in docs]
