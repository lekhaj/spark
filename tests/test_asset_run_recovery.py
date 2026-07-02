"""Stuck-stage recovery + idempotent fan-out tests for asset_run_routes._refresh.

No GPU / Redis / AWS: Mongo is mongomock, the manual-gen submitters are patched
to record calls (and insert a fresh queued stage-run, mimicking the real
queue_*), and the GPU heartbeat is monkeypatched. Proves the self-healing and
exactly-once behavior added to fix the mid-pipeline-stall incident.
"""
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")

import mongomock
import pytest

from app.routes import asset_run_routes as arr
from app.routes.asset_run_routes import GENERATORS
from worker.lib import manual_gen_schema as mgs


@pytest.fixture()
def env(monkeypatch):
    mdb = mongomock.MongoClient()["test"]
    calls = {"image": 0, "3d": [], "rig": 0, "fanout": 0}

    def fake_image(db, doc, output):
        calls["image"] += 1
        _insert_run(db, doc["asset_id"], doc["stages"]["image"]["stage"], "queued")
        return "task-img"

    def fake_fanout(db, doc):
        calls["fanout"] += 1
        for g in GENERATORS:
            _insert_run(db, doc["asset_id"], g, "queued")

    def fake_one_3d(db, doc, gen):
        calls["3d"].append(gen)
        _insert_run(db, doc["asset_id"], gen, "queued")

    def fake_rig(db, doc, gen):
        calls["rig"] += 1
        _insert_run(db, doc["asset_id"], "rig", "queued")

    monkeypatch.setattr(arr, "_submit_image_job", fake_image)
    monkeypatch.setattr(arr, "_submit_3d_jobs", fake_fanout)
    monkeypatch.setattr(arr, "_submit_one_3d", fake_one_3d)
    monkeypatch.setattr(arr, "_submit_rig_job", fake_rig)
    # Recovery prerequisites: Redis "reachable", GPU not busy (so recovery runs).
    monkeypatch.setattr(arr, "_hb_redis", lambda: object())
    monkeypatch.setattr(arr.gpu_heartbeat, "is_busy", lambda *a, **k: False)
    return mdb, calls


def _insert_run(db, asset_id, stage, status, age_s=0.0, _id=None):
    """Insert a manual_gen_stage_run. age_s back-dates created_at to fake staleness."""
    rid = _id or f"{stage}-{int(time.time()*1e6)}-{status}"
    db[mgs.COLLECTION].insert_one({
        "_id": rid, "char_label": asset_id, "stage": stage, "status": status,
        "image_url": ("u" if status == "done" else None),
        "created_at": time.time() - age_s,
    })
    return rid


def _make_doc(mdb, **over):
    doc = {
        "asset_run_id": "ar1", "project_id": "p", "asset_id": "diablo-2__zombie",
        "spec_run_id": "spec1", "spec_version": "1.0", "kind": "character",
        "morphology": "B1_humanoid", "rig_required": True, "status": "generating",
        "stages": {
            "image": {"stage": "flux_pose", "status": "done", "url": "u"},
            "model3d": {g: {"status": "pending", "url": None} for g in GENERATORS},
            "model3d_chosen": None,
            "rigged": {"status": "pending", "url": None, "fbx_url": None, "rig_status": None},
        },
        "manifest_entry": None,
        "created_at": datetime.now(timezone.utc), "completed_at": None,
        "_major": 1, "_minor": 0,
    }
    doc.update(over)
    mdb[arr.COLLECTION].insert_one(dict(doc))
    return doc


def test_fanout_is_idempotent(env):
    mdb, calls = env
    doc = _make_doc(mdb)  # image done, not yet fanned out
    arr._refresh(mdb, doc)
    arr._refresh(mdb, doc)
    arr._refresh(mdb, doc)
    assert calls["fanout"] == 1            # exactly one fan-out despite 3 refreshes
    assert doc["_model3d_queued"] is True


def test_stuck_3d_is_recovered(env):
    mdb, calls = env
    doc = _make_doc(mdb, _model3d_queued=True)
    doc["stages"]["model3d"] = {
        "trellis":   {"status": "done", "url": "u"},
        "pixal3d":   {"status": "running", "url": None},
        "hunyuan3d": {"status": "done", "url": "u"},
    }
    _insert_run(mdb, doc["asset_id"], "trellis", "done")
    _insert_run(mdb, doc["asset_id"], "hunyuan3d", "done")
    stuck = _insert_run(mdb, doc["asset_id"], "pixal3d", "running", age_s=3000)

    arr._refresh(mdb, doc)

    assert calls["3d"] == ["pixal3d"]                       # only the lost one re-queued
    assert doc["_retries"]["pixal3d"] == 1
    assert mdb[mgs.COLLECTION].find_one({"_id": stuck})["status"] == "error"


def test_recovery_respects_retry_cap(env):
    mdb, calls = env
    doc = _make_doc(mdb, _model3d_queued=True, _retries={"pixal3d": 2})
    doc["stages"]["model3d"] = {
        "trellis":   {"status": "done", "url": "u"},
        "pixal3d":   {"status": "running", "url": None},
        "hunyuan3d": {"status": "done", "url": "u"},
    }
    _insert_run(mdb, doc["asset_id"], "pixal3d", "running", age_s=3000)

    arr._refresh(mdb, doc)

    assert calls["3d"] == []                                # cap hit → not re-queued
    assert doc["stages"]["model3d"]["pixal3d"]["status"] == "error"


def test_fresh_heartbeat_suppresses_recovery(env, monkeypatch):
    mdb, calls = env
    monkeypatch.setattr(arr.gpu_heartbeat, "is_busy", lambda *a, **k: True)  # GPU busy
    doc = _make_doc(mdb, _model3d_queued=True)
    doc["stages"]["model3d"]["pixal3d"] = {"status": "running", "url": None}
    _insert_run(mdb, doc["asset_id"], "pixal3d", "running", age_s=3000)

    arr._refresh(mdb, doc)

    assert calls["3d"] == []                                # busy GPU → never clobbered


def test_all_3d_failed_fails_run(env):
    mdb, calls = env
    doc = _make_doc(mdb, _model3d_queued=True, _retries={g: 2 for g in GENERATORS})
    doc["stages"]["model3d"] = {g: {"status": "error", "url": None} for g in GENERATORS}
    for g in GENERATORS:
        _insert_run(mdb, doc["asset_id"], g, "error")

    out = arr._refresh(mdb, doc)

    assert out["status"] == "failed"
    assert out["stages"]["model3d_chosen"] is None
    assert calls["rig"] == 0


def test_image_hard_error_fails_run(env):
    mdb, calls = env
    doc = _make_doc(mdb)
    doc["stages"]["image"]["status"] = "queued"
    # An errored image run that is NOT past timeout (recent) → not recovered,
    # treated as a hard failure → run fails (and stops holding the GPU).
    _insert_run(mdb, doc["asset_id"], "flux_pose", "error", age_s=5)

    out = arr._refresh(mdb, doc)

    assert out["status"] == "failed"
    assert calls["fanout"] == 0


# ── _run_has_gpu_work: the single stop-decision predicate ───────────────────────

def _doc(**stages_over):
    base = {
        "status": "generating",
        "stages": {
            "image": {"status": "done", "url": "u", "stage": "flux_pose"},
            "model3d": {g: {"status": "done", "url": "u"} for g in GENERATORS},
            "model3d_chosen": "trellis",
            "rigged": {"status": "done", "url": "u", "fbx_url": "f"},
        },
        "_model3d_queued": True,
        "_rig_queued": True,
    }
    base.update(stages_over)
    return base


def test_work_false_when_terminal():
    assert arr._run_has_gpu_work(_doc(status="complete")) is False
    assert arr._run_has_gpu_work(_doc(status="failed")) is False


def test_work_true_image_running():
    d = _doc()
    d["stages"]["image"]["status"] = "running"
    assert arr._run_has_gpu_work(d) is True


def test_work_false_image_errored():
    d = _doc()
    d["stages"]["image"]["status"] = "error"
    assert arr._run_has_gpu_work(d) is False  # terminal fail → no work → may stop


def test_work_true_before_fanout():
    d = _doc(_model3d_queued=False)
    assert arr._run_has_gpu_work(d) is True  # image done, about to fan out


def test_work_true_gen_running():
    d = _doc(_rig_queued=False)
    d["stages"]["model3d_chosen"] = None
    d["stages"]["model3d"]["pixal3d"]["status"] = "running"
    assert arr._run_has_gpu_work(d) is True


def test_work_false_all_gens_failed_none_chosen():
    d = _doc(_rig_queued=False)
    d["stages"]["model3d_chosen"] = None
    d["stages"]["model3d"] = {g: {"status": "error", "url": None} for g in GENERATORS}
    assert arr._run_has_gpu_work(d) is False  # nothing left → may stop


def test_work_true_before_rig():
    d = _doc(_rig_queued=False)
    d["stages"]["rigged"]["status"] = "pending"
    assert arr._run_has_gpu_work(d) is True  # chosen, rig enqueue imminent
