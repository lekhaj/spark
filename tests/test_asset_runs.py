"""
U05 asset-run bridge tests — manual-gen submit mocked, mongomock.
Covers the stage machine: image (flux_pose) → 3D fan-out → chosen → rig → manifest.
"""

import os

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

from datetime import datetime, timezone

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import asset_run_routes, schema_routes, spec_gen_routes

ASSET_SPEC_SCHEMA = {
    "type": "object",
    "required": ["asset_id", "kind", "display_name", "generation_prompt", "rig_required"],
    "properties": {
        "asset_id": {"type": "string"},
        "kind": {"enum": ["character", "prop", "environment"]},
        "morphology": {"type": "string"},
        "display_name": {"type": "string"},
        "generation_prompt": {"type": "string", "minLength": 1},
        "style_refs": {"type": "array", "items": {"type": "string"}},
        "rig_required": {"type": "boolean"},
        "attach_scripts": {"type": "array", "items": {"type": "string"}},
        "supersedes": {"type": ["string", "null"]},
    },
}

KAEL_SPEC = {
    "asset_id": "char_kael",
    "kind": "character",
    "morphology": "B1_humanoid",
    "display_name": "Kael (older)",
    "generation_prompt": "older battle-worn ranger, grey-streaked beard, semi-realistic 3D game asset",
    "style_refs": [],
    "rig_required": True,
    "attach_scripts": ["src/actors/kael.ts"],
    "supersedes": "char_kael@1.0",
}


@pytest.fixture()
def env(monkeypatch):
    db = mongomock.MongoClient()["World_builder_test"]
    monkeypatch.setattr(schema_routes, "_db", lambda: db)
    monkeypatch.setattr(spec_gen_routes, "_db", lambda: db)
    monkeypatch.setattr(asset_run_routes, "_db", lambda: db)
    calls = {"image": [], "fanout": [], "rig": []}

    def fake_image(db_, doc, output):
        calls["image"].append({"asset_id": doc["asset_id"], "stage": doc["stages"]["image"]["stage"],
                                "morphology": doc["morphology"], "prompt": output["generation_prompt"]})
        return "task-123"

    def fake_fanout(db_, doc):
        calls["fanout"].append(doc["asset_id"])

    def fake_rig(db_, doc, generator):
        calls["rig"].append({"asset_id": doc["asset_id"], "generator": generator})

    monkeypatch.setattr(asset_run_routes, "_submit_image_job", fake_image)
    monkeypatch.setattr(asset_run_routes, "_submit_3d_jobs", fake_fanout)
    monkeypatch.setattr(asset_run_routes, "_submit_rig_job", fake_rig)
    app = FastAPI()
    app.include_router(schema_routes.router, prefix="/schemas")
    app.include_router(spec_gen_routes.router, prefix="/spec-gen")
    app.include_router(asset_run_routes.router, prefix="/asset-runs")
    client = TestClient(app)
    client.post("/schemas/asset_spec", json={"json_schema": ASSET_SPEC_SCHEMA, "changelog": "seed"})
    return client, db, calls


def _accepted_spec_run(client, spec=KAEL_SPEC):
    run = client.post("/spec-gen/runs", json={
        "project_id": "cyclezero", "entity_id": "asset:char_kael",
        "stage": "asset_spec", "mode": "paste", "output": spec,
    }).json()
    client.post(f"/spec-gen/runs/{run['run_id']}/accept")
    return run


def _pipeline_run(db, asset_id, stage, status, url=None, **extra):
    doc = {
        "_id": f"{stage}-{status}-{url}", "char_label": asset_id, "stage": stage,
        "status": status, "image_url": url,
        "created_at": datetime.now(timezone.utc),
    }
    doc.update(extra)
    db["manual_gen_stage_runs"].insert_one(doc)


def test_create_requires_accepted_asset_spec(env):
    client, db, calls = env
    run = client.post("/spec-gen/runs", json={
        "project_id": "cyclezero", "entity_id": "asset:char_kael",
        "stage": "asset_spec", "mode": "paste", "output": KAEL_SPEC,
    }).json()
    # valid but NOT accepted → 409
    assert client.post("/asset-runs", json={"spec_run_id": run["run_id"]}).status_code == 409
    assert calls["image"] == []

    client.post(f"/spec-gen/runs/{run['run_id']}/accept")
    r = client.post("/asset-runs", json={"spec_run_id": run["run_id"]})
    assert r.status_code == 200
    doc = r.json()
    assert doc["asset_id"] == "char_kael"
    # character → Union-Pro image stage
    assert doc["stages"]["image"]["stage"] == "flux_pose"
    assert doc["stages"]["image"]["status"] == "queued"
    assert doc["morphology"] == "B1_humanoid"
    assert calls["image"][0]["morphology"] == "B1_humanoid"


def test_prop_uses_schnell_flux(env):
    client, db, calls = env
    prop = {**KAEL_SPEC, "asset_id": "prop_torch", "kind": "prop", "rig_required": False}
    prop.pop("morphology", None)
    run = _accepted_spec_run(client, prop)
    doc = client.post("/asset-runs", json={"spec_run_id": run["run_id"]}).json()
    assert doc["stages"]["image"]["stage"] == "flux"
    assert doc["rig_required"] is False


def test_wrong_stage_409_and_missing_404(env):
    client, db, _ = env
    client.post("/schemas/zone_spec", json={
        "json_schema": {"type": "object"}, "changelog": "seed"})
    run = client.post("/spec-gen/runs", json={
        "project_id": "cyclezero", "entity_id": "zone:x",
        "stage": "zone_spec", "mode": "paste", "output": {},
    }).json()
    client.post(f"/spec-gen/runs/{run['run_id']}/accept")
    assert client.post("/asset-runs", json={"spec_run_id": run["run_id"]}).status_code == 409
    assert client.post("/asset-runs", json={"spec_run_id": "nope"}).status_code == 404


def test_image_done_fans_out_3d_and_polls_candidates(env):
    client, db, calls = env
    spec_run = _accepted_spec_run(client)
    ar = client.post("/asset-runs", json={"spec_run_id": spec_run["run_id"]}).json()

    _pipeline_run(db, "char_kael", "flux_pose", "done", "https://s3/kael.png")
    out = client.get(f"/asset-runs/{ar['asset_run_id']}").json()
    assert out["stages"]["image"] == {"stage": "flux_pose", "status": "done", "url": "https://s3/kael.png"}
    # fan-out fired once; all three candidates queued
    assert calls["fanout"] == ["char_kael"]
    assert all(out["stages"]["model3d"][g]["status"] == "queued" for g in ("trellis", "pixal3d", "hunyuan3d"))

    # auto-pick waits for ALL three terminal; once they are, first done wins
    _pipeline_run(db, "char_kael", "trellis", "done", "https://s3/kael_trellis.glb")
    _pipeline_run(db, "char_kael", "hunyuan3d", "done", "https://s3/kael_huny.glb")
    # only two done so far → no auto-pick yet (artist still has the window)
    out = client.get(f"/asset-runs/{ar['asset_run_id']}").json()
    assert out["stages"]["model3d_chosen"] is None
    assert calls["rig"] == []
    # third finishes (failed) → all terminal → trellis (first done) auto-chosen
    _pipeline_run(db, "char_kael", "pixal3d", "failed")
    out = client.get(f"/asset-runs/{ar['asset_run_id']}").json()
    assert out["stages"]["model3d_chosen"] == "trellis"
    assert calls["rig"] == [{"asset_id": "char_kael", "generator": "trellis"}]


def test_choose_model3d_overrides_auto_pick(env):
    client, db, calls = env
    spec_run = _accepted_spec_run(client)
    ar = client.post("/asset-runs", json={"spec_run_id": spec_run["run_id"]}).json()
    _pipeline_run(db, "char_kael", "flux_pose", "done", "https://s3/kael.png")
    client.get(f"/asset-runs/{ar['asset_run_id']}")  # fan out
    _pipeline_run(db, "char_kael", "hunyuan3d", "done", "https://s3/kael_huny.glb")

    # pick hunyuan before trellis finishes
    out = client.post(f"/asset-runs/{ar['asset_run_id']}/choose-model3d",
                      json={"generator": "hunyuan3d"}).json()
    assert out["stages"]["model3d_chosen"] == "hunyuan3d"
    assert calls["rig"] == [{"asset_id": "char_kael", "generator": "hunyuan3d"}]

    # unknown generator 422, not-done generator 409
    assert client.post(f"/asset-runs/{ar['asset_run_id']}/choose-model3d",
                       json={"generator": "bogus"}).status_code == 422


def test_rig_completion_writes_dual_export_manifest(env):
    client, db, _ = env
    spec_run = _accepted_spec_run(client)
    ar = client.post("/asset-runs", json={"spec_run_id": spec_run["run_id"]}).json()

    _pipeline_run(db, "char_kael", "flux_pose", "done", "https://s3/kael.png")
    client.get(f"/asset-runs/{ar['asset_run_id']}")
    _pipeline_run(db, "char_kael", "trellis", "done", "https://s3/kael.glb")
    client.post(f"/asset-runs/{ar['asset_run_id']}/choose-model3d", json={"generator": "trellis"})
    _pipeline_run(db, "char_kael", "rig", "done", "https://s3/kael_rigged.glb",
                  fbx_url="https://s3/kael_rigged.fbx", rig_status="auto")

    out = client.get(f"/asset-runs/{ar['asset_run_id']}").json()
    assert out["status"] == "complete"
    m = out["manifest_entry"]
    assert m["asset_id"] == "char_kael"
    assert m["glb_url"] == "https://s3/kael_rigged.glb"
    assert m["fbx_url"] == "https://s3/kael_rigged.fbx"
    assert m["morphology"] == "B1_humanoid"
    assert m["char_type"] == "humanoid"
    assert m["rig_status"] == "auto"
    assert m["model3d_generator"] == "trellis"
    assert m["attach_scripts"] == ["src/actors/kael.ts"]

    lst = client.get("/asset-runs", params={"project_id": "cyclezero", "asset_id": "char_kael"}).json()
    assert len(lst) == 1 and lst[0]["status"] == "complete"


def test_rig_manual_fallback_recorded_in_manifest(env):
    client, db, _ = env
    spec_run = _accepted_spec_run(client)
    ar = client.post("/asset-runs", json={"spec_run_id": spec_run["run_id"]}).json()
    _pipeline_run(db, "char_kael", "flux_pose", "done", "https://s3/kael.png")
    client.get(f"/asset-runs/{ar['asset_run_id']}")
    _pipeline_run(db, "char_kael", "trellis", "done", "https://s3/kael.glb")
    client.post(f"/asset-runs/{ar['asset_run_id']}/choose-model3d", json={"generator": "trellis"})
    # rig couldn't auto-rig — mesh shipped unrigged, rig_status=manual
    _pipeline_run(db, "char_kael", "rig", "done", "https://s3/kael_mesh.glb",
                  fbx_url="https://s3/kael_mesh.fbx", rig_status="manual")

    out = client.get(f"/asset-runs/{ar['asset_run_id']}").json()
    assert out["status"] == "complete"
    assert out["manifest_entry"]["rig_status"] == "manual"
