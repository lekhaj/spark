"""
U05 asset-run bridge tests — manual-gen submit mocked, mongomock.
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
    submitted = []

    def fake_submit(db_, asset_id, prompt, major, minor):
        submitted.append({"asset_id": asset_id, "prompt": prompt, "major": major, "minor": minor})
        return "task-123"

    monkeypatch.setattr(asset_run_routes, "_submit_image_job", fake_submit)
    app = FastAPI()
    app.include_router(schema_routes.router, prefix="/schemas")
    app.include_router(spec_gen_routes.router, prefix="/spec-gen")
    app.include_router(asset_run_routes.router, prefix="/asset-runs")
    client = TestClient(app)
    client.post("/schemas/asset_spec", json={"json_schema": ASSET_SPEC_SCHEMA, "changelog": "seed"})
    return client, db, submitted


def _accepted_spec_run(client):
    run = client.post("/spec-gen/runs", json={
        "project_id": "cyclezero", "entity_id": "asset:char_kael",
        "stage": "asset_spec", "mode": "paste", "output": KAEL_SPEC,
    }).json()
    client.post(f"/spec-gen/runs/{run['run_id']}/accept")
    return run


def test_create_requires_accepted_asset_spec(env):
    client, db, submitted = env
    run = client.post("/spec-gen/runs", json={
        "project_id": "cyclezero", "entity_id": "asset:char_kael",
        "stage": "asset_spec", "mode": "paste", "output": KAEL_SPEC,
    }).json()
    # valid but NOT accepted → 409
    assert client.post("/asset-runs", json={"spec_run_id": run["run_id"]}).status_code == 409
    assert submitted == []

    client.post(f"/spec-gen/runs/{run['run_id']}/accept")
    r = client.post("/asset-runs", json={"spec_run_id": run["run_id"]})
    assert r.status_code == 200
    doc = r.json()
    assert doc["asset_id"] == "char_kael"
    assert doc["stages"]["image"]["status"] == "queued"
    assert submitted[0]["prompt"] == KAEL_SPEC["generation_prompt"]


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


def _pipeline_run(db, asset_id, stage, status, url=None):
    db["manual_gen_stage_runs"].insert_one({
        "_id": f"{stage}-{status}", "char_label": asset_id, "stage": stage,
        "status": status, "image_url": url,
        "created_at": datetime.now(timezone.utc),
    })


def test_get_refreshes_stages_from_pipeline(env):
    client, db, _ = env
    spec_run = _accepted_spec_run(client)
    ar = client.post("/asset-runs", json={"spec_run_id": spec_run["run_id"]}).json()

    _pipeline_run(db, "char_kael", "flux", "done", "https://s3/kael.png")
    _pipeline_run(db, "char_kael", "hunyuan3d", "running")

    out = client.get(f"/asset-runs/{ar['asset_run_id']}").json()
    assert out["stages"]["image"] == {"status": "done", "url": "https://s3/kael.png"}
    assert out["stages"]["model3d"]["status"] == "running"
    assert out["status"] == "generating"
    assert out["manifest_entry"] is None


def test_rig_completion_writes_manifest(env):
    client, db, _ = env
    spec_run = _accepted_spec_run(client)
    ar = client.post("/asset-runs", json={"spec_run_id": spec_run["run_id"]}).json()

    _pipeline_run(db, "char_kael", "flux", "done", "https://s3/kael.png")
    _pipeline_run(db, "char_kael", "hunyuan3d", "done", "https://s3/kael.glb")
    _pipeline_run(db, "char_kael", "rig", "done", "https://s3/kael_rigged.glb")

    out = client.get(f"/asset-runs/{ar['asset_run_id']}").json()
    assert out["status"] == "complete"
    m = out["manifest_entry"]
    assert m == {
        "asset_id": "char_kael",
        "glb_url": "https://s3/kael_rigged.glb",
        "kind": "character",
        "attach_scripts": ["src/actors/kael.ts"],
        "supersedes": "char_kael@1.0",
    }
    # History list endpoint
    lst = client.get("/asset-runs", params={"project_id": "cyclezero", "asset_id": "char_kael"}).json()
    assert len(lst) == 1 and lst[0]["status"] == "complete"
