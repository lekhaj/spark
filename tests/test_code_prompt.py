"""
T10 backend tests — auto-draft on engine-bound bump, template render, applied.
mongomock, no live Mongo.
"""

import os

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

import json

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.lib.code_prompt_template import render_system_spec_prompt
from app.routes import schema_routes, spec_gen_routes

ZONE_V1 = {
    "type": "object",
    "required": ["zone_id"],
    "properties": {"zone_id": {"type": "string"}},
}
ZONE_V2 = {
    "type": "object",
    "required": ["zone_id", "timer_seconds"],
    "properties": {
        "zone_id": {"type": "string"},
        "timer_seconds": {"type": "integer", "minimum": 0},
    },
}

SYSTEM_SPEC = json.load(
    open(os.path.join(os.path.dirname(__file__), "fixtures", "system_behavior_spec.valid.1.json"))
)


@pytest.fixture()
def client(monkeypatch):
    db = mongomock.MongoClient()["World_builder_test"]
    monkeypatch.setattr(schema_routes, "_db", lambda: db)
    monkeypatch.setattr(spec_gen_routes, "_db", lambda: db)
    app = FastAPI()
    app.include_router(schema_routes.router, prefix="/schemas")
    app.include_router(spec_gen_routes.router, prefix="/spec-gen")
    return TestClient(app)


def _post_schema(client, key, schema, engine_bound=None, changelog="test"):
    body = {"json_schema": schema, "changelog": changelog}
    if engine_bound is not None:
        body["engine_bound"] = engine_bound
    return client.post(f"/schemas/{key}", json=body)


def test_engine_bound_bump_auto_drafts_code_change_prompt(client):
    _post_schema(client, "zone_spec", ZONE_V1, engine_bound=True)
    r = _post_schema(client, "zone_spec", ZONE_V2, changelog="add timer_seconds")
    out = r.json()
    assert out["engine_sync_required"] is True
    run_id = out["code_change_prompt_run_id"]

    run = client.get(f"/spec-gen/runs/{run_id}").json()
    assert run["stage"] == "code_change_prompt"
    assert run["status"] == "valid"  # draft, not accepted
    assert run["output"]["source_kind"] == "schema_bump"
    assert run["output"]["applied"] is False
    # Both schema versions present in the prompt.
    assert "zone_spec v1" in run["output"]["prompt_text"]
    assert "zone_spec v2" in run["output"]["prompt_text"]
    assert "timer_seconds" in run["output"]["prompt_text"]


def test_non_engine_bound_bump_drafts_nothing(client):
    _post_schema(client, "mission_spec", ZONE_V1, engine_bound=False)
    out = _post_schema(client, "mission_spec", ZONE_V2).json()
    assert "engine_sync_required" not in out
    assert "code_change_prompt_run_id" not in out
    runs = client.get("/spec-gen/cyclezero/entities").json()
    assert runs == []


def test_engine_bound_v1_drafts_nothing(client):
    out = _post_schema(client, "terrain_spec", ZONE_V1, engine_bound=True).json()
    assert out["engine_sync_required"] is True
    assert "code_change_prompt_run_id" not in out


def test_system_spec_template_renders_everything():
    text = render_system_spec_prompt(SYSTEM_SPEC)
    assert SYSTEM_SPEC["module_path"] in text
    assert SYSTEM_SPEC["interface_name"] in text
    for rule in SYSTEM_SPEC["behavior_rules"]:
        assert rule["when"] in text
        assert rule["then"] in text
    for t in SYSTEM_SPEC["acceptance_tests"]:
        assert t["name"] in text
        assert t["given"] in text
        assert t["expect"] in text


def test_applied_requires_accepted(client):
    _post_schema(client, "zone_spec", ZONE_V1, engine_bound=True)
    out = _post_schema(client, "zone_spec", ZONE_V2).json()
    run_id = out["code_change_prompt_run_id"]

    # Draft (valid, not accepted) → 409.
    r = client.post(f"/spec-gen/runs/{run_id}/applied", json={"commit": "abc1234"})
    assert r.status_code == 409

    client.post(f"/spec-gen/runs/{run_id}/accept")
    r = client.post(f"/spec-gen/runs/{run_id}/applied", json={"commit": "abc1234"})
    assert r.status_code == 200
    out = r.json()["output"]
    assert out["applied"] is True
    assert out["applied_commit"] == "abc1234"


def test_applied_wrong_stage_409(client):
    _post_schema(client, "zone_spec", ZONE_V1)
    r = client.post(
        "/spec-gen/runs",
        json={
            "project_id": "cyclezero",
            "entity_id": "zone:x",
            "stage": "zone_spec",
            "mode": "paste",
            "output": {"zone_id": "x"},
        },
    )
    run_id = r.json()["run_id"]
    client.post(f"/spec-gen/runs/{run_id}/accept")
    assert client.post(f"/spec-gen/runs/{run_id}/applied", json={"commit": "abc"}).status_code == 409
