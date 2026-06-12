"""
T03 spec-gen run tests — creation, versioning, reads. mongomock, no live Mongo.
"""

import os

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import schema_routes, spec_gen_routes

ZONE_SCHEMA = {
    "type": "object",
    "required": ["zone_id", "timer_seconds"],
    "properties": {
        "zone_id": {"type": "string"},
        "timer_seconds": {"type": "integer", "minimum": 0},
    },
}

VALID_OUTPUT = {"zone_id": "ruined_outpost", "timer_seconds": 480}


@pytest.fixture()
def client(monkeypatch):
    db = mongomock.MongoClient()["World_builder_test"]
    monkeypatch.setattr(schema_routes, "_db", lambda: db)
    monkeypatch.setattr(spec_gen_routes, "_db", lambda: db)
    app = FastAPI()
    app.include_router(schema_routes.router, prefix="/schemas")
    app.include_router(spec_gen_routes.router, prefix="/spec-gen")
    c = TestClient(app)
    # Seed an active zone_spec schema for runs to pin against.
    c.post("/schemas/zone_spec", json={"json_schema": ZONE_SCHEMA, "changelog": "test seed"})
    return c


def _create(client, output=VALID_OUTPUT, entity="zone:ruined_outpost", stage="zone_spec", **extra):
    return client.post(
        "/spec-gen/runs",
        json={
            "project_id": "cyclezero",
            "entity_id": entity,
            "stage": stage,
            "mode": "paste",
            "output": output,
            **extra,
        },
    )


def test_first_run_is_1_0_with_pinned_schema(client):
    r = _create(client)
    assert r.status_code == 200
    doc = r.json()
    assert (doc["major"], doc["minor"]) == (1, 0)
    assert doc["schema_version"] == 1
    # T04 validates inline on create, so a valid paste lands as "valid".
    assert doc["status"] == "valid"


def test_second_create_same_entity_stage_is_1_1(client):
    _create(client)
    doc = _create(client).json()
    assert (doc["major"], doc["minor"]) == (1, 1)


def test_unknown_stage_404(client):
    r = _create(client, stage="nonexistent_spec")
    assert r.status_code == 404


def test_agent_mode_501(client):
    r = client.post(
        "/spec-gen/runs",
        json={
            "project_id": "cyclezero",
            "entity_id": "zone:x",
            "stage": "zone_spec",
            "mode": "agent",
            "output": {},
        },
    )
    assert r.status_code == 501
    assert "agent mode" in r.json()["detail"]


def test_versions_endpoint_newest_first(client):
    _create(client)
    _create(client)
    _create(client)
    versions = client.get("/spec-gen/cyclezero/zone:ruined_outpost/zone_spec/versions").json()
    pairs = [(v["major"], v["minor"]) for v in versions]
    assert pairs == [(1, 2), (1, 1), (1, 0)]


def test_entities_endpoint_aggregates_latest_status(client):
    _create(client, entity="zone:ruined_outpost")
    _create(client, output={"zone_id": "x"}, entity="zone:ruined_outpost")  # invalid (missing timer)
    _create(client, entity="mission:day1", stage="zone_spec")
    ents = client.get("/spec-gen/cyclezero/entities").json()
    by_id = {e["entity_id"]: e for e in ents}
    assert set(by_id) == {"zone:ruined_outpost", "mission:day1"}
    # Latest run for ruined_outpost is the invalid 1.1
    zs = by_id["zone:ruined_outpost"]["stages"]["zone_spec"]
    assert zs["version"] == "1.1"
    assert zs["status"] == "invalid"


def test_input_ref_and_prompt_text_round_trip(client):
    prompt = "Génère la zone 🏰 — survive 8 minutes\nwith newlines"
    doc = _create(client, input_ref="abc123", prompt_text=prompt).json()
    fetched = client.get(f"/spec-gen/runs/{doc['run_id']}").json()
    assert fetched["input_ref"] == "abc123"
    assert fetched["prompt_text"] == prompt


def test_different_entities_have_independent_counters(client):
    a = _create(client, entity="zone:a").json()
    b = _create(client, entity="zone:b").json()
    assert (a["major"], a["minor"]) == (1, 0)
    assert (b["major"], b["minor"]) == (1, 0)
