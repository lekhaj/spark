"""
T04 spec-gen validation tests — fix packet, feedback, accept/reject, pinning.
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

ZONE_SCHEMA_V1 = {
    "type": "object",
    "required": ["zone_id", "timer_seconds"],
    "properties": {
        "zone_id": {"type": "string"},
        "timer_seconds": {"type": "integer", "minimum": 0},
    },
}

VALID = {"zone_id": "ruined_outpost", "timer_seconds": 480}
INVALID = {"zone_id": "ruined_outpost", "timer_seconds": "8 minutes"}


@pytest.fixture()
def client(monkeypatch):
    db = mongomock.MongoClient()["World_builder_test"]
    monkeypatch.setattr(schema_routes, "_db", lambda: db)
    monkeypatch.setattr(spec_gen_routes, "_db", lambda: db)
    app = FastAPI()
    app.include_router(schema_routes.router, prefix="/schemas")
    app.include_router(spec_gen_routes.router, prefix="/spec-gen")
    c = TestClient(app)
    c.post("/schemas/zone_spec", json={"json_schema": ZONE_SCHEMA_V1, "changelog": "v1"})
    return c


def _create(client, output, entity="zone:ruined_outpost"):
    return client.post(
        "/spec-gen/runs",
        json={
            "project_id": "cyclezero",
            "entity_id": entity,
            "stage": "zone_spec",
            "mode": "paste",
            "output": output,
        },
    ).json()


def test_valid_output_status_valid_no_errors(client):
    doc = _create(client, VALID)
    assert doc["status"] == "valid"
    assert doc["validation_errors"] == []


def test_invalid_output_errors_have_json_path(client):
    doc = _create(client, INVALID)
    assert doc["status"] == "invalid"
    paths = [e["json_path"] for e in doc["validation_errors"]]
    assert "/timer_seconds" in paths


def test_fix_packet_contains_three_blocks(client):
    doc = _create(client, INVALID)
    packet = client.get(f"/spec-gen/runs/{doc['run_id']}/fix-packet").json()["text"]
    assert "ERRORS:" in packet
    assert "/timer_seconds" in packet
    assert "THE SCHEMA (authoritative):" in packet
    assert '"timer_seconds"' in packet
    assert "YOUR PREVIOUS RESPONSE:" in packet
    assert "8 minutes" in packet


def test_fix_packet_for_valid_run_409(client):
    doc = _create(client, VALID)
    r = client.get(f"/spec-gen/runs/{doc['run_id']}/fix-packet")
    assert r.status_code == 409


def test_accept_valid_run_and_409_on_invalid(client):
    ok = _create(client, VALID)
    r = client.post(f"/spec-gen/runs/{ok['run_id']}/accept")
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"

    bad = _create(client, INVALID)
    r = client.post(f"/spec-gen/runs/{bad['run_id']}/accept")
    assert r.status_code == 409


def test_one_accepted_per_major(client):
    a = _create(client, VALID)
    b = _create(client, VALID)
    client.post(f"/spec-gen/runs/{a['run_id']}/accept")
    client.post(f"/spec-gen/runs/{b['run_id']}/accept")
    a_after = client.get(f"/spec-gen/runs/{a['run_id']}").json()
    b_after = client.get(f"/spec-gen/runs/{b['run_id']}").json()
    assert a_after["status"] == "rejected"
    assert b_after["status"] == "accepted"


def test_create_after_accept_starts_major_2(client):
    a = _create(client, VALID)
    client.post(f"/spec-gen/runs/{a['run_id']}/accept")
    nxt = _create(client, VALID)
    assert (nxt["major"], nxt["minor"]) == (2, 0)


def test_feedback_appends_in_order(client):
    doc = _create(client, VALID)
    client.post(f"/spec-gen/runs/{doc['run_id']}/feedback", json={"note": "first"})
    client.post(f"/spec-gen/runs/{doc['run_id']}/feedback", json={"note": "second"})
    fetched = client.get(f"/spec-gen/runs/{doc['run_id']}").json()
    assert [f["note"] for f in fetched["feedback"]] == ["first", "second"]
    assert all("at" in f for f in fetched["feedback"])


def test_validation_pins_schema_version_across_schema_bumps(client):
    doc = _create(client, VALID)
    assert doc["schema_version"] == 1

    # Schema v2 adds a required field the old artifact doesn't have.
    v2 = {
        "type": "object",
        "required": ["zone_id", "timer_seconds", "task"],
        "properties": {
            "zone_id": {"type": "string"},
            "timer_seconds": {"type": "integer"},
            "task": {"type": "string"},
        },
    }
    client.post("/schemas/zone_spec", json={"json_schema": v2, "changelog": "add task"})

    # Old run revalidates against its PINNED v1 → stays valid.
    r = client.post(f"/spec-gen/runs/{doc['run_id']}/revalidate").json()
    assert r["status"] == "valid"
    assert r["schema_version"] == 1

    # A NEW run pins v2 and the same output is now invalid (missing task).
    new = _create(client, VALID)
    assert new["schema_version"] == 2
    assert new["status"] == "invalid"
