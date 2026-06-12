"""
T00 Schema Manager tests — runs against mongomock, no live Mongo needed.
"""

import os

# Importing app.routes pulls in app.config.Settings, which requires these.
# Harmless dummies — the tests run entirely against mongomock.
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

from app.routes import schema_routes


@pytest.fixture()
def client(monkeypatch):
    db = mongomock.MongoClient()["World_builder_test"]
    monkeypatch.setattr(schema_routes, "_db", lambda: db)
    app = FastAPI()
    app.include_router(schema_routes.router, prefix="/schemas")
    return TestClient(app)


VALID = {"type": "object", "properties": {"name": {"type": "string"}}}


def _post(client, key="mission_spec", schema=VALID, changelog="initial", **extra):
    body = {"json_schema": schema, "changelog": changelog, **extra}
    return client.post(f"/schemas/{key}", json=body)


def test_post_new_key_creates_v1_active(client):
    r = _post(client)
    assert r.status_code == 200
    doc = r.json()
    assert doc["version"] == 1
    assert doc["active"] is True


def test_post_same_key_bumps_version_and_deactivates_previous(client):
    _post(client)
    r = _post(client, changelog="second")
    assert r.json()["version"] == 2
    versions = client.get("/schemas/mission_spec").json()
    by_v = {d["version"]: d for d in versions}
    assert by_v[2]["active"] is True
    assert by_v[1]["active"] is False


def test_post_invalid_json_schema_422(client):
    r = _post(client, schema={"type": 42})
    assert r.status_code == 422


def test_post_without_changelog_422(client):
    r = client.post("/schemas/mission_spec", json={"json_schema": VALID})
    assert r.status_code == 422
    # Empty changelog is also rejected
    r = client.post("/schemas/mission_spec", json={"json_schema": VALID, "changelog": ""})
    assert r.status_code == 422


def test_activate_rolls_back_active_pointer(client):
    _post(client)
    _post(client, changelog="v2")
    r = client.post("/schemas/mission_spec/activate/1")
    assert r.status_code == 200
    versions = client.get("/schemas/mission_spec").json()
    by_v = {d["version"]: d for d in versions}
    assert by_v[1]["active"] is True
    assert by_v[2]["active"] is False
    active = client.get("/schemas/mission_spec/active").json()
    assert active["version"] == 1


def test_engine_bound_post_flags_engine_sync_required(client):
    r = _post(client, key="zone_spec", engine_bound=True)
    assert r.json()["engine_sync_required"] is True
    # Inherited on subsequent versions without explicit flag
    r2 = _post(client, key="zone_spec", changelog="v2")
    assert r2.json()["engine_sync_required"] is True


def test_get_nonexistent_key_404(client):
    assert client.get("/schemas/nope").status_code == 404
    assert client.get("/schemas/nope/active").status_code == 404
    assert client.get("/schemas/nope/v/1").status_code == 404
    assert client.post("/schemas/nope/activate/1").status_code == 404
