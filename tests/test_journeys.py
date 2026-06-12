"""
U01 journeys + impact items tests (+ U04 escalate). mongomock, no live Mongo.
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

from app.routes import journey_routes

STAMINA_JOURNEY = {
    "project_id": "cyclezero",
    "kind": "system",
    "title": "Stamina system",
    "user_intent": "add a stamina bar that drains on sprint",
    "rail": [
        {"label": "Understand", "sub": "refiner chat"},
        {"label": "Spec", "sub": "system_behavior_spec"},
        {"label": "Impacts", "sub": "resolve feed"},
        {"label": "Done", "sub": ""},
    ],
    "items": [
        {"impact": "direct", "icon": "⚙", "title": "New system spec",
         "body": "stamina system_behavior_spec needed",
         "target": {"entity_id": "system:stamina", "stage": "system_behavior_spec"},
         "workspace": "spec_code", "suggested_intent": "stamina system"},
        {"impact": "high", "icon": "⛰", "title": "Zone revalidation",
         "body": "ruined_outpost references player stats",
         "target": {"entity_id": "zone:ruined_outpost", "stage": "zone_spec"},
         "workspace": "revalidate"},
        {"impact": "ripple", "icon": "ℹ", "title": "HUD note",
         "body": "HUD will show one more bar", "workspace": "none"},
    ],
}


@pytest.fixture()
def client(monkeypatch):
    db = mongomock.MongoClient()["World_builder_test"]
    monkeypatch.setattr(journey_routes, "_db", lambda: db)
    app = FastAPI()
    app.include_router(journey_routes.router, prefix="")
    return TestClient(app)


def _create(client, **overrides):
    return client.post("/journeys", json={**STAMINA_JOURNEY, **overrides}).json()


def test_create_links_items_and_list_filters(client):
    j = _create(client)
    assert j["status"] == "active"
    assert len(j["items"]) == 3
    assert all(i["journey_id"] == j["journey_id"] for i in j["items"])
    # First rail step activates on create.
    assert j["rail"][0]["state"] == "active"

    full = client.get(f"/journeys/id/{j['journey_id']}").json()
    assert len(full["items"]) == 3

    assert len(client.get("/journeys/cyclezero", params={"status": "active"}).json()) == 1
    assert client.get("/journeys/cyclezero", params={"status": "complete"}).json() == []
    assert client.get("/journeys/otherproject").json() == []


def test_resolve_stores_note_and_double_resolve_409(client):
    j = _create(client)
    item = j["items"][0]
    r = client.post(f"/impact-items/{item['item_id']}/resolve", json={"note": "spec v1.0 accepted"})
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "resolved"
    assert out["resolution_note"] == "spec v1.0 accepted"
    assert out["resolved_at"]

    assert client.post(f"/impact-items/{item['item_id']}/resolve", json={"note": "again"}).status_code == 409


def test_dismiss_counts_as_closed(client):
    j = _create(client)
    items = j["items"]
    client.post(f"/impact-items/{items[0]['item_id']}/resolve", json={"note": "done"})
    r = client.post(f"/impact-items/{items[1]['item_id']}/dismiss", json={"reason": "intentional inconsistency"})
    assert r.json()["status"] == "dismissed"
    # Both blocking items closed (one dismissed) → journey complete.
    assert r.json()["journey"]["status"] == "complete"


def test_auto_complete_marks_rail_and_ripples_dont_block(client):
    j = _create(client)
    items = j["items"]
    # Resolve only the two blocking items; the ripple stays open and must not block.
    client.post(f"/impact-items/{items[0]['item_id']}/resolve", json={"note": "a"})
    out = client.post(f"/impact-items/{items[1]['item_id']}/resolve", json={"note": "b"}).json()
    journey = out["journey"]
    assert journey["status"] == "complete"
    assert journey["completed_at"]
    assert all(s["state"] == "done" for s in journey["rail"])
    # Ripple auto-closed with the journey.
    full = client.get(f"/journeys/id/{j['journey_id']}").json()
    ripple = next(i for i in full["items"] if i["impact"] == "ripple")
    assert ripple["status"] == "resolved"


def test_rail_update_validates(client):
    j = _create(client)
    jid = j["journey_id"]
    r = client.post(f"/journeys/{jid}/rail/1", json={"state": "done"})
    assert r.json()["rail"][1]["state"] == "done"
    assert client.post(f"/journeys/{jid}/rail/9", json={"state": "done"}).status_code == 422
    assert client.post(f"/journeys/{jid}/rail/1", json={"state": "bogus"}).status_code == 422


def test_escalate_switches_workspace_open_only(client):
    j = _create(client)
    reval = next(i for i in j["items"] if i["workspace"] == "revalidate")
    r = client.post(f"/impact-items/{reval['item_id']}/escalate", json={"workspace": "spec_code"})
    assert r.status_code == 200
    assert r.json()["workspace"] == "spec_code"

    client.post(f"/impact-items/{reval['item_id']}/resolve", json={"note": "fixed"})
    assert client.post(
        f"/impact-items/{reval['item_id']}/escalate", json={"workspace": "spec_code"}
    ).status_code == 409
