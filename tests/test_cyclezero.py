"""CycleZero game-authoring backend tests.

DB layer runs on in-memory SQLite (portable column types make the same models
work there and on Postgres). Contract (P6) and matching (P7) are pure logic.
"""
import os

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cyclezero import contract, generation, matching
from app.cyclezero.db import Base, get_db
from app.cyclezero.routes import router


@pytest.fixture()
def client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    # Don't hit real Mongo during generate tests.
    monkeypatch.setattr(
        generation,
        "submit",
        lambda job, entity, game=None: {"submitted": True, "mongo_request_id": "x"},
    )

    app = FastAPI()
    app.include_router(router, prefix="/cyclezero")
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


# ── games (P2) ────────────────────────────────────────────────────────────────
def test_create_game_gets_uri(client):
    r = client.post("/cyclezero/games", json={"title": "Ruined Outpost"})
    assert r.status_code == 201
    body = r.json()
    assert body["slug"] == "ruined-outpost"
    assert body["uri"] == "/cyclezero/games/ruined-outpost"
    assert body["status"] == "draft"


def test_duplicate_slug_conflicts(client):
    client.post("/cyclezero/games", json={"title": "X", "slug": "dup"})
    r = client.post("/cyclezero/games", json={"title": "Y", "slug": "dup"})
    assert r.status_code == 409


def test_list_and_get_and_update_game(client):
    client.post("/cyclezero/games", json={"title": "Alpha", "slug": "alpha"})
    assert len(client.get("/cyclezero/games").json()) == 1
    assert client.get("/cyclezero/games/alpha").json()["title"] == "Alpha"
    r = client.patch("/cyclezero/games/alpha", json={"status": "live"})
    assert r.json()["status"] == "live"
    assert client.get("/cyclezero/games/missing").status_code == 404


# ── entities (P3) + relations (P4) ──────────────────────────────────────────
def _seed_scene(client, slug="game"):
    client.post("/cyclezero/games", json={"title": "Game", "slug": slug})
    base = f"/cyclezero/games/{slug}/entities"
    client.post(base, json={"layer": "scene", "name": "Main", "key": "main",
                            "data": {"camera": {"viewHeight": 28}, "quality": "high"}})
    client.post(base, json={"layer": "character", "name": "Kael", "key": "kael",
                            "data": {"role": "player", "spawn": [0, 1, 0], "speed": 6,
                                     "glb": "assets/kael.glb"}})
    client.post(base, json={"layer": "collider", "name": "W Wall", "key": "w_wall",
                            "data": {"shape": "box", "position": [-12, 1, 0], "size": [1, 3, 10]}})
    client.post(base, json={"layer": "trigger", "name": "NE Tower", "key": "ne_tower",
                            "data": {"shape": "sphere", "center": [12, 0, 12], "radius": 4,
                                     "event": "reveal_hidden_tower"}})
    return slug


def test_entities_crud_and_filter(client):
    slug = _seed_scene(client)
    allents = client.get(f"/cyclezero/games/{slug}/entities").json()
    assert len(allents) == 4
    chars = client.get(f"/cyclezero/games/{slug}/entities?layer=character").json()
    assert [c["key"] for c in chars] == ["kael"]


def test_relations(client):
    slug = _seed_scene(client)
    r = client.post(f"/cyclezero/games/{slug}/relations",
                    json={"src": "kael", "dst": "main", "kind": "APPEARS_IN"})
    assert r.status_code == 201
    assert len(client.get(f"/cyclezero/games/{slug}/relations").json()) == 1
    # unknown entity → 400
    bad = client.post(f"/cyclezero/games/{slug}/relations",
                      json={"src": "ghost", "dst": "main", "kind": "X"})
    assert bad.status_code == 400


# ── generation (P5) ──────────────────────────────────────────────────────────
def test_generate_creates_job(client):
    slug = _seed_scene(client)
    r = client.post(f"/cyclezero/games/{slug}/entities/kael/generate",
                    json={"kind": "3d", "params": {"morphology": "B1_humanoid"}})
    assert r.status_code == 202
    job = r.json()
    assert job["kind"] == "3d"
    assert job["status"] == "queued"
    jobs = client.get(f"/cyclezero/games/{slug}/jobs").json()
    assert len(jobs) == 1


# ── contract (P6) ────────────────────────────────────────────────────────────
def test_contract_endpoint_builds_from_graph(client):
    slug = _seed_scene(client)
    c = client.get(f"/cyclezero/games/{slug}/contract").json()
    assert c["contractVersion"] == "1.0"
    assert c["id"] == slug
    assert c["camera"]["viewHeight"] == 28
    assert c["player"]["asset"] == "kael"
    assert [e["id"] for e in c["entities"]] == ["w_wall"]
    assert [t["id"] for t in c["triggers"]] == ["ne_tower"]
    assert any(a["id"] == "kael" for a in c["assets"])


def test_build_contract_defaults_without_scene():
    c = contract.build_contract({"slug": "s", "title": "T"}, [])
    assert c["camera"]["viewHeight"] == 30
    assert c["player"]["asset"] is None
    assert c["environment"]["ground"]["size"] == [40, 40]
    assert len(c["environment"]["lights"]) == 2


# ── matching (P7) ────────────────────────────────────────────────────────────
def test_match_full_coverage():
    entities = [
        {"layer": "collider", "key": "w_wall", "name": "W", "data": {}},
        {"layer": "trigger", "key": "ne_tower", "name": "T", "data": {}},
        {"layer": "character", "key": "kael", "name": "K", "data": {"glb": "k.glb"}},
    ]
    built = contract.build_contract({"slug": "s", "title": "T"}, entities)
    report = matching.match(entities, built)
    assert report["ok"] is True
    assert report["coverage"] == 1.0
    assert report["counts"]["missing"] == 0


def test_match_reports_missing():
    entities = [
        {"layer": "collider", "key": "w_wall", "name": "W", "data": {}},
        {"layer": "trigger", "key": "ne_tower", "name": "T", "data": {}},
    ]
    built = {"entities": [{"id": "w_wall"}], "triggers": [], "assets": [], "player": {}}
    report = matching.match(entities, built)
    assert report["ok"] is False
    assert {"kind": "trigger", "key": "ne_tower"} in report["missing"]
    assert report["coverage"] == 0.5
