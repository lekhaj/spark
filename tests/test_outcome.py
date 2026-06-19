"""X5 — Factors + Outcome model: pure resolver/projection/contributors plus the
scene-hub aggregation and outcome-project route. SQLite + mongomock, mirroring
``test_cyclezero_graph.py``.
"""
import os

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cyclezero import outcome
from app.cyclezero import routes as cz_routes
from app.cyclezero.db import Base, get_db


# ── pure: project ─────────────────────────────────────────────────────────────
def _factor(key, kind="numeric", **data):
    return {"layer": "factor", "key": key, "name": key, "data": {"kind": kind, **data}}


def test_project_add_accumulates_and_clamps():
    entities = [
        _factor("trust", min=0, max=10, default=5),
        {"layer": "story", "key": "b1", "name": "b1", "data": {}},
        {"layer": "mission", "key": "m1", "name": "m1", "data": {}},
    ]
    rels = [
        {"src": "b1", "dst": "trust", "kind": "AFFECTS", "data": {"op": "add", "value": 3}},
        {"src": "m1", "dst": "trust", "kind": "AFFECTS", "data": {"op": "add", "value": 9}},
    ]
    # 5 + 3 + 9 = 17, clamped to max 10
    assert outcome.project(entities, rels)["trust"] == 10


def test_project_set_overrides_and_flag():
    entities = [_factor("alarm", kind="flag", default=False), _factor("hp", default=50),
                {"layer": "interaction", "key": "i1", "name": "i1", "data": {}}]
    rels = [
        {"src": "i1", "dst": "alarm", "kind": "AFFECTS", "data": {"op": "add", "value": True}},
        {"src": "i1", "dst": "hp", "kind": "AFFECTS", "data": {"op": "set", "value": 20}},
    ]
    state = outcome.project(entities, rels)
    assert state["alarm"] is True
    assert state["hp"] == 20


def test_project_defaults_when_no_edges():
    entities = [_factor("morale", default=7), _factor("flagged", kind="flag")]
    state = outcome.project(entities, [])
    assert state == {"morale": 7, "flagged": False}


# ── pure: resolve ─────────────────────────────────────────────────────────────
def test_resolve_priority_and_first_match():
    rules = [
        {"when": [{"factor": "trust", "op": ">=", "value": 5}], "ending": "ally", "priority": 1},
        {"when": [{"factor": "trust", "op": ">=", "value": 8}], "ending": "hero", "priority": 5},
    ]
    # trust=9 satisfies both; higher priority (hero) wins despite later declaration
    assert outcome.resolve({"trust": 9}, rules)["ending"] == "hero"
    # trust=6 satisfies only the ally rule
    assert outcome.resolve({"trust": 6}, rules)["ending"] == "ally"


def test_resolve_default_ending_fallback():
    rules = [{"when": [{"factor": "trust", "op": ">=", "value": 9}], "ending": "hero"}]
    res = outcome.resolve({"trust": 2}, rules, default_ending="neutral")
    assert res["ending"] == "neutral"
    assert res["matched_rule"] is None
    assert res["trace"][0]["ok"] is False


def test_resolve_empty_when_is_catch_all():
    rules = [
        {"when": [{"factor": "x", "op": ">", "value": 100}], "ending": "rare", "priority": 9},
        {"when": [], "ending": "default-ish", "priority": 0},
    ]
    assert outcome.resolve({"x": 1}, rules)["ending"] == "default-ish"


# ── pure: contributors ────────────────────────────────────────────────────────
def test_contributors_ranked_by_magnitude():
    entities = [_factor("trust"),
                {"layer": "story", "key": "b1", "name": "Beat 1", "data": {}},
                {"layer": "mission", "key": "m1", "name": "Mission 1", "data": {}}]
    rels = [
        {"src": "b1", "dst": "trust", "kind": "AFFECTS", "data": {"op": "add", "value": 2}},
        {"src": "m1", "dst": "trust", "kind": "AFFECTS", "data": {"op": "add", "value": -7}},
    ]
    ranked = outcome.contributors("trust", entities, rels)
    assert [c["src_key"] for c in ranked] == ["m1", "b1"]  # |−7| > |2|
    assert ranked[0]["src_layer"] == "mission"


# ── integration: scene hub + outcome-project route ────────────────────────────
@pytest.fixture()
def client(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool, future=True,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    mongo = mongomock.MongoClient()["World_builder_test"]
    monkeypatch.setattr(cz_routes, "_mongo", lambda: mongo)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(cz_routes.router, prefix="/cyclezero")
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_scene_hub_groups_members_and_inherits_globals(client):
    slug = "g"
    client.post("/cyclezero/games", json={"title": "G", "slug": slug})
    base = f"/cyclezero/games/{slug}/entities"
    client.post(base, json={"layer": "scene", "name": "Town", "key": "town"})
    client.post(base, json={"layer": "character", "name": "Kael", "key": "kael"})
    client.post(base, json={"layer": "system", "name": "Economy", "key": "economy",
                            "data": {"scope": "global"}})
    client.post(base, json={"layer": "factor", "name": "Trust", "key": "trust",
                            "data": {"kind": "numeric", "default": 5}})
    client.post(f"/cyclezero/games/{slug}/relations",
                json={"src": "town", "dst": "kael", "kind": "CONTAINS"})

    hub = client.get(f"/cyclezero/games/{slug}/scenes/town/hub").json()
    assert hub["scene"]["key"] == "town"
    assert [m["key"] for m in hub["members"]["character"]] == ["kael"]
    # global system + the factor are inherited (not CONTAINS-ed by the scene)
    assert any(s["key"] == "economy" for s in hub["inherited"]["systems"])
    assert any(f["key"] == "trust" for f in hub["inherited"]["factors"])

    # not-a-scene → 400
    assert client.get(f"/cyclezero/games/{slug}/scenes/kael/hub").status_code == 400


def test_outcome_project_route_with_overrides(client):
    slug = "g2"
    client.post("/cyclezero/games", json={"title": "G2", "slug": slug})
    base = f"/cyclezero/games/{slug}/entities"
    client.post(base, json={"layer": "factor", "name": "Trust", "key": "trust",
                            "data": {"kind": "numeric", "min": 0, "max": 10, "default": 5}})
    client.post(base, json={"layer": "story", "name": "Beat", "key": "beat"})
    client.post(base, json={"layer": "outcome", "name": "Endings", "key": "endings",
                            "data": {"rules": [
                                {"when": [{"factor": "trust", "op": ">=", "value": 8}],
                                 "ending": "hero"}],
                                "default_ending": "neutral"}})
    client.post(f"/cyclezero/games/{slug}/relations",
                json={"src": "beat", "dst": "trust", "kind": "AFFECTS",
                      "data": {"op": "add", "value": 1}})

    # baseline: 5 + 1 = 6 → below 8 → neutral
    base_proj = client.post(f"/cyclezero/games/{slug}/outcome/project", json={}).json()
    assert base_proj["factor_state"]["trust"] == 6
    assert base_proj["ending"] == "neutral"

    # what-if override trust=9 → hero
    over = client.post(f"/cyclezero/games/{slug}/outcome/project",
                       json={"overrides": {"trust": 9}}).json()
    assert over["ending"] == "hero"
    assert over["matched_rule"] == 0


def test_factor_contributors_route(client):
    slug = "g3"
    client.post("/cyclezero/games", json={"title": "G3", "slug": slug})
    base = f"/cyclezero/games/{slug}/entities"
    client.post(base, json={"layer": "factor", "name": "Trust", "key": "trust",
                            "data": {"kind": "numeric"}})
    client.post(base, json={"layer": "mission", "name": "Heist", "key": "heist"})
    client.post(f"/cyclezero/games/{slug}/relations",
                json={"src": "heist", "dst": "trust", "kind": "AFFECTS",
                      "data": {"op": "add", "value": -4}})
    contribs = client.get(f"/cyclezero/games/{slug}/factors/trust/contributors").json()
    assert contribs[0]["src_key"] == "heist"
    assert contribs[0]["value"] == -4
