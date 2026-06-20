"""Creator orchestrator tests — deterministic dispatch over a fake LLM.

No live Bedrock: a ``FakeProvider`` returns canned tool calls so we exercise the
deterministic layer (``creator_agent.run_turn``) and the routes. SQLite (graph) +
mongomock (memory/sessions), mirroring ``test_cyclezero_graph.py``.
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

from app.cyclezero import creator_agent, service, schemas
from app.cyclezero.db import Base, get_db as get_sql_db
from app.routes import creator_routes


# ── fakes / fixtures ──────────────────────────────────────────────────────────
class FakeProvider:
    """Returns one scripted {text, tool_calls} per chat_tools call."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.seen = []

    def chat_tools(self, system, messages, tools, tool_choice="auto", max_tokens=4096):
        self.seen.append({"system": system, "messages": messages})
        return self.scripts.pop(0)


def _tc(tool, **inp):
    return {"name": tool, "input": inp, "id": tool + "1"}


@pytest.fixture()
def sql():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool, future=True,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = Session()
    db._factory = Session
    yield db
    db.close()


@pytest.fixture()
def mongo():
    return mongomock.MongoClient()["World_builder_test"]


# ── tool dispatch ─────────────────────────────────────────────────────────────
def test_start_game_creates_game(sql, mongo):
    prov = FakeProvider({"text": "Starting it!", "tool_calls": [_tc("start_game", title="Diablo 2")]})
    out = creator_agent.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug=None, user_text="start a game called Diablo 2", known_layers=[],
    )
    assert out["game_slug"] == "diablo-2"
    assert any(s["kind"] == "game" for s in out["saved"])
    assert service.get_game(sql, "diablo-2") is not None


def test_save_facts_merges_into_memory(sql, mongo):
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    prov = FakeProvider({"text": "Got it.", "tool_calls": [
        _tc("save_facts", facts={"genre": "ARPG", "references": ["Diablo 2"]})]})
    out = creator_agent.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="it's an ARPG like Diablo 2", known_layers=[],
    )
    assert any(s["kind"] == "facts" for s in out["saved"])
    mem = creator_agent.load_memory(mongo, "u1", "diablo-2")
    assert mem["facts"]["genre"] == "ARPG"
    assert mem["facts"]["references"] == ["Diablo 2"]


def test_upsert_entity_creates_and_rejects_unknown_layer(sql, mongo):
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    prov = FakeProvider({"text": "", "tool_calls": [
        _tc("upsert_entity", layer="character", name="Necromancer", data={"role": "player"}),
        _tc("upsert_entity", layer="bogus_layer", name="Junk"),
    ]})
    out = creator_agent.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="add a Necromancer", known_layers=["character"],
    )
    kinds = {s["kind"] for s in out["saved"]}
    assert "entity" in kinds and "rejected" in kinds
    game = service.get_game(sql, "diablo-2")
    ents = service.list_entities(sql, game)
    assert [e.name for e in ents] == ["Necromancer"]  # bogus one was NOT written


def test_save_confident_and_ask_the_gap(sql, mongo):
    """Mode: save the sure part immediately AND ask only the uncertain field."""
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    prov = FakeProvider({"text": "Saved the genre; one question:", "tool_calls": [
        _tc("save_facts", facts={"genre": "ARPG"}),
        _tc("ask_clarification", field="perspective", header="View",
            question="Which camera?", options=["Isometric", "Over-the-shoulder"]),
    ]})
    out = creator_agent.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="ARPG, not sure on camera", known_layers=[],
    )
    # confident part persisted
    assert creator_agent.load_memory(mongo, "u1", "diablo-2")["facts"]["genre"] == "ARPG"
    # gap surfaced as a popup + tracked as an open question
    assert out["pending_question"]["field"] == "perspective"
    assert out["pending_question"]["options"] == ["Isometric", "Over-the-shoulder"]
    oq = creator_agent.load_memory(mongo, "u1", "diablo-2")["open_questions"]
    assert any(q["field"] == "perspective" for q in oq)


def test_answer_resolves_open_question(sql, mongo):
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    # seed an open question
    creator_agent.save_memory(mongo, "u1", "a@b.com", "diablo-2", {"genre": "ARPG"},
                              [{"field": "perspective", "question": "?", "options": []}])
    prov = FakeProvider({"text": "Locked in isometric.", "tool_calls": [
        _tc("save_facts", facts={"perspective": "Isometric"})]})
    creator_agent.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="Answer to 'perspective': Isometric",
        known_layers=[], resolve_field="perspective",
    )
    mem = creator_agent.load_memory(mongo, "u1", "diablo-2")
    assert mem["facts"]["perspective"] == "Isometric"
    assert mem["open_questions"] == []  # resolved


# ── memory / session persistence ──────────────────────────────────────────────
def test_turns_persist_and_cap_at_50(mongo):
    for i in range(60):
        creator_agent.append_turn(mongo, "u1", "g", {"role": "user", "text": f"m{i}"})
    turns = creator_agent.load_turns(mongo, "u1", "g")
    assert len(turns) == creator_agent.TURN_CAP == 50
    assert turns[-1]["text"] == "m59"  # newest kept
    assert turns[0]["text"] == "m10"   # oldest dropped


# ── routes ────────────────────────────────────────────────────────────────────
@pytest.fixture()
def client(sql, mongo, monkeypatch):
    monkeypatch.setattr(creator_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(creator_routes, "_known_layers", lambda: ["character"])

    def override_sql():
        db = sql._factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(creator_routes.router)
    app.dependency_overrides[get_sql_db] = override_sql
    c = TestClient(app)
    c._mongo = mongo
    return c


def _hdr():
    return {"X-Studio-Uid": "u1", "X-Studio-Email": "a@b.com"}


def test_route_turn_then_state_and_latest(client, monkeypatch):
    prov = FakeProvider(
        {"text": "Starting!", "tool_calls": [_tc("start_game", title="Diablo 2")]},
        {"text": "Saved.", "tool_calls": [_tc("save_facts", facts={"genre": "ARPG"})]},
    )
    monkeypatch.setattr(creator_routes, "_provider", lambda: prov)
    # turn 1: start the game
    r1 = client.post("/creator/turn", json={"text": "start a game called Diablo 2"}, headers=_hdr())
    assert r1.json()["game_slug"] == "diablo-2"

    # turn 2: save a fact into that game
    r2 = client.post("/creator/turn",
                     json={"game_slug": "diablo-2", "text": "it's an ARPG"}, headers=_hdr())
    assert any(s["kind"] == "facts" for s in r2.json()["saved"])

    # state reloads memory + history
    st = client.get("/creator/state", params={"game_slug": "diablo-2"}, headers=_hdr()).json()
    assert st["facts"]["genre"] == "ARPG"
    assert len(st["turns"]) >= 2

    # latest finds this user's game
    lt = client.get("/creator/latest", headers=_hdr()).json()
    assert lt["game_slug"] == "diablo-2"


def test_route_turn_never_500s_on_provider_error(client, monkeypatch):
    class Boom:
        def chat_tools(self, *a, **k):
            raise RuntimeError("bedrock down")

    monkeypatch.setattr(creator_routes, "_provider", lambda: Boom())
    r = client.post("/creator/turn", json={"game_slug": "diablo-2", "text": "hi"}, headers=_hdr())
    assert r.status_code == 200
    assert r.json()["saved"] == []
