"""Creator orchestrator tests — deterministic dispatch over a fake LLM.

No live Bedrock: a ``FakeProvider`` returns canned tool calls so we exercise the
orchestrator (``agents.orchestrator.run_turn``) + the shared deterministic gate
(``creator_agent.apply_tool_calls``) + the routes. SQLite (graph) + mongomock
(memory/sessions), mirroring ``test_cyclezero_graph.py``.
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
from app.cyclezero.agents import orchestrator, registry
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


# A minimal metamodel for link_entities tests: two relation kinds with layer contracts.
_MM = {
    "layers": {
        "system": {"layer": "system"},
        "factor": {"layer": "factor"},
        "character": {"layer": "character"},
        "scene": {"layer": "scene"},
    },
    "relation_types": {
        "REQUIRES": {"kind": "REQUIRES", "src_layers": ["system"], "dst_layers": ["system"],
                     "src_cardinality": "many", "dst_cardinality": "many"},
        "AFFECTS": {"kind": "AFFECTS", "src_layers": ["system"], "dst_layers": ["factor"],
                    "src_cardinality": "many", "dst_cardinality": "many"},
    },
}
_MM_LAYERS = list(_MM["layers"].keys())


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
    out = orchestrator.run_turn(
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
    out = orchestrator.run_turn(
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
    out = orchestrator.run_turn(
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
    out = orchestrator.run_turn(
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
    orchestrator.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="Answer to 'perspective': Isometric",
        known_layers=[], resolve_field="perspective",
    )
    mem = creator_agent.load_memory(mongo, "u1", "diablo-2")
    assert mem["facts"]["perspective"] == "Isometric"
    assert mem["open_questions"] == []  # resolved


# ── relations (the stamina example) ───────────────────────────────────────────
def test_link_entities_creates_and_rejects_illegal_edges(sql, mongo):
    """One turn: create three entities and wire them up; legal edges persist,
    illegal/unknown ones are rejected by the relation contract."""
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    prov = FakeProvider({"text": "Wiring stamina up.", "tool_calls": [
        _tc("upsert_entity", layer="system", name="Stamina"),
        _tc("upsert_entity", layer="system", name="Power Attack"),
        _tc("upsert_entity", layer="factor", name="Defense"),
        # legal: system REQUIRES system
        _tc("link_entities", src="Power Attack", kind="REQUIRES", dst="Stamina"),
        # legal: system AFFECTS factor (by name → resolves to slug key)
        _tc("link_entities", src="Stamina", kind="AFFECTS", dst="Defense",
            data={"op": "add", "value": -10}),
        # illegal: AFFECTS dst must be a factor, Stamina is a system
        _tc("link_entities", src="Power Attack", kind="AFFECTS", dst="Stamina"),
        # unknown kind
        _tc("link_entities", src="Stamina", kind="ZAPS", dst="Defense"),
        # missing endpoint
        _tc("link_entities", src="Stamina", kind="REQUIRES", dst="Nonexistent"),
    ]})
    out = orchestrator.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2",
        user_text="stamina drains on power attacks; low stamina weakens defense",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    rels = [s for s in out["saved"] if s["kind"] == "relation"]
    rejects = [s for s in out["saved"] if s["kind"] == "rejected"]
    assert {(r["src"], r["rel_kind"], r["dst"]) for r in rels} == {
        ("power-attack", "REQUIRES", "stamina"),
        ("stamina", "AFFECTS", "defense"),
    }
    assert len(rejects) == 3  # illegal layer + unknown kind + missing endpoint
    # the edges are actually in the graph
    game = service.get_game(sql, "diablo-2")
    assert len(service.list_relations(sql, game)) == 2
    # AFFECTS edge carried its data delta
    affects = next(r for r in service.list_relations(sql, game) if r.kind == "AFFECTS")
    assert affects.data == {"op": "add", "value": -10}


def test_link_rejected_when_no_metamodel(sql, mongo):
    """Without a metamodel the contract can't be enforced → relations are rejected,
    never silently written."""
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    prov = FakeProvider({"text": "", "tool_calls": [
        _tc("upsert_entity", layer="system", name="Stamina"),
        _tc("upsert_entity", layer="system", name="Power Attack"),
        _tc("link_entities", src="Power Attack", kind="REQUIRES", dst="Stamina"),
    ]})
    out = orchestrator.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="link them", known_layers=["system"],
        metamodel=None,
    )
    assert any(s["kind"] == "rejected" for s in out["saved"])
    assert not any(s["kind"] == "relation" for s in out["saved"])


def test_play_hints_track_scene_and_player(sql, mongo):
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    # nothing yet → both hints
    p0 = FakeProvider({"text": "ok", "tool_calls": []})
    o0 = orchestrator.run_turn(
        provider=p0, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="hi", known_layers=_MM_LAYERS, metamodel=_MM,
    )
    assert o0["playable"] is True  # contract renders from defaults
    assert set(o0["play_hints"]) == {"add a scene", "add a player character"}

    # add a scene + a player character → no hints left
    p1 = FakeProvider({"text": "building", "tool_calls": [
        _tc("upsert_entity", layer="scene", name="Tristram"),
        _tc("upsert_entity", layer="character", name="Hero", data={"role": "player"}),
    ]})
    o1 = orchestrator.run_turn(
        provider=p1, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="add a town and a hero",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    assert o1["play_hints"] == []


def test_review_intent_routes_to_critic_without_writing(sql, mongo):
    """A 'is it fun / review' turn short-circuits to the read-only Critic: it returns a
    scorecard + prose, attributes the turn to the experience agent, and writes NO graph."""
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    # seed a tiny graph so there's something to score
    seed = FakeProvider({"text": "wiring", "tool_calls": [
        _tc("upsert_entity", layer="system", name="Power Attack"),
        _tc("upsert_entity", layer="factor", name="Defense"),
        _tc("link_entities", src="Power Attack", kind="AFFECTS", dst="Defense"),
    ]})
    orchestrator.run_turn(
        provider=seed, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="power attack affects defense",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    game = service.get_game(sql, "diablo-2")
    rels_before = len(service.list_relations(sql, game))

    # the review turn — provider returns prose; it must NOT be asked for tool calls
    critic_prov = FakeProvider({"text": "Solid start; CHOICE is weakest.", "tool_calls": []})
    out = orchestrator.run_turn(
        provider=critic_prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="is this fun yet? give me a review",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    assert out["agent"] == "experience" and out["routed_to"] == "experience"
    assert out["reply"] == "Solid start; CHOICE is weakest."
    assert "scorecard" in out and 0 <= out["scorecard"]["headline"] <= 100
    assert out["saved"][0]["kind"] == "scorecard"
    # the critic wrote nothing to the graph
    assert len(service.list_relations(sql, game)) == rels_before


def test_generate_asset_captures_dimensions_and_queues_record_only(sql, mongo):
    """The Art path: create a prop, then generate_asset captures native dimensions on the
    entity and queues a job (record-only — status 'queued', no GPU submit)."""
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    prov = FakeProvider({"text": "Queuing the barrel.", "tool_calls": [
        _tc("upsert_entity", layer="prop", name="Barrel"),
        _tc("generate_asset", entity="Barrel", asset_kind="glb",
            dimensions={"w": 0.6, "h": 0.9, "d": 0.6},
            descriptor={"style": "weathered oak"}),
    ]})
    out = orchestrator.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="generate the barrel asset",
        known_layers=["prop"], metamodel=_MM,
    )
    jobs = [s for s in out["saved"] if s["kind"] == "asset_job"]
    assert len(jobs) == 1 and jobs[0]["status"] == "queued"
    assert jobs[0]["dimensions"] == {"w": 0.6, "h": 0.9, "d": 0.6}
    # dimensions captured on the entity so scaling is well-defined
    game = service.get_game(sql, "diablo-2")
    barrel = service.get_entity(sql, game, "barrel")
    assert barrel.data["dimensions"] == {"w": 0.6, "h": 0.9, "d": 0.6}
    # a durable job row exists, queued (not submitted to GPU)
    assert len(service.list_jobs(sql, game)) == 1


def test_validate_intent_routes_to_validator_without_writing(sql, mongo):
    """A 'validate / any errors' turn short-circuits to the read-only Validator."""
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    seed = FakeProvider({"text": "ok", "tool_calls": [
        _tc("upsert_entity", layer="system", name="Stamina"),
        _tc("upsert_entity", layer="factor", name="Defense"),
        _tc("link_entities", src="Stamina", kind="AFFECTS", dst="Defense"),
    ]})
    orchestrator.run_turn(
        provider=seed, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="stamina affects defense",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    game = service.get_game(sql, "diablo-2")
    rels_before = len(service.list_relations(sql, game))

    vprov = FakeProvider({"text": "All checks pass.", "tool_calls": []})
    out = orchestrator.run_turn(
        provider=vprov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="validate the game — any errors?",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    assert out["agent"] == "validator" and out["routed_to"] == "validator"
    assert "validation" in out and out["validation"]["ok"] is True
    assert out["saved"][0]["kind"] == "validation"
    assert len(service.list_relations(sql, game)) == rels_before  # wrote nothing


# ── agent layer: routing + active game ────────────────────────────────────────
def test_router_classifies_disciplines():
    # mechanics → systems (implemented)
    ag, routed = registry.route("add a stamina system that drains on attacks")
    assert ag.name == "systems" and routed == "systems"
    # narrative intent → narrative agent (now implemented, handles it directly)
    ag, routed = registry.route("write the opening quest and the villain's dialogue")
    assert routed == "narrative" and ag.name == "narrative"
    # world intent → world agent (now implemented)
    ag, routed = registry.route("design the dungeon layout and its rooms")
    assert routed == "world" and ag.name == "world"
    # art intent → art agent (now implemented)
    ag, routed = registry.route("generate a 3d model and its texture")
    assert routed == "art" and ag.name == "art"


def test_orchestrator_reports_handling_agent(sql, mongo):
    service.create_game(sql, schemas.GameCreate(title="Diablo 2", owner_id="u1"))
    prov = FakeProvider({"text": "ok", "tool_calls": [
        _tc("upsert_entity", layer="system", name="Stamina")]})
    out = orchestrator.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug="diablo-2", user_text="add a stamina system",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    assert out["agent"] == "systems"
    assert out["routed_to"] == "systems"


def test_active_game_pointer_set_on_turn(sql, mongo):
    prov = FakeProvider({"text": "Starting!", "tool_calls": [
        _tc("start_game", title="Diablo 2")]})
    orchestrator.run_turn(
        provider=prov, sql_db=sql, mongo_db=mongo, uid="u1", email="a@b.com",
        game_slug=None, user_text="start a game called Diablo 2",
        known_layers=_MM_LAYERS, metamodel=_MM,
    )
    assert creator_agent.get_active_game(mongo, "u1") == "diablo-2"


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
    monkeypatch.setattr(creator_routes, "_metamodel", lambda: None)
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


def test_route_games_list_and_switch_active(client, monkeypatch):
    prov = FakeProvider(
        {"text": "g1", "tool_calls": [_tc("start_game", title="Diablo 2")]},
        {"text": "g2", "tool_calls": [_tc("start_game", title="Baldurs Gate")]},
    )
    monkeypatch.setattr(creator_routes, "_provider", lambda: prov)
    client.post("/creator/turn", json={"text": "start Diablo 2"}, headers=_hdr())
    client.post("/creator/turn", json={"text": "start Baldurs Gate"}, headers=_hdr())

    # both games listed for this uid; active = the last one worked on
    g = client.get("/creator/games", headers=_hdr()).json()
    slugs = {x["game_slug"] for x in g["games"]}
    assert {"diablo-2", "baldurs-gate"} <= slugs
    assert g["active"] == "baldurs-gate"
    assert client.get("/creator/latest", headers=_hdr()).json()["game_slug"] == "baldurs-gate"

    # switch back to diablo-2 → latest follows the pointer
    s = client.post("/creator/active", json={"game_slug": "diablo-2"}, headers=_hdr())
    assert s.json()["active"] == "diablo-2"
    assert client.get("/creator/latest", headers=_hdr()).json()["game_slug"] == "diablo-2"
