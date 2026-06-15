"""CycleZero S-series (stitching) tests: metamodel, graph algorithms, the
spec-bridge, and graph-sourced contract. Pure graph logic needs no DB; the
integration cases use in-memory SQLite (graph) + mongomock (metamodel/specs).
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

from app.cyclezero import graph, metamodel, service
from app.cyclezero import routes as cz_routes
from app.cyclezero.db import Base, get_db


# ── metamodel snapshot used by the pure tests ─────────────────────────────────
def _mm(extra_relation_types=None):
    db = mongomock.MongoClient()["mm_test"]
    if extra_relation_types:
        for rt in extra_relation_types:
            metamodel.upsert_relation_type(db, rt)
    return metamodel.load_metamodel(db)


# ── S0: metamodel ─────────────────────────────────────────────────────────────
def test_metamodel_seeds_defaults():
    db = mongomock.MongoClient()["mm_test"]
    layers = metamodel.list_layers(db)
    rtypes = metamodel.list_relation_types(db)
    assert any(l["layer"] == "scene" and l["schema_key"] == "scene_spec" for l in layers)
    assert any(r["kind"] == "APPEARS_IN" for r in rtypes)


def test_metamodel_upsert_relation_type_validates_cardinality():
    db = mongomock.MongoClient()["mm_test"]
    with pytest.raises(ValueError):
        metamodel.upsert_relation_type(db, {"kind": "X", "src_cardinality": "lots"})


# ── S2: validate_graph ────────────────────────────────────────────────────────
def _ents(*pairs):
    return [{"layer": l, "key": k, "name": k, "data": {}} for l, k in pairs]


def test_validate_illegal_layer_edge():
    mm = _mm()
    entities = _ents(("character", "kael"), ("character", "bron"))
    # OWNS goes character→item; character→character is illegal.
    rels = [{"src": "kael", "dst": "bron", "kind": "OWNS"}]
    rep = graph.validate_graph(entities, rels, mm)
    assert rep["ok"] is False
    assert rep["illegal_edges"][0]["reason"].startswith("dst layer")


def test_validate_unknown_kind_and_missing_endpoint():
    mm = _mm()
    entities = _ents(("scene", "main"))
    rels = [
        {"src": "main", "dst": "ghost", "kind": "APPEARS_IN"},  # missing endpoint
        {"src": "main", "dst": "main", "kind": "NOPE"},          # unknown kind
    ]
    rep = graph.validate_graph(entities, rels, mm)
    reasons = {e["reason"] for e in rep["illegal_edges"]}
    assert any("endpoint not found" in r for r in reasons)
    assert any("unknown relation kind" in r for r in reasons)


def test_validate_cardinality_one():
    mm = _mm([{"kind": "MAIN_SCENE", "src_layers": ["character"], "dst_layers": ["scene"],
               "src_cardinality": "one", "dependency": True}])
    entities = _ents(("character", "kael"), ("scene", "a"), ("scene", "b"))
    rels = [
        {"src": "kael", "dst": "a", "kind": "MAIN_SCENE"},
        {"src": "kael", "dst": "b", "kind": "MAIN_SCENE"},
    ]
    rep = graph.validate_graph(entities, rels, mm)
    assert rep["ok"] is False
    assert rep["cardinality_violations"][0]["key"] == "kael"


def test_validate_missing_required_edge():
    mm = _mm([{"kind": "PART_OF", "src_layers": ["quest"], "dst_layers": ["system"],
               "required": True, "dependency": True}])
    entities = _ents(("quest", "q1"), ("system", "combat"))
    rep = graph.validate_graph(entities, [], mm)
    assert {"kind": "PART_OF", "key": "q1", "layer": "quest"} in rep["missing_required_edges"]
    assert rep["ok"] is False


def test_validate_complete_requires_accepted_spec():
    mm = _mm()
    entities = [{"layer": "scene", "key": "main", "name": "Main", "data": {},
                 "accepted_spec_run_id": None}]
    rep = graph.validate_graph(entities, [], mm)
    assert rep["ok"] is True            # structurally fine
    assert rep["complete"] is False     # but no accepted spec
    assert rep["nodes_without_accepted_spec"] == ["main"]


# ── S3: order / cycles / ripple ───────────────────────────────────────────────
def test_topo_order_prerequisites_first():
    mm = _mm()
    entities = _ents(("scene", "main"), ("character", "kael"))
    rels = [{"src": "kael", "dst": "main", "kind": "APPEARS_IN"}]  # kael depends on main
    res = graph.topo_order(entities, rels, mm)
    assert res["has_cycle"] is False
    assert res["order"].index("main") < res["order"].index("kael")


def test_find_cycles_detects_loop():
    mm = _mm([{"kind": "REQUIRES", "src_layers": ["quest"], "dst_layers": ["quest"],
               "dependency": True}])
    entities = _ents(("quest", "a"), ("quest", "b"))
    rels = [
        {"src": "a", "dst": "b", "kind": "REQUIRES"},
        {"src": "b", "dst": "a", "kind": "REQUIRES"},
    ]
    assert graph.find_cycles(entities, rels, mm)
    assert graph.topo_order(entities, rels, mm)["has_cycle"] is True


def test_ripple_reverse_reachability():
    mm = _mm()
    # main ← kael ← (sword references nothing dep) ; build a chain via APPEARS_IN
    entities = _ents(("scene", "main"), ("character", "kael"), ("prop", "torch"))
    rels = [
        {"src": "kael", "dst": "main", "kind": "APPEARS_IN"},
        {"src": "torch", "dst": "main", "kind": "APPEARS_IN"},
    ]
    rep = graph.ripple("main", entities, rels, mm)
    assert set(rep["downstream"]) == {"kael", "torch"}
    assert rep["count"] == 2
    # A leaf change ripples to nobody.
    assert graph.ripple("kael", entities, rels, mm)["downstream"] == []


# ── integration: graph routes + bridge + S6 contract over SQLite+mongomock ────
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
    c = TestClient(app)
    c._mongo = mongo  # expose for assertions
    c._Session = TestingSession
    return c


def _seed_graph(client, slug="g"):
    client.post("/cyclezero/games", json={"title": "G", "slug": slug})
    base = f"/cyclezero/games/{slug}/entities"
    client.post(base, json={"layer": "scene", "name": "Main", "key": "main"})
    client.post(base, json={"layer": "character", "name": "Kael", "key": "kael",
                            "data": {"role": "player"}})
    client.post(f"/cyclezero/games/{slug}/relations",
                json={"src": "kael", "dst": "main", "kind": "APPEARS_IN"})
    return slug


def test_routes_metamodel_and_graph(client):
    slug = _seed_graph(client)
    assert any(l["layer"] == "scene" for l in client.get("/cyclezero/metamodel/layers").json())

    val = client.get(f"/cyclezero/games/{slug}/graph/validate").json()
    assert val["ok"] is True and val["counts"]["relations"] == 1

    order = client.get(f"/cyclezero/games/{slug}/graph/order").json()
    assert order["order"].index("main") < order["order"].index("kael")

    rip = client.get(f"/cyclezero/games/{slug}/graph/ripple", params={"entity": "main"}).json()
    assert rip["downstream"] == ["kael"]


def test_route_rejects_illegal_relation(client):
    slug = _seed_graph(client)
    base = f"/cyclezero/games/{slug}/entities"
    client.post(base, json={"layer": "character", "name": "Bron", "key": "bron"})
    # OWNS is character→item; character→character must 400.
    r = client.post(f"/cyclezero/games/{slug}/relations",
                    json={"src": "kael", "dst": "bron", "kind": "OWNS"})
    assert r.status_code == 400


def test_entity_gets_spec_stage_from_metamodel(client):
    slug = _seed_graph(client)
    ent = client.get(f"/cyclezero/games/{slug}/entities/kael").json()
    assert ent["spec_stage"] == "character_spec"


def test_contract_sources_from_accepted_spec(client):
    """S6: when a node has an accepted spec, the contract uses the validated body."""
    slug = _seed_graph(client)
    # Stamp an accepted spec for the scene with a different camera height.
    run_id = "run_scene_1"
    client._mongo["spec_gen_runs"].insert_one(
        {"run_id": run_id, "output": {"camera": {"viewHeight": 99}, "quality": "performance"}}
    )
    session = client._Session()
    from app.cyclezero.models import Game
    from sqlalchemy import select
    game = session.scalar(select(Game).where(Game.slug == slug))
    scene = service.get_entity_by_key(session, game.id, "main")
    service.set_accepted_spec(session, scene, run_id)
    session.close()

    c = client.get(f"/cyclezero/games/{slug}/contract").json()
    assert c["camera"]["viewHeight"] == 99
    assert c["quality"] == "performance"


def test_packet_includes_graph_neighborhood(client):
    slug = _seed_graph(client)
    # Give the upstream scene an accepted body so the packet carries it.
    client._mongo["spec_gen_runs"].insert_one(
        {"run_id": "r1", "output": {"camera": {"viewHeight": 30}}}
    )
    client._mongo["spec_schemas"].insert_one(
        {"schema_key": "character_spec", "version": 1, "active": True,
         "title": "Character", "json_schema": {"type": "object"}}
    )
    session = client._Session()
    from app.cyclezero.models import Game
    from sqlalchemy import select
    game = session.scalar(select(Game).where(Game.slug == slug))
    scene = service.get_entity_by_key(session, game.id, "main")
    service.set_accepted_spec(session, scene, "r1")
    session.close()

    pkt = client.get(f"/cyclezero/games/{slug}/entities/kael/packet").json()
    assert pkt["stage"] == "character_spec"
    assert pkt["schemaVersion"] == 1
    # kael APPEARS_IN main (a dependency edge) → main is an upstream with its body.
    assert any(u["key"] == "main" and u["accepted_body"] for u in pkt["inputSpec"]["upstream"])
    assert any(ref["dst"] == "main" for ref in pkt["inputSpec"]["references"])


# ── S7: releases ──────────────────────────────────────────────────────────────
def test_release_cut_lists_and_versions(client):
    slug = _seed_graph(client)
    r1 = client.post(f"/cyclezero/games/{slug}/releases", json={"label": "0.1", "notes": "first"})
    assert r1.status_code == 201
    body = r1.json()
    assert body["version"] == 1
    assert body["label"] == "0.1"
    # manifest freezes the whole authored state.
    assert {"entities", "relations", "contract", "validation"} <= set(body["manifest"])
    assert body["manifest"]["contract"]["id"] == slug
    # no accepted specs yet → not complete.
    assert body["complete"] is False

    r2 = client.post(f"/cyclezero/games/{slug}/releases", json={})
    assert r2.json()["version"] == 2

    summaries = client.get(f"/cyclezero/games/{slug}/releases").json()
    assert [s["version"] for s in summaries] == [2, 1]  # newest first
    full = client.get(f"/cyclezero/games/{slug}/releases/1").json()
    assert full["notes"] == "first"
    assert client.get(f"/cyclezero/games/{slug}/releases/99").status_code == 404


def test_release_records_spec_version(client):
    slug = _seed_graph(client)
    client._mongo["spec_gen_runs"].insert_one(
        {"run_id": "rk", "output": {"role": "player"}, "major": 2, "minor": 1}
    )
    session = client._Session()
    from app.cyclezero.models import Game
    from sqlalchemy import select
    game = session.scalar(select(Game).where(Game.slug == slug))
    kael = service.get_entity_by_key(session, game.id, "kael")
    service.set_accepted_spec(session, kael, "rk")
    session.close()

    rel = client.post(f"/cyclezero/games/{slug}/releases", json={}).json()
    kael_entry = next(e for e in rel["manifest"]["entities"] if e["key"] == "kael")
    assert kael_entry["spec_version"] == "2.1"
