"""CycleZero asset-generation bridge tests (Phase 0).

Proves the seam between the game graph and the existing asset-run orchestrator,
*without* GPU/redis: the manual-gen submitters are patched out and Mongo is
mongomock. Covers game-segmented identity, spec synthesis, the create→reconcile
loop, and the GLB write-back onto the entity via the public REST surface.
"""
import os

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cyclezero import generation
from app.cyclezero.db import Base, get_db
from app.cyclezero.routes import router
from app.routes import asset_run_routes


@pytest.fixture()
def env(monkeypatch):
    # ── Postgres (sqlite in-memory) ──────────────────────────────────────────
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

    # ── Mongo (one mongomock db for both World_builder + cyclezero handles) ───
    mdb = mongomock.MongoClient()["test"]
    monkeypatch.setattr(asset_run_routes, "_db", lambda: mdb)
    monkeypatch.setattr(generation, "get_mongo", lambda: mdb)

    # ── GPU submitters are no-ops (no redis) ─────────────────────────────────
    monkeypatch.setattr(asset_run_routes, "_submit_image_job", lambda db, doc, output: "task-img")
    monkeypatch.setattr(asset_run_routes, "_submit_3d_jobs", lambda db, doc: None)
    monkeypatch.setattr(asset_run_routes, "_submit_rig_job", lambda db, doc, gen: None)

    app = FastAPI()
    app.include_router(router, prefix="/cyclezero")
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app), mdb


def _make_game_with_char(client, slug="diablo-2", key="zombie-shambler"):
    client.post("/cyclezero/games", json={"title": slug.title(), "slug": slug})
    client.post(
        f"/cyclezero/games/{slug}/entities",
        json={
            "layer": "character",
            "key": key,
            "name": "Zombie Shambler",
            "data": {"description": "A gaunt, slack-jawed corpse in rotted rags."},
        },
    )


# ── identity / segmentation ─────────────────────────────────────────────────
def test_asset_id_is_game_scoped():
    assert generation.asset_id_for("diablo-2", "zombie") == "diablo-2__zombie"
    # two games with the same entity key never collide
    assert generation.asset_id_for("game-a", "zombie") != generation.asset_id_for("game-b", "zombie")


# ── submit synthesizes an accepted asset_spec + creates a run ───────────────
def test_generate_creates_segmented_spec_and_run(env):
    client, mdb = env
    _make_game_with_char(client)

    r = client.post("/cyclezero/games/diablo-2/entities/zombie-shambler/generate",
                    json={"kind": "character"})
    assert r.status_code == 202
    result = r.json()["result"]
    assert result["submitted"] is True
    assert result["asset_id"] == "diablo-2__zombie-shambler"
    assert result["asset_run_id"]

    # spec landed in the World_builder DB, accepted, game-scoped
    spec = mdb[asset_run_routes.SPEC_RUNS].find_one({"run_id": result["spec_run_id"]})
    assert spec["status"] == "accepted"
    assert spec["stage"] == "asset_spec"
    assert spec["project_id"] == "diablo-2"
    out = spec["output"]
    assert out["asset_id"] == "diablo-2__zombie-shambler"
    assert out["rig_required"] is True
    assert out["generation_prompt"]  # non-empty, deterministic

    # asset_run keys on the segmented char_label (== asset_id)
    run = mdb[asset_run_routes.COLLECTION].find_one({"asset_run_id": result["asset_run_id"]})
    assert run["asset_id"] == "diablo-2__zombie-shambler"


def test_two_games_do_not_collide(env):
    client, mdb = env
    _make_game_with_char(client, slug="diablo-2", key="zombie")
    _make_game_with_char(client, slug="parallel-game", key="zombie")

    client.post("/cyclezero/games/diablo-2/entities/zombie/generate", json={"kind": "character"})
    client.post("/cyclezero/games/parallel-game/entities/zombie/generate", json={"kind": "character"})

    ids = mdb[asset_run_routes.COLLECTION].distinct("asset_id")
    assert "diablo-2__zombie" in ids
    assert "parallel-game__zombie" in ids  # disjoint S3/Mongo namespaces


# ── reconcile writes the finished GLB back onto the entity ──────────────────
def test_reconcile_writes_glb_back_to_entity(env, monkeypatch):
    client, mdb = env
    _make_game_with_char(client)
    gen = client.post("/cyclezero/games/diablo-2/entities/zombie-shambler/generate",
                      json={"kind": "character"}).json()
    job_id = gen["id"]

    # Simulate the pipeline finishing: patch _refresh to mark the run complete
    # with a manifest entry carrying the rigged GLB.
    def fake_refresh(db, doc):
        doc["status"] = "complete"
        doc["manifest_entry"] = {
            "glb_url": "https://s3/chars/diablo-2__zombie-shambler/v1.0/rigged.glb",
            "fbx_url": "https://s3/chars/diablo-2__zombie-shambler/v1.0/rigged.fbx",
            "model3d_generator": "trellis",
        }
        return doc

    monkeypatch.setattr(asset_run_routes, "_refresh", fake_refresh)

    r = client.get(f"/cyclezero/games/diablo-2/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "done"

    # the entity now carries the GLB → the contract builder will serve it
    ent = client.get("/cyclezero/games/diablo-2/entities/zombie-shambler").json()
    assert ent["data"]["glb"].endswith("rigged.glb")
    assert ent["data"]["fbx"].endswith("rigged.fbx")

    # …and it surfaces in the compiled contract's assets[]
    contract = client.get("/cyclezero/games/diablo-2/contract").json()
    asset_ids = {a["id"] for a in contract.get("assets", [])}
    assert "zombie-shambler" in asset_ids


def test_submit_best_effort_when_pipeline_unavailable(env, monkeypatch):
    client, mdb = env
    _make_game_with_char(client)

    # create_asset_run blows up (e.g. redis down) → job stays queued, no 500
    def boom(body):
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(asset_run_routes, "create_asset_run", boom)
    r = client.post("/cyclezero/games/diablo-2/entities/zombie-shambler/generate",
                    json={"kind": "character"})
    assert r.status_code == 202
    assert r.json()["result"]["submitted"] is False
