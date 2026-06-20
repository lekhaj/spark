"""Usage + event-audit tests (mongomock; no live infra).

Covers: Bedrock pricing math, model-family mapping, usage_recorder insert shape +
contextvar attribution, /events batch ingest, /usage/me aggregation (LLM $ vs GPU
counts kept separate), and owner-gating on the admin reads.
"""
from datetime import datetime, timezone

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.lib import usage_recorder as ur
from app.lib.identity import OWNER_EMAIL
from app.routes import usage_routes


# ── pricing ───────────────────────────────────────────────────────────────────
def test_model_family_maps_haiku():
    assert ur.model_family("global.anthropic.claude-haiku-4-5-20251001-v1:0") == "haiku-4-5"
    assert ur.model_family("us.anthropic.claude-sonnet-4-6") == "sonnet-4-6"
    assert ur.model_family("something-unknown") == "haiku-4-5"  # safe default


def test_cost_math_haiku_defaults():
    # 1M input @ $1, 1M output @ $5 → $6 exactly
    assert ur.cost_usd("haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)
    # cache read/write priced on their own fields
    c = ur.cost_usd("haiku-4-5", 0, 0, cache_read=1_000_000, cache_write=1_000_000)
    assert c == pytest.approx(0.10 + 1.25)


def test_pricing_env_override(monkeypatch):
    monkeypatch.setenv("PRICE_HAIKU45_INPUT", "2.0")
    import importlib
    importlib.reload(ur)
    try:
        assert ur.cost_usd("haiku-4-5", 1_000_000, 0) == pytest.approx(2.0)
    finally:
        monkeypatch.delenv("PRICE_HAIKU45_INPUT", raising=False)
        importlib.reload(ur)


# ── recorder + attribution ─────────────────────────────────────────────────────
def test_record_inserts_priced_row_with_attribution(monkeypatch):
    db = mongomock.MongoClient()["usage_test"]
    monkeypatch.setattr(ur, "get_db", lambda: db)
    with ur.attribution(email="A@B.com", uid="u1", game_slug="g1", agent="compile"):
        ur.record("global.anthropic.claude-haiku-4-5-x", "C",
                  ur.Usage(input_tokens=1000, output_tokens=500))
    row = db["llm_usage"].find_one()
    assert row["source"] == "bedrock"
    assert row["email"] == "a@b.com"        # normalized
    assert row["agent"] == "compile"
    assert row["tier"] == "C"
    assert row["model_family"] == "haiku-4-5"
    assert row["total_tokens"] == 1500
    assert row["cost_usd"] == pytest.approx(ur.cost_usd("haiku-4-5", 1000, 500))


def test_record_never_raises_without_db(monkeypatch):
    monkeypatch.setattr(ur, "get_db", lambda: None)
    ur.record("haiku", None, ur.Usage(input_tokens=1))  # must not raise


# ── routes ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client(monkeypatch):
    db = mongomock.MongoClient()["usage_routes_test"]
    monkeypatch.setattr(usage_routes, "get_db", lambda: db)
    app = FastAPI()
    app.include_router(usage_routes.router)
    return TestClient(app), db


def _owner_headers():
    return {"X-Studio-Email": OWNER_EMAIL, "X-Studio-Uid": "owner"}


def test_events_ingest_and_audit(client):
    c, db = client
    r = c.post("/events", json={"events": [
        {"type": "refiner.chat", "text": "add a stamina system", "game_slug": "g1"},
        {"type": "compile.run", "game_slug": "g1"},
    ]}, headers={"X-Studio-Email": "u@x.com", "X-Studio-Uid": "u1"})
    assert r.json()["inserted"] == 2
    assert db["studio_events"].count_documents({}) == 2
    # surface gets resolved server-side
    ev = db["studio_events"].find_one({"type": "refiner.chat"})
    assert "refiner_routes" in ev["surface"]
    assert ev["email"] == "u@x.com"

    # audit feed is owner-only
    assert c.get("/events").status_code == 403
    feed = c.get("/events", headers=_owner_headers()).json()["events"]
    assert len(feed) == 2


def test_usage_me_separates_llm_and_gpu(client):
    c, db = client
    now = datetime.now(timezone.utc)
    db["llm_usage"].insert_many([
        {"ts": now, "email": "u@x.com", "model_family": "haiku-4-5", "agent": "refiner",
         "input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 0,
         "cache_write_tokens": 0, "total_tokens": 1500, "cost_usd": 0.0035},
    ])
    db["manual_gen_stage_runs"].insert_many([
        {"created_at": now, "stage": "flux", "status": "done"},
        {"created_at": now, "stage": "flux", "status": "done"},
        {"created_at": now, "stage": "trellis", "status": "done"},
    ])
    out = c.get("/usage/me", headers={"X-Studio-Email": "u@x.com"}).json()
    assert out["llm"]["totals"]["total_tokens"] == 1500
    assert out["llm"]["totals"]["cost_usd"] == pytest.approx(0.0035)
    stages = {s["stage"]: s["runs"] for s in out["gpu"]["by_stage"]}
    assert stages == {"flux": 2, "trellis": 1}
    assert out["gpu"]["total_runs"] == 3
    # GPU summary carries no dollar figure
    assert "cost_usd" not in out["gpu"]


def test_usage_admin_owner_gated(client):
    c, _ = client
    assert c.get("/usage").status_code == 403
    assert c.get("/usage", headers=_owner_headers()).status_code == 200


def test_aws_cost_owner_gated_and_parses(client, monkeypatch):
    c, _ = client
    # non-owner blocked
    assert c.get("/usage/aws-cost").status_code == 403

    class _FakeCE:
        def get_cost_and_usage(self, **kw):
            return {"ResultsByTime": [
                {"TimePeriod": {"Start": "2026-06-19"},
                 "Total": {"NetUnblendedCost": {"Amount": "0.42", "Unit": "USD"}}},
                {"TimePeriod": {"Start": "2026-06-20"},
                 "Total": {"NetUnblendedCost": {"Amount": "1.08", "Unit": "USD"}}},
            ]}

    monkeypatch.setattr(usage_routes, "_ce_client", lambda: _FakeCE())
    out = c.get("/usage/aws-cost", headers=_owner_headers()).json()
    assert out["available"] is True
    assert out["total_usd"] == pytest.approx(1.50)
    assert len(out["by_day"]) == 2
    assert out["currency"] == "USD"


def test_aws_cost_unavailable_never_500s(client, monkeypatch):
    c, _ = client

    def _boom():
        raise RuntimeError("no ce permission")

    monkeypatch.setattr(usage_routes, "_ce_client", _boom)
    r = c.get("/usage/aws-cost", headers=_owner_headers())
    assert r.status_code == 200
    assert r.json()["available"] is False
