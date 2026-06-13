"""
U02 refiner tests — provider registry + chat proxy. SDKs/HTTP mocked.
"""

import os

os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "World_builder_test")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("REDIS_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.lib import refiner_providers
from app.routes import refiner_routes


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(refiner_routes.router, prefix="/refiner")
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "REFINER_ANTHROPIC_MODEL",
                "REFINER_OPENAI_BASE", "REFINER_OPENAI_MODEL", "REFINER_OPENAI_KEY",
                "REFINER_BEDROCK", "REFINER_BEDROCK_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_registry_env_driven(client, monkeypatch):
    assert client.get("/refiner/providers").json() == []

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = client.get("/refiner/providers").json()
    assert out == [{"id": "anthropic", "model": "claude-haiku-4-5"}]

    monkeypatch.setenv("REFINER_OPENAI_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("REFINER_OPENAI_MODEL", "qwen2.5")
    ids = [p["id"] for p in client.get("/refiner/providers").json()]
    assert ids == ["anthropic", "openai_compat"]


def test_bedrock_is_default_no_key(client, monkeypatch):
    monkeypatch.setenv("REFINER_BEDROCK", "1")
    out = client.get("/refiner/providers").json()
    assert out == [{"id": "bedrock", "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0"}]

    # When both Bedrock and a direct key are set, Bedrock wins as default (first).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    ids = [p["id"] for p in client.get("/refiner/providers").json()]
    assert ids[0] == "bedrock"


def test_bedrock_provider_passthrough(client, monkeypatch):
    monkeypatch.setenv("REFINER_BEDROCK", "1")
    monkeypatch.setenv("REFINER_BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    captured = {}

    class FakeBlock:
        type = "text"
        text = "hello from bedrock"

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            class R:
                content = [FakeBlock()]
            return R()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(
        refiner_providers.AnthropicBedrockProvider, "_client", lambda self: FakeClient()
    )

    r = client.post("/refiner/chat", json={
        "messages": [{"role": "user", "content": "add a stamina system"}],
        "context": {"project_id": "cyclezero", "schemas": ["zone_spec"], "entities": []},
    })
    assert r.status_code == 200
    assert r.json()["reply"] == "hello from bedrock"
    assert captured["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert captured["max_tokens"] == 2000
    assert "journey_plan" in captured["system"]


def test_anthropic_provider_passthrough(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured = {}

    class FakeBlock:
        type = "text"
        text = "hello from haiku"

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            class R:
                content = [FakeBlock()]
            return R()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(refiner_providers.AnthropicProvider, "_client", lambda self: FakeClient())

    r = client.post("/refiner/chat", json={
        "messages": [{"role": "user", "content": "add a stamina system"}],
        "context": {"project_id": "cyclezero", "schemas": ["zone_spec"], "entities": []},
    })
    assert r.status_code == 200
    assert r.json()["reply"] == "hello from haiku"
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["max_tokens"] == 2000
    assert "journey_plan" in captured["system"]
    assert "zone_spec" in captured["system"]
    assert captured["messages"][-1]["content"] == "add a stamina system"


def test_openai_compat_provider_url_and_model(client, monkeypatch):
    monkeypatch.setenv("REFINER_OPENAI_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("REFINER_OPENAI_MODEL", "qwen2.5")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": "local reply"}}]}
        return R()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    r = client.post("/refiner/chat", json={
        "provider_id": "openai_compat",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.json()["reply"] == "local reply"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["model"] == "qwen2.5"
    assert captured["json"]["messages"][0]["role"] == "system"


def test_unknown_provider_404_and_none_503(client, monkeypatch):
    r = client.post("/refiner/chat", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 503

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = client.post("/refiner/chat", json={
        "provider_id": "bogus",
        "messages": [{"role": "user", "content": "x"}],
    })
    assert r.status_code == 404


def test_journey_plan_block_passes_through_unmodified(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    reply = 'Here you go.\n```journey_plan\n{"kind": "system", "title": "Stamina"}\n```\n'

    class FakeBlock:
        type = "text"
        text = reply

    class FakeMessages:
        def create(self, **kwargs):
            class R:
                content = [FakeBlock()]
            return R()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(refiner_providers.AnthropicProvider, "_client", lambda self: FakeClient())
    r = client.post("/refiner/chat", json={"messages": [{"role": "user", "content": "stamina"}]})
    assert r.json()["reply"] == reply
