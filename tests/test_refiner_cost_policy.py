"""Cost policy (enforced 2026-06-19): LLM usage must bill via Bedrock (AWS credits),
never the direct Anthropic API. These tests lock that the direct provider stays OFF
unless explicitly opted in."""
import importlib

from app.lib import refiner_providers as rp


def test_direct_anthropic_disabled_even_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    monkeypatch.delenv("REFINER_ALLOW_DIRECT_ANTHROPIC", raising=False)
    monkeypatch.delenv("REFINER_CONVERSE", raising=False)
    monkeypatch.delenv("REFINER_BEDROCK", raising=False)
    monkeypatch.delenv("REFINER_OPENAI_BASE", raising=False)
    providers = rp.get_providers()
    assert "anthropic" not in providers  # direct billing path NOT registered


def test_direct_anthropic_requires_explicit_optin(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("REFINER_ALLOW_DIRECT_ANTHROPIC", "1")
    providers = rp.get_providers()
    assert "anthropic" in providers  # only when explicitly allowed


def test_converse_is_bedrock_aws_billed(monkeypatch):
    monkeypatch.setenv("REFINER_CONVERSE", "1")
    p = rp.get_tier_provider("C")
    assert isinstance(p, rp.BedrockConverseProvider)  # boto3 Bedrock = AWS credits
    assert p.model.startswith("global.anthropic") or "anthropic" in p.model
