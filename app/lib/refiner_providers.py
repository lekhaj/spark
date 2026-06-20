"""
refiner_providers.py — pluggable LLM providers for the refiner chat (U02/T07)
=============================================================================

The refiner is the only live-LLM piece: a cheap model turns the user's message
into clarifying chat or a ``journey_plan`` fenced block (parsed studio-side).
Providers are swappable via env config — Anthropic (Haiku 4.5) today, any
OpenAI-compatible local model (Ollama / LM Studio on the L40S) later.

Env:
  REFINER_BEDROCK          — "1"/"true" enables AnthropicBedrockProvider (no key;
                             uses the instance IAM role / standard AWS cred chain)
  REFINER_BEDROCK_MODEL    — Bedrock model id or inference-profile id
                             (default "us.anthropic.claude-haiku-4-5-20251001-v1:0")
  AWS_REGION / AWS_DEFAULT_REGION — region for Bedrock (default "us-east-1")
  ANTHROPIC_API_KEY        — enables AnthropicProvider (direct API, alternative to Bedrock)
  REFINER_ANTHROPIC_MODEL  — default "claude-haiku-4-5"
  REFINER_OPENAI_BASE      — e.g. http://localhost:11434/v1 — enables OpenAICompatProvider
  REFINER_OPENAI_MODEL     — model name for the compat endpoint

  REFINER_CONVERSE         — "1"/"true" enables BedrockConverseProvider (boto3
                             Converse API; any vendor; IAM creds). Preferred path.
  REFINER_TIER_A_MODEL     — cheap chat (refiner). Default = Haiku (start simple).
  REFINER_TIER_B_MODEL     — structured (schema/rule/brief draft + validator). Haiku.
  REFINER_TIER_C_MODEL     — reasoning (compile: plan + code-gen prompt). Sonnet.
  AWS_REGION / AWS_DEFAULT_REGION — region (default us-east-1)

Tiers (inference-architecture.md): start Haiku-everywhere, then drop A→Llama 4 and
lift C→Sonnet by env, one line each. Actual code-writing stays in Claude Code.
"""

from __future__ import annotations

import os
from typing import Dict, List, Protocol

MAX_TOKENS = 2000


class RefinerProvider(Protocol):
    id: str
    model: str

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str: ...


def _record_usage(model: str, tier, usage) -> None:
    """Best-effort: price + persist one external (Bedrock) LLM call. Import is
    lazy + fully guarded so the provider never breaks if recording is unavailable."""
    try:
        from app.lib import usage_recorder

        usage_recorder.record(model, tier, usage)
    except Exception:  # noqa: BLE001
        pass


class AnthropicBedrockProvider:
    """Anthropic via AWS Bedrock — no API key; uses the CPU box's IAM role
    (standard boto3 credential chain). Same Messages API shape as the direct SDK."""

    id = "bedrock"
    tier = None

    def __init__(self, model: str, region: str):
        self.model = model
        self._region = region

    def _client(self):
        import anthropic  # lazy — only needed when this provider is configured

        return anthropic.AnthropicBedrock(aws_region=self._region)

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        response = self._client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        )
        try:
            from app.lib.usage_recorder import Usage

            u = response.usage
            _record_usage(self.model, self.tier, Usage(
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ))
        except Exception:  # noqa: BLE001
            pass
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        )


class BedrockConverseProvider:
    """Bedrock via the **Converse API** (boto3) — one interface for ANY vendor on
    Bedrock (Claude / Llama / Nova / Mistral …). No API key; uses the instance IAM
    role (standard boto3 chain). Swapping models is just a different ``model`` id, so
    the studio's A/B/C tiers (cheap chat / structured / reasoning) are pure config.

    Optional ``cache`` adds a Bedrock cachePoint after the system prompt — the reused
    system prompt is then billed ~90% cheaper on subsequent calls (where supported)."""

    id = "converse"
    tier = None

    def __init__(self, model: str, region: str, cache: bool = True):
        self.model = model
        self._region = region
        self._cache = cache

    def _client(self):
        import boto3  # lazy — only needed when this provider is configured

        return boto3.client("bedrock-runtime", region_name=self._region)

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        system_blocks: List[Dict] = [{"text": system}]
        if self._cache:
            system_blocks.append({"cachePoint": {"type": "default"}})
        conv = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
        ]
        resp = self._client().converse(
            modelId=self.model,
            system=system_blocks,
            messages=conv,
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )
        try:
            from app.lib.usage_recorder import Usage

            u = resp.get("usage", {}) or {}
            _record_usage(self.model, self.tier, Usage(
                input_tokens=u.get("inputTokens", 0) or 0,
                output_tokens=u.get("outputTokens", 0) or 0,
                cache_read=u.get("cacheReadInputTokens", 0) or 0,
                cache_write=u.get("cacheWriteInputTokens", 0) or 0,
            ))
        except Exception:  # noqa: BLE001
            pass
        return "".join(
            block.get("text", "")
            for block in resp["output"]["message"]["content"]
        )


class AnthropicProvider:
    id = "anthropic"
    tier = None

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._api_key = api_key

    def _client(self):
        import anthropic  # lazy — only needed when this provider is configured

        return anthropic.Anthropic(api_key=self._api_key)

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        response = self._client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        )
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        )


class OpenAICompatProvider:
    id = "openai_compat"
    tier = None

    def __init__(self, base: str, model: str, api_key: str = ""):
        self.base = base.rstrip("/")
        self.model = model
        self._api_key = api_key

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        resp = requests.post(
            f"{self.base}/chat/completions",
            json={
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": system}, *messages],
            },
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# Tier → Bedrock Converse model id. Locked plan (2026-06-19): start Haiku-everywhere;
# A→Llama 4 Maverick and C→Sonnet 4.6 are env overrides when tiering up.
# global. cross-region profile = ~10% cheaper than us. + best availability (live-verified).
_HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
TIER_DEFAULTS = {
    "A": os.environ.get("REFINER_TIER_A_MODEL", _HAIKU),  # cheap chat (refiner)
    "B": os.environ.get("REFINER_TIER_B_MODEL", _HAIKU),  # structured (draft/validate)
    "C": os.environ.get("REFINER_TIER_C_MODEL", _HAIKU),  # reasoning (compile)
}


def _region() -> str:
    return (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1")


def get_tier_provider(tier: str) -> "RefinerProvider":
    """Return the Converse provider for an inference tier (A/B/C). Used by the
    compile/validate agents; falls back to the first configured provider if
    Converse isn't enabled, so callers degrade gracefully."""
    if os.environ.get("REFINER_CONVERSE", "").lower() in ("1", "true", "yes"):
        p = BedrockConverseProvider(
            model=TIER_DEFAULTS.get(tier, _HAIKU), region=_region()
        )
        p.tier = tier
        return p
    providers = get_providers()
    if not providers:
        raise RuntimeError("no LLM providers configured (set REFINER_CONVERSE=1)")
    return next(iter(providers.values()))


def get_providers() -> Dict[str, RefinerProvider]:
    """Env-driven registry. Order matters: first configured = default."""
    providers: Dict[str, RefinerProvider] = {}
    if os.environ.get("REFINER_CONVERSE", "").lower() in ("1", "true", "yes"):
        _converse = BedrockConverseProvider(model=TIER_DEFAULTS["A"], region=_region())
        _converse.tier = "A"
        providers["converse"] = _converse
    if os.environ.get("REFINER_BEDROCK", "").lower() in ("1", "true", "yes"):
        providers["bedrock"] = AnthropicBedrockProvider(
            model=os.environ.get(
                "REFINER_BEDROCK_MODEL",
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            region=os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1",
        )
    # ⚠️ COST POLICY (enforced 2026-06-19): all LLM usage must bill via **Bedrock
    # = AWS credits**, never the **direct Anthropic API = direct charge**. The
    # direct provider is therefore OFF unless someone explicitly opts in with
    # REFINER_ALLOW_DIRECT_ANTHROPIC=1 — a stray ANTHROPIC_API_KEY alone will NOT
    # enable it, so it can never silently cause direct charges.
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    allow_direct = os.environ.get("REFINER_ALLOW_DIRECT_ANTHROPIC", "").lower() in (
        "1", "true", "yes"
    )
    if anthropic_key and allow_direct:
        providers["anthropic"] = AnthropicProvider(
            model=os.environ.get("REFINER_ANTHROPIC_MODEL", "claude-haiku-4-5"),
            api_key=anthropic_key,
        )
    openai_base = os.environ.get("REFINER_OPENAI_BASE", "")
    if openai_base:
        providers["openai_compat"] = OpenAICompatProvider(
            base=openai_base,
            model=os.environ.get("REFINER_OPENAI_MODEL", ""),
            api_key=os.environ.get("REFINER_OPENAI_KEY", ""),
        )
    return providers
