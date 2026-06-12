"""
refiner_providers.py — pluggable LLM providers for the refiner chat (U02/T07)
=============================================================================

The refiner is the only live-LLM piece: a cheap model turns the user's message
into clarifying chat or a ``journey_plan`` fenced block (parsed studio-side).
Providers are swappable via env config — Anthropic (Haiku 4.5) today, any
OpenAI-compatible local model (Ollama / LM Studio on the L40S) later.

Env:
  ANTHROPIC_API_KEY        — enables AnthropicProvider
  REFINER_ANTHROPIC_MODEL  — default "claude-haiku-4-5"
  REFINER_OPENAI_BASE      — e.g. http://localhost:11434/v1 — enables OpenAICompatProvider
  REFINER_OPENAI_MODEL     — model name for the compat endpoint
"""

from __future__ import annotations

import os
from typing import Dict, List, Protocol

MAX_TOKENS = 2000


class RefinerProvider(Protocol):
    id: str
    model: str

    def chat(self, system: str, messages: List[Dict[str, str]]) -> str: ...


class AnthropicProvider:
    id = "anthropic"

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


def get_providers() -> Dict[str, RefinerProvider]:
    """Env-driven registry. Order matters: first configured = default."""
    providers: Dict[str, RefinerProvider] = {}
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
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
