"""Validator discipline — read-only correctness check.

Like the Critic, it writes nothing: it runs the DETERMINISTIC ``validate_agent.static_validate``
(graph legality + contract compiles + outcome resolves) and the LLM explains the issues in
plain language with fixes. The Critic judges whether the game is *good*; the Validator judges
whether it is *correct*. Routed before the write-router on a "validate / is it broken" turn.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .. import spatial, validate_agent
from .base import keyword_hit

log = logging.getLogger("validator_agent")

NAME = "validator"
LABEL = "Validator"

VALIDATE_INTENTS: Tuple[str, ...] = (
    "validate", "is it valid", "does it compile", "compiles", "any errors",
    "errors", "what's broken", "is it broken", "broken", "lint",
    "sanity check", "health check", "check the graph", "check for errors", "what's wrong",
)

VALIDATOR_SYSTEM = """You are the Spark Studio VALIDATOR — you check a game-in-progress for
CORRECTNESS (not quality). You are GIVEN a deterministic validation result. You MUST NOT
invent issues — only explain the ones given.

- If ok: say so in one line and note what was checked.
- Otherwise: list each issue in plain language, say why it matters, and give the concrete
  fix (which entity/relation to change). Be concise and specific."""


def is_validate_intent(text: str) -> bool:
    return keyword_hit(text, VALIDATE_INTENTS)


def _deterministic_prose(res: Dict[str, Any]) -> str:
    if res.get("ok"):
        names = ", ".join(c["name"] for c in res.get("checks", []))
        return f"Validation passed ✓ — checked: {names or 'graph'}."
    issues = res.get("issues", [])
    head = f"Validation found {len(issues)} issue(s):"
    return " ".join([head] + [f"• {i}" for i in issues[:8]])


def _llm_prose(provider, res: Dict[str, Any]) -> Optional[str]:
    try:
        messages = [{"role": "user",
                     "content": "Explain this validation result:\n" + json.dumps(res, default=str)}]
        out = provider.chat_tools(VALIDATOR_SYSTEM, messages, [], tool_choice="auto")
        return ((out or {}).get("text") or "").strip() or None
    except Exception:  # noqa: BLE001
        log.exception("validator prose failed; using deterministic explanation")
        return None


def validate(
    *,
    entities: List[dict],
    relations: List[dict],
    metamodel: Optional[Dict[str, Any]] = None,
    game: Optional[Dict[str, Any]] = None,
    provider=None,
) -> Dict[str, Any]:
    """Deterministic validation + plain-language explanation. Pure inputs + injected
    provider; writes nothing."""
    mm = metamodel or {"layers": {}, "relation_types": {}}
    res = validate_agent.static_validate(entities, relations, mm, game)

    # spatial completeness — placed nodes need a transform, sized assets need dimensions.
    # These are design-completeness warnings (don't flip structural `ok`), surfaced so
    # scale/placement is never silently dropped.
    sp = spatial.spatial_health(entities, relations)
    res["spatial"] = sp
    if sp["issues"]:
        res = {**res, "issues": list(res.get("issues", [])) + sp["issues"]}

    reply = (_llm_prose(provider, res) if provider else None) or _deterministic_prose(res)
    saved = [{"kind": "validation", "ok": bool(res.get("ok")),
              "issues": res.get("issues", []),
              "spatial_warnings": sp["issues"]}]
    return {"result": res, "reply": reply, "saved": saved}
