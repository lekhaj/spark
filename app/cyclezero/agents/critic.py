"""Experience / Critic — the read-only "mind" that reviews a game.

Unlike the write disciplines (Systems, World, …) this agent never touches the graph.
It is a critic, not an author: it runs the DETERMINISTIC structural scorer
(``experience.score_structural``) and the LLM writes prose ON TOP of that scorecard —
explaining the weakest axes, citing the evidence, teaching the design pitfall, and giving
the smallest next step. The LLM may never change a number; if no provider is available
(offline/tests) a deterministic templated review is returned instead.

The orchestrator routes "is it fun / review / score / boring" turns here BEFORE the
write-router, and short-circuits (no tool-use, no writes). Also exposed as
``GET /creator/experience/{slug}`` for the raw scorecard.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .. import experience
from .base import keyword_hit

log = logging.getLogger("critic_agent")

NAME = "experience"
LABEL = "Experience"

# keywords that mean "judge what I have so far" (checked before the write-router)
REVIEW_INTENTS: Tuple[str, ...] = (
    "review", "is it fun", "is this fun", "is it boring", "is this boring", "boring",
    "engaging", "score", "rate", "rating", "critique", "feedback", "how good",
    "how am i doing", "playtest review", "is it good", "assess", "evaluate", "grade",
)

CRITIC_SYSTEM = """You are the Spark Studio CRITIC — a senior game designer reviewing a
game-in-progress. You are GIVEN a deterministic scorecard (0-100 per axis) computed from
the game's structure. You MUST NOT invent or change any number — quote the ones given.

Your job:
- Lead with the headline and an honest one-line verdict.
- Focus on the 1-2 WEAKEST axes. For each: state the score, cite the evidence given, name
  the design pitfall it reveals (teach it briefly), and give the concrete next step.
- Mention one genuine strength so it's encouraging.
- Be concise (4-7 sentences), specific, and kind. No fluff, no invented data.

Axes: CHOICE (interesting decisions), MASTERY (learning curve), AUTONOMY (un-forced),
FEEL (actions have consequences), TENSION (stakes + comeback/hope), IMMERSION (a world),
DISCOVERY (emergent interactions)."""


def is_review_intent(text: str) -> bool:
    return keyword_hit(text, REVIEW_INTENTS)


def _deterministic_prose(sc: experience.Scorecard) -> str:
    """A real review without an LLM — used offline/in tests and as a safe fallback."""
    ax = sc.axes[sc.weakest]
    lines = [f"Overall experience score: {sc.headline}/100."]
    lines.append(f"Weakest axis is {sc.weakest.upper()} ({ax.score}/100): {ax.evidence}")
    if sc.suggestion:
        lines.append(f"Next step: {sc.suggestion}")
    if sc.pitfalls:
        lines.append(f"Design pitfalls detected: {', '.join(sc.pitfalls)}.")
    strongest = max(sc.axes, key=lambda k: sc.axes[k].score)
    lines.append(f"Strongest is {strongest.upper()} ({sc.axes[strongest].score}/100).")
    return " ".join(lines)


def _llm_prose(provider, sc: experience.Scorecard, facts: Dict[str, Any]) -> Optional[str]:
    """Ask the provider to narrate the scorecard. Returns None on any failure so the
    caller falls back to the deterministic review (the chat must never break)."""
    try:
        payload = {"scorecard": sc.as_dict(), "design_facts": facts or {}}
        messages = [{"role": "user",
                     "content": "Review this game. Scorecard JSON:\n" + json.dumps(payload, default=str)}]
        out = provider.chat_tools(CRITIC_SYSTEM, messages, [], tool_choice="auto")
        text = (out or {}).get("text") or ""
        return text.strip() or None
    except Exception:  # noqa: BLE001
        log.exception("critic prose generation failed; using deterministic review")
        return None


def review(
    *,
    entities: List[dict],
    relations: List[dict],
    facts: Optional[Dict[str, Any]] = None,
    provider=None,
    metamodel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the game (deterministic) + narrate it (LLM or fallback). Pure inputs +
    injected provider, so it's unit-testable offline. Writes nothing."""
    sc = experience.score_structural(entities, relations, metamodel)
    reply = (_llm_prose(provider, sc, facts or {}) if provider else None) or _deterministic_prose(sc)
    saved = [{
        "kind": "scorecard",
        "headline": sc.headline,
        "weakest": sc.weakest,
        "pitfalls": sc.pitfalls,
        "axes": {k: v.score for k, v in sc.axes.items()},
    }]
    return {"scorecard": sc.as_dict(), "reply": reply, "saved": saved}
