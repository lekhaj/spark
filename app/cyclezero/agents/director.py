"""Director — the read-only "core agent" that tracks overall progress.

It writes nothing. It reads the whole board deterministically — playability, correctness
(Validator), quality (Critic), spatial completeness, narrative orphans, and missing required
relations — and produces a single PRIORITISED "what's next" list. The LLM narrates it.

Priority order: playable first → correct → complete → good. So the Director always points at
the cheapest highest-leverage next step, like a producer running stand-up.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .. import compile_tools, experience, spatial, validate_agent
from . import narrative
from .base import keyword_hit

log = logging.getLogger("director_agent")

NAME = "director"
LABEL = "Director"

PROGRESS_INTENTS: Tuple[str, ...] = (
    "what's next", "whats next", "what next", "next step", "next steps", "progress",
    "status", "todo", "to do", "what should i do", "what now", "where am i",
    "what's missing", "whats missing", "what's left", "roadmap", "overview",
)

DIRECTOR_SYSTEM = """You are the Spark Studio DIRECTOR — the producer tracking the whole game.
You are GIVEN a deterministic progress report (a prioritised list of next steps + coverage
numbers). Do not invent items — present the ones given. Lead with where the game stands in one
line, then walk the top 3 next steps in order (most important first) and why each matters.
Be concise and motivating."""


def is_progress_intent(text: str) -> bool:
    return keyword_hit(text, PROGRESS_INTENTS)


def _missing_required_relations(
    entities: List[dict], relations: List[dict], metamodel: Optional[Dict[str, Any]]
) -> List[str]:
    """For each relation type flagged ``required``, every entity in its src_layers must have
    an outgoing edge of that kind. Returns 'KIND on key' for each gap."""
    rtypes = (metamodel or {}).get("relation_types", {})
    have = {(r.get("kind"), r.get("src")) for r in relations}
    missing: List[str] = []
    for kind, rt in rtypes.items():
        if not rt.get("required"):
            continue
        src_layers = set(rt.get("src_layers", []))
        for e in entities:
            if e.get("layer") in src_layers and (kind, e["key"]) not in have:
                missing.append(f"{kind} on {e['key']}")
    return missing


def progress(
    entities: List[dict],
    relations: List[dict],
    *,
    metamodel: Optional[Dict[str, Any]] = None,
    game: Optional[Dict[str, Any]] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    engine: str = "babylon",
) -> Dict[str, Any]:
    """The deterministic progress report: prioritised next steps + coverage. Pure.

    ``capabilities`` is the merged engine registry (base seed + the living ledger of what
    Claude Code has built, ingested via /capabilities/ingest). The graph layers it does NOT
    yet ``consume`` are the engine gaps — exactly the build packet to hand the coding agent.
    Passing it closes the loop: after a done-note is ingested, the gaps shrink here."""
    layers = {e.get("layer") for e in entities}
    has_scene = "scene" in layers
    has_player = any((e.get("data") or {}).get("role") == "player"
                     for e in entities if e.get("layer") == "character")

    sc = experience.score_structural(entities, relations, metamodel)
    val = validate_agent.static_validate(entities, relations,
                                         metamodel or {"layers": {}, "relation_types": {}}, game)
    sp = spatial.spatial_health(entities, relations)
    narr = narrative.narrative_health(entities, relations)
    missing_req = _missing_required_relations(entities, relations, metamodel)

    # engine gaps = layers used by the graph that the runtime can't render yet → Claude Code.
    registry = capabilities or compile_tools.get_base_registry(engine)
    cap = compile_tools.diff_capabilities(sorted(l for l in layers if l), registry)

    steps: List[str] = []
    # 1 playable
    if not has_scene:
        steps.append("Add a scene so there's a place to play.")
    if not has_player:
        steps.append("Add a character with data.role='player'.")
    # 2 correct
    if not val.get("ok"):
        steps.append(f"Fix {len(val.get('issues', []))} validation issue(s): "
                     f"{'; '.join(val.get('issues', [])[:2])}.")
    # 3 complete
    if missing_req:
        steps.append(f"Add required relation(s): {', '.join(missing_req[:3])}.")
    if sp["missing_transform"]:
        steps.append(f"{len(sp['missing_transform'])} placed object(s) need a transform "
                     f"(position/scale): {', '.join(sp['missing_transform'][:3])}.")
    if sp["missing_dimensions"]:
        steps.append(f"{len(sp['missing_dimensions'])} asset(s) need native dimensions.")
    if narr["orphans"]:
        steps.append(f"Connect {len(narr['orphans'])} orphan quest(s) into the story flow.")
    # 3b engine build — layers the runtime can't render yet → compile a packet for Claude Code
    if cap["gaps"]:
        steps.append(f"Compile a build packet — the {engine} engine can't render these yet: "
                     f"{', '.join(cap['gaps'])}. Hand it to the coding agent, then paste its "
                     f"done-note back to /capabilities/ingest.")
    # 4 good (quality) — the weakest experience axis
    steps.append(f"Improve {sc.weakest.upper()} ({sc.axes[sc.weakest].score}/100): {sc.suggestion}")

    return {
        "headline": sc.headline,
        "playable": has_scene and has_player,
        "next_steps": steps,
        "coverage": {
            "experience_headline": sc.headline,
            "weakest_axis": sc.weakest,
            "valid": bool(val.get("ok")),
            "validation_issues": len(val.get("issues", [])),
            "missing_required_relations": missing_req,
            "spatial_missing_transform": sp["missing_transform"],
            "spatial_missing_dimensions": sp["missing_dimensions"],
            "orphan_quests": narr["orphans"],
            "engine_gaps": cap["gaps"],
            "engine_ready": cap["fully_covered"],
        },
    }


def _deterministic_prose(rep: Dict[str, Any]) -> str:
    state = "playable" if rep["playable"] else "not yet playable"
    head = f"The game scores {rep['headline']}/100 and is {state}. Next:"
    return " ".join([head] + [f"{i+1}. {s}" for i, s in enumerate(rep["next_steps"][:3])])


def _llm_prose(provider, rep: Dict[str, Any]) -> Optional[str]:
    try:
        messages = [{"role": "user", "content": "Progress report:\n" + json.dumps(rep, default=str)}]
        out = provider.chat_tools(DIRECTOR_SYSTEM, messages, [], tool_choice="auto")
        return ((out or {}).get("text") or "").strip() or None
    except Exception:  # noqa: BLE001
        log.exception("director prose failed; using deterministic report")
        return None


def review(
    *,
    entities: List[dict],
    relations: List[dict],
    metamodel: Optional[Dict[str, Any]] = None,
    game: Optional[Dict[str, Any]] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    provider=None,
) -> Dict[str, Any]:
    """Progress report (deterministic) + optional narration. Pure inputs + injected
    provider. Narration is opt-in (provider=None → deterministic, zero LLM cost)."""
    rep = progress(entities, relations, metamodel=metamodel, game=game, capabilities=capabilities)
    reply = (_llm_prose(provider, rep) if provider else None) or _deterministic_prose(rep)
    saved = [{"kind": "progress", "headline": rep["headline"], "playable": rep["playable"],
              "next_steps": rep["next_steps"]}]
    return {"progress": rep, "reply": reply, "saved": saved}
