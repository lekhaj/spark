"""Creator substrate — the shared deterministic write layer + tool catalog.

This module is the **blackboard writer** that every discipline agent sits on top of.
It owns the tool schemas, the deterministic dispatch (``apply_tool_calls``) that turns
LLM-proposed tool calls into Postgres/Mongo writes, the per-(uid, game) memory helpers,
and the active-game pointer. It does **not** call the LLM or decide routing — that's
the orchestrator's job (``agents/orchestrator.py``), which picks a discipline agent
(``agents/systems.py`` …), runs ``provider.chat_tools`` with that agent's prompt + tool
subset, then hands the proposed tool calls back here to apply.

Three planes (per the approved design):
  • Conversation  — Haiku, non-deterministic, fine (lives in the agent/orchestrator).
  • Memory        — per-(uid, game_slug) Mongo ``game_memory`` doc re-fed each turn.
  • Game graph    — Postgres entities/relations (the source of truth).

Tools (Converse ``toolConfig``):
  start_game(title)                  → ``service.create_game``
  save_facts(facts)                  → merge into ``game_memory.facts``
  upsert_entity(layer,name,key?,data)→ ``service.create_entity``
  link_entities(src,kind,dst,data?)  → metamodel-gated ``service.create_relation``
  ask_clarification(field,question,…)→ NO write; surfaces the Claude-Code-style popup

Mode = "save the confident parts, ask only the gap". The LLM never touches persistence —
it only emits tool *calls*; ``apply_tool_calls`` is the deterministic gate. Everything
here is unit-testable offline (no LLM), and the agents on top inject a fake provider.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from . import schemas, service

log = logging.getLogger("creator_agent")

MEMORY = "game_memory"
SESSIONS = "creator_sessions"
ACTIVE = "creator_active_game"   # per-uid pointer: the game the user is working on now
TURN_CAP = 50          # how much chat history we keep / reload ("to some extent")
HISTORY_FOR_LLM = 12   # recent turns fed back to Haiku as conversation context

# The contract renders with graceful defaults from an empty graph, but a game is
# meaningfully playable once there's a place to be (scene) and someone to control
# (a character with data.role == "player"). These drive the "what's left to play"
# hints — not hard gates.
PLAY_SCENE_LAYER = "scene"
PLAY_PLAYER_LAYER = "character"

# Deterministic dispatch order within a single turn: a game must exist before we
# write to it, entities must exist before we can link them, and the clarification
# popup is resolved last (after everything we *could* save this turn).
_DISPATCH_ORDER = {
    "start_game": 0,
    "upsert_entity": 1,
    "save_facts": 2,
    "link_entities": 3,
    "ask_clarification": 4,
}


# ── tool schemas (Anthropic-style; the provider translates to Converse) ───────
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "start_game",
        "description": "Create a NEW game when the user asks to start/create one. "
                       "Call this before saving anything if no game exists yet.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "the game's name"}},
            "required": ["title"],
        },
    },
    {
        "name": "save_facts",
        "description": "Persist confirmed high-level design facts you are SURE about "
                       "(e.g. genre, pillars, tone, references, target_feel). Merge-safe: "
                       "send only the keys you're confident in. Do NOT guess — if unsure, "
                       "use ask_clarification instead.",
        "input_schema": {
            "type": "object",
            "properties": {"facts": {"type": "object", "description":
                           "flat map of confirmed design facts, e.g. "
                           "{\"genre\":\"ARPG\",\"references\":[\"Diablo 2\"]}"}},
            "required": ["facts"],
        },
    },
    {
        "name": "upsert_entity",
        "description": "Add a concrete game object to the graph (a character, system, "
                       "mission, prop, environment, …) when its identity is clear. "
                       "'layer' must be one of the known layers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "layer": {"type": "string", "description": "known layer name"},
                "name": {"type": "string"},
                "key": {"type": "string", "description": "optional stable key; derived from name if omitted"},
                "data": {"type": "object", "description": "the entity's fields"},
            },
            "required": ["layer", "name"],
        },
    },
    {
        "name": "link_entities",
        "description": "Create a typed relationship between two existing entities — this is "
                       "how mechanics actually connect (e.g. a power-attack REQUIRES a stamina "
                       "system; a system AFFECTS a factor; a scene CONTAINS a character). "
                       "'kind' must be one of the known relation types and the two entities' "
                       "layers must be legal for it. Create BOTH endpoints with upsert_entity "
                       "first (in the same turn is fine).",
        "input_schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "source entity key or name"},
                "kind": {"type": "string", "description": "known relation kind (UPPER_SNAKE)"},
                "dst": {"type": "string", "description": "destination entity key or name"},
                "data": {"type": "object", "description":
                         "optional edge data, e.g. {\"op\":\"add\",\"value\":-10} for an AFFECTS delta"},
            },
            "required": ["src", "kind", "dst"],
        },
    },
    {
        "name": "ask_clarification",
        "description": "Ask the user ONE question when you are unsure what to save. "
                       "Provide 2-4 concrete options. Use this instead of guessing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "what the answer will set, e.g. 'genre'"},
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "header": {"type": "string", "description": "≤12-char chip label"},
            },
            "required": ["field", "question"],
        },
    },
]

# The discipline prompts live with the agents (``agents/base.py`` COMMON_RULES +
# each ``agents/<discipline>.py`` intro). This substrate stays prompt-free.


# ── Mongo memory + session helpers ────────────────────────────────────────────
def _key(uid: Optional[str], game_slug: str) -> Dict[str, Any]:
    return {"uid": uid, "game_slug": game_slug}


def load_memory(mongo_db, uid: Optional[str], game_slug: str) -> Dict[str, Any]:
    """The per-(uid, game) memory doc; an empty skeleton if none yet."""
    if mongo_db is None or not game_slug:
        return {"facts": {}, "open_questions": []}
    doc = mongo_db[MEMORY].find_one(_key(uid, game_slug)) or {}
    return {
        "facts": doc.get("facts") or {},
        "open_questions": doc.get("open_questions") or [],
    }


def save_memory(
    mongo_db, uid: Optional[str], email: Optional[str], game_slug: str,
    facts: Dict[str, Any], open_questions: List[Dict[str, Any]],
) -> None:
    if mongo_db is None or not game_slug:
        return
    mongo_db[MEMORY].update_one(
        _key(uid, game_slug),
        {"$set": {
            "uid": uid, "email": email, "game_slug": game_slug,
            "facts": facts, "open_questions": open_questions,
            "updated_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )


def append_turn(mongo_db, uid: Optional[str], game_slug: str, turn: Dict[str, Any]) -> None:
    """Append one chat turn, keeping only the last ``TURN_CAP`` turns."""
    if mongo_db is None or not game_slug:
        return
    mongo_db[SESSIONS].update_one(
        _key(uid, game_slug),
        {
            "$set": {"uid": uid, "game_slug": game_slug, "updated_at": datetime.now(timezone.utc)},
            "$push": {"turns": {"$each": [turn], "$slice": -TURN_CAP}},
        },
        upsert=True,
    )


def load_turns(mongo_db, uid: Optional[str], game_slug: str, limit: int = TURN_CAP) -> List[Dict[str, Any]]:
    if mongo_db is None or not game_slug:
        return []
    doc = mongo_db[SESSIONS].find_one(_key(uid, game_slug)) or {}
    turns = doc.get("turns") or []
    return turns[-limit:]


# ── active-game pointer (which game the user is working on now) ────────────────
def set_active_game(mongo_db, uid: Optional[str], game_slug: str) -> None:
    if mongo_db is None or not uid or not game_slug:
        return
    mongo_db[ACTIVE].update_one(
        {"uid": uid},
        {"$set": {"uid": uid, "game_slug": game_slug,
                  "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


def get_active_game(mongo_db, uid: Optional[str]) -> Optional[str]:
    if mongo_db is None or not uid:
        return None
    doc = mongo_db[ACTIVE].find_one({"uid": uid})
    return (doc or {}).get("game_slug")


# ── relation + play helpers ───────────────────────────────────────────────────
def _render_relations(metamodel: Optional[Dict[str, Any]]) -> str:
    """Render the relation vocabulary for the system prompt:
    ``KIND (src_layers → dst_layers)`` per known relation type."""
    rts = (metamodel or {}).get("relation_types") or {}
    parts: List[str] = []
    for kind, rt in sorted(rts.items()):
        src = "/".join(rt.get("src_layers") or []) or "*"
        dst = "/".join(rt.get("dst_layers") or []) or "*"
        parts.append(f"{kind} ({src} → {dst})")
    return "; ".join(parts) or "(no relation types installed yet)"


def _resolve_key(sql_db: Session, game, ref: str) -> Optional[str]:
    """Resolve an entity reference (a stored key OR a display name) to its key.
    upsert_entity derives ``key = slugify(name)``, so a name resolves the same way."""
    ref = (ref or "").strip()
    if not ref:
        return None
    ent = service.get_entity(sql_db, game, ref)
    if ent is None:
        ent = service.get_entity(sql_db, game, schemas.slugify(ref))
    return ent.key if ent else None


def _play_hints(sql_db: Session, game_slug: str) -> List[str]:
    """What's still missing for a *fun* playtest (the contract already renders an
    empty scene from defaults, so these are nudges, not blockers)."""
    game = service.get_game(sql_db, game_slug)
    if game is None:
        return []
    layers = {e.layer for e in service.list_entities(sql_db, game)}
    hints: List[str] = []
    if PLAY_SCENE_LAYER not in layers:
        hints.append("add a scene")
    if PLAY_PLAYER_LAYER not in layers:
        hints.append("add a player character")
    return hints


# ── deterministic dispatch (the shared blackboard writer) ─────────────────────
def apply_tool_calls(
    *,
    sql_db: Session,
    mongo_db,
    uid: Optional[str],
    email: Optional[str],
    game_slug: Optional[str],
    tool_calls: List[Dict[str, Any]],
    facts: Dict[str, Any],
    open_questions: List[Dict[str, Any]],
    known_layers: List[str],
    metamodel: Optional[Dict[str, Any]] = None,
    resolve_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a discipline agent's proposed tool calls deterministically — the gate
    every agent shares. No LLM here. Mutates ``facts``/``open_questions`` (passed in
    as the caller's working copies) and returns
    ``{saved[], pending_question?, game_slug, facts, open_questions}``. The caller
    (orchestrator) owns memory persistence + the LLM call.

    ``metamodel`` (``{layers, relation_types}``) gates ``link_entities`` against the
    relation contract; None → relations rejected. ``resolve_field`` (set when the user
    is answering a popup) drops that open question once handled.
    """
    saved: List[Dict[str, Any]] = []
    pending_question: Optional[Dict[str, Any]] = None

    # Deterministic order: start_game → upsert_entity → save_facts → link_entities →
    # ask_clarification, so a single turn can create a game, populate it, link the new
    # entities, and still ask one gap. Stable sort preserves intra-group order.
    tool_calls = sorted(tool_calls, key=lambda c: _DISPATCH_ORDER.get(c.get("name"), 2))

    for call in tool_calls:
        name = call.get("name")
        args = call.get("input") or {}
        try:
            if name == "start_game":
                title = (args.get("title") or "").strip()
                if not title:
                    continue
                existing = service.get_game(sql_db, schemas.slugify(title))
                if existing is None:
                    game = service.create_game(
                        sql_db, schemas.GameCreate(title=title, owner_id=uid, status="draft")
                    )
                    saved.append({"kind": "game", "title": game.title, "slug": game.slug})
                else:
                    game = existing
                game_slug = game.slug

            elif name == "save_facts":
                new_facts = args.get("facts") or {}
                if isinstance(new_facts, dict) and game_slug:
                    facts.update(new_facts)
                    saved.append({"kind": "facts", "keys": sorted(new_facts.keys())})

            elif name == "upsert_entity":
                layer = (args.get("layer") or "").strip()
                ename = (args.get("name") or "").strip()
                if not game_slug or not layer or not ename:
                    continue
                if known_layers and layer not in known_layers:
                    saved.append({"kind": "rejected", "reason": f"unknown layer '{layer}'",
                                  "name": ename})
                    continue
                game = service.get_game(sql_db, game_slug)
                if game is None:
                    continue
                key = (args.get("key") or "").strip() or None
                try:
                    ent = service.create_entity(
                        sql_db, game,
                        schemas.EntityCreate(layer=layer, name=ename, key=key,
                                             data=args.get("data") or {}),
                    )
                    saved.append({"kind": "entity", "layer": layer, "name": ent.name, "key": ent.key})
                except Exception as e:  # noqa: BLE001 — e.g. duplicate key
                    sql_db.rollback()
                    saved.append({"kind": "rejected", "reason": str(e), "name": ename})

            elif name == "link_entities":
                kind = (args.get("kind") or "").strip()
                src_ref = (args.get("src") or "").strip()
                dst_ref = (args.get("dst") or "").strip()
                label = f"{src_ref} {kind} {dst_ref}".strip()
                if not game_slug or not kind or not src_ref or not dst_ref:
                    continue
                # gate on the relation contract: kind must be known
                rel_types = (metamodel or {}).get("relation_types") or {}
                if kind not in rel_types:
                    saved.append({"kind": "rejected", "name": label,
                                  "reason": f"unknown relation '{kind}'"})
                    continue
                game = service.get_game(sql_db, game_slug)
                if game is None:
                    continue
                src_key = _resolve_key(sql_db, game, src_ref)
                dst_key = _resolve_key(sql_db, game, dst_ref)
                if not src_key or not dst_key:
                    miss = src_ref if not src_key else dst_ref
                    saved.append({"kind": "rejected", "name": label,
                                  "reason": f"entity not found: {miss}"})
                    continue
                try:
                    # passing the metamodel makes create_relation enforce the edge
                    # contract (legal src/dst layers + cardinality).
                    service.create_relation(
                        sql_db, game,
                        schemas.RelationCreate(src=src_key, dst=dst_key, kind=kind,
                                               data=args.get("data") or {}),
                        metamodel=metamodel,
                    )
                    saved.append({"kind": "relation", "rel_kind": kind,
                                  "src": src_key, "dst": dst_key})
                except Exception as e:  # noqa: BLE001 — contract violation / bad edge
                    sql_db.rollback()
                    saved.append({"kind": "rejected", "name": label, "reason": str(e)})

            elif name == "ask_clarification":
                if pending_question is None:  # first question wins; one popup at a time
                    pending_question = {
                        "id": uuid.uuid4().hex[:8],
                        "field": args.get("field") or "",
                        "question": args.get("question") or "",
                        "options": args.get("options") or [],
                        "header": args.get("header") or (args.get("field") or "Question")[:12],
                        "asked_ts": datetime.now(timezone.utc).isoformat(),
                    }
        except Exception:  # noqa: BLE001 — a single bad tool call never breaks the turn
            log.exception("creator tool '%s' failed", name)
            sql_db.rollback()

    # resolve the answered question; register any new one
    if resolve_field:
        open_questions = [q for q in open_questions if q.get("field") != resolve_field]
    if pending_question:
        open_questions = [q for q in open_questions if q.get("field") != pending_question["field"]]
        open_questions.append(pending_question)

    return {
        "saved": saved,
        "pending_question": pending_question,
        "game_slug": game_slug,
        "facts": facts,
        "open_questions": open_questions,
    }


def make_bedrock_creator():
    """Tier-A creator provider via Bedrock Converse (AWS credits). Returns the
    provider (the route wraps the call in ``usage_recorder.attribution``)."""
    from app.lib import refiner_providers

    return refiner_providers.get_tier_provider("A")
