"""Orchestrator — the only entry point for a creator chat turn.

Sits on top of the shared deterministic substrate. Per turn it:
  1. loads the per-(uid, game) memory (facts + open questions),
  2. routes the turn to a discipline agent (``registry.route``),
  3. runs THAT agent's LLM tool-use (its prompt + its tool subset),
  4. applies the proposed tool calls through the shared gate
     (``creator_agent.apply_tool_calls`` — the blackboard writer),
  5. persists memory + chat turns + the active-game pointer,
  6. returns the reply + saved chips + which agent handled it + play status.

The provider is the only non-deterministic seam (injected), so this is unit-testable
offline with a fake provider that returns scripted tool calls.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .. import creator_agent
from . import critic, director, info, registry, validator

log = logging.getLogger("creator_orchestrator")


def run_turn(
    *,
    provider,
    sql_db: Session,
    mongo_db,
    uid: Optional[str],
    email: Optional[str],
    game_slug: Optional[str],
    user_text: str,
    known_layers: List[str],
    metamodel: Optional[Dict[str, Any]] = None,
    metamodel_db=None,
    capabilities: Optional[Dict[str, Any]] = None,
    resolve_field: Optional[str] = None,
) -> Dict[str, Any]:
    """One creator turn through the routed discipline agent. Returns
    ``{reply, saved[], pending_question?, game_slug, agent, routed_to, playable, play_hints}``."""
    facts: Dict[str, Any] = {}
    open_questions: List[Dict[str, Any]] = []
    if game_slug:
        mem = creator_agent.load_memory(mongo_db, uid, game_slug)
        facts, open_questions = dict(mem["facts"]), list(mem["open_questions"])

    # 1a. read-only INFO queries (zero LLM, zero writes): "list my games" works with or
    #     without an active game; "what characters/systems do I have" reads the active
    #     graph. These are pure catalog reads — the cheapest path, so they go first.
    if info.is_games_intent(user_text):
        return _info_games_turn(
            sql_db=sql_db, mongo_db=mongo_db, uid=uid, email=email,
            game_slug=game_slug, user_text=user_text,
        )
    if game_slug and info.is_catalog_intent(user_text):
        return _info_catalog_turn(
            sql_db=sql_db, mongo_db=mongo_db, uid=uid, game_slug=game_slug, user_text=user_text,
        )

    # 1b. read-only short-circuits (no writes, no tools): a "review / is it fun" turn goes
    #     to the Critic (quality); a "validate / is it broken" turn goes to the Validator
    #     (correctness). Both score the graph deterministically and narrate it.
    if game_slug and critic.is_review_intent(user_text):
        return _review_turn(
            provider=provider, sql_db=sql_db, mongo_db=mongo_db, uid=uid,
            game_slug=game_slug, user_text=user_text, facts=facts, metamodel=metamodel,
        )
    if game_slug and validator.is_validate_intent(user_text):
        return _validate_turn(
            provider=provider, sql_db=sql_db, mongo_db=mongo_db, uid=uid,
            game_slug=game_slug, user_text=user_text, metamodel=metamodel,
        )
    if game_slug and director.is_progress_intent(user_text):
        return _director_turn(
            sql_db=sql_db, mongo_db=mongo_db, uid=uid, game_slug=game_slug,
            user_text=user_text, metamodel=metamodel, capabilities=capabilities,
        )

    # 2. route → discipline agent
    agent, routed_to = registry.route(user_text, facts)
    system = agent.system_prompt(
        known_layers=", ".join(known_layers),
        known_relations=creator_agent._render_relations(metamodel),
        facts_json=json.dumps(facts, default=str) if facts else "(none yet)",
    )

    # recent history (LLM-visible conversation context) + this user turn
    messages: List[Dict[str, str]] = [
        {"role": t["role"], "content": t["text"]}
        for t in creator_agent.load_turns(mongo_db, uid, game_slug, creator_agent.HISTORY_FOR_LLM)
        if t.get("role") in ("user", "assistant") and t.get("text")
    ]
    messages.append({"role": "user", "content": user_text})

    # 3. the agent proposes (LLM tool-use, scoped to its tool subset)
    out = provider.chat_tools(system, messages, agent.tools, tool_choice="auto")
    reply: str = out.get("text") or ""
    tool_calls: List[Dict[str, Any]] = out.get("tool_calls") or []

    # 4. apply deterministically through the shared gate
    res = creator_agent.apply_tool_calls(
        sql_db=sql_db, mongo_db=mongo_db, uid=uid, email=email, game_slug=game_slug,
        tool_calls=tool_calls, facts=facts, open_questions=open_questions,
        known_layers=known_layers, metamodel=metamodel, metamodel_db=metamodel_db,
        resolve_field=resolve_field,
    )
    game_slug = res["game_slug"]
    saved = res["saved"]
    pending_question = res["pending_question"]

    # 5. persist memory + turns + active-game pointer (best-effort; never breaks reply)
    if game_slug:
        creator_agent.save_memory(mongo_db, uid, email, game_slug, res["facts"], res["open_questions"])
        creator_agent.append_turn(mongo_db, uid, game_slug,
                                  {"role": "user", "text": user_text,
                                   "ts": datetime.now(timezone.utc).isoformat()})
        creator_agent.append_turn(mongo_db, uid, game_slug,
                                  {"role": "assistant", "text": reply,
                                   "ts": datetime.now(timezone.utc).isoformat(),
                                   "saved": saved, "question": pending_question,
                                   "agent": agent.name})
        creator_agent.set_active_game(mongo_db, uid, game_slug)

    return {
        "reply": reply,
        "saved": saved,
        "pending_question": pending_question,
        "game_slug": game_slug,
        "agent": agent.name,        # who handled it
        "routed_to": routed_to,     # intended discipline (may differ until modules land)
        "playable": bool(game_slug),
        "play_hints": creator_agent._play_hints(sql_db, game_slug) if game_slug else [],
    }


def _persist_readonly_turn(
    mongo_db, uid, game_slug: Optional[str], user_text: str, reply: str,
    saved: List[Dict[str, Any]], agent_name: str,
) -> None:
    """Append a user+assistant turn for a read-only agent (best-effort). Only persists
    when there's an active game to attach to — a games-list query with no game is
    answered but not stored."""
    if not game_slug:
        return
    now = datetime.now(timezone.utc).isoformat()
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "user", "text": user_text, "ts": now})
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "assistant", "text": reply, "ts": now,
                               "saved": saved, "question": None, "agent": agent_name})
    creator_agent.set_active_game(mongo_db, uid, game_slug)


def _info_games_turn(
    *, sql_db: Session, mongo_db, uid: Optional[str], email: Optional[str],
    game_slug: Optional[str], user_text: str,
) -> Dict[str, Any]:
    """The Info path for "list my games" — a deterministic DB read, no LLM. Returns the
    games list so the frontend switcher can refresh from the same call."""
    games = creator_agent.list_user_games(sql_db, uid)
    ans = info.answer_games(games)
    reply, saved = ans["reply"], ans["saved"]
    _persist_readonly_turn(mongo_db, uid, game_slug, user_text, reply, saved, info.NAME)
    return {
        "reply": reply, "saved": saved, "pending_question": None, "game_slug": game_slug,
        "agent": info.NAME, "routed_to": info.NAME,
        "playable": bool(game_slug),
        "play_hints": creator_agent._play_hints(sql_db, game_slug) if game_slug else [],
        "games": games,
    }


def _info_catalog_turn(
    *, sql_db: Session, mongo_db, uid: Optional[str], game_slug: str, user_text: str,
) -> Dict[str, Any]:
    """The Info path for "what characters/systems do I have" — reads the active graph
    deterministically and lists a layer (or an overview), no LLM."""
    entities, _ = creator_agent.load_graph_dicts(sql_db, game_slug)
    ans = info.answer_catalog(user_text, entities, game_slug=game_slug)
    reply, saved = ans["reply"], ans["saved"]
    _persist_readonly_turn(mongo_db, uid, game_slug, user_text, reply, saved, info.NAME)
    return {
        "reply": reply, "saved": saved, "pending_question": None, "game_slug": game_slug,
        "agent": info.NAME, "routed_to": info.NAME,
        "playable": True, "play_hints": creator_agent._play_hints(sql_db, game_slug),
    }


def _review_turn(
    *, provider, sql_db: Session, mongo_db, uid: Optional[str], game_slug: str,
    user_text: str, facts: Dict[str, Any], metamodel: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """The Critic path: deterministic scorecard + narrated review, persisted like any
    turn but writing nothing to the game graph."""
    entities, relations = creator_agent.load_graph_dicts(sql_db, game_slug)
    rv = critic.review(entities=entities, relations=relations, facts=facts,
                       provider=provider, metamodel=metamodel)
    reply, saved = rv["reply"], rv["saved"]
    now = datetime.now(timezone.utc).isoformat()
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "user", "text": user_text, "ts": now})
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "assistant", "text": reply, "ts": now,
                               "saved": saved, "question": None, "agent": critic.NAME})
    creator_agent.set_active_game(mongo_db, uid, game_slug)
    return {
        "reply": reply, "saved": saved, "pending_question": None, "game_slug": game_slug,
        "agent": critic.NAME, "routed_to": critic.NAME,
        "playable": True, "play_hints": creator_agent._play_hints(sql_db, game_slug),
        "scorecard": rv["scorecard"],
    }


def _validate_turn(
    *, provider, sql_db: Session, mongo_db, uid: Optional[str], game_slug: str,
    user_text: str, metamodel: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """The Validator path: deterministic correctness check + plain-language explanation,
    persisted like any turn but writing nothing to the game graph."""
    entities, relations = creator_agent.load_graph_dicts(sql_db, game_slug)
    # cost posture: correctness is a factual status — narrate it deterministically (no LLM).
    vr = validator.validate(entities=entities, relations=relations, metamodel=metamodel,
                            game={"slug": game_slug}, provider=None)
    reply, saved = vr["reply"], vr["saved"]
    now = datetime.now(timezone.utc).isoformat()
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "user", "text": user_text, "ts": now})
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "assistant", "text": reply, "ts": now,
                               "saved": saved, "question": None, "agent": validator.NAME})
    creator_agent.set_active_game(mongo_db, uid, game_slug)
    return {
        "reply": reply, "saved": saved, "pending_question": None, "game_slug": game_slug,
        "agent": validator.NAME, "routed_to": validator.NAME,
        "playable": True, "play_hints": creator_agent._play_hints(sql_db, game_slug),
        "validation": vr["result"],
    }


def _director_turn(
    *, sql_db: Session, mongo_db, uid: Optional[str], game_slug: str,
    user_text: str, metamodel: Optional[Dict[str, Any]],
    capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The Director path: deterministic prioritised 'what's next' report (incl. engine gaps
    from the capability ledger), persisted like any turn but writing nothing to the graph."""
    entities, relations = creator_agent.load_graph_dicts(sql_db, game_slug)
    # cost posture: "what's next" is a deterministic coverage read — no LLM call.
    dr = director.review(entities=entities, relations=relations, metamodel=metamodel,
                         game={"slug": game_slug}, capabilities=capabilities, provider=None)
    reply, saved = dr["reply"], dr["saved"]
    now = datetime.now(timezone.utc).isoformat()
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "user", "text": user_text, "ts": now})
    creator_agent.append_turn(mongo_db, uid, game_slug,
                              {"role": "assistant", "text": reply, "ts": now,
                               "saved": saved, "question": None, "agent": director.NAME})
    creator_agent.set_active_game(mongo_db, uid, game_slug)
    return {
        "reply": reply, "saved": saved, "pending_question": None, "game_slug": game_slug,
        "agent": director.NAME, "routed_to": director.NAME,
        "playable": dr["progress"]["playable"],
        "play_hints": creator_agent._play_hints(sql_db, game_slug),
        "progress": dr["progress"],
    }
