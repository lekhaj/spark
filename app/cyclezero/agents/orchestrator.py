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
from . import registry

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
    resolve_field: Optional[str] = None,
) -> Dict[str, Any]:
    """One creator turn through the routed discipline agent. Returns
    ``{reply, saved[], pending_question?, game_slug, agent, routed_to, playable, play_hints}``."""
    facts: Dict[str, Any] = {}
    open_questions: List[Dict[str, Any]] = []
    if game_slug:
        mem = creator_agent.load_memory(mongo_db, uid, game_slug)
        facts, open_questions = dict(mem["facts"]), list(mem["open_questions"])

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
        known_layers=known_layers, metamodel=metamodel, resolve_field=resolve_field,
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
