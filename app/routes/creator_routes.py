"""
creator_routes.py — chat-driven game builder (the always-on creator orchestrator)
=================================================================================

The studio chat posts each turn here. The deterministic orchestrator
(``cyclezero.creator_agent.run_turn``) calls Haiku with tool-use, then writes the
confirmed parts to the Postgres game graph + the per-(uid, game) Mongo memory doc,
and surfaces an ``ask_clarification`` as a popup when Haiku is unsure.

  POST /creator/turn    one chat turn → {reply, saved[], pending_question?, game_slug}
  GET  /creator/state   memory doc + recent turns for a game (history reload)
  GET  /creator/latest  most-recent game for the caller (auto-load on mount)

Identity via the studio headers (attribution, not auth). All writes are
best-effort and must never 500 the chat.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.cyclezero import creator_agent, service
from app.cyclezero.db import get_db as get_sql_db
from app.lib import usage_recorder
from app.lib.identity import StudioUser, current_user
from app.services.mongo_service import get_db as get_mongo

log = logging.getLogger("creator_routes")
router = APIRouter()


def _provider():
    """The Tier-A creator provider (Bedrock = AWS credits). Overridable in tests."""
    return creator_agent.make_bedrock_creator()


def _metamodel() -> Optional[Dict[str, Any]]:
    """The shared relation metamodel (``{layers, relation_types}``), best-effort.
    Gates ``link_entities`` against the edge contract; None → relations rejected."""
    try:
        from app.cyclezero import metamodel
        from worker.lib import manual_gen_schema as mgs

        return metamodel.load_metamodel(mgs.get_db())
    except Exception:  # noqa: BLE001
        return None


def _known_layers() -> List[str]:
    """Known layer names from the shared metamodel (best-effort)."""
    return list((_metamodel() or {}).get("layers", {}).keys())


@router.post("/creator/turn")
def creator_turn(
    body: Dict[str, Any] = Body(...),
    sql_db: Session = Depends(get_sql_db),
    user: StudioUser = Depends(current_user),
) -> Dict[str, Any]:
    game_slug: Optional[str] = (body.get("game_slug") or "").strip() or None
    answer = body.get("answer")
    field = (body.get("field") or "").strip() or None
    if answer is not None and str(answer).strip():
        # the user answered a clarification popup
        user_text = f"Answer to '{field or 'question'}': {answer}"
        resolve_field = field
    else:
        user_text = (body.get("text") or "").strip()
        resolve_field = None

    if not user_text:
        return {"reply": "", "saved": [], "pending_question": None, "game_slug": game_slug}

    try:
        provider = _provider()
        mm = _metamodel()
        known_layers = list((mm or {}).get("layers", {}).keys()) or _known_layers()
        with usage_recorder.attribution(
            uid=user.uid, email=user.email, game_slug=game_slug or "(new)", agent="creator",
        ):
            return creator_agent.run_turn(
                provider=provider,
                sql_db=sql_db,
                mongo_db=get_mongo(),
                uid=user.uid,
                email=user.email,
                game_slug=game_slug,
                user_text=user_text,
                known_layers=known_layers,
                metamodel=mm,
                resolve_field=resolve_field,
            )
    except Exception as e:  # noqa: BLE001 — chat must degrade, never 500
        log.exception("creator turn failed")
        return {
            "reply": "Sorry — I hit an error saving that. Try again?",
            "saved": [], "pending_question": None, "game_slug": game_slug,
            "error": str(e),
        }


@router.get("/creator/state")
def creator_state(
    game_slug: str = Query(...),
    user: StudioUser = Depends(current_user),
) -> Dict[str, Any]:
    """Memory + recent chat history for a game — used to rehydrate the chat."""
    mongo = get_mongo()
    mem = creator_agent.load_memory(mongo, user.uid, game_slug)
    turns = creator_agent.load_turns(mongo, user.uid, game_slug)
    return {"game_slug": game_slug, "facts": mem["facts"],
            "open_questions": mem["open_questions"], "turns": turns}


@router.get("/creator/latest")
def creator_latest(
    sql_db: Session = Depends(get_sql_db),
    user: StudioUser = Depends(current_user),
) -> Dict[str, Any]:
    """The caller's most-recently-created game (list_games is created_at desc)."""
    try:
        for g in service.list_games(sql_db):
            if g.owner_id == user.uid:
                return {"game_slug": g.slug, "title": g.title}
    except Exception:  # noqa: BLE001
        log.exception("creator latest failed")
    return {"game_slug": None, "title": None}
