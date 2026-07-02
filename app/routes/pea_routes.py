"""
pea_routes.py — Player-Experience Analytics read-API (mounted at /pea).

Reads ONLY the derived pea_* tables in the CycleZero Postgres via the shared
SQLAlchemy session — never live Mixpanel, so the studio stays fast. All read-only.

  GET /pea/health
  GET /pea/digest?date=&game_id=
  GET /pea/mood-trends?days=
  GET /pea/personality?date=
  GET /pea/player/{distinct_id}
  GET /pea/friction?days=
  GET /pea/watch-list?date=
  GET /pea/bringback?date=
  GET /pea/funnel?date=
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.cyclezero.db import get_db as get_sql_db
from app.pea import config as C

router = APIRouter()

BANNER = C.PRELAUNCH_BANNER
GAME = C.GAME_ID


def _rows(db: Session, sql: str, params: dict):
    return [dict(r) for r in db.execute(text(sql), params).mappings().all()]


@router.get("/health")
def health():
    return {"ok": True, "game_id": GAME, "banner": BANNER}


@router.get("/digest")
def digest(date: Optional[dt.date] = None, game_id: str = GAME, db: Session = Depends(get_sql_db)):
    if date:
        return _rows(db, "SELECT * FROM pea_daily_digest WHERE game_id=:g AND date=:d",
                     {"g": game_id, "d": date})
    return _rows(db, "SELECT * FROM pea_daily_digest WHERE game_id=:g ORDER BY date DESC LIMIT 30",
                 {"g": game_id})


@router.get("/mood-trends")
def mood_trends(days: int = 30, game_id: str = GAME, db: Session = Depends(get_sql_db)):
    return _rows(db, "SELECT date, entry_mood_dist, exit_mood_dist, during_tension_dist, dau, "
                 "confidence FROM pea_daily_digest WHERE game_id=:g ORDER BY date DESC LIMIT :n",
                 {"g": game_id, "n": days})


@router.get("/personality")
def personality(date: Optional[dt.date] = None, game_id: str = GAME, db: Session = Depends(get_sql_db)):
    d = date or dt.date.today()
    dist = _rows(db, "SELECT personality, count(*) AS players FROM pea_player_state "
                 "WHERE game_id=:g AND date=:d GROUP BY personality ORDER BY players DESC",
                 {"g": game_id, "d": d})
    return {"date": str(d), "banner": BANNER, "distribution": dist}


@router.get("/player/{distinct_id}")
def player(distinct_id: str, game_id: str = GAME, db: Session = Depends(get_sql_db)):
    daily = _rows(db, "SELECT * FROM pea_player_state WHERE game_id=:g AND distinct_id=:id "
                  "ORDER BY date DESC", {"g": game_id, "id": distinct_id})
    sessions = _rows(db, "SELECT * FROM pea_session_state WHERE game_id=:g AND distinct_id=:id "
                     "ORDER BY started_at", {"g": game_id, "id": distinct_id})
    return {"distinct_id": distinct_id, "banner": BANNER, "daily": daily, "sessions": sessions}


@router.get("/friction")
def friction(days: int = 14, date: Optional[dt.date] = None, game_id: str = GAME,
             db: Session = Depends(get_sql_db)):
    if date:
        return _rows(db, "SELECT * FROM pea_level_friction WHERE game_id=:g AND date=:d ORDER BY level_id",
                     {"g": game_id, "d": date})
    return _rows(db, "SELECT * FROM pea_level_friction WHERE game_id=:g AND date > :since "
                 "ORDER BY date DESC, level_id", {"g": game_id, "since": dt.date.today() - dt.timedelta(days=days)})


@router.get("/watch-list")
def watch_list(date: Optional[dt.date] = None, game_id: str = GAME, db: Session = Depends(get_sql_db)):
    d = date or dt.date.today()
    return _rows(db, "SELECT distinct_id, exit_mood, persona, personality, overall_feeling, narrative "
                 "FROM pea_player_state WHERE game_id=:g AND date=:d AND flipped_to_risk "
                 "ORDER BY feeling_score", {"g": game_id, "d": d})


@router.get("/bringback")
def bringback(date: Optional[dt.date] = None, game_id: str = GAME, db: Session = Depends(get_sql_db)):
    d = date or dt.date.today()
    return _rows(db, "SELECT * FROM pea_bringback_list WHERE game_id=:g AND date=:d AND included "
                 "ORDER BY lapse_risk DESC", {"g": game_id, "d": d})


@router.get("/funnel")
def funnel(date: Optional[dt.date] = None, game_id: str = GAME, db: Session = Depends(get_sql_db)):
    if date:
        return _rows(db, "SELECT * FROM pea_funnel_retention WHERE game_id=:g AND date=:d",
                     {"g": game_id, "d": date})
    return _rows(db, "SELECT * FROM pea_funnel_retention WHERE game_id=:g ORDER BY date DESC LIMIT 1",
                 {"g": game_id})
