"""
pea_routes.py — Player-Experience Analytics read-API (mounted at /pea).

Reads ONLY the derived pea_* tables in the CycleZero Postgres via the shared
SQLAlchemy session — never live Mixpanel, so the studio stays fast. Owner-gated
(studio analytics is owner-only today); `/pea/health` stays open for probes.

Date defaults use the latest date PRESENT in the data (not the box's UTC 'today'),
so the dashboard is never blank when the server clock leads the IST data by a day.

  GET   /pea/health
  GET   /pea/digest?date=            GET /pea/mood-trends?days=
  GET   /pea/personality?date=       GET /pea/players?q=&limit=
  GET   /pea/player/{distinct_id}    GET /pea/friction?days=
  GET   /pea/watch-list?date=        GET /pea/funnel?date=
  GET   /pea/bringback?date=         PATCH /pea/bringback  (persist hand-picks)
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.cyclezero.db import get_db as get_sql_db
from app.lib.identity import StudioUser, current_user, require_owner
from app.pea import config as C, store
from app.pea import felt_axes as felt_axes_mod
from app.pea import grow as grow_mod

router = APIRouter()

BANNER = C.PRELAUNCH_BANNER
GAME = C.GAME_ID


def _rows(db: Session, sql: str, params: dict):
    return [dict(r) for r in db.execute(text(sql), params).mappings().all()]


def _default_date(date: Optional[dt.date], table: str = "pea_daily_digest") -> Optional[dt.date]:
    return date or store.latest_date(GAME, table)


@router.get("/health")
def health():
    return {"ok": True, "game_id": GAME, "banner": BANNER}


@router.get("/felt-axes")
def felt_axes(game_id: str = GAME, user: StudioUser = Depends(current_user)):
    """The FELT (telemetry) half of the Critic's 7-axis scorecard — how players actually
    felt, on the same axes, so the studio can see GAP = |structural - felt|. Aggregate-only
    (no distinct_ids), so signed-in creators can see it in the Experience view."""
    return felt_axes_mod.felt_axes(game_id)


@router.get("/digest")
def digest(date: Optional[dt.date] = None, game_id: str = GAME,
           user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    d = _default_date(date)
    if d:
        return _rows(db, "SELECT * FROM pea_daily_digest WHERE game_id=:g AND date=:d",
                     {"g": game_id, "d": d})
    return []


@router.get("/mood-trends")
def mood_trends(days: int = 30, game_id: str = GAME,
                user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    return _rows(db, "SELECT date, entry_mood_dist, exit_mood_dist, during_tension_dist, dau, "
                 "confidence FROM pea_daily_digest WHERE game_id=:g ORDER BY date DESC LIMIT :n",
                 {"g": game_id, "n": days})


@router.get("/personality")
def personality(date: Optional[dt.date] = None, game_id: str = GAME,
                user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    d = _default_date(date, "pea_player_state")
    dist = _rows(db, "SELECT personality, count(*) AS players FROM pea_player_state "
                 "WHERE game_id=:g AND date=:d GROUP BY personality ORDER BY players DESC",
                 {"g": game_id, "d": d}) if d else []
    return {"date": str(d) if d else None, "banner": BANNER, "distribution": dist}


@router.get("/players")
def players(q: str = "", limit: int = 50, game_id: str = GAME,
            user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    """Player picker for the deep-dive search: most-recent row per distinct_id."""
    require_owner(user)
    return _rows(db,
                 "SELECT DISTINCT ON (distinct_id) distinct_id, date, personality, exit_mood, "
                 "overall_feeling, flipped_to_risk FROM pea_player_state "
                 "WHERE game_id=:g AND (:q='' OR distinct_id ILIKE :like) "
                 "ORDER BY distinct_id, date DESC LIMIT :n",
                 {"g": game_id, "q": q, "like": f"%{q}%", "n": limit})


@router.get("/player/{distinct_id}")
def player(distinct_id: str, game_id: str = GAME,
           user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    daily = _rows(db, "SELECT * FROM pea_player_state WHERE game_id=:g AND distinct_id=:id "
                  "ORDER BY date DESC", {"g": game_id, "id": distinct_id})
    sessions = _rows(db, "SELECT * FROM pea_session_state WHERE game_id=:g AND distinct_id=:id "
                     "ORDER BY started_at", {"g": game_id, "id": distinct_id})
    return {"distinct_id": distinct_id, "banner": BANNER, "daily": daily, "sessions": sessions}


@router.get("/friction")
def friction(days: int = 14, date: Optional[dt.date] = None, game_id: str = GAME,
             user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    if date:
        return _rows(db, "SELECT * FROM pea_level_friction WHERE game_id=:g AND date=:d ORDER BY level_id",
                     {"g": game_id, "d": date})
    latest = store.latest_date(game_id, "pea_level_friction") or dt.date.today()
    return _rows(db, "SELECT * FROM pea_level_friction WHERE game_id=:g AND date > :since "
                 "ORDER BY date DESC, level_id", {"g": game_id, "since": latest - dt.timedelta(days=days)})


@router.get("/watch-list")
def watch_list(date: Optional[dt.date] = None, game_id: str = GAME,
               user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    d = _default_date(date, "pea_player_state")
    if not d:
        return []
    return _rows(db, "SELECT distinct_id, exit_mood, persona, personality, overall_feeling, narrative "
                 "FROM pea_player_state WHERE game_id=:g AND date=:d AND flipped_to_risk "
                 "ORDER BY feeling_score", {"g": game_id, "d": d})


@router.get("/bringback")
def bringback(date: Optional[dt.date] = None, game_id: str = GAME,
              user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    d = _default_date(date, "pea_bringback_list")
    if not d:
        return []
    return _rows(db, "SELECT * FROM pea_bringback_list WHERE game_id=:g AND date=:d "
                 "ORDER BY lapse_risk DESC", {"g": game_id, "d": d})


@router.patch("/bringback")
def bringback_override(payload: dict = Body(...), game_id: str = GAME,
                       user: StudioUser = Depends(current_user)):
    """Persist a hand-pick: {date, distinct_id, included}. Survives nightly recompute."""
    require_owner(user)
    n = store.set_bringback_override(game_id, payload["date"], payload["distinct_id"],
                                     bool(payload.get("included", True)))
    return {"updated": n}


@router.get("/funnel")
def funnel(date: Optional[dt.date] = None, game_id: str = GAME,
           user: StudioUser = Depends(current_user), db: Session = Depends(get_sql_db)):
    require_owner(user)
    if date:
        return _rows(db, "SELECT * FROM pea_funnel_retention WHERE game_id=:g AND date=:d",
                     {"g": game_id, "d": date})
    return _rows(db, "SELECT * FROM pea_funnel_retention WHERE game_id=:g ORDER BY date DESC LIMIT 1",
                 {"g": game_id})


# ── GROW: capture-worthy moments → package → publish link → share funnel ──────────
@router.get("/grow/moments")
def grow_moments(game_id: str = GAME, user: StudioUser = Depends(current_user)):
    require_owner(user)
    return {"banner": BANNER, "channels": grow_mod.CHANNELS, "moments": grow_mod.detect_moments(game_id)}


@router.post("/grow/share")
def grow_share(payload: dict = Body(...), game_id: str = GAME, user: StudioUser = Depends(current_user)):
    """Log a share package the creator built: {moment:{...}, channel, caption}."""
    require_owner(user)
    return grow_mod.log_share(game_id, payload.get("moment", {}), payload["channel"], payload.get("caption", ""))


@router.get("/grow/shares")
def grow_shares(game_id: str = GAME, user: StudioUser = Depends(current_user)):
    require_owner(user)
    return {"shares": grow_mod.list_shares(game_id), "funnel": grow_mod.share_funnel(game_id)}
