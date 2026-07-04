"""
GROW — the organic-growth engine. PEA already knows the emotionally charged moments
(deterministically), so Grow surfaces "share-worthy moments", helps the creator package
them (caption + a /play deep-link with UTM), and logs the shares for a share funnel.

Phase 1 (no platform APIs, no in-game capture yet): detect moments from telemetry, build
the share package client-side from these, log created shares. Real clip capture + click/play
attribution (via the /play landing + Mixpanel UTM join) are Phase 2, flagged as data gaps.
"""
from __future__ import annotations

import datetime as dt
import uuid

from . import config as C
from . import store

# What makes a moment worth sharing — deterministic, from FELT labels + arcs.
_MOMENT_RULES = [
    ("earned-relief", "Clutch comeback — near-miss then a win", "😤➡️🏆 So close, then clutch!"),
    ("comeback",      "Ended on a high after a rough patch",     "Never gave up 💪"),
    ("first-win",     "First level cleared — the aha moment",    "First win in AuraBeam! ✨"),
    ("mastery-run",   "Flawless run, zero retries",              "Flawless. No retries. 🎯"),
]


def detect_moments(game_id: str = C.GAME_ID, limit: int = 20) -> list[dict]:
    """Rank recent sessions into share-worthy moments. Newest + most charged first."""
    with store.connect() as cx, cx.cursor() as cur:
        cur.execute(
            "SELECT distinct_id, session_id, session_date, level_reached, felt_tension, "
            "exit_mood, wins, retries, fails FROM pea_session_state "
            "WHERE game_id=%s AND session_date IS NOT NULL "
            "ORDER BY session_date DESC LIMIT 400", (game_id,))
        rows = cur.fetchall()

    moments = []
    for (did, sid, sdate, level, tension, exit_mood, wins, retries, fails) in rows:
        kind = caption = why = None
        if tension == "earned-relief" and wins:
            kind, why, caption = "earned-relief", _MOMENT_RULES[0][1], _MOMENT_RULES[0][2]
        elif exit_mood in ("comeback-tomorrow", "happy") and (retries or fails) and wins:
            kind, why, caption = "comeback", _MOMENT_RULES[1][1], _MOMENT_RULES[1][2]
        elif wins and retries == 0 and fails == 0 and (level or 0) >= 3:
            kind, why, caption = "mastery-run", _MOMENT_RULES[3][1], _MOMENT_RULES[3][2]
        elif wins and (level or 0) <= 2:
            kind, why, caption = "first-win", _MOMENT_RULES[2][1], _MOMENT_RULES[2][2]
        if not kind:
            continue
        # a charge score to rank moments
        charge = {"earned-relief": 3, "comeback": 3, "mastery-run": 2, "first-win": 1}[kind]
        moments.append({
            "moment_key": sid, "distinct_id": did, "date": str(sdate),
            "level_id": level, "felt_tension": tension, "exit_mood": exit_mood,
            "kind": kind, "why": why, "suggested_caption": caption,
            "charge": charge, "wins": wins, "retries": retries, "fails": fails,
        })
    moments.sort(key=lambda m: (m["charge"], m["date"]), reverse=True)
    return moments[:limit]


# Channel presets: aspect + hashtag flavor (used client-side to build the package).
CHANNELS = {
    "youtube_short": {"label": "YouTube Short", "aspect": "9:16", "tags": "#shorts #indiegame #puzzle"},
    "tiktok":        {"label": "TikTok",        "aspect": "9:16", "tags": "#gaming #puzzlegame #indiedev"},
    "reels":         {"label": "Instagram Reels", "aspect": "9:16", "tags": "#reels #indiegame #mobilegame"},
    "x":             {"label": "X / Twitter",   "aspect": "16:9", "tags": "#indiegame #gamedev"},
}


def play_url(game_id: str, moment_key: str, channel: str) -> str:
    """The growth atom: a deep-link back into the game, UTM-tagged so Mixpanel attributes
    the installs it drives (App Opened already captures utm_*)."""
    base = f"https://spark-studio.pages.dev/play/{game_id}"
    return f"{base}?utm_source={channel}&utm_medium=organic&utm_campaign=grow&m={moment_key}"


def log_share(game_id: str, moment: dict, channel: str, caption: str) -> dict:
    share_id = uuid.uuid4().hex[:12]
    url = play_url(game_id, moment.get("moment_key", ""), channel)
    row = {
        "game_id": game_id, "share_id": share_id, "created_at": dt.datetime.now(dt.timezone.utc),
        "moment_key": moment.get("moment_key"), "distinct_id": moment.get("distinct_id"),
        "level_id": moment.get("level_id"), "felt_tension": moment.get("felt_tension"),
        "exit_mood": moment.get("exit_mood"), "channel": channel, "caption": caption,
        "play_url": url, "clicks": 0, "plays": 0,
    }
    store._write("pea_share_log", [row], ("game_id", "share_id"))
    return row


def list_shares(game_id: str = C.GAME_ID, limit: int = 100) -> list[dict]:
    with store.connect() as cx, cx.cursor() as cur:
        cur.execute("SELECT share_id, created_at, channel, caption, play_url, level_id, "
                    "felt_tension, exit_mood, clicks, plays FROM pea_share_log "
                    "WHERE game_id=%s ORDER BY created_at DESC LIMIT %s", (game_id, limit))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def share_funnel(game_id: str = C.GAME_ID) -> dict:
    shares = list_shares(game_id, 1000)
    return {
        "shares_created": len(shares),
        "clicks": sum(s["clicks"] for s in shares),   # data gap until /play landing lands
        "plays": sum(s["plays"] for s in shares),
        "by_channel": _count_by(shares, "channel"),
        "k_factor": None,  # data gap: needs shares-generated-by-referred-players
        "data_gap": "clicks/plays/K-factor need the /play landing page + Mixpanel UTM join (Phase 2)",
    }


def _count_by(rows, key):
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out
