"""
Roll session_state -> player_state (daily), daily_digest, and level_friction.
Applies small-N 'low confidence', day-over-day mood deltas, and the pre-launch banner.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

from . import config as C

M = C.MOODS


def _confidence(n_users: int) -> str:
    return "low" if n_users < C.LOW_CONFIDENCE_USER_THRESHOLD else "high"


# FELT tension severity — pick the DAY's worst so "during" reflects the hardest moment,
# not just how the last session happened to end.
_TENSION_RANK = {"defeated": 3, "frustrated": 2, "earned-relief": 1, "flat": 0}


def _worst_tension(day_sessions: list[dict]) -> str | None:
    ts = [s.get("felt_tension") for s in day_sessions if s.get("felt_tension")]
    return max(ts, key=lambda t: _TENSION_RANK.get(t, 0)) if ts else None


def _first_present(day_sessions: list[dict], key: str):
    for s in day_sessions:
        if s.get(key):
            return s[key]
    return None


def rollup_players(sessions: list[dict]) -> list[dict]:
    """Collapse a player's sessions on a given date into one player_state row."""
    by_day: dict[tuple, list[dict]] = defaultdict(list)
    for s in sessions:
        if s["session_date"]:
            by_day[(s["distinct_id"], s["session_date"])].append(s)

    # rolling persona window per player
    all_by_player: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        all_by_player[s["distinct_id"]].append(s)

    from .moods import persona
    from .personality import personality as personality_fn
    rows = []
    for (did, date), day_sessions in by_day.items():
        day_sessions.sort(key=lambda s: s["started_at"])
        first, last = day_sessions[0], day_sessions[-1]
        window_start = date - dt.timedelta(days=M.persona_window_days)
        window = [s for s in all_by_player[did]
                  if s["session_date"] and window_start <= s["session_date"] <= date]
        p_str, axes = persona(window)
        lapsing = last["exit_mood"] == "churn-risk"
        pers = personality_fn(window, lapsing=lapsing)
        exit_m = last["exit_mood"]
        rows.append({
            "game_id": C.GAME_ID, "distinct_id": did, "date": date,
            "sessions_today": len(day_sessions),
            "level_reached": max((s["level_reached"] or 0) for s in day_sessions) or None,
            "retries": sum(s["retries"] for s in day_sessions),
            "fails": sum(s["fails"] for s in day_sessions),
            "wins": sum(s["wins"] for s in day_sessions),
            "entry_mood": first["entry_mood"], "exit_mood": exit_m,
            "overall_feeling": last["overall_feeling"], "feeling_score": last["feeling_score"],
            "build_version": _first_present(day_sessions, "build_version"),
            "platform": _first_present(day_sessions, "platform"),
            "persona": p_str, "persona_axes": axes, "prev_persona": None,
            "personality": pers["personality"],
            "personality_runner_up": pers.get("personality_runner_up"),
            "personality_spectrum": pers["spectrum"],
            "felt_tension": _worst_tension(day_sessions) or last["felt_tension"],
            "felt_mastery": last["felt_mastery"], "felt_autonomy": last["felt_autonomy"],
            "confidence": pers["confidence"], "is_new": any(s["is_new"] for s in day_sessions),
            "flipped_to_risk": exit_m in ("frustrated", "churn-risk"),
            "evidence": [e for s in day_sessions for e in s["evidence"]],
            "narrative": None,  # filled by narrate.py
        })
    return rows


def build_digest(player_rows: list[dict], date: dt.date, prev_dist: dict | None = None) -> dict:
    today = [p for p in player_rows if p["date"] == date]
    dau = len(today)
    entry = Counter(p["entry_mood"] for p in today)            # how they ARRIVED (start)
    during = Counter(p.get("felt_tension") for p in today if p.get("felt_tension"))  # DURING gameplay
    exit_ = Counter(p["exit_mood"] for p in today)             # how they ENDED
    personality_dist = Counter(p.get("personality") for p in today if p.get("personality"))
    by_build = Counter(p.get("build_version") or "unknown" for p in today)
    by_platform = Counter(p.get("platform") or "unknown" for p in today)
    watch = [{"distinct_id": p["distinct_id"], "exit_mood": p["exit_mood"], "persona": p["persona"]}
             for p in today if p["flipped_to_risk"]]
    friction = _friction_from_players(today)
    conf = _confidence(dau)

    def dod(cur: Counter, prev: dict | None):
        if not prev:
            return {}
        return {k: cur.get(k, 0) - prev.get(k, 0) for k in set(cur) | set(prev)}

    return {
        "game_id": C.GAME_ID, "date": date, "dau": dau,
        "new_users": sum(1 for p in today if p["is_new"]),
        "returning_users": sum(1 for p in today if not p["is_new"]),
        "by_build": dict(by_build), "by_platform": dict(by_platform),
        "entry_mood_dist": dict(entry), "exit_mood_dist": dict(exit_),
        "during_tension_dist": dict(during), "personality_dist": dict(personality_dist),
        "entry_mood_dod": dod(entry, (prev_dist or {}).get("entry")),
        "exit_mood_dod": dod(exit_, (prev_dist or {}).get("exit")),
        "top_friction_levels": friction[:5],
        "watch_list": watch, "insights": [],  # narrate.py fills insights
        "confidence": conf, "banner": C.PRELAUNCH_BANNER,
    }


def _friction_from_players(players: list[dict]) -> list[dict]:
    agg = defaultdict(lambda: {"frustrated_sessions": 0, "retries": 0, "fails": 0, "players": set()})
    for p in players:
        lvl = p["level_reached"]
        if not lvl:
            continue
        a = agg[lvl]
        a["retries"] += p["retries"]
        a["fails"] += p["fails"]
        a["players"].add(p["distinct_id"])
        if p["flipped_to_risk"]:
            a["frustrated_sessions"] += 1
    out = [{"level_id": lvl, "retries": a["retries"], "fails": a["fails"],
            "frustrated_sessions": a["frustrated_sessions"], "unique_players": len(a["players"])}
           for lvl, a in agg.items()]
    out.sort(key=lambda x: (x["frustrated_sessions"], x["retries"] + x["fails"]), reverse=True)
    return out


def build_level_friction(sessions: list[dict], date: dt.date) -> list[dict]:
    agg = defaultdict(lambda: {"attempts": 0, "retries": 0, "fails": 0, "wins": 0,
                               "frustrated_sessions": 0, "churn_risk_sessions": 0, "players": set()})
    for s in sessions:
        if s["session_date"] != date:
            continue
        for lvl in s["levels_played"]:
            a = agg[lvl]
            a["attempts"] += 1
            a["players"].add(s["distinct_id"])
        top = s["level_reached"]
        if top:
            a = agg[top]
            a["retries"] += s["retries"]
            a["fails"] += s["fails"]
            a["wins"] += s["wins"]
            if s["exit_mood"] == "frustrated":
                a["frustrated_sessions"] += 1
            if s["exit_mood"] == "churn-risk":
                a["churn_risk_sessions"] += 1
    rows = []
    for lvl, a in agg.items():
        rows.append({"game_id": C.GAME_ID, "date": date, "level_id": lvl,
                     "attempts": a["attempts"], "retries": a["retries"], "fails": a["fails"],
                     "wins": a["wins"], "frustrated_sessions": a["frustrated_sessions"],
                     "churn_risk_sessions": a["churn_risk_sessions"],
                     "unique_players": len(a["players"]),
                     "confidence": _confidence(len(a["players"]))})
    return rows


def build_funnel_retention(sessions: list[dict], df, date: dt.date, max_level: int = 5) -> dict:
    """Funnel: install -> first_open -> level_1..N complete (unique users, cumulative).
    Retention: of players first-seen on (date-1)/(date-7), fraction active again on `date`.
    df is the raw_events DataFrame (for funnel counts); sessions drive retention."""
    steps = []
    installs = df["distinct_id"].nunique() if not df.empty else 0
    opened = df[df["event_name"] == C.E_APP_OPENED]["distinct_id"].nunique() if not df.empty else 0
    steps.append({"step": "install (distinct_id seen)", "users": int(installs)})
    steps.append({"step": "first_open (App Opened)", "users": int(opened)})
    if not df.empty:
        comp = df[df["event_name"] == C.E_LEVEL_COMPLETED]
        for lvl in range(1, max_level + 1):
            u = comp[comp["level_id"] == lvl]["distinct_id"].nunique()
            steps.append({"step": f"level_{lvl}_complete", "users": int(u)})

    # retention from session dates
    first_seen: dict[str, dt.date] = {}
    active: dict[dt.date, set] = defaultdict(set)
    for s in sessions:
        d = s["session_date"]
        if not d:
            continue
        active[d].add(s["distinct_id"])
        if s["distinct_id"] not in first_seen or d < first_seen[s["distinct_id"]]:
            first_seen[s["distinct_id"]] = d

    def _ret(offset: int):
        cohort_day = date - dt.timedelta(days=offset)
        cohort = {p for p, fs in first_seen.items() if fs == cohort_day}
        if not cohort:
            return None, 0
        returned = len(cohort & active.get(cohort_day + dt.timedelta(days=offset), set()))
        return round(returned / len(cohort), 3), len(cohort)

    d1, c1 = _ret(1)
    d7, c7 = _ret(7)
    return {"game_id": C.GAME_ID, "date": date, "funnel_steps": steps,
            "d1_retention": d1, "d7_retention": d7, "cohort_size": max(c1, c7),
            "confidence": _confidence(int(installs))}
