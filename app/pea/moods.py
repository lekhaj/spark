"""
DETERMINISTIC feeling engine — Level 2, thresholds only, NO LLM.

Computes session_state, player_state (daily rollup) and persona from cached raw events.
The LLM (narrate.py) may ONLY narrate these labels; it must never invent a mood.
All thresholds come from config.MOODS so they can be retuned in one place.

Inputs: a pandas DataFrame of normalized raw_events (see extract._to_row).
Outputs: list[dict] session rows, list[dict] player-day rows.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import pandas as pd

from . import config as C

M = C.MOODS


# ----------------------------------------------------------------------------- sessions
def stitch_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign session_id per (distinct_id): use $session_start/$session_end when present,
    else split on SESSION_INACTIVITY_MINUTES. Marks `stitched=True` when no $session_end."""
    df = df.sort_values(["distinct_id", "ts_client"]).copy()
    df["session_id"] = None
    df["stitched"] = False
    gap = pd.Timedelta(minutes=C.SESSION_INACTIVITY_MINUTES)

    for did, g in df.groupby("distinct_id", sort=False):
        sid_num, prev_ts, has_end = 0, None, True
        for idx, row in g.iterrows():
            ts = row["ts_client"] or row["ts_server"]
            new_session = (
                prev_ts is None
                or (ts - prev_ts) > gap
                or row["event_name"] == C.E_SESSION_START
            )
            if new_session:
                sid_num += 1
                has_end = False
            df.at[idx, "session_id"] = f"{did}:{sid_num}"
            if row["event_name"] == C.E_SESSION_END:
                has_end = True
            df.at[idx, "stitched"] = not has_end
            prev_ts = ts
    return df


def _session_metrics(g: pd.DataFrame) -> dict:
    ev = g["event_name"]
    levels = sorted({int(l) for l in g["level_id"].dropna().tolist() if l and l >= 1})
    dur = None
    end_rows = g[ev == C.E_SESSION_END]
    if not end_rows.empty:
        dur = end_rows.iloc[-1]["properties"].get("$duration_s")
    if dur is None and len(g) > 1:
        span = (g["ts_client"].max() - g["ts_client"].min())
        dur = int(span.total_seconds()) if pd.notna(span) else None
    # per-level attempt tally for frustration detection
    per_level_attempts = defaultdict(int)
    for _, r in g.iterrows():
        if r["event_name"] in (C.RETRY_EVENTS | C.FAIL_EVENTS) and r["level_id"]:
            per_level_attempts[int(r["level_id"])] += 1
    return {
        "levels_played": levels,
        "level_reached": max(levels) if levels else None,
        "retries": int(ev.isin(C.RETRY_EVENTS).sum()),
        "fails": int(ev.isin(C.FAIL_EVENTS).sum()),
        "wins": int(ev.isin(C.WIN_EVENTS).sum()),
        "is_new": bool(ev.isin(C.NEW_PLAYER_EVENTS).any()),
        "duration_s": int(dur) if dur is not None else None,
        "per_level_attempts": dict(per_level_attempts),
        "ended_on_win": (ev[ev.isin(C.WIN_EVENTS | C.FAIL_EVENTS)].tolist()[-1:] == [next(iter(C.WIN_EVENTS))])
                        if ev.isin(C.WIN_EVENTS | C.FAIL_EVENTS).any() else False,
        "has_end": bool((ev == C.E_SESSION_END).any()),
        "abandoned": bool(ev.isin(C.ABANDON_EVENTS).any()),
    }


# ----------------------------------------------------------------------------- entry mood
def entry_mood(prior: dict, m: dict) -> tuple[str, str | None, list[str]]:
    """prior = per-player history at session start: {days_since_install, session_count,
    normal_gap_days, gap_days, last_exit_positive, avg_retries_per_level}."""
    candidates: list[tuple[str, str]] = []  # (mood, evidence)
    dsi = prior.get("days_since_install")
    sc = prior.get("session_count", 0)

    if m["is_new"] or (dsi is not None and dsi <= M.new_max_days_since_install and sc <= M.new_max_session_count):
        candidates.append(("new-excited", f"days_since_install={dsi}, session_count={sc}"))
    if (prior.get("gap_days") is not None and prior.get("normal_gap_days")
            and prior["gap_days"] > M.cadence_multiplier_at_risk * prior["normal_gap_days"]):
        candidates.append(("at-risk-returner",
                           f"gap={prior['gap_days']:.1f}d > {M.cadence_multiplier_at_risk}x norm {prior['normal_gap_days']:.1f}d"))
    if sc >= M.returning_engaged_min_sessions and prior.get("last_exit_positive"):
        candidates.append(("returning-engaged", f"session_count={sc}, last exit positive"))
    if (prior.get("avg_retries_per_level") is not None
            and prior["avg_retries_per_level"] <= M.puzzle_seeker_max_retries_per_level):
        candidates.append(("puzzle-seeker", f"avg_retries/level={prior['avg_retries_per_level']:.2f}"))
    if (len(m["levels_played"]) >= M.explorer_min_distinct_levels
            and (m["duration_s"] or 0) <= M.explorer_max_session_minutes * 60):
        candidates.append(("explorer", f"{len(m['levels_played'])} levels in short session"))

    if not candidates:
        candidates.append(("returning-engaged", "default: has prior history, no other signal"))
    best = candidates[0]
    runner = candidates[1][0] if len(candidates) > 1 else None
    return best[0], runner, [c[1] for c in candidates[:2]]


# ----------------------------------------------------------------------------- exit mood
def exit_mood(m: dict, will_return_in_cadence: bool | None) -> tuple[str, list[str]]:
    ev: list[str] = []
    max_attempts = max(m["per_level_attempts"].values(), default=0)
    frustrated = max_attempts >= M.frustrated_min_attempts_one_level and m["wins"] == 0
    interrupted = (not m["has_end"]) and (m["duration_s"] or 0) <= M.interrupted_max_session_minutes * 60 \
        and m["wins"] == 0 and m["fails"] == 0

    if frustrated:
        worst = max(m["per_level_attempts"], key=m["per_level_attempts"].get)
        ev.append(f"{max_attempts} attempts on level {worst}, 0 wins")
        if will_return_in_cadence is False:
            return "churn-risk", ev + ["no return within expected cadence"]
        return "frustrated", ev
    if interrupted:
        return "interrupted", [f"abrupt end, {m['duration_s']}s, no win/fail, no $session_end"]
    if m["wins"] >= 1 and m["ended_on_win"]:
        if m["wins"] >= M.comeback_min_trailing_wins and m["has_end"]:
            return "comeback-tomorrow", [f"{m['wins']} win(s), ended on win at a natural break"]
        return "happy", [f"{m['wins']} win(s), ended on a win"]
    if m["wins"] >= 1 or m["level_reached"]:
        return "ok-satisfied", [f"{m['wins']} win(s), neutral stop"]
    return "ok-satisfied", ["no strong signal"]


# ----------------------------------------------------------------------------- overall feeling
def overall_feeling(exit_m: str, entry_m: str, m: dict) -> tuple[str, int]:
    score = M.feeling_scores.get(exit_m, 0)
    if m["wins"] and m["fails"] == 0:
        score = min(2, score + 1)
    if m["fails"] >= 3:
        score = max(-2, score - 1)
    score = max(-2, min(2, score))
    return M.feeling_labels[score], score


# ----------------------------------------------------------------------------- persona (rolling)
def persona(window_rows: list[dict]) -> tuple[str, dict]:
    """Derive persona axes from a rolling window of this player's session rows."""
    if not window_rows:
        return "unknown", {}
    total_retries = sum(r["retries"] for r in window_rows)
    total_fails = sum(r["fails"] for r in window_rows)
    total_attempts = sum(r["retries"] + r["fails"] + r["wins"] for r in window_rows) or 1
    levels = [r["level_reached"] for r in window_rows if r["level_reached"]]
    new_levels = len(set(levels))
    active_days = len({r["session_date"] for r in window_rows})
    sessions_per_day = len(window_rows) / max(active_days, 1)
    retries_per_level = total_retries / max(new_levels, 1)

    skill = ("climbing" if new_levels >= M.climbing_min_new_levels and retries_per_level < M.struggling_min_retries_per_level
             else "struggling" if retries_per_level >= M.struggling_min_retries_per_level
             else "plateaued")
    rhythm = ("binge" if sessions_per_day >= M.binge_min_sessions_per_active_day
              else "grazer" if sessions_per_day <= M.grazer_max_sessions_per_active_day
              else "steady")
    fail_rate = total_fails / total_attempts
    risk = "cautious" if fail_rate <= M.cautious_max_fail_rate else "exploratory"
    booster = "unknown"  # DATA GAP: no booster events exist yet

    axes = {"skill_trajectory": skill, "booster_relationship": booster,
            "session_rhythm": rhythm, "risk_posture": risk}
    return f"{skill}/{booster}/{rhythm}/{risk}", axes


# ----------------------------------------------------------------------------- orchestration
def build_session_state(df: pd.DataFrame, tz=C.TIMEZONE) -> list[dict]:
    """Full pipeline for a batch of raw events -> session_state rows with moods."""
    df = stitch_sessions(df)
    out: list[dict] = []
    # player history accumulates as we walk sessions chronologically
    hist: dict[str, dict] = defaultdict(lambda: {"session_count": 0, "gaps": [], "last_date": None,
                                                  "retries": 0, "levels": set(), "last_exit_positive": None})
    for (did, sid), g in df.groupby(["distinct_id", "session_id"], sort=False):
        g = g.sort_values("ts_client")
        m = _session_metrics(g)
        started = g["ts_server"].min()
        sdate = started.astimezone(tz).date() if started is not None else None

        h = hist[did]
        gap_days = ((sdate - h["last_date"]).days if h["last_date"] else None)
        normal_gap = (sum(h["gaps"]) / len(h["gaps"])) if h["gaps"] else None
        prior = {
            "days_since_install": None,  # filled from engage/People.FirstLogin if available
            "session_count": h["session_count"],
            "gap_days": gap_days,
            "normal_gap_days": normal_gap,
            "last_exit_positive": h["last_exit_positive"],
            "avg_retries_per_level": (h["retries"] / len(h["levels"])) if h["levels"] else None,
        }
        em, em_runner, em_ev = entry_mood(prior, m)
        xm, xm_ev = exit_mood(m, will_return_in_cadence=None)  # cadence resolved in a 2nd pass
        feeling, score = overall_feeling(xm, em, m)
        conf = "low" if False else "high"  # per-bucket small-N set later in aggregate()

        from .felt import session_felt
        felt = session_felt(prior, m)

        out.append({
            "game_id": C.GAME_ID, "distinct_id": did, "session_id": sid, "session_date": sdate,
            "started_at": started, "ended_at": g["ts_server"].max(), "duration_s": m["duration_s"],
            "build_version": g["build_version"].dropna().iloc[0] if g["build_version"].notna().any() else None,
            "platform": g["platform"].dropna().iloc[0] if g["platform"].notna().any() else None,
            "is_new": m["is_new"], "level_reached": m["level_reached"], "levels_played": m["levels_played"],
            "retries": m["retries"], "fails": m["fails"], "wins": m["wins"],
            "entry_mood": em, "entry_mood_runner_up": em_runner, "exit_mood": xm,
            "overall_feeling": feeling, "feeling_score": score,
            "felt_tension": felt["felt_tension"], "felt_mastery": felt["felt_mastery"],
            "felt_autonomy": felt["felt_autonomy"],
            "stitched": bool(g["stitched"].iloc[-1]), "confidence": conf,
            "evidence": [{"entry": em_ev}, {"exit": xm_ev}, {"felt": felt["felt_evidence"]},
                         {"counts": {"retries": m["retries"], "fails": m["fails"], "wins": m["wins"],
                                     "level_reached": m["level_reached"]}}],
        })
        # update history
        h["session_count"] += 1
        if gap_days is not None:
            h["gaps"].append(gap_days)
        h["last_date"] = sdate
        h["retries"] += m["retries"]
        h["levels"].update(m["levels_played"])
        h["last_exit_positive"] = score >= 1
    return out
