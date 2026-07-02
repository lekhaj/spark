"""
FELT labels — per-session deterministic emotion labels, aligned with the
'Player Experience Telemetry & FELT Scoring' plan. NO LLM. The narrator may
restate these; it may never invent them.

  TENSION  : flat | earned-relief | frustrated | defeated
  MASTERY  : climbing | plateaued | struggling
  AUTONOMY : confident-independent | strategic-user | dependent | stuck-without-tools
             -> DATA GAP: needs booster-decision events; returns 'unknown' today.
"""
from __future__ import annotations

from . import config as C

F = C.FELT


def tension_label(m: dict) -> tuple[str, list[str]]:
    """m = session metrics from moods._session_metrics."""
    max_attempts = max(m["per_level_attempts"].values(), default=0)
    if max_attempts >= F.tension_frustrated_min_attempts and m["wins"] == 0:
        # defeated if they also gave up (abandoned/quit or ran fully out), else frustrated
        if m["abandoned"] or m["fails"] >= 1:
            return "defeated", [f"{max_attempts} attempts, no win, abandoned/out-of-hearts"]
        return "frustrated", [f"{max_attempts} attempts on one level, no win"]
    if 1 <= max_attempts <= F.tension_relief_max_attempts and m["wins"] >= 1:
        return "earned-relief", [f"{max_attempts} near-miss then recovered to a win"]
    if m["wins"] >= 1 and max_attempts == 0:
        return "flat", ["first-try win, no near-miss"]
    return "flat", ["no strong tension signal"]


def mastery_label(prior: dict, m: dict) -> tuple[str, list[str]]:
    """Session-trend mastery. Uses rolling retries/level from prior history."""
    arpl = prior.get("avg_retries_per_level")
    reached_new = m["level_reached"] is not None
    if arpl is not None and arpl >= C.MOODS.struggling_min_retries_per_level:
        return "struggling", [f"avg retries/level={arpl:.1f}"]
    if reached_new and (arpl is None or arpl < 1.0):
        return "climbing", [f"progressing, low retries (avg={arpl if arpl is None else round(arpl,1)})"]
    return "plateaued", ["progress steady, no clear climb or struggle"]


def autonomy_label(m: dict) -> tuple[str, list[str]]:
    """DATA GAP: real autonomy needs booster-decision events (offered/used/declined).
    Until AuraBeam emits them, we cannot tell independent from stuck-without-tools."""
    return F.autonomy_default, ["data gap: no booster-decision events yet"]


def session_felt(prior: dict, m: dict) -> dict:
    t, t_ev = tension_label(m)
    ms, ms_ev = mastery_label(prior, m)
    au, au_ev = autonomy_label(m)
    return {
        "felt_tension": t, "felt_mastery": ms, "felt_autonomy": au,
        "felt_evidence": {"tension": t_ev, "mastery": ms_ev, "autonomy": au_ev},
    }
