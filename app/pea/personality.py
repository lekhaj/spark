"""
Personality spectrum — DERIVED, deterministic, rolling window. NO LLM invention.

Produces, per player: six 0-100 dimension scores + a best-fit archetype label
(+ runner-up + confidence). Archetypes: puzzle-solver, analytical-serious,
creative-experimenter, determined-grinder, casual-gamer, booster-reliant,
at-risk-drifter, steady-improver.

Dimensions map onto FELT/persona so the whole system speaks one vocabulary:
  persistence, mastery, method, exploration, independence(DATA GAP), rhythm.
The LLM later only NARRATES this; it cannot change the label or the scores.
"""
from __future__ import annotations

from . import config as C

P = C.PERSONALITY


def _clamp(x: float) -> int:
    return int(max(0, min(100, round(x))))


def dimensions(window: list[dict]) -> dict:
    """window = this player's session rows within P.window_days (each has retries/fails/
    wins/level_reached/levels_played/duration_s/session_date/exit_mood)."""
    if not window:
        return {}
    n = len(window)
    retries = sum(s["retries"] for s in window)
    fails = sum(s["fails"] for s in window)
    wins = sum(s["wins"] for s in window)
    attempts = retries + fails + wins or 1
    levels = [s["level_reached"] for s in window if s["level_reached"]]
    distinct_levels = len(set(levels))
    all_levels_played = [lv for s in window for lv in (s.get("levels_played") or [])]
    revisits = len(all_levels_played) - len(set(all_levels_played))
    active_days = len({s["session_date"] for s in window if s["session_date"]})
    sessions_per_day = n / max(active_days, 1)
    quits = sum(1 for s in window if s["exit_mood"] in ("frustrated", "churn-risk", "interrupted"))
    avg_dur_min = sum((s["duration_s"] or 0) for s in window) / n / 60.0
    fail_rate = fails / attempts
    retries_per_level = retries / max(distinct_levels, 1)

    # persistence: retries endured relative to quitting; more retries with fewer quits = grind
    persistence = _clamp(100 * (retries / (retries + quits + 1)))
    # mastery: high when progressing many levels with few retries
    mastery = _clamp(100 * (distinct_levels / (distinct_levels + retries_per_level + 1)))
    # method/caution: proxy = low fail rate (TRUE method needs aim-time/deliberation events)
    method = _clamp(100 * (1 - fail_rate))
    # exploration: variety + revisiting levels (creative replay)
    exploration = _clamp(100 * ((distinct_levels + revisits) / (distinct_levels + revisits + 3)))
    # rhythm: 100 = binge, 0 = grazer(casual)
    rhythm = _clamp(100 * (sessions_per_day / (sessions_per_day + 2)))
    # independence: DATA GAP (no booster events) -> None
    independence = None

    return {
        "persistence": persistence, "mastery": mastery, "method": method,
        "exploration": exploration, "rhythm": rhythm, "independence": independence,
        "_meta": {"sessions": n, "avg_session_min": round(avg_dur_min, 1),
                  "fail_rate": round(fail_rate, 2), "distinct_levels": distinct_levels,
                  "sessions_per_day": round(sessions_per_day, 2)},
    }


def _score_archetypes(d: dict) -> list[tuple[str, float, str]]:
    """Return (label, fit_score, evidence) sorted best-first. Fit is a simple additive match."""
    hi, mid = P.high, P.mid
    m = d["_meta"]
    cands: list[tuple[str, float, str]] = []

    def add(label, score, ev):
        cands.append((label, score, ev))

    add("puzzle-solver", (d["mastery"] >= hi) * 2 + (d["method"] >= mid),
        f"mastery={d['mastery']}, method={d['method']}")
    add("analytical-serious", (d["method"] >= hi) + (d["persistence"] >= mid) + (m["fail_rate"] <= 0.2),
        f"method={d['method']}, persistence={d['persistence']}, fail_rate={m['fail_rate']}")
    add("creative-experimenter", (d["exploration"] >= hi) * 2,
        f"exploration={d['exploration']} (variety/revisits)")
    add("determined-grinder", (d["persistence"] >= hi) + (d["mastery"] < mid),
        f"persistence={d['persistence']}, mastery={d['mastery']}")
    add("casual-gamer", (d["rhythm"] < mid) + (m["avg_session_min"] <= 4) + (d["persistence"] < mid),
        f"grazer rhythm={d['rhythm']}, avg_session={m['avg_session_min']}min")
    # low-engagement drift: very few, very short sessions in the window (or lapsing, added by caller)
    add("at-risk-drifter", (m["sessions"] <= 2) + (m["avg_session_min"] <= 1.5),
        f"sessions={m['sessions']}, avg_session={m['avg_session_min']}min")
    add("steady-improver", 0.5, "default when no dimension dominates")  # baseline
    cands.sort(key=lambda c: c[1], reverse=True)
    return cands


def personality(window: list[dict], lapsing: bool = False) -> dict:
    d = dimensions(window)
    if not d:
        return {"personality": "unknown", "confidence": "low", "spectrum": {},
                "evidence": ["no sessions in window"]}
    cands = _score_archetypes(d)
    if lapsing:
        cands = [("at-risk-drifter", 99, "lapsing: no return within cadence")] + cands
    best, runner = cands[0], (cands[1] if len(cands) > 1 else None)
    n = d["_meta"]["sessions"]
    conf = "low" if n < P.min_sessions_for_confidence else "high"
    spectrum = {k: v for k, v in d.items() if k != "_meta"}
    return {
        "personality": best[0],
        "personality_runner_up": runner[0] if runner else None,
        "confidence": conf,
        "spectrum": spectrum,           # six dimensions (independence=None = data gap)
        "meta": d["_meta"],
        "evidence": [best[2]] + ([runner[2]] if runner else [])
                    + (["independence dimension = DATA GAP (no booster events)"]),
    }
