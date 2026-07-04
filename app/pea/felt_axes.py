"""
FELT axes — the telemetry (FELT) half of the Critic's 7-axis scorecard.

The Critic scores a game's *potential* structurally (app/cyclezero/experience.py). This
module scores what players *actually felt*, on the SAME axes, from the pea_* tables — so
the studio can see GAP = |structural - felt|, the most valuable output per the FELT plan.

Deterministic (no LLM). Only the axes telemetry can honestly measure are returned; the
rest are reported as data_gaps (never faked). Aggregate-only — no distinct_ids — so it is
safe to surface in the per-slug Experience view, not just owner analytics.
"""
from __future__ import annotations

from . import config as C
from . import store

# weight each FELT label by "how well this axis was delivered" (0-100)
_TENSION_W = {"earned-relief": 90, "frustrated": 55, "flat": 40, "defeated": 25}
_MASTERY_W = {"climbing": 85, "plateaued": 55, "struggling": 30}


def _wavg(counts: dict, weights: dict):
    total = sum(counts.values())
    if not total:
        return None, 0
    return round(sum(weights.get(k, 50) * v for k, v in counts.items()) / total), total


def felt_axes(game_id: str = C.GAME_ID) -> dict:
    with store.connect() as cx, cx.cursor() as cur:
        cur.execute("SELECT felt_tension, count(*) FROM pea_session_state WHERE game_id=%s GROUP BY 1", (game_id,))
        tension = {k: v for k, v in cur.fetchall() if k}
        cur.execute("SELECT felt_mastery, count(*) FROM pea_session_state WHERE game_id=%s GROUP BY 1", (game_id,))
        mastery = {k: v for k, v in cur.fetchall() if k}
        cur.execute("SELECT count(*) FROM pea_session_state WHERE game_id=%s", (game_id,))
        n_sess = cur.fetchone()[0] or 0
        cur.execute("SELECT count(*) FROM pea_session_state WHERE game_id=%s AND exit_mood='interrupted'", (game_id,))
        interrupted = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM pea_raw_events WHERE game_id=%s AND event_name=%s", (game_id, C.E_CRASH))
        crashes = cur.fetchone()[0]

    axes: dict = {}
    ts, tn = _wavg(tension, _TENSION_W)
    if ts is not None:
        axes["tension"] = {"score": ts, "evidence": f"FELT tension mix {tension} over {tn} sessions"}
    ms, mn = _wavg(mastery, _MASTERY_W)
    if ms is not None:
        axes["mastery"] = {"score": ms, "evidence": f"FELT mastery mix {mastery} over {mn} sessions"}

    if n_sess:
        crash_rate = crashes / n_sess
        interrupted_share = interrupted / n_sess
        feel = max(0, min(100, round(100 - 120 * crash_rate - 60 * interrupted_share)))
        axes["feel"] = {"score": feel,
                        "evidence": f"{crashes} crashes + {interrupted} interrupted exits / {n_sess} sessions"}

    return {
        "game_id": game_id, "axes": axes, "sessions": n_sess,
        # honestly unmeasurable from today's event contract:
        "data_gaps": {
            "autonomy": "needs booster-decision events",
            "choice": "needs decision_point events",
            "immersion": "not a telemetry signal",
            "discovery": "needs level-variety / emergent-path events",
        },
        "confidence": "low" if n_sess < C.LOW_CONFIDENCE_USER_THRESHOLD else "high",
        "banner": C.PRELAUNCH_BANNER,
    }
