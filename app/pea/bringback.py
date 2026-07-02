"""
Bring-back list — decide WHO/WHEN/WHAT. No auto-send (app has no push channel yet):
we output a CSV the user can hand-edit before a manual send. Targeting lives in
config.BRINGBACK so it can later feed FCM automatically.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from . import config as C

BB = C.BRINGBACK


def _active_hour_local(player_sessions: list[dict]) -> int:
    hours = [s["started_at"].astimezone(C.TIMEZONE).hour
             for s in player_sessions if s.get("started_at")]
    if not hours:
        return BB.default_send_hour
    hour = max(set(hours), key=hours.count)  # modal active hour
    lo, hi = BB.quiet_hours
    if hour >= lo or hour < hi:  # inside quiet window -> push to default
        return BB.default_send_hour
    return hour


def _lapse_risk(exit_mood: str, days_since_last: int | None) -> float:
    base = {"churn-risk": 0.9, "frustrated": 0.7, "interrupted": 0.4,
            "ok-satisfied": 0.3, "happy": 0.15, "comeback-tomorrow": 0.2}.get(exit_mood, 0.4)
    if days_since_last and days_since_last >= BB.lapse_risk_days:
        base = min(1.0, base + 0.2)
    return round(base, 2)


def _message(persona: str, exit_mood: str) -> tuple[str, str]:
    for key in (exit_mood, persona.split("/")[0] if persona else "", "default"):
        if key in BB.templates:
            return BB.templates[key]
    return BB.templates["default"]


def build_bringback(player_rows: list[dict], sessions_by_player: dict[str, list[dict]],
                    date: dt.date) -> list[dict]:
    rows = []
    returned_today = {p["distinct_id"] for p in player_rows if p["date"] == date}
    for p in player_rows:
        if BB.suppress_if_returned_today and p["distinct_id"] in returned_today and p["date"] == date:
            # they're active today; only target lapsing candidates (handled by caller's date logic)
            pass
        exit_mood = p["exit_mood"]
        if exit_mood not in ("churn-risk", "frustrated", "comeback-tomorrow", "at-risk-returner"):
            continue
        sess = sessions_by_player.get(p["distinct_id"], [])
        msg, incentive = _message(p["persona"], exit_mood)
        rows.append({
            "game_id": C.GAME_ID, "date": date, "distinct_id": p["distinct_id"],
            "persona": p["persona"],
            "mood_history": [{"date": str(p["date"]), "entry": p["entry_mood"], "exit": exit_mood}],
            "lapse_risk": _lapse_risk(exit_mood, None),
            "recommended_send_hour_local": _active_hour_local(sess),
            "recommended_message": msg, "recommended_incentive": incentive,
            "included": True, "overridden": False,
        })
    rows.sort(key=lambda r: r["lapse_risk"], reverse=True)
    return rows


def export_csv(rows: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["distinct_id", "persona", "lapse_risk", "recommended_send_hour_local",
              "recommended_message", "recommended_incentive", "included"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if r.get("included", True):
                w.writerow(r)
    return path
