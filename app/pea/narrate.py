"""
LLM narration — Level 3. STRICTLY grounded: the model may only restate the
deterministic labels + evidence it is given. It cannot introduce a feeling the
thresholds did not produce. If ANTHROPIC_API_KEY is unset, we fall back to a
template so the pipeline still runs offline.
"""
from __future__ import annotations

import json

from . import config as C

_SYSTEM = (
    "You are a game-analytics narrator. You will be given a player's DETERMINISTIC mood "
    "labels and the evidence rows that produced them. Write ONE plain-English sentence that "
    "restates what happened, citing the evidence. HARD RULES: (1) Never state a feeling that "
    "is not in the provided labels. (2) Never invent numbers; only use the evidence given. "
    "(3) If evidence is thin, say so. Output only the sentence."
)


def _fallback(row: dict) -> str:
    return (f"Started {row['entry_mood']}, reached level {row.get('level_reached')} "
            f"({row['retries']} retries, {row['fails']} fails, {row['wins']} wins), "
            f"left {row['exit_mood']}.")


def narrate_player(row: dict) -> str:
    if not C.ANTHROPIC_API_KEY:
        return _fallback(row)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)
        payload = {k: row[k] for k in ("entry_mood", "exit_mood", "overall_feeling",
                                       "persona", "level_reached", "retries", "fails",
                                       "wins", "sessions_today", "evidence")}
        msg = client.messages.create(
            model=C.NARRATION_MODEL, max_tokens=120,
            system=_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return _fallback(row)


def narrate_digest_insights(digest: dict) -> list[str]:
    """3-6 insights that narrate the digest numbers (each cites a metric). Deterministic
    fallback covers the offline / pre-launch case; all are prefixed with the QA banner."""
    lines: list[str] = []
    d = digest
    lines.append(f"[{d['banner']}] DAU={d['dau']} ({d['new_users']} new, {d['returning_users']} returning).")
    if d["exit_mood_dist"]:
        top = max(d["exit_mood_dist"], key=d["exit_mood_dist"].get)
        lines.append(f"Most common exit mood was '{top}' ({d['exit_mood_dist'][top]} players).")
    if d["watch_list"]:
        lines.append(f"{len(d['watch_list'])} player(s) flipped to frustrated/churn-risk today — see watch list.")
    if d["top_friction_levels"]:
        f = d["top_friction_levels"][0]
        lines.append(f"Level {f['level_id']} is the top friction point "
                     f"({f.get('frustrated_sessions',0)} frustrated, {f['retries']} retries).")
    if d["confidence"] == "low":
        lines.append(f"Confidence LOW: only {d['dau']} users (< {C.LOW_CONFIDENCE_USER_THRESHOLD}); distributions indicative only.")
    if not C.ANTHROPIC_API_KEY:
        return lines[:6]
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=C.NARRATION_MODEL, max_tokens=400,
            system=("Narrate these game daily-digest numbers as 3-6 short bullet insights. "
                    "Cite the metric each came from. Do not speculate beyond the data. "
                    "Keep the QA banner caveat."),
            messages=[{"role": "user", "content": json.dumps(d, default=str)}],
        )
        return [l.strip("-• ") for l in msg.content[0].text.splitlines() if l.strip()][:6]
    except Exception:
        return lines[:6]
