"""
LLM narration — Level 3. STRICTLY grounded: the model may only restate the
deterministic labels + evidence it is given; it can never introduce a feeling the
thresholds didn't produce.

Runs through Spark's Bedrock provider (Tier-A, AWS credits) — the same provider the
Critic uses — so narration is consistent + costs nothing extra. Falls back to a
deterministic template on ANY provider failure so the pipeline never breaks.
"""
from __future__ import annotations

import json

from . import config as C

_SYSTEM = (
    "You are a game-analytics narrator for a mini game studio. You are given a player's "
    "DETERMINISTIC mood/personality labels and the evidence rows that produced them. Write "
    "ONE plain, designer-useful sentence that restates what happened, citing the evidence. "
    "HARD RULES: (1) Never state a feeling not in the provided labels. (2) Never invent "
    "numbers; use only the evidence given. (3) If evidence is thin, say so. Output only the sentence."
)

_INSIGHTS_SYSTEM = (
    "You narrate a game's daily player-experience digest as 3-6 short, concrete bullet "
    "insights for the studio. Cite the metric each insight came from. Never speculate beyond "
    "the numbers. Keep the pre-launch/QA caveat if present. Output bullets only, one per line."
)

_provider = None


def _get_provider():
    """Cached Bedrock Tier-A provider; None if unavailable (offline/tests)."""
    global _provider
    if _provider is None:
        try:
            from app.lib import refiner_providers
            _provider = refiner_providers.get_tier_provider("A")
        except Exception:
            _provider = False  # sentinel: tried and failed
    return _provider or None


def _chat(system: str, user_payload) -> str | None:
    prov = _get_provider()
    if not prov:
        return None
    try:
        return prov.chat(system, [{"role": "user", "content": json.dumps(user_payload, default=str)}]).strip()
    except Exception:
        return None


def _fallback(row: dict) -> str:
    return (f"Started {row['entry_mood']}, reached level {row.get('level_reached')} "
            f"({row['retries']} retries, {row['fails']} fails, {row['wins']} wins), "
            f"left {row['exit_mood']} — personality {row.get('personality','?')}.")


def narrate_player(row: dict) -> str:
    payload = {k: row.get(k) for k in ("entry_mood", "exit_mood", "overall_feeling", "persona",
                                       "personality", "felt_tension", "level_reached", "retries",
                                       "fails", "wins", "sessions_today", "evidence")}
    return _chat(_SYSTEM, payload) or _fallback(row)


def narrate_digest_insights(digest: dict) -> list[str]:
    """Deterministic insights first (always present + cited); the LLM may rephrase/extend them."""
    d = digest
    lines: list[str] = [
        f"[{d['banner']}] DAU={d['dau']} ({d['new_users']} new, {d['returning_users']} returning)."]
    if d["exit_mood_dist"]:
        top = max(d["exit_mood_dist"], key=d["exit_mood_dist"].get)
        lines.append(f"Most players ended '{top}' ({d['exit_mood_dist'][top]}); full exit mix: {d['exit_mood_dist']}.")
    if d.get("personality_dist"):
        lines.append(f"Personality mix today: {d['personality_dist']}.")
    if d["watch_list"]:
        lines.append(f"{len(d['watch_list'])} player(s) flipped to frustrated/churn-risk — see watch list.")
    if d["top_friction_levels"]:
        f = d["top_friction_levels"][0]
        lines.append(f"Level {f['level_id']} is the top friction point "
                     f"({f.get('frustrated_sessions',0)} frustrated exits, {f.get('retries',0)} retries).")
    if d["confidence"] == "low":
        lines.append(f"Confidence LOW: only {d['dau']} users (< {C.LOW_CONFIDENCE_USER_THRESHOLD}); treat distributions as indicative.")

    llm = _chat(_INSIGHTS_SYSTEM, d)
    if llm:
        bullets = [l.strip("-•* ").strip() for l in llm.splitlines() if l.strip()]
        if bullets:
            return bullets[:6]
    return lines[:6]
