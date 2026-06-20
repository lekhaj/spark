"""
usage_recorder.py — record + price external LLM (Bedrock) token usage
=====================================================================

ONE rule this module enforces: we only ever track **external, billed** LLM usage
here — i.e. Anthropic models served via **AWS Bedrock = AWS credits** (Haiku 4.5
today). Internal generation (Flux / 3D / rigging) runs on *our own GPU*, costs no
external dollars, and is tracked separately by counting ``manual_gen_stage_runs``
(see usage_routes.gpu_usage) — never here, never dollar-costed.

Capture point: ``refiner_providers`` calls ``record(...)`` after every Bedrock
Converse / Messages call, reading the current attribution context (who / which
game / which agent) from a contextvar the HTTP route sets. Capture is strictly
best-effort: it must never raise into, or slow, the request path.

Pricing is per-million-tokens and **configurable via env** so a rate is never
silently wrong. Defaults below are Claude Haiku 4.5 list prices — confirm against
the AWS Bedrock pricing page before trusting the dollar figures.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.services.mongo_service import get_db

log = logging.getLogger("usage_recorder")

_COLLECTION = "llm_usage"


# ── Pricing ($ per 1,000,000 tokens), env-overridable ────────────────────────
# CONFIRMED 2026-06-20 (Anthropic price list): Haiku 4.5 = $1.00 input / $5.00
# output per MTok; cache read = 0.1x input ($0.10), cache write (5-min) = 1.25x
# input ($1.25). The Bedrock *runtime* API does NOT return $ rates — only token
# counts — so rates live here. For the actual credit-spend AWS billed (after
# credits, ~24h delayed, aggregate), reconcile via Cost Explorer:
# usage_routes.aws_cost (ce:GetCostAndUsage, service "Amazon Bedrock").
def _price(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    cache_write: float


# Keyed by a normalized model family. Bedrock ids look like
# "global.anthropic.claude-haiku-4-5-20251001-v1:0" → family "haiku-4-5".
PRICING: Dict[str, ModelPrice] = {
    "haiku-4-5": ModelPrice(
        input=_price("PRICE_HAIKU45_INPUT", 1.00),
        output=_price("PRICE_HAIKU45_OUTPUT", 5.00),
        cache_read=_price("PRICE_HAIKU45_CACHE_READ", 0.10),
        cache_write=_price("PRICE_HAIKU45_CACHE_WRITE", 1.25),
    ),
    # Add other families here ONLY if a tier is lifted off Haiku later.
    "sonnet-4-6": ModelPrice(
        input=_price("PRICE_SONNET46_INPUT", 3.00),
        output=_price("PRICE_SONNET46_OUTPUT", 15.00),
        cache_read=_price("PRICE_SONNET46_CACHE_READ", 0.30),
        cache_write=_price("PRICE_SONNET46_CACHE_WRITE", 3.75),
    ),
}
_DEFAULT_FAMILY = "haiku-4-5"


def model_family(model_id: str) -> str:
    """Collapse a Bedrock/Anthropic model id to a pricing family key."""
    m = (model_id or "").lower()
    if "haiku-4-5" in m or "haiku-4.5" in m:
        return "haiku-4-5"
    if "sonnet-4-6" in m or "sonnet-4.6" in m or "sonnet-4-5" in m:
        return "sonnet-4-6"
    return _DEFAULT_FAMILY


def cost_usd(family: str, input_tokens: int, output_tokens: int,
             cache_read: int = 0, cache_write: int = 0) -> float:
    """Compute dollar cost from token counts. Cache reads/writes are billed at
    their own rates and are NOT double-counted in input_tokens (Bedrock reports
    them as distinct fields)."""
    p = PRICING.get(family) or PRICING[_DEFAULT_FAMILY]
    return round(
        (input_tokens * p.input
         + output_tokens * p.output
         + cache_read * p.cache_read
         + cache_write * p.cache_write) / 1_000_000.0,
        6,
    )


# ── Attribution context (set by the HTTP route, read by the provider) ─────────
@dataclass
class Attribution:
    uid: Optional[str] = None
    email: Optional[str] = None
    game_slug: Optional[str] = None
    agent: str = "unknown"          # refiner | compile | validate | propose
    meta: Dict[str, Any] = field(default_factory=dict)


_ctx: "contextvars.ContextVar[Attribution]" = contextvars.ContextVar(
    "llm_attribution", default=Attribution()
)


@contextlib.contextmanager
def attribution(**kwargs):
    """Set the attribution for any Bedrock calls made inside this block.

        with usage_recorder.attribution(email=u.email, game_slug=slug, agent="compile"):
            stitch_fn(sections)
    """
    base = _ctx.get()
    merged = Attribution(
        uid=kwargs.get("uid", base.uid),
        email=kwargs.get("email", base.email),
        game_slug=kwargs.get("game_slug", base.game_slug),
        agent=kwargs.get("agent", base.agent),
        meta={**base.meta, **(kwargs.get("meta") or {})},
    )
    token = _ctx.set(merged)
    try:
        yield merged
    finally:
        _ctx.reset(token)


def current() -> Attribution:
    return _ctx.get()


# ── Recording ────────────────────────────────────────────────────────────────
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0


def record(model_id: str, tier: Optional[str], usage: Usage) -> None:
    """Insert one priced usage row. Best-effort: swallows every error so a slow
    or down Mongo can never break an LLM response."""
    try:
        fam = model_family(model_id)
        attr = current()
        doc = {
            "ts": datetime.now(timezone.utc),
            "source": "bedrock",          # external / AWS credits (vs internal GPU)
            "model_id": model_id,
            "model_family": fam,
            "tier": tier,
            "agent": attr.agent,
            "uid": attr.uid,
            "email": (attr.email or "").lower() or None,
            "game_slug": attr.game_slug,
            "input_tokens": int(usage.input_tokens or 0),
            "output_tokens": int(usage.output_tokens or 0),
            "cache_read_tokens": int(usage.cache_read or 0),
            "cache_write_tokens": int(usage.cache_write or 0),
            "total_tokens": int((usage.input_tokens or 0) + (usage.output_tokens or 0)),
            "cost_usd": cost_usd(fam, usage.input_tokens, usage.output_tokens,
                                 usage.cache_read, usage.cache_write),
        }
        db = get_db()
        if db is None:
            log.warning("usage record skipped — no Mongo")
            return
        db[_COLLECTION].insert_one(doc)
    except Exception as exc:  # noqa: BLE001 — recording must never break a request
        log.warning("usage record failed: %s", exc)
