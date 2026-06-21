"""Tests for the read-only Experience/Critic agent (Step 2).

Covers: deterministic fallback review (no provider), LLM-narrated review (fake provider,
numbers unchanged), the saved scorecard chip, and that a provider blowing up still yields
a review (chat never breaks).
"""
from __future__ import annotations

from app.cyclezero.agents import critic


def _ent(layer, key, name=None, **data):
    return {"layer": layer, "key": key, "name": name or key, "data": data}


def _rel(src, kind, dst):
    return {"src": src, "dst": dst, "kind": kind}


class _FakeProvider:
    def __init__(self, text="Looks promising. CHOICE is your weak spot."):
        self.text = text
        self.calls = []

    def chat_tools(self, system, messages, tools, tool_choice="auto"):
        self.calls.append((system, messages, tools))
        return {"text": self.text, "tool_calls": []}


class _BoomProvider:
    def chat_tools(self, *a, **k):
        raise RuntimeError("bedrock down")


_GOOD_GRAPH = (
    [_ent("system", "power_attack"), _ent("system", "block"), _ent("factor", "defense")],
    [_rel("power_attack", "AFFECTS", "defense"), _rel("block", "AFFECTS", "defense")],
)


def test_review_without_provider_is_deterministic_and_cites_numbers():
    ents, rels = _GOOD_GRAPH
    out = critic.review(entities=ents, relations=rels, facts={"genre": "arpg"})
    assert out["reply"]
    assert "/100" in out["reply"]              # cites the headline
    sc = out["scorecard"]
    assert 0 <= sc["headline"] <= 100
    assert sc["weakest"] in sc["axes"]
    # the saved chip carries the headline + per-axis scores for the UI
    chip = out["saved"][0]
    assert chip["kind"] == "scorecard"
    assert set(chip["axes"]) == {"choice", "mastery", "autonomy", "feel",
                                 "tension", "immersion", "discovery"}


def test_review_uses_provider_prose_when_available():
    ents, rels = _GOOD_GRAPH
    fp = _FakeProvider()
    out = critic.review(entities=ents, relations=rels, provider=fp)
    assert out["reply"] == fp.text
    assert fp.calls, "provider should have been called"
    # the deterministic scorecard is still attached untouched
    assert 0 <= out["scorecard"]["headline"] <= 100


def test_review_falls_back_when_provider_errors():
    ents, rels = _GOOD_GRAPH
    out = critic.review(entities=ents, relations=rels, provider=_BoomProvider())
    assert "/100" in out["reply"]              # fell back to deterministic prose


def test_is_review_intent():
    assert critic.is_review_intent("is this fun yet?")
    assert critic.is_review_intent("Give me a review")
    assert critic.is_review_intent("score my game")
    assert not critic.is_review_intent("add a stamina system")
    assert not critic.is_review_intent("create a sword item")
