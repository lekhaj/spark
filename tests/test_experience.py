"""Tests for the deterministic structural experience scorer (Critic, Step 1).

Pure logic — no Mongo, no SQL, no LLM. Each test crafts a small graph and asserts an
axis behaves and the right pitfall fires. Mirrors the determinism contract: the SCORE is
a function of the graph alone.
"""
from __future__ import annotations

from app.cyclezero import experience


def _ent(layer, key, name=None, **data):
    return {"layer": layer, "key": key, "name": name or key, "data": data}


def _rel(src, kind, dst):
    return {"src": src, "dst": dst, "kind": kind}


def test_empty_graph_is_all_low_and_flags_empty_sandbox():
    sc = experience.score_structural([], [])
    assert sc.headline <= 20
    assert sc.axes["choice"].score == 0
    assert "EmptySandbox" in sc.pitfalls
    # never crashes, always produces a suggestion
    assert sc.suggestion


def test_choice_rewards_real_decision_points():
    # two abilities both feed one defense factor → a convergent decision point
    ents = [_ent("system", "power_attack"), _ent("system", "block"),
            _ent("factor", "defense")]
    rels = [_rel("power_attack", "AFFECTS", "defense"),
            _rel("block", "AFFECTS", "defense")]
    sc = experience.score_structural(ents, rels)
    assert sc.axes["choice"].score >= 60
    assert "SolvedGame" not in sc.axes["choice"].pitfalls


def test_choice_flags_solved_game_when_no_convergence():
    # a single verb hitting a single outcome → no competing choice
    ents = [_ent("system", "shoot"), _ent("outcome", "hit")]
    rels = [_rel("shoot", "AFFECTS", "hit")]
    sc = experience.score_structural(ents, rels)
    assert "SolvedGame" in sc.axes["choice"].pitfalls


def test_feel_flags_dead_verbs():
    ents = [_ent("system", "wave"), _ent("system", "shoot"), _ent("outcome", "hit")]
    rels = [_rel("shoot", "AFFECTS", "hit")]  # 'wave' produces nothing
    sc = experience.score_structural(ents, rels)
    assert "VerbWithoutConsequence" in sc.axes["feel"].pitfalls
    assert sc.axes["feel"].score < 100


def test_autonomy_railroad_detected():
    # a strict corridor: each scene has exactly one forward path
    ents = [_ent("scene", "s1"), _ent("scene", "s2"), _ent("scene", "s3")]
    rels = [_rel("s1", "LEADS_TO", "s2"), _rel("s2", "LEADS_TO", "s3")]
    sc = experience.score_structural(ents, rels)
    assert "Railroad" in sc.axes["autonomy"].pitfalls
    assert sc.axes["autonomy"].score <= 20


def test_autonomy_high_when_branching():
    ents = [_ent("scene", "hub"), _ent("scene", "a"), _ent("scene", "b")]
    rels = [_rel("hub", "LEADS_TO", "a"), _rel("hub", "LEADS_TO", "b")]
    sc = experience.score_structural(ents, rels)
    assert "Railroad" not in sc.axes["autonomy"].pitfalls
    assert sc.axes["autonomy"].score >= 80


def test_tension_no_stakes_then_comeback():
    # no failure outcome → NoStakes
    base = [_ent("system", "attack"), _ent("outcome", "win")]
    sc = experience.score_structural(base, [_rel("attack", "REWARDS", "win")])
    assert "NoStakes" in sc.axes["tension"].pitfalls

    # add a death sink with no way back → NoComeback
    ents = base + [_ent("outcome", "death")]
    sc2 = experience.score_structural(ents, [_rel("attack", "REWARDS", "win")])
    assert "NoComeback" in sc2.axes["tension"].pitfalls

    # give death a forward edge → stakes + hope, pitfall clears
    rels3 = [_rel("attack", "REWARDS", "win"), _rel("death", "LEADS_TO", "win")]
    sc3 = experience.score_structural(ents, rels3)
    assert not sc3.axes["tension"].pitfalls
    assert sc3.axes["tension"].score >= 80


def test_discovery_flags_all_authored_vs_emergent():
    # all 1-hop authored: system→outcome only
    flat = [_ent("system", "atk"), _ent("outcome", "dmg")]
    sc = experience.score_structural(flat, [_rel("atk", "AFFECTS", "dmg")])
    assert "AllAuthored" in sc.axes["discovery"].pitfalls

    # chained: fire affects heat (factor), heat affects spread (outcome) → emergent
    ents = [_ent("system", "fire"), _ent("factor", "heat"), _ent("outcome", "spread")]
    rels = [_rel("fire", "AFFECTS", "heat"), _rel("heat", "AFFECTS", "spread")]
    sc2 = experience.score_structural(ents, rels)
    assert "AllAuthored" not in sc2.axes["discovery"].pitfalls
    assert sc2.axes["discovery"].score > 10


def test_weakest_axis_drives_suggestion_and_serialization():
    ents = [_ent("system", "a"), _ent("system", "b"), _ent("factor", "f")]
    rels = [_rel("a", "AFFECTS", "f"), _rel("b", "AFFECTS", "f")]
    sc = experience.score_structural(ents, rels)
    d = sc.as_dict()
    assert d["weakest"] in d["axes"]
    assert d["suggestion"]
    # every axis serializes with score/evidence/pitfalls
    for ax in d["axes"].values():
        assert set(ax) == {"score", "evidence", "pitfalls"}
        assert 0 <= ax["score"] <= 100
