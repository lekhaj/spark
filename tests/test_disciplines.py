"""Tests for the discipline agents added on top of the shared substrate.

World + Narrative are thin "minds" (prompt + tool subset + intents) that route real turns
and write through the same deterministic gate as Systems. Plus the narrative-health check.
"""
from __future__ import annotations

import mongomock

from app.cyclezero import creator_agent, metamodel, spatial
from app.cyclezero.agents import director, narrative, registry, validator, world

# minimal metamodel for validator tests: AFFECTS must go system → factor
_VMM = {
    "layers": {"system": {"layer": "system"}, "factor": {"layer": "factor"}},
    "relation_types": {
        "AFFECTS": {"kind": "AFFECTS", "src_layers": ["system"], "dst_layers": ["factor"],
                    "src_cardinality": "many", "dst_cardinality": "many"},
    },
}


def test_disciplines_are_registered_and_route():
    names = {a.name for a in registry.AGENTS}
    assert {"systems", "world", "narrative", "art", "propose"} <= names

    ag, routed = registry.route("lay out the town scene and place the player spawn")
    assert ag.name == "world" and routed == "world"

    ag, routed = registry.route("the hero accepts the first quest from the village elder")
    assert ag.name == "narrative" and routed == "narrative"

    ag, routed = registry.route("generate the 3d model and texture for the barrel")
    assert ag.name == "art" and routed == "art"

    ag, routed = registry.route("I need a new layer for factions that doesn't exist yet")
    assert ag.name == "propose" and routed == "propose"

    # mechanics still win for Systems even with a character mentioned
    ag, routed = registry.route("stamina drains when the hero does a power attack")
    assert ag.name == "systems"


def test_agents_only_use_known_tools():
    # every agent's tool subset must resolve against the shared catalog
    catalog = {t["name"] for t in creator_agent.TOOLS}
    for ag in registry.AGENTS:
        assert set(ag.tool_names) <= catalog
        assert {t["name"] for t in ag.tools} == set(ag.tool_names)


def _ent(layer, key):
    return {"layer": layer, "key": key, "name": key, "data": {}}


def _rel(src, kind, dst):
    return {"src": src, "dst": dst, "kind": kind}


def test_narrative_health_flags_orphan_quests():
    ents = [_ent("quest", "intro"), _ent("quest", "rescue"), _ent("quest", "lost_side")]
    rels = [_rel("intro", "LEADS_TO", "rescue")]  # lost_side is connected to nothing
    h = narrative.narrative_health(ents, rels)
    assert h["quests"] == 3
    assert h["connected"] == 2
    assert h["orphans"] == ["lost_side"]


def test_narrative_health_empty():
    assert narrative.narrative_health([], []) == {"quests": 0, "connected": 0, "orphans": []}


# ── Validator (read-only correctness) ──────────────────────────────────────────
def test_validator_passes_clean_graph():
    ents = [_ent("system", "atk"), _ent("factor", "defense")]
    rels = [_rel("atk", "AFFECTS", "defense")]
    out = validator.validate(entities=ents, relations=rels, metamodel=_VMM, game={"slug": "g"})
    assert out["result"]["ok"] is True
    chip = out["saved"][0]
    assert chip["kind"] == "validation" and chip["ok"] is True
    assert chip["issues"] == [] and chip["spatial_warnings"] == []


def test_validator_flags_illegal_edge():
    # AFFECTS dst must be a factor; pointing at a system is illegal
    ents = [_ent("system", "atk"), _ent("system", "other")]
    rels = [_rel("atk", "AFFECTS", "other")]
    out = validator.validate(entities=ents, relations=rels, metamodel=_VMM, game={"slug": "g"})
    assert out["result"]["ok"] is False
    assert out["saved"][0]["ok"] is False
    assert out["result"]["issues"]


def test_is_validate_intent():
    assert validator.is_validate_intent("validate my game")
    assert validator.is_validate_intent("any errors in the graph?")
    assert not validator.is_validate_intent("add a stamina system")


# ── spatial contract (scale/placement never silently dropped) ──────────────────
def test_spatial_health_flags_missing_transform_and_dimensions():
    ents = [
        _ent_d("prop", "barrel", dimensions={"w": 0.6, "h": 0.9, "d": 0.6},
               transform={"position": [1, 0, 2], "scale": 1.0}),   # complete
        _ent_d("prop", "crate"),                                    # missing both
        _ent_d("character", "hero", transform={"position": [0, 0, 0]}),  # missing dims
    ]
    h = spatial.spatial_health(ents, [])
    assert h["placed"] == 3
    assert "crate" in h["missing_transform"]
    assert "barrel" not in h["missing_transform"]
    assert set(h["missing_dimensions"]) == {"crate", "hero"}
    assert h["issues"]


def test_spatial_transform_can_ride_on_contains_edge():
    ents = [_ent_d("scene", "town"),
            _ent_d("prop", "barrel", dimensions={"w": 1, "h": 1, "d": 1})]
    # the prop has no own transform, but the scene places it via a CONTAINS edge
    rels = [{"src": "town", "dst": "barrel", "kind": "CONTAINS",
             "data": {"transform": {"position": [3, 0, 3]}}}]
    h = spatial.spatial_health(ents, rels)
    assert h["missing_transform"] == []        # placement satisfied by the edge


def test_validator_surfaces_spatial_warnings():
    ents = [_ent_d("prop", "crate")]           # placed but no transform / dimensions
    out = validator.validate(entities=ents, relations=[], metamodel=_VMM, game={"slug": "g"})
    assert out["saved"][0]["spatial_warnings"]
    assert any("crate" in w for w in out["result"]["issues"])


def _ent_d(layer, key, **data):
    return {"layer": layer, "key": key, "name": key, "data": data}


# ── Propose (installs new vocabulary) ──────────────────────────────────────────
def test_propose_system_installs_layer_and_relation():
    mm_db = mongomock.MongoClient()["mm_test"]
    res = creator_agent.apply_tool_calls(
        sql_db=None, mongo_db=None, uid="u1", email="a@b.com", game_slug="g",
        tool_calls=[{"name": "propose_system", "input": {
            "layers": [{"layer": "faction", "title": "Faction", "schema": {"type": "object"}}],
            "relations": [{"kind": "ALLIES_WITH", "src_layers": ["faction"],
                           "dst_layers": ["faction"]}],
        }}],
        facts={}, open_questions=[], known_layers=["faction"], metamodel_db=mm_db,
    )
    kinds = {s["kind"] for s in res["saved"]}
    assert "layer_installed" in kinds and "relation_type_installed" in kinds
    # actually installed into the metamodel store
    assert metamodel.get_layer(mm_db, "faction") is not None
    assert metamodel.get_relation_type(mm_db, "ALLIES_WITH") is not None


def test_propose_system_rejected_without_metamodel_store():
    res = creator_agent.apply_tool_calls(
        sql_db=None, mongo_db=None, uid="u1", email="a@b.com", game_slug="g",
        tool_calls=[{"name": "propose_system", "input": {"layers": [{"layer": "x"}]}}],
        facts={}, open_questions=[], known_layers=[], metamodel_db=None,
    )
    # no store handed in and worker.lib unavailable offline → rejected, never crashes
    assert any(s["kind"] == "rejected" for s in res["saved"])


# ── Director (read-only "what's next") ─────────────────────────────────────────
def test_director_prioritises_playability_then_quality():
    # empty-ish game: no scene, no player → those lead the list
    ents = [_ent_d("system", "atk")]
    rep = director.progress(ents, [], metamodel={"relation_types": {}})
    assert rep["playable"] is False
    assert "scene" in rep["next_steps"][0].lower()
    assert any("player" in s.lower() for s in rep["next_steps"])
    # the quality step (weakest axis) is always present at the tail
    assert any("/100" in s for s in rep["next_steps"])


def test_director_flags_missing_required_relation():
    mm = {"layers": {}, "relation_types": {
        "OWNS": {"kind": "OWNS", "src_layers": ["character"], "dst_layers": ["item"],
                 "required": True}}}
    ents = [_ent_d("character", "hero", role="player"), _ent_d("scene", "town")]
    rep = director.progress(ents, [], metamodel=mm)
    assert any("OWNS on hero" in m for m in rep["coverage"]["missing_required_relations"])


def test_is_progress_intent():
    assert director.is_progress_intent("what's next?")
    assert director.is_progress_intent("show me the progress")
    assert not director.is_progress_intent("add a stamina system")


def test_director_reports_engine_gap_and_closes_on_ingest():
    # a custom 'faction' layer the babylon engine doesn't render yet → an engine gap
    ents = [_ent_d("faction", "rebels"), _ent_d("scene", "town"),
            _ent_d("character", "hero", role="player")]
    rep = director.progress(ents, [], metamodel={"relation_types": {}})
    assert "faction" in rep["coverage"]["engine_gaps"]
    assert any("can't render" in s for s in rep["next_steps"])

    # simulate Claude Code building faction support, ingested into the ledger → gap closes
    reg = {"engine": "babylon", "consumes": ["scene", "character", "faction"]}
    rep2 = director.progress(ents, [], metamodel={"relation_types": {}}, capabilities=reg)
    assert rep2["coverage"]["engine_gaps"] == []
    assert rep2["coverage"]["engine_ready"] is True
