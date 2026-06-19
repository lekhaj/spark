"""U4-BE2 — pure unit tests for the compile agent. The LLM seam is injected, so
these run with no Bedrock/DB (mirrors test_compile_tools.py)."""
from app.cyclezero import compile_agent as ca


def _ent(layer, key, **data):
    return {"layer": layer, "key": key, "name": key.title(), "data": data,
            "accepted_spec_run_id": None}


def _rel(kind, src, dst):
    return {"kind": kind, "src": src, "dst": dst, "data": None}


# A tiny non-Pokémon (Diablo-ish) slice: scene + a custom 'monster' + 'loot_table'.
def _graph():
    entities = [
        _ent("scene", "blood_moor"),
        _ent("character", "hero", role="player"),
        _ent("monster", "fallen", hp=12),       # custom layer → a gap
        _ent("loot_table", "fallen_drops"),      # custom layer → a gap
    ]
    relations = [
        _rel("CONTAINS", "blood_moor", "hero"),
        _rel("CONTAINS", "blood_moor", "fallen"),
    ]
    # minimal metamodel: CONTAINS known, layers registered enough for validate_graph
    mm = {"layers": {}, "relation_types": {"CONTAINS": {"src_layers": [], "dst_layers": []}}}
    return entities, relations, mm


def test_compile_deterministic_only_renders_skeleton():
    entities, relations, mm = _graph()
    out = ca.compile_prompt(entities, relations, mm, {}, stitch_fn=None)
    assert "Build Packet" in out["prompt"]
    assert out["stitched"] is False
    # monster + loot_table are not engine-consumed → gaps
    assert "monster" in out["gaps"] and "loot_table" in out["gaps"]
    assert out["needs_llm_stitch"] is True
    assert out["gate"]["ok"] is True


def test_compile_with_injected_stitch_appends_plan():
    entities, relations, mm = _graph()
    calls = {}

    def fake_stitch(sections):
        calls["sections"] = sections
        return "1. Build the monster system.\n2. Wire loot_table on death."

    out = ca.compile_prompt(entities, relations, mm, {}, stitch_fn=fake_stitch)
    assert out["stitched"] is True
    assert "## Implementation plan" in out["prompt"]
    assert "Build the monster system" in out["prompt"]
    # the stitch saw the assembled sections, not raw graph
    assert "gaps" in calls["sections"]


def test_compile_stitch_failure_degrades_gracefully():
    entities, relations, mm = _graph()

    def boom(_):
        raise RuntimeError("bedrock down")

    out = ca.compile_prompt(entities, relations, mm, {}, stitch_fn=boom)
    assert out["stitched"] is False
    assert "Build Packet" in out["prompt"]          # still usable
    assert "LLM stitch unavailable" in out["prompt"]


def test_compile_scoped_to_scene():
    entities, relations, mm = _graph()
    out = ca.compile_prompt(
        entities, relations, mm, {},
        scope={"kind": "scene", "key": "blood_moor"}, stitch_fn=None,
    )
    # loot_table isn't CONTAINS-linked to the scene → out of scope, not a gap
    assert "loot_table" not in out["gaps"]
    assert "monster" in out["gaps"]
    assert out["scope"]["kind"] == "scene"


def test_compile_no_gaps_no_stitch_needed():
    entities = [_ent("scene", "s"), _ent("prop", "p", glb="p.glb")]
    out = ca.compile_prompt(entities, [], {"layers": {}, "relation_types": {}}, {},
                            stitch_fn=lambda s: "should not be called")
    assert out["needs_llm_stitch"] is False
    assert out["stitched"] is False
