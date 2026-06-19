"""U4-BE1 — pure unit tests for the deterministic compile tool layer.

No DB / no LLM: ``compile_tools`` is pure logic over dicts, so these tests need
nothing but the module (mirrors the pure half of ``test_outcome.py``).
"""
from app.cyclezero import compile_tools as ct


def _ent(layer, key, **data):
    return {"layer": layer, "key": key, "name": key.title(), "data": data}


def _rel(kind, src, dst, data=None):
    return {"kind": kind, "src": src, "dst": dst, "data": data}


def _graph():
    entities = [
        _ent("scene", "dawnvale"),
        _ent("character", "hero", role="player"),
        _ent("prop", "fountain", glb="fountain.glb"),
        _ent("npc", "elder"),
        _ent("system", "weather"),
    ]
    relations = [
        _rel("CONTAINS", "dawnvale", "hero"),
        _rel("CONTAINS", "dawnvale", "fountain"),
        _rel("CONTAINS", "dawnvale", "elder"),
    ]
    return entities, relations


# ── gather_scope ──────────────────────────────────────────────────────────────
def test_gather_scope_game_is_everything():
    entities, relations = _graph()
    b = ct.gather_scope(entities, relations, {"kind": "game"})
    assert b["counts"]["entities"] == 5
    assert b["counts"]["relations"] == 3
    assert "npc" in b["layers"] and "scene" in b["layers"]


def test_gather_scope_scene_pulls_contained_members():
    entities, relations = _graph()
    b = ct.gather_scope(entities, relations, {"kind": "scene", "key": "dawnvale"})
    keys = {e["key"] for e in b["entities"]}
    assert keys == {"dawnvale", "hero", "fountain", "elder"}
    # weather (uncontained system) excluded
    assert "weather" not in keys


def test_gather_scope_relations_closed_under_keys():
    entities, relations = _graph()
    # only hero in scope → CONTAINS edges (dawnvale->hero) excluded since src out
    b = ct.gather_scope(entities, relations, {"kind": "entities", "keys": ["hero"]})
    assert b["counts"]["entities"] == 1
    assert b["counts"]["relations"] == 0


def test_gather_scope_defaults_to_game():
    entities, relations = _graph()
    assert ct.gather_scope(entities, relations)["counts"]["entities"] == 5


# ── capability registry + diff ────────────────────────────────────────────────
def test_registry_known_and_unknown_engine():
    assert "IsoCamera" in ct.get_capability_registry("babylon")["systems"]
    unk = ct.get_capability_registry("unreal")
    assert unk["unknown_engine"] is True and unk["consumes"] == []


def test_diff_capabilities_splits_provided_and_gaps():
    reg = ct.get_capability_registry("babylon")
    d = ct.diff_capabilities(["scene", "character", "npc", "system"], reg)
    assert "scene" in d["provided"] and "character" in d["provided"]
    assert set(d["gaps"]) == {"npc", "system"}
    assert d["fully_covered"] is False


def test_diff_capabilities_fully_covered():
    reg = ct.get_capability_registry("babylon")
    d = ct.diff_capabilities(["scene", "prop"], reg)
    assert d["fully_covered"] is True and d["gaps"] == []


# ── assemble_prompt_skeleton ──────────────────────────────────────────────────
def test_assemble_skeleton_renders_and_flags_stitch():
    entities, relations = _graph()
    bundle = ct.gather_scope(entities, relations)
    reg = ct.get_capability_registry("babylon")
    diff = ct.diff_capabilities(bundle["layers"], reg)
    schemas = {"npc": {"type": "object", "properties": {"dialogue": {"type": "string"}}}}
    out = ct.assemble_prompt_skeleton(bundle, schemas, reg, diff, acceptance=["talks to elder"])
    assert "Build Packet" in out["rendered"]
    assert "DO NOT re-implement" in out["rendered"]
    # npc + system are gaps → needs a stitch for the code-gen output
    assert out["needs_llm_stitch"] is True
    assert "npc" in out["sections"]["gaps"]
    assert "talks to elder" in out["rendered"]


def test_assemble_skeleton_no_gaps_no_stitch():
    entities = [_ent("scene", "s"), _ent("prop", "p", glb="p.glb")]
    bundle = ct.gather_scope(entities, [])
    reg = ct.get_capability_registry("babylon")
    diff = ct.diff_capabilities(bundle["layers"], reg)
    out = ct.assemble_prompt_skeleton(bundle, {}, reg, diff)
    assert out["needs_llm_stitch"] is False


# ── lint_schema ───────────────────────────────────────────────────────────────
def test_lint_schema_ok():
    s = {"type": "object", "properties": {"hp": {"type": "integer"}}, "required": ["hp"]}
    assert ct.lint_schema(s) == {"ok": True, "errors": []}


def test_lint_schema_custom_keywords_allowed():
    s = {"type": "object", "properties": {"home": {"x-ref": "town"}}}
    assert ct.lint_schema(s)["ok"] is True


def test_lint_schema_catches_problems():
    r = ct.lint_schema({"type": "widget", "properties": {"a": {"type": "blob"}},
                        "required": ["missing"]})
    assert r["ok"] is False
    assert any("unsupported type 'widget'" in e for e in r["errors"])
    assert any("required field 'missing'" in e for e in r["errors"])


def test_lint_schema_non_dict():
    assert ct.lint_schema("nope")["ok"] is False


# ── lint_rule ─────────────────────────────────────────────────────────────────
def test_lint_rule_ok():
    rule = {"name": "win", "when": [{"factor": "hope", "op": ">=", "value": 10}],
            "then": [{"effect": "transition", "target": "ending_good"}], "priority": 5}
    assert ct.lint_rule(rule) == {"ok": True, "errors": []}


def test_lint_rule_catches_problems():
    r = ct.lint_rule({"when": [{"factor": "x", "op": "≈", "value": 1}],
                      "then": [{"effect": "explode", "target": "y"}],
                      "priority": "high"})
    assert r["ok"] is False
    assert any("missing 'name'" in e for e in r["errors"])
    assert any("op '≈'" in e for e in r["errors"])
    assert any("effect 'explode'" in e for e in r["errors"])
    assert any("'priority' must be numeric" in e for e in r["errors"])
