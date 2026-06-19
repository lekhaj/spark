"""U5 — pure tests for the validate agent (static gate + injected semantic seam)."""
from app.cyclezero import validate_agent as va


def _ent(layer, key, **data):
    return {"layer": layer, "key": key, "name": key.title(), "data": data,
            "accepted_spec_run_id": None}


def test_static_passes_clean_graph():
    entities = [_ent("scene", "s"), _ent("character", "hero", role="player")]
    out = va.validate(entities, [], {"layers": {}, "relation_types": {}})
    assert out["verdict"] == "pass"
    assert out["static"]["ok"] is True
    assert out["fix_packet"] == ""


def test_static_fails_on_illegal_edge():
    entities = [_ent("scene", "s")]
    relations = [{"kind": "BOGUS", "src": "s", "dst": "missing", "data": None}]
    out = va.validate(entities, relations, {"layers": {}, "relation_types": {}})
    assert out["verdict"] == "fail"
    assert any("illegal edge" in i or "BOGUS" in i for i in out["static"]["issues"])
    assert "Fix Packet" in out["fix_packet"]


def test_semantic_fail_overrides_pass():
    entities = [_ent("scene", "s")]
    out = va.validate(
        entities, [], {"layers": {}, "relation_types": {}},
        acceptance=["player can fly"], done_note="added swimming",
        semantic_fn=lambda acc, note: "- player can fly: FAIL no flight\nVERDICT: fail",
    )
    assert out["verdict"] == "fail"
    assert "Fix Packet" in out["fix_packet"]
    assert "semantic" in out["fix_packet"].lower()


def test_semantic_pass_keeps_pass():
    entities = [_ent("scene", "s")]
    out = va.validate(
        entities, [], {"layers": {}, "relation_types": {}},
        acceptance=["scene exists"], done_note="built the scene",
        semantic_fn=lambda acc, note: "- scene exists: PASS\nVERDICT: pass",
    )
    assert out["verdict"] == "pass"


def test_semantic_skipped_without_done_note():
    entities = [_ent("scene", "s")]
    called = {"n": 0}

    def fn(a, n):
        called["n"] += 1
        return "VERDICT: fail"

    out = va.validate(entities, [], {"layers": {}, "relation_types": {}},
                      acceptance=["x"], semantic_fn=fn)  # no done_note
    assert called["n"] == 0
    assert out["verdict"] == "pass"
