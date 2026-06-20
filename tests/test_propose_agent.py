"""U6 — pure tests for the system-proposal agent (injected LLM seam, no Bedrock)."""
import json

from app.cyclezero import propose_agent as pa


_GOOD = {
    "layers": [
        {"layer": "monster", "title": "Monster",
         "schema": {"type": "object", "properties": {"hp": {"type": "integer"}}, "required": ["hp"]}},
        {"layer": "loot_table", "title": "Loot Table",
         "schema": {"type": "object", "properties": {"drops": {"type": "array"}}}},
    ],
    "relations": [
        {"kind": "DROPS", "src_layers": ["monster"], "dst_layers": ["loot_table"]},
    ],
    "notes": "ARPG slice",
}


def test_propose_parses_and_validates():
    out = pa.propose_systems("a diablo-like arpg", ["scene", "character"],
                             propose_fn=lambda _: "```json\n" + json.dumps(_GOOD) + "\n```")
    assert [l["layer"] for l in out["layers"]] == ["monster", "loot_table"]
    assert out["relations"][0]["kind"] == "DROPS"
    assert out["relations"][0]["src_cardinality"] == "many"  # defaulted
    assert out["warnings"] == []


def test_propose_flags_bad_schema_and_unknown_relation_layer():
    bad = {
        "layers": [{"layer": "thing", "schema": {"type": "widget"}}],
        "relations": [{"kind": "USES", "src_layers": ["thing"], "dst_layers": ["ghost"]}],
    }
    out = pa.propose_systems("x", ["scene"], propose_fn=lambda _: json.dumps(bad))
    assert any("schema issues" in w for w in out["warnings"])
    assert any("unknown layer 'ghost'" in w for w in out["warnings"])


def test_propose_tolerates_prose_around_json():
    reply = "Sure! Here are the systems:\n```json\n" + json.dumps(_GOOD) + "\n```\nHope that helps!"
    out = pa.propose_systems("x", [], propose_fn=lambda _: reply)
    assert len(out["layers"]) == 2


def test_propose_empty_on_garbage():
    out = pa.propose_systems("x", [], propose_fn=lambda _: "no json here")
    assert out["layers"] == [] and out["relations"] == []


def test_feedback_and_prior_passed_to_llm():
    seen = {}
    def fn(user):
        seen["user"] = user
        return json.dumps(_GOOD)
    pa.propose_systems("base game", ["scene"], feedback="add a town layer",
                       prior=_GOOD, propose_fn=fn)
    assert "add a town layer" in seen["user"]
    assert "Current proposed systems" in seen["user"]


# ── BYO-LLM (zero-Bedrock) path ────────────────────────────────────────────────
def test_build_prompt_is_self_contained():
    """Prompt-only mode embeds the system rules, known layers, and the user's ask —
    so it can be pasted into an external LLM with no extra context."""
    p = pa.build_prompt("a diablo arpg", ["scene", "character"],
                        feedback="add waypoints", prior=_GOOD)
    assert "scene, character" in p           # known layers injected (no str.format crash)
    assert "a diablo arpg" in p              # the description
    assert "add waypoints" in p              # feedback
    assert "Current proposed systems" in p   # prior
    assert "ONLY the JSON" in p              # output contract


def test_lint_raw_matches_propose_systems():
    """Pasted-back JSON lints to the same shape the in-Studio run produces."""
    raw = "```json\n" + json.dumps(_GOOD) + "\n```"
    out = pa.lint_raw(raw, ["scene", "character"])
    ref = pa.propose_systems("x", ["scene", "character"], propose_fn=lambda _: raw)
    assert out == ref
    assert out["warnings"] == []
