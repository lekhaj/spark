"""Mental-model loop — pure tests for the living capability ledger (no DB/LLM)."""
from app.cyclezero import capability_store as cs
from app.cyclezero import compile_tools as ct
from app.cyclezero import compile_agent as ca


def test_merge_registry_folds_ledger_into_base():
    base = ct.get_base_registry("babylon")
    led = cs.empty_ledger("babylon")
    led = cs.apply_report(led, {"systems": ["DialogueSystem"], "consumes": ["npc"]})
    reg = cs.merge_registry(base, led)
    assert "DialogueSystem" in reg["systems"]
    assert "npc" in reg["consumes"]
    assert reg["entry_count"] == 1
    # base untouched
    assert "DialogueSystem" not in ct.get_base_registry("babylon")["systems"]


def test_apply_report_dedupes_and_records_entry():
    led = cs.empty_ledger("babylon")
    led = cs.apply_report(led, {"consumes": ["npc"], "summary": "added npc dialogue",
                                "repo": "x/y", "commit": "abc1234", "files": ["a.ts"]})
    led = cs.apply_report(led, {"consumes": ["npc", "quest"]})  # npc dup
    assert led["extra_consumes"] == ["npc", "quest"]
    assert len(led["entries"]) == 2
    assert led["entries"][0]["commit"] == "abc1234"


def test_ingest_shrinks_gaps_on_next_compile():
    entities = [{"layer": "scene", "key": "s", "name": "S", "data": {}, "accepted_spec_run_id": None},
                {"layer": "npc", "key": "elder", "name": "Elder", "data": {}, "accepted_spec_run_id": None}]
    mm = {"layers": {}, "relation_types": {}}
    # before: npc is a gap
    before = ca.compile_prompt(entities, [], mm, {}, stitch_fn=None)
    assert "npc" in before["gaps"]
    # report that npc was built → merge into ledger
    led = cs.apply_report(cs.empty_ledger("babylon"), {"consumes": ["npc"]})
    after = ca.compile_prompt(entities, [], mm, {}, ledger=led, stitch_fn=None)
    assert "npc" not in after["gaps"]  # mental model updated → no longer a gap


def test_extract_from_note_finds_layers_repo_commit_files():
    note = ("Implemented the npc dialogue system and quest tracking. "
            "Repo https://github.com/lekhaj/cyclezero commit 9f3a1c2 "
            "Files: src/systems/dialogue.ts, src/contract/types.ts")
    out = cs.extract_from_note(note, known_layers=["npc", "quest", "monster"],
                               known_systems=["IsoCamera"])
    assert set(out["consumes"]) == {"npc", "quest"}
    assert "github.com/lekhaj/cyclezero" in out["repo"]
    assert out["commit"] == "9f3a1c2"
    assert "src/systems/dialogue.ts" in out["files"]


def test_get_capability_registry_backward_compatible():
    # no ledger arg → just the base seed (signature stable)
    reg = ct.get_capability_registry("babylon")
    assert "IsoCamera" in reg["systems"]
    assert "entry_count" not in reg
