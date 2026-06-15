"""P7 — Matching v1 (spec ↔ built game coverage).

Compares the authored graph (spec) against a built/deployed Scene Contract and
reports what the game covers, what's missing, and what's extra. This is the
agent-readable verification report: a coding agent reads `missing`, fills the
gaps, and re-runs until `ok` is true. Pure logic over dicts.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _ids(items: List[dict]) -> set:
    return {i.get("id") for i in items if i.get("id")}


def match(entities: List[Dict[str, Any]], contract: Dict[str, Any]) -> Dict[str, Any]:
    """Return a coverage report comparing authored entities to a contract.

    Each authored renderable entity (collider/trigger/character-with-glb) is a
    requirement; we check it appears in the corresponding contract section.
    """
    missing: List[Dict[str, str]] = []
    covered: List[Dict[str, str]] = []

    contract_entity_ids = _ids(contract.get("entities", []))
    contract_trigger_ids = _ids(contract.get("triggers", []))
    contract_asset_ids = _ids(contract.get("assets", []))
    player_asset = (contract.get("player") or {}).get("asset")

    def _check(kind: str, key: str, present: bool) -> None:
        (covered if present else missing).append({"kind": kind, "key": key})

    for e in entities:
        layer = e.get("layer")
        key = e.get("key")
        data = e.get("data", {})
        if layer == "collider":
            _check("collider", key, key in contract_entity_ids)
        elif layer == "trigger":
            _check("trigger", key, key in contract_trigger_ids)
        elif layer in ("character", "prop") and data.get("glb"):
            present = key in contract_asset_ids or key == player_asset
            _check(f"{layer}_asset", key, present)
        elif layer == "character" and data.get("role") == "player":
            # A player without a GLB is still satisfied by the player block.
            _check("player", key, bool(contract.get("player")))

    # Extras: contract items with no authored source (informational).
    authored_keys = {e.get("key") for e in entities}
    extra = [
        {"kind": "entity", "id": i}
        for i in contract_entity_ids | contract_trigger_ids | contract_asset_ids
        if i not in authored_keys
    ]

    total = len(covered) + len(missing)
    coverage = round(len(covered) / total, 3) if total else 1.0
    return {
        "ok": not missing,
        "coverage": coverage,
        "counts": {"covered": len(covered), "missing": len(missing), "extra": len(extra)},
        "covered": covered,
        "missing": missing,
        "extra": extra,
    }
