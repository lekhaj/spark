"""Spatial contract + completeness check.

Scale and placement must be captured at design time or they're lost (the contract today
registers a prop's GLB but no transform, and assets carry no native size). This module
defines the convention and a deterministic guardrail so the Critic/Validator can flag
missing spatial data. Pure logic over plain dicts — no migration; it all lives in
``entity.data`` / edge ``data`` (the generic-graph invariant).

Convention (v1):
- INTRINSIC size  → ``entity.data["dimensions"] = {"w","h","d"}`` (+ optional ``"units"``,
  default "m") on a prop/character — the asset's native bounding box. Captured by Art.
- PLACEMENT       → ``entity.data["transform"] = {"position":[x,y,z], "rotation":[x,y,z],
  "scale":[x,y,z] | number}`` per placed node. Captured by World. (For one asset instanced
  many times, the same ``transform`` may instead ride on a ``CONTAINS`` edge's ``data``.)
- WORLD extent    → ``scene.data["bounds"] = {"size":[x,z], "height"?}`` (+ units). Captured by World.

A ``collider``'s existing ``data.position`` / ``data.size`` already count as a transform.
"""
from __future__ import annotations

from typing import Any, Dict, List

# things that occupy space and therefore need a placement transform
PLACED_LAYERS = frozenset({"prop", "character", "collider"})
# things that render a mesh and therefore need a native size to scale correctly
SIZED_LAYERS = frozenset({"prop", "character"})


def has_transform(data: Dict[str, Any]) -> bool:
    """A node is placed if it carries a transform.position, or (collider) a data.position."""
    data = data or {}
    t = data.get("transform") or {}
    return bool(t.get("position")) or bool(data.get("position"))


def has_dimensions(data: Dict[str, Any]) -> bool:
    d = (data or {}).get("dimensions") or {}
    return bool(d) and all(k in d for k in ("w", "h", "d"))


def _contained_keys(relations: List[dict]) -> Dict[str, Dict[str, Any]]:
    """dst_key -> the CONTAINS edge data (a scene placing an entity may carry the
    transform on the edge instead of the node)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in relations:
        if r.get("kind") == "CONTAINS" and r.get("dst"):
            out[r["dst"]] = r.get("data") or {}
    return out


def spatial_health(entities: List[dict], relations: List[dict]) -> Dict[str, Any]:
    """Deterministic spatial completeness. Returns counts + the keys that are missing a
    placement transform or native dimensions, plus human-readable ``issues`` for the
    Validator. Empty graph → all zero, no issues."""
    edge_placement = _contained_keys(relations)
    placed = [e for e in entities if e.get("layer") in PLACED_LAYERS]
    sized = [e for e in entities if e.get("layer") in SIZED_LAYERS]

    missing_transform: List[str] = []
    for e in placed:
        edata = e.get("data") or {}
        edge = edge_placement.get(e["key"], {})
        if not (has_transform(edata) or bool((edge.get("transform") or {}).get("position"))):
            missing_transform.append(e["key"])

    missing_dimensions = [e["key"] for e in sized if not has_dimensions(e.get("data") or {})]

    issues: List[str] = []
    for k in missing_transform:
        issues.append(f"{k}: placed object has no transform (position/rotation/scale) — "
                      f"its location and scale are undefined")
    for k in missing_dimensions:
        issues.append(f"{k}: asset has no native dimensions {{w,h,d}} — scaling it is ambiguous")

    return {
        "placed": len(placed),
        "missing_transform": sorted(missing_transform),
        "sized": len(sized),
        "missing_dimensions": sorted(missing_dimensions),
        "issues": issues,
    }
