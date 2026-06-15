"""P6 — Spark→Contract pipe.

Turns a game's authored graph (entities + their JSON data) into the engine-
agnostic Scene Contract the CycleZero runtime plays (cyclezero
src/contract/types.ts). This is pure logic over plain dicts so it is fully
unit-testable and engine-free.

Authoring convention (v1):
- one ``scene`` entity carries camera / quality / ground / lights / scatter
  (in its ``data``);
- ``collider`` entities → contract ``entities`` (shape/position/size in data);
- ``trigger`` entities  → contract ``triggers`` (shape/center/radius/event);
- a ``character`` entity with ``data.role == "player"`` → the player
  (spawn/speed; ``asset`` = its key when it has ``data.glb``);
- any entity with ``data.glb`` → an ``assets[]`` entry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

CONTRACT_VERSION = "1.0"

_DEFAULT_LIGHTS = [
    {"kind": "hemispheric", "intensity": 0.5},
    {
        "kind": "directional",
        "direction": [-0.6, -1, -0.4],
        "intensity": 2.2,
        "shadows": True,
    },
]


def _first(entities: List[dict], layer: str) -> Optional[dict]:
    for e in entities:
        if e.get("layer") == layer:
            return e
    return None


def _by_layer(entities: List[dict], layer: str) -> List[dict]:
    return [e for e in entities if e.get("layer") == layer]


def build_contract(game: Dict[str, Any], entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble a Scene Contract dict from a game + its entities.

    ``game`` needs ``slug`` and ``title``; each entity is ``{layer, key, name, data}``.
    """
    scene = _first(entities, "scene")
    sd: Dict[str, Any] = (scene or {}).get("data", {}) if scene else {}

    camera = {"kind": "iso", "viewHeight": sd.get("camera", {}).get("viewHeight", 30)}
    quality = sd.get("quality", "high")

    ground_in = sd.get("ground", {})
    ground = {
        "size": ground_in.get("size", [40, 40]),
        "albedo": ground_in.get("albedo", [0.18, 0.2, 0.24]),
        "roughness": ground_in.get("roughness", 0.9),
    }
    lights = sd.get("lights") or _DEFAULT_LIGHTS

    assets: List[dict] = []

    def _register_asset(e: dict) -> Optional[str]:
        data = e.get("data", {})
        glb = data.get("glb")
        if not glb:
            return None
        assets.append({"id": e["key"], "glb": glb, "lod": data.get("lod", "auto")})
        return e["key"]

    # Player (first character flagged role=player, else first character).
    characters = _by_layer(entities, "character")
    player_ent = next(
        (c for c in characters if c.get("data", {}).get("role") == "player"),
        characters[0] if characters else None,
    )
    if player_ent:
        pd = player_ent.get("data", {})
        player_asset = _register_asset(player_ent)
        player = {
            "spawn": pd.get("spawn", [0, 1, 0]),
            "speed": pd.get("speed", 6),
            "asset": player_asset,
        }
    else:
        player = {"spawn": [0, 1, 0], "speed": 6, "asset": None}

    # Colliders → contract entities.
    contract_entities: List[dict] = []
    for e in _by_layer(entities, "collider"):
        d = e.get("data", {})
        contract_entities.append(
            {
                "id": e["key"],
                "kind": "collider",
                "shape": d.get("shape", "box"),
                "position": d.get("position", [0, 0, 0]),
                "size": d.get("size", [1, 1, 1]),
            }
        )

    # Props with a GLB also register as assets (rendered later by the engine).
    for e in _by_layer(entities, "prop"):
        _register_asset(e)

    # Triggers.
    triggers: List[dict] = []
    for e in _by_layer(entities, "trigger"):
        d = e.get("data", {})
        triggers.append(
            {
                "id": e["key"],
                "shape": d.get("shape", "sphere"),
                "center": d.get("center", [0, 0, 0]),
                "radius": d.get("radius", 4),
                "event": d.get("event", e["key"]),
            }
        )

    scatter = sd.get("scatter", [])

    return {
        "contractVersion": CONTRACT_VERSION,
        "id": game["slug"],
        "name": game["title"],
        "camera": camera,
        "quality": quality,
        "environment": {"ground": ground, "lights": lights},
        "player": player,
        "entities": contract_entities,
        "scatter": scatter,
        "triggers": triggers,
        "assets": assets,
    }
