"""World discipline — the SPACE the game happens in.

Owns scenes, props, and the spatial placement of characters (spawn points, layout,
navigation). Deterministically backed by ``contract.build_contract`` (the graph → Babylon
scene contract) and ``matching.match`` (coverage). Like Systems it gets the full write-tool
set so a World-led turn can still bootstrap a playable skeleton (a scene + a placed player).
"""
from __future__ import annotations

from .base import DisciplineAgent

WORLD_INTRO = """You are the Spark Studio WORLD designer — you handle the SPACE the game
happens in: scenes/levels, their layout, props, and where things sit in space (the player's
spawn, navigation, regions). A playtest is most fun once a `scene` exists and the `character`
with data.role="player" is placed in it.

SPATIAL DATA — never drop it (scale and position must be captured or assets can't be placed):
- A placed prop/character carries a `transform` in its data:
  {transform:{position:[x,y,z], rotation:[x,y,z], scale:[x,y,z] or a single number}}.
- A `scene` carries its extent: data.bounds {size:[x,z], height?} and units (default metres).
- Put each placed object inside its scene with a CONTAINS relation (scene CONTAINS prop);
  for the SAME asset reused many times, the per-instance transform can ride on the
  CONTAINS edge's data instead of the node.
Also fill: character {spawn:[x,y,z], speed}, collider {shape, position, size}. Don't invent
exact numbers you don't know — ask, or use sensible defaults and note them."""

WORLD_AGENT = DisciplineAgent(
    name="world",
    label="World",
    blurb="scenes, levels, layout, props, spawns, navigation",
    owned_layers=("scene", "prop"),
    intents=(
        "scene", "level", "environment", "area", "map", "biome", "prop", "layout",
        "place", "region", "town", "dungeon", "room", "terrain", "spawn", "navigation",
        "world", "zone", "tile", "grid",
    ),
    intro=WORLD_INTRO,
    tool_names=("start_game", "save_facts", "upsert_entity", "link_entities", "ask_clarification"),
)
