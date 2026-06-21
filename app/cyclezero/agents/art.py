"""Art discipline — visual assets and their SCALE.

Owns the look of props and characters and, crucially, their NATIVE DIMENSIONS so they can
be scaled and placed correctly in the world. Uses ``generate_asset`` to queue a job
(record-only — no GPU spend yet; generation is batched/optimised later). Deterministically
backed by ``generation.submit`` / the AssetJob pipeline and the ``spatial`` contract.
"""
from __future__ import annotations

from .base import DisciplineAgent

ART_INTRO = """You are the Spark Studio ART designer — you handle the VISUAL assets (3D
models, textures, concepts) for props and characters. Two things you must always capture so
assets aren't unusable later:
1. NATIVE DIMENSIONS — every asset's real-world size {w,h,d} in metres. A door is ~1x2x0.1,
   a barrel ~0.6x0.9x0.6. Without this, the asset can't be scaled to fit the world.
2. A clear visual DESCRIPTOR — silhouette, palette, style, references.
Workflow: create the prop/character with upsert_entity (put its `dimensions` and any
`transform` in data), then call generate_asset with the entity, its dimensions, and the
descriptor. Generation is queued and batched — you are capturing the spec, not rendering now."""

ART_AGENT = DisciplineAgent(
    name="art",
    label="Art",
    blurb="visual assets, look, and native dimensions for scaling",
    owned_layers=("prop", "character"),
    intents=(
        "asset", "model", "render", "look", "texture", "concept", "visual", "3d",
        "sprite", "art", "mesh", "palette", "silhouette", "portrait", "generate",
        "image", "skin", "material", "dimension", "size", "scale",
    ),
    intro=ART_INTRO,
    tool_names=("start_game", "save_facts", "upsert_entity", "link_entities",
                "generate_asset", "ask_clarification"),
)
