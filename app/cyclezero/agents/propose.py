"""Propose discipline — installs NEW vocabulary when the design outgrows the metamodel.

When a turn needs a layer or relation kind that doesn't exist (e.g. the user invents a
"reputation" system that should AFFECTS a faction layer that isn't seeded), this mind
proposes the new layer(s)/relation-type(s) directly in a ``propose_system`` tool call; the
shared gate lints them (``propose_agent.validate_proposal``) and installs them via the
metamodel store. It has the write tools too, so a single turn can install vocab AND use it
is NOT possible (known_layers is fixed per turn) — propose first, then build next turn.
"""
from __future__ import annotations

from .base import DisciplineAgent

PROPOSE_INTRO = """You are the Spark Studio SYSTEMS-DESIGN (vocabulary) designer. Most turns
should reuse the known layers and relation kinds. But when the design genuinely needs a NEW
kind of thing or a NEW kind of connection that doesn't exist yet, call propose_system to
install it:
- a new LAYER: {layer:"faction", title:"Faction", schema:{...optional JSON schema...}}.
- a new RELATION type: {kind:"ALLIES_WITH", src_layers:["faction"], dst_layers:["faction"],
  src_cardinality:"many", dst_cardinality:"many"}.
Propose the minimum needed, with clear UPPER_SNAKE relation kinds and lowercase layer names.
After it's installed, the next turn can create entities/relations using it."""

PROPOSE_AGENT = DisciplineAgent(
    name="propose",
    label="Vocabulary",
    blurb="installs new layers & relation types when the design needs them",
    owned_layers=(),  # it defines layers rather than owning any
    intents=(
        "new layer", "new relation", "new kind", "new type", "doesn't exist",
        "does not exist", "vocabulary", "define a layer", "define a relation",
        "add a layer", "add a relation type", "custom relation", "propose",
    ),
    intro=PROPOSE_INTRO,
    tool_names=("start_game", "propose_system", "save_facts", "upsert_entity",
                "link_entities", "ask_clarification"),
)
