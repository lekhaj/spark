"""Systems discipline — game MECHANICS and how they connect.

The first real "mind" extracted from the mono-agent. Owns systems, factors, outcomes,
items, and gameplay loops, plus the relations among them (REQUIRES, AFFECTS, MODIFIES,
GATES, READS…). This is where the stamina→power-attack/defense wiring happens.

It is offered the full tool set (including start_game and scene/character creation) so a
single Systems-led turn can still bootstrap a playable skeleton — until the World and
Narrative disciplines land and take those layers over.
"""
from __future__ import annotations

from .base import DisciplineAgent

SYSTEMS_INTRO = """You are the Spark Studio SYSTEMS designer — you handle game MECHANICS:
systems (stamina, combat, health, economy), factors (numeric/state signals), outcomes,
items, and gameplay loops, and crucially the RELATIONSHIPS among them. Capturing how
mechanics connect is the whole point: e.g. "stamina drains on power attacks and low
stamina weakens defense" → upsert a `system` "Stamina", a `system` "Power Attack" and a
`factor` "Defense", then link Power Attack REQUIRES Stamina and Stamina AFFECTS Defense."""

SYSTEMS_AGENT = DisciplineAgent(
    name="systems",
    label="Systems",
    blurb="mechanics, systems, factors, economy, and their relations",
    owned_layers=("system", "factor", "outcome", "item", "gameplay_loop"),
    intents=(
        "system", "mechanic", "stamina", "combat", "damage", "attack", "defense",
        "health", "mana", "energy", "economy", "currency", "gold", "loot", "drop",
        "factor", "stat", "rule", "loop", "balance", "ability", "skill", "resource",
        "cooldown", "level up", "xp", "experience", "craft", "inventory",
    ),
    intro=SYSTEMS_INTRO,
    tool_names=("start_game", "save_facts", "upsert_entity", "link_entities", "ask_clarification"),
)
