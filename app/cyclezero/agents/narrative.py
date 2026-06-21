"""Narrative discipline — STORY, quests, and characters.

Owns story, quests, and characters (hero/villain/NPCs), and how they connect: a quest
LEADS_TO another, REQUIRES an item/state, REWARDS an outcome, GATES progress. Deterministically
backed by the graph relations + ``narrative_health`` (a small reachability check that flags
orphan quests and unreachable objectives, the narrative analogue of the experience scorer).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .base import DisciplineAgent

NARRATIVE_INTRO = """You are the Spark Studio NARRATIVE designer — you handle STORY: the
quests/missions, their objectives, dialogue beats, and the characters (hero, villains, NPCs).
Capture the SHAPE of the story with relations: a `quest` LEADS_TO the next, REQUIRES an item
or state, REWARDS an outcome, and GATES later content. Every quest should be reachable from a
start and lead somewhere — don't leave orphan objectives. Create characters as `character`
entities (NPCs without data.role; the player has data.role="player")."""

NARRATIVE_AGENT = DisciplineAgent(
    name="narrative",
    label="Narrative",
    blurb="story, quests, objectives, dialogue, characters",
    owned_layers=("story", "quest", "character"),
    intents=(
        "story", "quest", "dialogue", "lore", "plot", "mission", "npc", "hero",
        "villain", "arc", "cutscene", "objective", "narrative", "chapter", "branch",
        "choice", "ending", "faction", "character",
    ),
    intro=NARRATIVE_INTRO,
    tool_names=("start_game", "save_facts", "upsert_entity", "link_entities", "ask_clarification"),
)

# ── deterministic backing: a tiny quest-reachability / orphan check ─────────────
_FLOW_EDGES = frozenset({"LEADS_TO", "REQUIRES", "REWARDS", "GATES", "TRIGGERS"})


def narrative_health(entities: List[dict], relations: List[dict]) -> Dict[str, Any]:
    """Pure check over story/quest nodes: which quests are connected into the flow vs
    orphaned (no flow edge in or out). Returns ``{quests, connected, orphans[]}``."""
    quests = [e["key"] for e in entities if e.get("layer") in ("quest", "story")]
    if not quests:
        return {"quests": 0, "connected": 0, "orphans": []}
    touched = set()
    for r in relations:
        if r.get("kind") in _FLOW_EDGES:
            if r.get("src") in quests:
                touched.add(r["src"])
            if r.get("dst") in quests:
                touched.add(r["dst"])
    orphans = sorted(q for q in quests if q not in touched)
    return {"quests": len(quests), "connected": len(touched), "orphans": orphans}
