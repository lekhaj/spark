"""DisciplineAgent — one "mind" in the creator studio.

A discipline = a system prompt (its expertise) + a subset of the shared tools it may
call + the layers it owns + the keyword intents that route turns to it. Every agent
writes through the SAME deterministic substrate (``creator_agent.apply_tool_calls``),
so they coordinate through the shared graph/memory (a blackboard), never peer-to-peer.

The prompt is ``intro`` (discipline-specific) + ``COMMON_RULES`` (cross-cutting: how to
save, the playable target, ask-don't-guess). Placeholders are filled by the orchestrator
via ``.replace()`` (the templates contain literal ``{ }`` JSON, so never ``str.format``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .. import creator_agent


def keyword_hit(text: str, keywords: Tuple[str, ...]) -> bool:
    """Whole-word/phrase match — used for routing + read-only intents. Naive substring
    matching is unsafe here ("rate" is inside "generate", "art" inside "start"), so every
    keyword is matched on word boundaries."""
    low = (text or "").lower()
    return any(re.search(r"\b" + re.escape(k) + r"\b", low) for k in keywords)

# Cross-cutting rules shared by every discipline. Each agent prepends its own intro.
COMMON_RULES = """How you work:
- If the user wants to start a new game and none exists yet, call start_game FIRST.
- Save high-level design facts (genre, pillars, tone, references) with save_facts.
- Create concrete objects with upsert_entity. 'layer' MUST be one of the known layers:
  {known_layers}.
- Capture how things RELATE with link_entities — relations are how the design actually
  connects, so don't drop them. 'kind' MUST be one of the known relation types (each
  shows which layers it may connect):
  {known_relations}
  Always create BOTH endpoints (upsert_entity) before linking them.
- PLAYABLE TARGET: the user can playtest the moment a game exists, but it's most fun
  once there's a `scene` to be in and a `character` whose data.role is "player". When
  the user is building, make sure those exist (sensible defaults are fine) so they can
  try it early. Fill play-relevant fields when known: character {role, spawn:[x,y,z],
  speed}, collider {shape, position, size}, prop {glb}.
- When you are UNSURE what to persist, DO NOT GUESS — call ask_clarification with a
  short question and 2-4 concrete options. You may save the parts you're sure about AND
  ask about the one gap in the same turn.
- Keep your spoken reply short (1-3 sentences); the tools do the saving.

Confirmed facts so far for this game (already saved — don't re-save unless changing):
{facts_json}"""


@dataclass(frozen=True)
class DisciplineAgent:
    name: str                       # stable id, e.g. "systems"
    label: str                      # human label, e.g. "Systems"
    blurb: str                      # one-line description (shown in UI)
    owned_layers: Tuple[str, ...]   # the layers this discipline is responsible for
    intents: Tuple[str, ...]        # keywords that route a turn to this agent
    intro: str                      # discipline-specific prompt preamble
    tool_names: Tuple[str, ...]     # which shared tools this agent may call

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """The agent's tool subset, taken from the shared catalog."""
        return [t for t in creator_agent.TOOLS if t["name"] in self.tool_names]

    def system_prompt(self, *, known_layers: str, known_relations: str, facts_json: str) -> str:
        return (
            (self.intro + "\n\n" + COMMON_RULES)
            .replace("{known_layers}", known_layers or "(none)")
            .replace("{known_relations}", known_relations)
            .replace("{facts_json}", facts_json or "(none yet)")
        )
