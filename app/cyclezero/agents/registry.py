"""Agent registry + the router.

``route(text)`` classifies a turn to a discipline. Today only Systems is implemented,
so everything resolves to it — but the router is real: it recognises narrative / world /
art intents and reports them (``routed_to``) even while they fall back to the default.
That keeps routing honest now and lights up the moment those modules are dropped in
(grow-into-the-sweet-spot, per the agentic-architecture field guide).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .base import DisciplineAgent
from .systems import SYSTEMS_AGENT

# Implemented disciplines, in routing priority order.
AGENTS: Tuple[DisciplineAgent, ...] = (SYSTEMS_AGENT,)
DEFAULT: DisciplineAgent = SYSTEMS_AGENT

# Disciplines we recognise but haven't implemented yet → classified for transparency,
# handled by DEFAULT until their module is added to AGENTS.
_FUTURE_INTENTS: Dict[str, Tuple[str, ...]] = {
    "narrative": ("story", "quest", "dialogue", "lore", "plot", "mission", "npc",
                  "character", "hero", "villain", "arc", "cutscene", "objective"),
    "world": ("scene", "level", "environment", "area", "map", "biome", "prop",
              "layout", "place", "region", "town", "dungeon", "room", "terrain"),
    "art": ("asset", "model", "render", "look", "texture", "concept", "visual",
            "3d", "sprite", "art", "mesh", "palette", "silhouette", "portrait"),
}


def _matches(low: str, kws: Tuple[str, ...]) -> bool:
    return any(k in low for k in kws)


def route(text: str, facts: Optional[Dict[str, Any]] = None) -> Tuple[DisciplineAgent, str]:
    """Pick the discipline for this turn. Returns ``(agent, routed_to)`` where
    ``routed_to`` is the *intended* discipline (may be unimplemented) and ``agent`` is
    who actually handles it."""
    low = (text or "").lower()
    # An implemented agent whose intents match wins.
    for ag in AGENTS:
        if _matches(low, ag.intents):
            return ag, ag.name
    # Otherwise classify the intended discipline (even if unimplemented) for honesty.
    for disc, kws in _FUTURE_INTENTS.items():
        if _matches(low, kws):
            return DEFAULT, disc
    return DEFAULT, DEFAULT.name
