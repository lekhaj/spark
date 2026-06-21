"""Agent registry + the router.

``route(text)`` classifies a turn to a discipline. Today only Systems is implemented,
so everything resolves to it — but the router is real: it recognises narrative / world /
art intents and reports them (``routed_to``) even while they fall back to the default.
That keeps routing honest now and lights up the moment those modules are dropped in
(grow-into-the-sweet-spot, per the agentic-architecture field guide).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .art import ART_AGENT
from .base import DisciplineAgent, keyword_hit
from .narrative import NARRATIVE_AGENT
from .propose import PROPOSE_AGENT
from .systems import SYSTEMS_AGENT
from .world import WORLD_AGENT

# Implemented disciplines, in routing priority order (most specific first; Systems is the
# fallback default). Propose is checked early so an explicit "new layer/relation" request
# wins over a generic mechanics match.
AGENTS: Tuple[DisciplineAgent, ...] = (
    PROPOSE_AGENT, SYSTEMS_AGENT, NARRATIVE_AGENT, WORLD_AGENT, ART_AGENT)
DEFAULT: DisciplineAgent = SYSTEMS_AGENT

# Disciplines we recognise but haven't implemented yet → classified for transparency,
# handled by DEFAULT until their module is added to AGENTS. (All current disciplines are
# implemented; this stays as the seam for the next ones, e.g. audio.)
_FUTURE_INTENTS: Dict[str, Tuple[str, ...]] = {}


def route(text: str, facts: Optional[Dict[str, Any]] = None) -> Tuple[DisciplineAgent, str]:
    """Pick the discipline for this turn. Returns ``(agent, routed_to)`` where
    ``routed_to`` is the *intended* discipline (may be unimplemented) and ``agent`` is
    who actually handles it."""
    # An implemented agent whose intents match wins.
    for ag in AGENTS:
        if keyword_hit(text, ag.intents):
            return ag, ag.name
    # Otherwise classify the intended discipline (even if unimplemented) for honesty.
    for disc, kws in _FUTURE_INTENTS.items():
        if keyword_hit(text, kws):
            return DEFAULT, disc
    return DEFAULT, DEFAULT.name
