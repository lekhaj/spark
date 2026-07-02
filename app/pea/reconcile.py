"""
Level-id reconciliation — collapse three encodings into ONE canonical int level_id.

Verified against project 3631004 (2026-07-02):
  - Level Completed.LevelNumber (int)         -> level_id == LevelNumber                 [canonical]
  - Journey.*  scene "Level N" @ buildIndex N+1 -> level_id == buildIndex - 1 (MainMenu buildIndex 1)
  - Game Start.Level (numeric STRING "1".."14") -> AMBIGUOUS; used only as fallback + logged.
"""
from __future__ import annotations

import logging
import re

from . import config as C

log = logging.getLogger("pea.reconcile")

_SCENE_RE = re.compile(r"level\s*(\d+)", re.IGNORECASE)

# Map the two undocumented start events to canonical Game Start for counting starts,
# but keep original names in raw_events; normalization is only for aggregation helpers.
_EVENT_ALIASES = {
    C.E_JOURNEY_STARTGAME: C.E_JOURNEY_STARTGAME,          # keep distinct; flagged in docs
    C.E_JOURNEY_NAV_STARTGAME: C.E_JOURNEY_NAV_STARTGAME,
}


def normalize_event(name: str | None) -> str:
    return _EVENT_ALIASES.get(name, name or "")


def _int_or_none(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def reconcile_level_id(event: str | None, props: dict) -> int | None:
    """Return canonical int level_id for an event, or None if not level-scoped."""
    if event == C.E_LEVEL_COMPLETED:
        return _int_or_none(props.get(C.P_LEVEL_NUM))

    # Journey events: prefer buildIndex-1, cross-check the scene string.
    bi = _int_or_none(props.get(C.P_BUILD_INDEX))
    scene = props.get(C.P_SCENE)
    if bi is not None:
        level_from_bi = bi - 1  # buildIndex 2 == Level 1
        if scene:
            m = _SCENE_RE.search(str(scene))
            if m and int(m.group(1)) != level_from_bi:
                log.warning("level mismatch: buildIndex-1=%s but scene=%r", level_from_bi, scene)
        return level_from_bi if level_from_bi >= 1 else None  # MainMenu(0)/negatives -> None

    if scene:
        m = _SCENE_RE.search(str(scene))
        if m:
            return int(m.group(1))

    # Game Start.Level — PARTIAL. Return as fallback but tag; transform layer validates
    # against the session's Journey stream and may override.
    if event == C.E_GAME_START:
        lv = _int_or_none(props.get(C.P_LEVEL_STR))
        return lv  # ambiguous; see SCHEMA_REALITY.md

    return None
