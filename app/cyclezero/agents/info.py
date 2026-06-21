"""Info — the read-only "librarian" that answers catalog/listing queries deterministically.

This is the cheapest agent: it writes nothing AND calls no LLM. It turns plain-language
questions like "list my games", "what characters do I have?", "show my systems", or
"what's in my game?" into pure reads of the Postgres graph + the games table, then narrates
the answer with a hand-written template. So the broad chat entry point (Haiku) can answer
"what do I have / where can I jump in" without spending a token.

It deliberately only fires on *enumerative* questions (list / how many / which / what do I
have) and never when a write verb is present — so "add a character" still routes to the
discipline agents, while "list my characters" is answered here for free.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import keyword_hit

NAME = "info"
LABEL = "Info"

# explicit games-list phrases (word-boundary matched) — never collide with "start a game"
GAMES_INTENTS: Tuple[str, ...] = (
    "my games", "list games", "list my games", "what games", "which games",
    "show games", "show my games", "all games", "all my games", "see my games",
    "view games", "my projects", "list projects", "what projects", "other games",
    "switch game", "switch games",
)

# enumerative cues that mark a "list/catalog" question (vs a create/modify request)
_LIST_CUES: Tuple[str, ...] = (
    "list", "how many", "which", "all my", "show my", "show all", "show me all",
    "do i have", "have i got", "what are my", "what's in", "whats in",
    "inventory", "catalog", "everything", "summary",
    # view/read intent phrases — asking to SEE something, not build it
    "show me", "let me see", "where can i see", "where is", "where are",
    "can i see", "view the", "see the", "what does", "what did",
)

# write verbs — if any is present it's an authoring turn, not a catalog read
_WRITE_VERBS: Tuple[str, ...] = (
    "add", "create", "make", "build", "remove", "delete", "rename", "change",
    "set", "give", "increase", "decrease", "drain", "spawn", "place", "generate",
    "design", "write", "update", "edit", "connect", "link",
)

# layer → the words a user might call it (singular + plural + synonyms)
_LAYER_WORDS: Dict[str, Tuple[str, ...]] = {
    "character": ("character", "characters", "hero", "heroes", "npc", "npcs", "cast"),
    "system": ("system", "systems", "mechanic", "mechanics"),
    "scene": ("scene", "scenes", "level", "levels", "location", "locations",
              "layout", "map", "area", "village", "town", "world"),
    "prop": ("prop", "props", "object", "objects"),
    "item": ("item", "items", "loot"),
    "quest": ("quest", "quests", "mission", "missions"),
    "story": ("story", "stories", "narrative", "narratives"),
    "outcome": ("outcome", "outcomes", "ending", "endings"),
    "factor": ("factor", "factors", "stat", "stats"),
    "gameplay_loop": ("loop", "loops", "gameplay loop", "gameplay loops"),
}

# overview cues → list a count per layer rather than one layer
_OVERVIEW_CUES: Tuple[str, ...] = (
    "what do i have", "what have i", "what's in", "whats in", "everything",
    "inventory", "catalog", "summary", "what is in",
)


# enumerative cues that, combined with a games signal, mean "show me my games"
_GAMES_CUES: Tuple[str, ...] = (
    "list", "show", "see", "view", "what", "which", "my", "all", "have",
)
# plural anchors only — singular "game" is too weak (appears in "game feel"/"game design")
_GAMES_ANCHORS: Tuple[str, ...] = ("games", "projects")
# the legit singulars, excluded from fuzzy matching so they don't read as a typo'd plural
_GAMES_SINGULAR: frozenset = frozenset({"game", "project"})


def _within_one(a: str, b: str) -> bool:
    """True if ``a`` is within edit distance 1 of ``b`` (one insert/delete/substitute or a
    transposition). Lets typos like 'gsames'/'gamse' still resolve to 'games'."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # one substitution OR one adjacent transposition
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1:
            i, j = diff
            return a[i] == b[j] and a[j] == b[i]
        return False
    # lengths differ by 1 → one insertion/deletion: the shorter must embed in the longer
    short, lng = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(lng):
        if short[i] == lng[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def is_games_intent(text: str) -> bool:
    # 1. explicit phrases (fast path, exact)
    if keyword_hit(text, GAMES_INTENTS):
        return True
    low = (text or "").lower()
    tokens = re.findall(r"[a-z]+", low)
    if not tokens:
        return False
    # never steal an authoring turn ("add a game mode", "make a new game")
    if _has_write_verb(text):
        return False
    has_cue = any(t in _GAMES_CUES for t in tokens) or keyword_hit(text, _LIST_CUES)
    if not has_cue:
        return False
    # "...in my profile" is an account-level listing → the user's games
    if "profile" in tokens:
        return True
    # typo-tolerant plural anchor: 'games'/'projects' (and near-misses like 'gsames'),
    # but never the legitimate singular ('game'/'project'), which appears in unrelated
    # phrases like "game feel" / "game design".
    return any(t not in _GAMES_SINGULAR and _within_one(t, a)
               for t in tokens for a in _GAMES_ANCHORS)


def _has_write_verb(text: str) -> bool:
    return keyword_hit(text, _WRITE_VERBS)


def catalog_target(text: str) -> Optional[str]:
    """Which layer the catalog question is about, or "" for a whole-game overview, or
    None when it isn't a catalog question at all."""
    if _has_write_verb(text):
        return None
    if not keyword_hit(text, _LIST_CUES):
        return None
    for layer, words in _LAYER_WORDS.items():
        if keyword_hit(text, words):
            return layer
    if keyword_hit(text, _OVERVIEW_CUES):
        return ""  # overview of all layers
    return None


def is_catalog_intent(text: str) -> bool:
    return catalog_target(text) is not None


# ── deterministic answers ─────────────────────────────────────────────────────
def answer_games(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Narrate the caller's games list. ``games`` = [{game_slug, title, status}]."""
    if not games:
        reply = ("You don't have any games yet. Say e.g. \"start a game called Nightfall\" "
                 "and I'll create it.")
        return {"reply": reply, "saved": [{"kind": "games", "games": []}]}
    lines = [f"You have {len(games)} game(s):"]
    for g in games:
        status = f" · {g['status']}" if g.get("status") else ""
        lines.append(f"  🎮 {g.get('title') or g['game_slug']} ({g['game_slug']}){status}")
    lines.append("Use the game switcher (top bar) or say \"switch to <name>\" to open one.")
    return {"reply": "\n".join(lines), "saved": [{"kind": "games", "games": games}]}


def _by_layer(entities: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for e in entities:
        out.setdefault(e.get("layer") or "?", []).append(e)
    return out


def answer_catalog(text: str, entities: List[dict], *, game_slug: Optional[str],
                   target: Optional[str] = None) -> Dict[str, Any]:
    """Narrate one layer's entities, or a whole-game overview (counts per layer).

    ``target`` overrides the parsed intent — used by the LLM-as-router fallback, where
    the model supplies the layer directly (``""`` = overview) instead of free text."""
    if target is None:
        target = catalog_target(text)
    grouped = _by_layer(entities)

    if target:  # a specific layer
        rows = grouped.get(target, [])
        if not rows:
            reply = f"No {target}s in this game yet."
            return {"reply": reply, "saved": [{"kind": "catalog", "layer": target, "items": []}]}
        names = [e.get("name") or e.get("key") for e in rows]
        reply = f"{len(rows)} {target}(s): " + ", ".join(str(n) for n in names) + "."
        # for scenes/layout, add the "how to view it" tip so the user knows where to look
        if target == "scene":
            reply += " Click ▶ Play in the header to see the spatial layout in the engine."
        return {"reply": reply, "saved": [{"kind": "catalog", "layer": target, "items": names}]}

    # overview: counts per layer
    if not entities:
        reply = "This game is empty so far — no entities yet. Try \"add a scene\" to start."
        return {"reply": reply, "saved": [{"kind": "catalog", "counts": {}}]}
    counts = {layer: len(rows) for layer, rows in sorted(grouped.items())}
    parts = [f"{n} {layer}{'s' if n != 1 else ''}" for layer, n in counts.items()]
    reply = "Here's what's in this game: " + ", ".join(parts) + "."
    return {"reply": reply, "saved": [{"kind": "catalog", "counts": counts}]}
