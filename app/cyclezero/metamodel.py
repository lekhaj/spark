"""S0 — the relation metamodel (CycleZero's "relation contract").

Two Mongo-backed registries that make the otherwise-generic graph *typed*:

  - ``cyclezero_layers``         layer  → bound ``schema_key`` (the JSON contract
                                 each node of that layer validates against).
  - ``cyclezero_relation_types`` kind   → which layers it may connect, cardinality,
                                 whether it is required, whether it is a generation
                                 dependency edge, and an optional edge-``data`` schema.

A new meta-layer is a new ``layer`` string + new relation-type rows — **no schema
migration** (preserves the generic entity/relation design). These live in the same
Mongo DB as the spec-gen registry (``World_builder`` via ``mgs.get_db()``) so all
authoring config is co-located and the spec bridge (S1) stays single-DB.

Functions take an explicit ``db`` (a pymongo ``Database``) so they are unit-testable
against mongomock, mirroring ``app/services``/``schema_routes`` style. ``_db()`` is
the production default and is monkeypatched in tests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

LAYERS_COLLECTION = "cyclezero_layers"
RELATION_TYPES_COLLECTION = "cyclezero_relation_types"

CARDINALITIES = ("one", "many")


def _db():
    """Open the authoring Mongo DB. Overridden in tests (mongomock)."""
    from worker.lib import manual_gen_schema as mgs

    return mgs.get_db()


# ── default seeds ─────────────────────────────────────────────────────────────
# Layers map 1:1 to a ``<layer>_spec`` schema key. Defaults are intentionally
# permissive — none force structure (``required`` stays False) so authoring is
# never blocked; teams tighten the registry as conventions settle.
DEFAULT_LAYERS: List[Dict[str, Any]] = [
    {"layer": "scene", "schema_key": "scene_spec", "title": "Scene"},
    {"layer": "character", "schema_key": "character_spec", "title": "Character"},
    {"layer": "npc", "schema_key": "npc_spec", "title": "NPC"},
    {"layer": "system", "schema_key": "system_spec", "title": "System"},
    {"layer": "environment", "schema_key": "environment_spec", "title": "Environment / Area theme"},
    {"layer": "story", "schema_key": "story_spec", "title": "Story"},
    {"layer": "quest", "schema_key": "quest_spec", "title": "Quest"},
    {"layer": "mission", "schema_key": "mission_spec", "title": "Mission"},
    {"layer": "interaction", "schema_key": "interaction_spec", "title": "Interaction"},
    {"layer": "factor", "schema_key": "factor_spec", "title": "Factor"},
    {"layer": "outcome", "schema_key": "outcome_spec", "title": "Outcome"},
    {"layer": "item", "schema_key": "item_spec", "title": "Item"},
    {"layer": "prop", "schema_key": "prop_spec", "title": "Prop"},
    {"layer": "collider", "schema_key": "collider_spec", "title": "Collider"},
    {"layer": "trigger", "schema_key": "trigger_spec", "title": "Trigger"},
    {"layer": "gameplay_loop", "schema_key": "gameplay_loop_spec", "title": "Gameplay Loop"},
]

# ``dependency: True`` means *src depends on dst* — dst must be accepted before src
# (drives topo order + ripple). Cardinality is per-endpoint: "one" caps how many
# edges of this kind a given src/dst node may participate in.
DEFAULT_RELATION_TYPES: List[Dict[str, Any]] = [
    {
        "kind": "APPEARS_IN",
        "src_layers": ["character", "prop", "collider", "trigger"],
        "dst_layers": ["scene"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": True,
        "data_schema": None,
    },
    {
        "kind": "PART_OF",
        "src_layers": ["quest", "character", "scene", "story"],
        "dst_layers": ["system", "story"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": True,
        "data_schema": None,
    },
    {
        "kind": "REQUIRES",
        "src_layers": ["quest", "gameplay_loop"],
        "dst_layers": ["quest", "item", "system"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": True,
        "data_schema": None,
    },
    {
        "kind": "REFERENCES",
        "src_layers": ["story", "quest", "scene"],
        "dst_layers": ["character", "item", "scene"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    {
        "kind": "TRIGGERS",
        "src_layers": ["trigger"],
        "dst_layers": ["quest", "story"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    {
        "kind": "OWNS",
        "src_layers": ["character", "npc"],
        "dst_layers": ["item"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    # ── Explore-layer relations (X0) ──────────────────────────────────────────
    # Scene-hub membership: a scene CONTAINS the things shown in its dashboard.
    # Aggregation edge, NOT a build-order dependency (APPEARS_IN drives order).
    {
        "kind": "CONTAINS",
        "src_layers": ["scene"],
        "dst_layers": ["character", "npc", "prop", "mission", "interaction",
                       "environment", "system", "story"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    # The progress/regress signal. Edge data carries the delta:
    # {"op": "add"|"set", "value": <number|bool>, "when"?: <choice/condition>}.
    {
        "kind": "AFFECTS",
        "src_layers": ["story", "mission", "interaction", "environment"],
        "dst_layers": ["factor"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    # Environment-driven system override. Edge data = param patch.
    {
        "kind": "MODIFIES",
        "src_layers": ["environment"],
        "dst_layers": ["system"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    # Content gated/unlocked by a factor or flag.
    {
        "kind": "GATES",
        "src_layers": ["factor"],
        "dst_layers": ["story", "mission", "interaction"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    # Story branch edges (choice → next beat or ending). Edge data: {"label": ...}.
    {
        "kind": "LEADS_TO",
        "src_layers": ["story"],
        "dst_layers": ["story", "outcome"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    # Outcome resolver wiring.
    {
        "kind": "READS",
        "src_layers": ["outcome"],
        "dst_layers": ["factor"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
    # NPC placement (scene before npc → dependency, like APPEARS_IN).
    {
        "kind": "ASSIGNED_TO",
        "src_layers": ["npc"],
        "dst_layers": ["scene"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": True,
        "data_schema": None,
    },
    # Mission rewards.
    {
        "kind": "REWARDS",
        "src_layers": ["mission", "quest"],
        "dst_layers": ["item"],
        "src_cardinality": "many",
        "dst_cardinality": "many",
        "required": False,
        "dependency": False,
        "data_schema": None,
    },
]


def ensure_seeded(db) -> None:
    """Idempotently seed the default layers + relation types into empty
    collections. Safe to call on every request (cheap count check)."""
    if db[LAYERS_COLLECTION].count_documents({}, limit=1) == 0:
        db[LAYERS_COLLECTION].insert_many([{**d} for d in DEFAULT_LAYERS])
    if db[RELATION_TYPES_COLLECTION].count_documents({}, limit=1) == 0:
        db[RELATION_TYPES_COLLECTION].insert_many([{**d} for d in DEFAULT_RELATION_TYPES])


def _clean(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {**doc}
    out.pop("_id", None)
    return out


# ── layers ────────────────────────────────────────────────────────────────────
def list_layers(db) -> List[Dict[str, Any]]:
    ensure_seeded(db)
    return [_clean(d) for d in db[LAYERS_COLLECTION].find({}).sort("layer", 1)]


def get_layer(db, layer: str) -> Optional[Dict[str, Any]]:
    ensure_seeded(db)
    doc = db[LAYERS_COLLECTION].find_one({"layer": layer})
    return _clean(doc) if doc else None


def upsert_layer(db, layer: str, schema_key: str, title: Optional[str] = None) -> Dict[str, Any]:
    doc = {"layer": layer, "schema_key": schema_key, "title": title or layer}
    db[LAYERS_COLLECTION].update_one({"layer": layer}, {"$set": doc}, upsert=True)
    return doc


# ── relation types ────────────────────────────────────────────────────────────
def list_relation_types(db) -> List[Dict[str, Any]]:
    ensure_seeded(db)
    return [_clean(d) for d in db[RELATION_TYPES_COLLECTION].find({}).sort("kind", 1)]


def get_relation_type(db, kind: str) -> Optional[Dict[str, Any]]:
    ensure_seeded(db)
    doc = db[RELATION_TYPES_COLLECTION].find_one({"kind": kind})
    return _clean(doc) if doc else None


def upsert_relation_type(db, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create/replace a relation type. ``body`` must carry ``kind``; missing
    optional fields fall back to permissive defaults. Raises ``ValueError`` on a
    bad cardinality so the route can answer 422."""
    kind = body.get("kind")
    if not kind:
        raise ValueError("relation type needs a 'kind'")
    for field in ("src_cardinality", "dst_cardinality"):
        val = body.get(field, "many")
        if val not in CARDINALITIES:
            raise ValueError(f"{field} must be one of {CARDINALITIES}, got {val!r}")
    doc = {
        "kind": kind,
        "src_layers": list(body.get("src_layers", [])),
        "dst_layers": list(body.get("dst_layers", [])),
        "src_cardinality": body.get("src_cardinality", "many"),
        "dst_cardinality": body.get("dst_cardinality", "many"),
        "required": bool(body.get("required", False)),
        "dependency": bool(body.get("dependency", False)),
        "data_schema": body.get("data_schema"),
    }
    db[RELATION_TYPES_COLLECTION].update_one({"kind": kind}, {"$set": doc}, upsert=True)
    return doc


# ── consumable snapshot ───────────────────────────────────────────────────────
def load_metamodel(db) -> Dict[str, Any]:
    """Return the whole metamodel as a plain dict keyed for O(1) lookup, so the
    pure graph algorithms in ``graph.py`` never touch Mongo:

        {"layers": {layer: {...}}, "relation_types": {kind: {...}}}
    """
    return {
        "layers": {d["layer"]: d for d in list_layers(db)},
        "relation_types": {d["kind"]: d for d in list_relation_types(db)},
    }
