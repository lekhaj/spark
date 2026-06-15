"""X0 — default per-layer JSON Schemas (the "contracts" each node's content fills).

These are *data* (they live in the spec-gen schema registry, Mongo ``spec_schemas``
in ``World_builder``), seeded once if absent so a fresh game has contracts to author
against. Authors can bump them later via ``POST /schemas/{key}`` like any schema.

The structured ones (mission/factor/outcome/environment) carry the bespoke fields the
Explore inspectors render; the rest are intentionally light so authoring isn't blocked.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

_OBJ = "object"

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "factor_spec": {
        "title": "Factor",
        "json_schema": {
            "type": _OBJ,
            "required": ["kind"],
            "properties": {
                "kind": {"enum": ["numeric", "flag"]},
                "min": {"type": "number"},
                "max": {"type": "number"},
                "default": {},
                "description": {"type": "string"},
            },
        },
    },
    "outcome_spec": {
        "title": "Outcome resolver",
        "json_schema": {
            "type": _OBJ,
            "required": ["rules"],
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": _OBJ,
                        "required": ["when", "ending"],
                        "properties": {
                            "when": {
                                "type": "array",
                                "items": {
                                    "type": _OBJ,
                                    "required": ["factor", "op", "value"],
                                    "properties": {
                                        "factor": {"type": "string"},
                                        "op": {"enum": [">=", ">", "<=", "<", "==", "!="]},
                                        "value": {},
                                    },
                                },
                            },
                            "ending": {"type": "string"},
                            "priority": {"type": "integer"},
                        },
                    },
                },
                "default_ending": {"type": "string"},
            },
        },
    },
    "mission_spec": {
        "title": "Mission",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "summary": {"type": "string"},
                "visible_objectives": {"type": "array", "items": {"type": _OBJ}},
                "hidden_objectives": {"type": "array", "items": {"type": _OBJ}},
                "conditions": {"type": "array", "items": {"type": _OBJ}},
                "carry_forward": {"type": "array", "items": {"type": "string"}},
                "timer": {
                    "type": _OBJ,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "seconds": {"type": "integer", "minimum": 0},
                        "on_expire": {"type": "string"},
                    },
                },
                "variants": {"type": "array", "items": {"type": _OBJ}},
                "rewards": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "environment_spec": {
        "title": "Environment / Area theme",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "summary": {"type": "string"},
                "palette": {"type": "array", "items": {"type": "string"}},
                "lighting": {"type": _OBJ},
                "scatter": {"type": "array", "items": {"type": _OBJ}},
                "ambient": {"type": _OBJ},
                "system_modifiers": {"type": "array", "items": {"type": _OBJ}},
            },
        },
    },
    "system_spec": {
        "title": "System",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "description": {"type": "string"},
                "scope": {"enum": ["global", "local"]},
                "params": {"type": _OBJ},
            },
        },
    },
    "character_spec": {
        "title": "Character",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "bio": {"type": "string"},
                "role": {"type": "string"},
                "portrait_prompt": {"type": "string"},
                "glb": {"type": "string"},
            },
        },
    },
    "npc_spec": {
        "title": "NPC",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "role": {"type": "string"},
                "faction": {"type": "string"},
                "behavior": {"type": "string"},
                "schedule": {"type": "array", "items": {"type": _OBJ}},
                "dialogue_hooks": {"type": "array", "items": {"type": "string"}},
                "glb": {"type": "string"},
            },
        },
    },
    "prop_spec": {
        "title": "Prop",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "description": {"type": "string"},
                "kind": {"type": "string"},
                "glb": {"type": "string"},
            },
        },
    },
    "story_spec": {
        "title": "Story beat",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "summary": {"type": "string"},
                "choices": {"type": "array", "items": {"type": _OBJ}},
            },
        },
    },
    "interaction_spec": {
        "title": "Interaction",
        "json_schema": {
            "type": _OBJ,
            "properties": {
                "trigger": {"type": "string"},
                "conditions": {"type": "array", "items": {"type": _OBJ}},
                "effects": {"type": "array", "items": {"type": _OBJ}},
            },
        },
    },
}


def seed_schemas(db) -> int:
    """Insert any missing per-layer schema as an active v1. Idempotent. Returns
    the count newly inserted. ``db`` is a pymongo Database (``World_builder``)."""
    col = db["spec_schemas"]
    inserted = 0
    for key, spec in SCHEMAS.items():
        if col.find_one({"schema_key": key}):
            continue
        col.insert_one({
            "schema_key": key,
            "version": 1,
            "title": spec["title"],
            "engine_bound": False,
            "json_schema": spec["json_schema"],
            "changelog": "X0 default seed",
            "created_at": datetime.now(timezone.utc),
            "active": True,
        })
        inserted += 1
    return inserted
