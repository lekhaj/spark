"""
spec_schema.py — MongoDB CRUD for the Character Spec / Style / Reference Library
=================================================================================

Collections (all in World_builder DB):
  character_specs      — structured character data (body, face, hair, clothing, etc.)
  style_library        — reusable style-DNA blocks
  reference_image_bank — grouped reference images with metadata
  prompt_templates     — tested prompt templates with variable slots
"""

import os
import time
from typing import Optional

import pymongo

# ── defaults ─────────────────────────────────────────────────────────────────
# Reads from env so the same code works from local (public IP) and on-instance (localhost)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "World_builder")


# ── connection ───────────────────────────────────────────────────────────────

def get_db(uri: str = MONGO_URI, db_name: str = MONGO_DB, timeout_ms: int = 10000):
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=timeout_ms,
                                  connectTimeoutMS=timeout_ms)
    return client[db_name]


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARACTER SPECS
# ═══════════════════════════════════════════════════════════════════════════════

def upsert_character_spec(db, spec: dict) -> str:
    """Insert or replace a character spec. `_id` = character name."""
    spec.setdefault("created_at", time.time())
    spec["updated_at"] = time.time()
    db.character_specs.replace_one({"_id": spec["_id"]}, spec, upsert=True)
    return spec["_id"]


def get_character_spec(db, char_id: str) -> Optional[dict]:
    return db.character_specs.find_one({"_id": char_id})


def list_character_specs(db, category: str = None) -> list:
    filt = {"category": category} if category else {}
    return list(db.character_specs.find(filt).sort("_id", 1))


def delete_character_spec(db, char_id: str):
    db.character_specs.delete_one({"_id": char_id})


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

def upsert_style(db, style: dict) -> str:
    style.setdefault("created_at", time.time())
    style["updated_at"] = time.time()
    db.style_library.replace_one({"_id": style["_id"]}, style, upsert=True)
    return style["_id"]


def get_style(db, style_id: str) -> Optional[dict]:
    return db.style_library.find_one({"_id": style_id})


def list_styles(db) -> list:
    return list(db.style_library.find().sort("_id", 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  REFERENCE IMAGE BANK
# ═══════════════════════════════════════════════════════════════════════════════

def upsert_reference(db, ref: dict) -> str:
    ref.setdefault("created_at", time.time())
    ref["updated_at"] = time.time()
    db.reference_image_bank.replace_one({"_id": ref["_id"]}, ref, upsert=True)
    return ref["_id"]


def get_reference(db, ref_id: str) -> Optional[dict]:
    return db.reference_image_bank.find_one({"_id": ref_id})


def list_references(db, group: str = None, tags: list = None) -> list:
    filt = {}
    if group:
        filt["group"] = group
    if tags:
        filt["tags"] = {"$all": tags}
    return list(db.reference_image_bank.find(filt).sort("_id", 1))


def search_references(db, query: str) -> list:
    """Simple text search across tags, group, subgroup, description."""
    terms = query.lower().split()
    # Build an $or across multiple fields
    conditions = []
    for t in terms:
        regex = {"$regex": t, "$options": "i"}
        conditions.append({"$or": [
            {"tags": regex},
            {"group": regex},
            {"subgroup": regex},
            {"description": regex},
        ]})
    if not conditions:
        return list_references(db)
    return list(db.reference_image_bank.find({"$and": conditions}).sort("_id", 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

def upsert_prompt_template(db, tpl: dict) -> str:
    tpl.setdefault("created_at", time.time())
    tpl["updated_at"] = time.time()
    db.prompt_templates.replace_one({"_id": tpl["_id"]}, tpl, upsert=True)
    return tpl["_id"]


def get_prompt_template(db, tpl_id: str) -> Optional[dict]:
    return db.prompt_templates.find_one({"_id": tpl_id})


def list_prompt_templates(db, stage: str = None, category: str = None) -> list:
    filt = {}
    if stage:
        filt["stage"] = stage
    if category:
        filt["category"] = category
    return list(db.prompt_templates.find(filt).sort("_id", 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER — assembles final prompt from spec + style + template
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten_spec_field(spec: dict, field: str) -> str:
    """Extract a flattened text description from a spec sub-dict."""
    val = spec.get(field, {})
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            if isinstance(v, list):
                parts.append(", ".join(str(x) for x in v))
            elif v:
                parts.append(str(v))
        return ", ".join(parts)
    if isinstance(val, list):
        return ", ".join(str(x) for x in val)
    return str(val) if val else ""


def build_body_desc(spec: dict) -> str:
    """Build a body description string from spec fields."""
    parts = []
    body = spec.get("body", {})
    if body.get("gender"):
        parts.append(body["gender"])
    if body.get("build"):
        parts.append(f"{body['build']} build")
    if body.get("age_range"):
        parts.append(f"{body['age_range']} years old")
    if body.get("species"):
        parts.append(body["species"])
    # Additional body traits
    for k, v in body.items():
        if k not in ("gender", "build", "age_range", "species", "height") and v:
            parts.append(str(v))
    return ", ".join(parts)


def build_face_desc(spec: dict) -> str:
    face = spec.get("face", {})
    parts = []
    for k, v in face.items():
        if v:
            parts.append(str(v))
    return ", ".join(parts)


def build_clothing_desc(spec: dict) -> str:
    clothing = spec.get("clothing", {})
    parts = []
    if clothing.get("primary"):
        parts.append(clothing["primary"])
    if clothing.get("details"):
        parts.extend(clothing["details"])
    if clothing.get("accessories"):
        parts.extend(clothing["accessories"])
    return ", ".join(parts)


def build_creature_desc(spec: dict) -> str:
    """For non-humanoid creatures — body + distinguishing features."""
    parts = []
    body = spec.get("body", {})
    for k, v in body.items():
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    features = spec.get("distinguishing_features", {})
    for k, v in features.items():
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    return ", ".join(parts)


def assemble_prompt(db, char_id: str, template_id: str,
                    style_override: str = None,
                    extra_tags: str = "") -> str:
    """
    Assemble a final prompt from:
      character_spec  — body, face, hair, clothing
      prompt_template — template string with {variables}
      style_library   — positive tags from style block
    """
    spec = get_character_spec(db, char_id)
    if not spec:
        raise ValueError(f"Character spec '{char_id}' not found")

    tpl = get_prompt_template(db, template_id)
    if not tpl:
        raise ValueError(f"Prompt template '{template_id}' not found")

    # Resolve style
    style_id = style_override or spec.get("style_ref", "default_3d_game")
    style = get_style(db, style_id) or {}
    style_tags = style.get("positive_tags", "")

    # Build variable substitutions
    is_creature = spec.get("category") in ("quadruped", "creature", "fantastical_animal")

    variables = {
        "body_desc":     build_creature_desc(spec) if is_creature else build_body_desc(spec),
        "face_desc":     build_face_desc(spec),
        "hair_desc":     _flatten_spec_field(spec, "hair"),
        "clothing_desc": build_clothing_desc(spec),
        "creature_desc": build_creature_desc(spec),
        "style_tags":    style_tags,
        "extra_tags":    extra_tags,
        "char_name":     spec.get("display_name", char_id),
        "theme":         spec.get("theme", ""),
    }

    # Fill template
    prompt = tpl["template"]
    for key, val in variables.items():
        prompt = prompt.replace(f"{{{key}}}", val)

    # Clean up empty commas, double spaces
    import re
    prompt = re.sub(r",\s*,", ",", prompt)
    prompt = re.sub(r"\s{2,}", " ", prompt)
    prompt = prompt.strip(", ")

    return prompt


def assemble_negative(db, char_id: str, template_id: str,
                      style_override: str = None) -> str:
    """Get negative prompt from template + style block."""
    spec = get_character_spec(db, char_id)
    tpl = get_prompt_template(db, template_id) or {}
    style_id = style_override or (spec or {}).get("style_ref", "default_3d_game")
    style = get_style(db, style_id) or {}

    parts = []
    if tpl.get("negative_template"):
        parts.append(tpl["negative_template"])
    if style.get("negative_tags"):
        parts.append(style["negative_tags"])
    return ", ".join(parts)
