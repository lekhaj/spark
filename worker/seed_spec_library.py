#!/usr/bin/env python3
"""
seed_spec_library.py — Populate MongoDB with initial character specs,
style blocks, reference image bank placeholders, and prompt templates.

Run once (idempotent via upsert):
  python worker/seed_spec_library.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from lib.spec_schema import (
    get_db,
    upsert_character_spec,
    upsert_style,
    upsert_reference,
    upsert_prompt_template,
)

S3_BASE = "https://sparkassets-us.s3.us-east-1.amazonaws.com"


def seed_styles(db):
    """Reusable style-DNA blocks — SAME block used everywhere for consistency."""

    upsert_style(db, {
        "_id": "default_3d_game",
        "name": "Semi-Realistic 3D Game",
        "positive_tags": (
            "semi-realistic, 3d render style, clean topology friendly, "
            "smooth surfaces, minimal noise, game character design, "
            "soft shading, no harsh lighting, consistent material definition"
        ),
        "negative_tags": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, "
            "sketch, photorealistic, noisy, grainy"
        ),
        "description": "Standard 3D game asset style. Semi-realistic, clean surfaces, soft lighting.",
        "version": 1,
    })

    upsert_style(db, {
        "_id": "xianxia_cultivation",
        "name": "Xianxia Cultivation World",
        "positive_tags": (
            "semi-realistic, 3d render style, xianxia aesthetic, "
            "eastern fantasy, clean topology friendly, smooth surfaces, "
            "game character design, soft shading, consistent material definition"
        ),
        "negative_tags": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, "
            "sketch, photorealistic, western armor, medieval european, noisy"
        ),
        "description": "Eastern fantasy / xianxia cultivation world style for game assets.",
        "version": 1,
    })

    upsert_style(db, {
        "_id": "dark_fantasy",
        "name": "Dark Fantasy",
        "positive_tags": (
            "semi-realistic, 3d render style, dark fantasy, "
            "moody lighting, desaturated palette, clean topology friendly, "
            "game character design, consistent material definition"
        ),
        "negative_tags": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, "
            "bright colors, cheerful, photorealistic, noisy"
        ),
        "description": "Dark fantasy aesthetic. Moody, desaturated, semi-realistic.",
        "version": 1,
    })

    print(f"  [styles] 3 styles upserted.")


def seed_character_specs(db):
    """Structured character specs — body/face/hair/clothing as data, not prose."""

    # ── cultivation_youth ─────────────────────────────────────────────────────
    upsert_character_spec(db, {
        "_id": "cultivation_youth",
        "display_name": "Cultivation Youth",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        "body": {
            "gender": "male",
            "build": "lean wiry",
            "age_range": "17-20",
            "height": "170cm",
        },
        "face": {
            "jaw": "sharp",
            "nose": "medium",
            "expression": "calm determined",
            "eyes": "dark sharp eyes",
        },
        "hair": {
            "style": "half-up topknot",
            "color": "black",
            "accessories": ["wooden hairpin"],
        },
        "clothing": {
            "primary": "plain ash-gray linen hanfu robe",
            "details": ["crossover collar", "rope belt"],
            "accessories": ["faded sect patch on chest", "worn leather herb pouch at hip"],
        },

        "pose_default": "T-pose",
        "view_default": "front",

        "flux_generation": {
            "width": 768,
            "height": 1024,
            "steps": 4,
            "guidance_scale": 0.0,
        },
        "controlnet": {
            "type": "bipedal_dual",
            "openpose_weight": 0.85,
            "canny_weight": 0.55,
        },

        "notes": "Outer disciple, mortal-born cultivator. No aura, no glow. Simple clothing.",
    })

    # ── suanni_lion ───────────────────────────────────────────────────────────
    upsert_character_spec(db, {
        "_id": "suanni_lion",
        "display_name": "Suanni Lion",
        "category": "fantastical_animal",
        "subcategory": "quadruped",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        "body": {
            "species": "fantastical lion beast",
            "build": "large muscular feline body",
            "fur": "deep amber-gold fur",
            "size": "large",
        },
        "face": {
            "eyes": "golden amber eyes with vertical slit pupils",
            "horn": "single spiraling bone-white horn on forehead",
            "whiskers": "dragon whiskers",
            "expression": "majestic serene",
        },
        "distinguishing_features": {
            "mane": "flame-shaped rust-orange mane with smoldering curling tips",
            "scales": "dragon-scale patches on chest and knees",
            "spine": "blue-white luminous spine markings",
            "hooves": "cloven rear hooves",
            "tail": "thick tasseled tail",
            "mark": "runic diamond mark at horn base",
            "aura": "faint incense smoke wisps at paws",
        },

        "pose_default": "neutral standing",
        "view_default": "side profile",

        "flux_generation": {
            "width": 1024,
            "height": 768,
            "steps": 4,
            "guidance_scale": 0.0,
        },
        "controlnet": {
            "type": "canny_only",
            "canny_weight": 0.70,
        },

        "notes": "Suanni-inspired. Quadruped — use side profile for Flux, Canny-only for CN.",
    })

    # ── placeholder: generic male warrior ─────────────────────────────────────
    upsert_character_spec(db, {
        "_id": "placeholder_male_warrior",
        "display_name": "Male Warrior (Placeholder)",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "generic_fantasy",
        "style_ref": "default_3d_game",
        "biome_ids": [],

        "body": {
            "gender": "male",
            "build": "athletic muscular",
            "age_range": "25-35",
            "height": "185cm",
        },
        "face": {
            "jaw": "strong square",
            "expression": "determined fierce",
            "eyes": "intense",
            "scars": "small scar on left cheek",
        },
        "hair": {
            "style": "short cropped",
            "color": "dark brown",
            "accessories": [],
        },
        "clothing": {
            "primary": "leather armor vest",
            "details": ["metal pauldrons", "belt with sword sheath"],
            "accessories": ["forearm bracers", "travel cloak"],
        },

        "pose_default": "T-pose",
        "view_default": "front",
        "flux_generation": {"width": 768, "height": 1024, "steps": 4, "guidance_scale": 0.0},
        "controlnet": {"type": "bipedal_dual", "openpose_weight": 0.85, "canny_weight": 0.55},
        "notes": "Generic placeholder. Customize for specific biomes.",
    })

    # ── placeholder: generic female mage ──────────────────────────────────────
    upsert_character_spec(db, {
        "_id": "placeholder_female_mage",
        "display_name": "Female Mage (Placeholder)",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "generic_fantasy",
        "style_ref": "default_3d_game",
        "biome_ids": [],

        "body": {
            "gender": "female",
            "build": "slender",
            "age_range": "20-30",
            "height": "168cm",
        },
        "face": {
            "expression": "calm focused",
            "eyes": "glowing pale blue",
        },
        "hair": {
            "style": "long flowing",
            "color": "silver white",
            "accessories": ["circlet"],
        },
        "clothing": {
            "primary": "dark blue hooded robe",
            "details": ["silver rune embroidery", "wide sleeves", "sash belt"],
            "accessories": ["staff holster on back", "potion pouch"],
        },

        "pose_default": "T-pose",
        "view_default": "front",
        "flux_generation": {"width": 768, "height": 1024, "steps": 4, "guidance_scale": 0.0},
        "controlnet": {"type": "bipedal_dual", "openpose_weight": 0.85, "canny_weight": 0.55},
        "notes": "Generic placeholder. Female mage archetype.",
    })

    # ── placeholder: wolf beast ───────────────────────────────────────────────
    upsert_character_spec(db, {
        "_id": "placeholder_dire_wolf",
        "display_name": "Dire Wolf (Placeholder)",
        "category": "fantastical_animal",
        "subcategory": "quadruped",
        "theme": "dark_fantasy",
        "style_ref": "dark_fantasy",
        "biome_ids": [],

        "body": {
            "species": "dire wolf beast",
            "build": "large powerful canine body",
            "fur": "dark gray-black thick fur",
            "size": "large",
        },
        "face": {
            "eyes": "piercing ice-blue eyes",
            "expression": "alert predatory",
        },
        "distinguishing_features": {
            "mane": "bristled dark mane along spine",
            "fangs": "elongated canine teeth",
            "tail": "thick bushy tail",
            "paws": "oversized paws with dark claws",
        },

        "pose_default": "neutral standing",
        "view_default": "side profile",
        "flux_generation": {"width": 1024, "height": 768, "steps": 4, "guidance_scale": 0.0},
        "controlnet": {"type": "canny_only", "canny_weight": 0.70},
        "notes": "Quadruped placeholder. Use Canny-only ControlNet.",
    })

    # ── placeholder: phoenix bird ─────────────────────────────────────────────
    upsert_character_spec(db, {
        "_id": "placeholder_phoenix",
        "display_name": "Phoenix (Placeholder)",
        "category": "fantastical_animal",
        "subcategory": "avian",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": [],

        "body": {
            "species": "phoenix firebird",
            "build": "elegant avian body",
            "plumage": "gradient crimson-to-gold feathers",
            "size": "medium-large",
        },
        "face": {
            "eyes": "bright golden eyes",
            "crest": "flame-shaped head crest",
            "beak": "sharp curved golden beak",
        },
        "distinguishing_features": {
            "wings": "wide spread wings with trailing flame feathers",
            "tail": "long flowing tail feathers with ember tips",
            "aura": "subtle heat shimmer",
        },

        "pose_default": "perched neutral",
        "view_default": "side profile",
        "flux_generation": {"width": 1024, "height": 768, "steps": 4, "guidance_scale": 0.0},
        "controlnet": {"type": "canny_only", "canny_weight": 0.65},
        "notes": "Avian creature. No skeleton ref available — Canny-only.",
    })

    print(f"  [character_specs] 6 specs upserted (2 active + 4 placeholders).")


def seed_reference_image_bank(db):
    """Placeholder reference images — grouped by Humans / Fantastical Animals.
    S3 keys are placeholders; upload actual reference images to fill them."""

    refs = [
        # ── Humans ────────────────────────────────────────────────────────────
        {
            "_id": "ref_human_male_tpose_front",
            "group": "humans",
            "subgroup": "male_bipedal",
            "pose": "T-pose",
            "view": "front",
            "description": "Male humanoid T-pose front view, arms extended, neutral expression",
            "s3_key": "references/humans/male_tpose_front.png",
            "s3_url": f"{S3_BASE}/references/humans/male_tpose_front.png",
            "tags": ["humanoid", "male", "T-pose", "front", "bipedal", "reference"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Use as init_image reference for male bipedal characters.",
        },
        {
            "_id": "ref_human_female_tpose_front",
            "group": "humans",
            "subgroup": "female_bipedal",
            "pose": "T-pose",
            "view": "front",
            "description": "Female humanoid T-pose front view, arms extended, neutral expression",
            "s3_key": "references/humans/female_tpose_front.png",
            "s3_url": f"{S3_BASE}/references/humans/female_tpose_front.png",
            "tags": ["humanoid", "female", "T-pose", "front", "bipedal", "reference"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Use as init_image reference for female bipedal characters.",
        },
        {
            "_id": "ref_human_male_side",
            "group": "humans",
            "subgroup": "male_bipedal",
            "pose": "neutral standing",
            "view": "side",
            "description": "Male humanoid neutral standing side view",
            "s3_key": "references/humans/male_side.png",
            "s3_url": f"{S3_BASE}/references/humans/male_side.png",
            "tags": ["humanoid", "male", "standing", "side", "bipedal", "reference"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Side view reference for multi-view Stage 3 generation.",
        },
        {
            "_id": "ref_human_male_back",
            "group": "humans",
            "subgroup": "male_bipedal",
            "pose": "neutral standing",
            "view": "back",
            "description": "Male humanoid neutral standing back view",
            "s3_key": "references/humans/male_back.png",
            "s3_url": f"{S3_BASE}/references/humans/male_back.png",
            "tags": ["humanoid", "male", "standing", "back", "bipedal", "reference"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Back view reference for multi-view Stage 3 generation.",
        },

        # ── Fantastical Animals — Felines ─────────────────────────────────────
        {
            "_id": "ref_feline_standing_side",
            "group": "fantastical_animals",
            "subgroup": "feline_quadruped",
            "pose": "neutral standing",
            "view": "side profile",
            "description": "Large feline quadruped, neutral standing, strict side profile, all four legs planted",
            "s3_key": "references/fantastical_animals/feline_standing_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/feline_standing_side.png",
            "tags": ["quadruped", "feline", "lion", "standing", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Side profile ref for lion/feline quads. Landscape orientation.",
        },
        {
            "_id": "ref_feline_standing_front",
            "group": "fantastical_animals",
            "subgroup": "feline_quadruped",
            "pose": "neutral standing",
            "view": "front",
            "description": "Large feline quadruped, neutral standing, front view",
            "s3_key": "references/fantastical_animals/feline_standing_front.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/feline_standing_front.png",
            "tags": ["quadruped", "feline", "lion", "standing", "front", "reference"],
            "metadata": {"width": 768, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Front view ref for lion/feline quads.",
        },

        # ── Fantastical Animals — Canines ─────────────────────────────────────
        {
            "_id": "ref_canine_standing_side",
            "group": "fantastical_animals",
            "subgroup": "canine_quadruped",
            "pose": "neutral standing",
            "view": "side profile",
            "description": "Large canine/wolf quadruped, neutral standing, strict side profile",
            "s3_key": "references/fantastical_animals/canine_standing_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/canine_standing_side.png",
            "tags": ["quadruped", "canine", "wolf", "standing", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Side profile ref for wolf/canine quads.",
        },

        # ── Fantastical Animals — Avian ───────────────────────────────────────
        {
            "_id": "ref_avian_perched_side",
            "group": "fantastical_animals",
            "subgroup": "avian",
            "pose": "perched neutral",
            "view": "side profile",
            "description": "Large bird, perched neutral pose, side profile",
            "s3_key": "references/fantastical_animals/avian_perched_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/avian_perched_side.png",
            "tags": ["avian", "bird", "phoenix", "perched", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Side profile ref for avian creatures.",
        },

        # ── Fantastical Animals — Serpentine / Dragon ─────────────────────────
        {
            "_id": "ref_dragon_eastern_side",
            "group": "fantastical_animals",
            "subgroup": "serpentine",
            "pose": "coiled hover",
            "view": "side profile",
            "description": "Eastern dragon, serpentine body, coiled hovering pose, side view",
            "s3_key": "references/fantastical_animals/dragon_eastern_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/dragon_eastern_side.png",
            "tags": ["serpentine", "dragon", "eastern", "coiled", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Ref for eastern/serpentine dragons. Unique body plan.",
        },
    ]

    for ref in refs:
        upsert_reference(db, ref)

    print(f"  [reference_image_bank] {len(refs)} references upserted (placeholder S3 keys).")


def seed_prompt_templates(db):
    """Tested prompt templates with variable slots."""

    templates = [
        # ── Flux: Humanoid T-Pose ─────────────────────────────────────────────
        {
            "_id": "flux_humanoid_tpose",
            "name": "Flux Humanoid T-Pose (Front)",
            "stage": "flux",
            "category": "humanoid",
            "template": (
                "{body_desc}, T-pose, arms extended horizontally, legs straight, "
                "orthographic front view, symmetrical, centered, "
                "{clothing_desc}, {style_tags}, "
                "clean silhouette, minimal detail, game-ready character design, "
                "white background, flat lighting, full body, head to toe"
            ),
            "variables": ["body_desc", "clothing_desc", "style_tags"],
            "negative_template": "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
            "tested": True,
            "notes": "Flux Schnell optimized. Keep prompt short (<40 words effective). body_desc from spec, clothing_desc from spec.",
            "version": 2,
        },

        # ── Flux: Quadruped Side Profile ──────────────────────────────────────
        {
            "_id": "flux_quadruped_side",
            "name": "Flux Quadruped Side Profile",
            "stage": "flux",
            "category": "quadruped",
            "template": (
                "{creature_desc}, mythical creature, "
                "neutral standing pose, orthographic strict side profile view, "
                "all four legs planted, spine horizontal, symmetrical, centered, "
                "{style_tags}, "
                "simple clean shapes, minimal detail, game-ready creature design, "
                "white background, flat lighting, full body, head to tail"
            ),
            "variables": ["creature_desc", "style_tags"],
            "negative_template": "deformed, extra limbs, text, watermark, background, shadows, blurry, human, bipedal, nsfw",
            "tested": True,
            "notes": "Side profile is better for quadrupeds. Landscape image (1024x768).",
            "version": 2,
        },

        # ── Flux: Avian Side Profile ──────────────────────────────────────────
        {
            "_id": "flux_avian_side",
            "name": "Flux Avian Side Profile",
            "stage": "flux",
            "category": "avian",
            "template": (
                "{creature_desc}, mythical bird, "
                "perched neutral pose, orthographic side profile view, "
                "wings folded, symmetrical, centered, "
                "{style_tags}, "
                "simple clean shapes, minimal detail, game-ready creature design, "
                "white background, flat lighting, full body"
            ),
            "variables": ["creature_desc", "style_tags"],
            "negative_template": "deformed, extra wings, text, watermark, background, shadows, blurry, human, nsfw",
            "tested": False,
            "notes": "Avian variant. Wings folded for clean silhouette.",
            "version": 1,
        },

        # ── Stage 1: Humanoid Pose Lock ───────────────────────────────────────
        {
            "_id": "stage1_humanoid_pose",
            "name": "SD1.5 Stage 1 — Humanoid Pose Lock",
            "stage": "stage1",
            "category": "humanoid",
            "template": (
                "same character, T-pose, arms extended horizontally, "
                "front view, orthographic, symmetrical, clean silhouette, white background"
            ),
            "variables": [],
            "negative_template": "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
            "tested": True,
            "notes": "MINIMAL prompt. SD only corrects pose. Flux provides design. denoise 0.15-0.25.",
            "version": 2,
        },

        # ── Stage 1: Quadruped Pose Lock ──────────────────────────────────────
        {
            "_id": "stage1_quadruped_pose",
            "name": "SD1.5 Stage 1 — Quadruped Pose Lock",
            "stage": "stage1",
            "category": "quadruped",
            "template": (
                "same creature, neutral standing, side profile view, orthographic, "
                "all four legs planted, clean silhouette, white background"
            ),
            "variables": [],
            "negative_template": "deformed, extra limbs, text, watermark, background, shadows, blurry, human, bipedal, nsfw, running, jumping, sitting",
            "tested": True,
            "notes": "Quad Stage 1. Canny-only, no skeleton ref. denoise 0.20.",
            "version": 2,
        },

        # ── Stage 2: Humanoid Detail ──────────────────────────────────────────
        {
            "_id": "stage2_humanoid_detail",
            "name": "SD1.5 Stage 2 — Humanoid Detail",
            "stage": "stage2",
            "category": "humanoid",
            "template": (
                "best quality, masterpiece, {style_tags}, "
                "{body_desc}, {clothing_desc}, {face_desc}, "
                "front view, white background, clean game asset"
            ),
            "variables": ["style_tags", "body_desc", "clothing_desc", "face_desc"],
            "negative_template": (
                "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
                "silk, brocade, aura, glow, heavy ornament"
            ),
            "tested": True,
            "notes": "Stage 2 adds character detail. denoise 0.35, CFG 7.0.",
            "version": 2,
        },

        # ── Stage 2: Quadruped Detail ─────────────────────────────────────────
        {
            "_id": "stage2_quadruped_detail",
            "name": "SD1.5 Stage 2 — Quadruped Detail",
            "stage": "stage2",
            "category": "quadruped",
            "template": (
                "best quality, masterpiece, {style_tags}, "
                "{creature_desc}, "
                "white background, clean game asset"
            ),
            "variables": ["style_tags", "creature_desc"],
            "negative_template": (
                "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
                "mundane animal, wings, human, sitting"
            ),
            "tested": True,
            "notes": "Quad Stage 2 detail. Keep under 77 CLIP tokens. denoise 0.35.",
            "version": 2,
        },

        # ── Stage 3: Multi-view Side ──────────────────────────────────────────
        {
            "_id": "stage3_side_view",
            "name": "SD1.5 Stage 3 — Side View",
            "stage": "stage3",
            "category": "any",
            "template": (
                "same character, side view, profile, {style_tags}, "
                "clean silhouette, white background, game asset"
            ),
            "variables": ["style_tags"],
            "negative_template": "deformed, extra limbs, text, watermark, blurry, nsfw",
            "tested": False,
            "notes": "Multi-view side. Uses FLUX concept (not SD output) as init. denoise 0.4-0.5.",
            "version": 1,
        },

        # ── Stage 3: Multi-view Back ──────────────────────────────────────────
        {
            "_id": "stage3_back_view",
            "name": "SD1.5 Stage 3 — Back View",
            "stage": "stage3",
            "category": "any",
            "template": (
                "same character, back view, rear view, {style_tags}, "
                "clean silhouette, white background, game asset"
            ),
            "variables": ["style_tags"],
            "negative_template": "deformed, extra limbs, text, watermark, blurry, nsfw, face visible",
            "tested": False,
            "notes": "Multi-view back. Uses FLUX concept as init. denoise 0.4-0.5.",
            "version": 1,
        },
    ]

    for tpl in templates:
        upsert_prompt_template(db, tpl)

    print(f"  [prompt_templates] {len(templates)} templates upserted.")


def dry_run():
    """Validate all data structures locally without MongoDB."""
    print("=" * 60)
    print("  Seed Spec Library — DRY RUN (no MongoDB)")
    print("=" * 60)

    # Collect all docs that would be inserted
    import inspect
    import types

    # We'll build a fake in-memory db using dicts
    class FakeCollection:
        def __init__(self, name):
            self.name = name
            self.docs = {}
        def replace_one(self, filt, doc, upsert=False):
            self.docs[doc["_id"]] = doc
        def find_one(self, filt):
            return self.docs.get(filt.get("_id"))
        def find(self, filt=None):
            class Cursor:
                def __init__(self, docs): self._docs = list(docs.values())
                def sort(self, *a): return self
                def limit(self, *a): return self
                def __iter__(self): return iter(self._docs)
            return Cursor(self.docs)
        def count_documents(self, filt):
            return len(self.docs)
        def distinct(self, field):
            return list(set(d.get(field) for d in self.docs.values() if d.get(field)))

    class FakeDB:
        def __init__(self):
            self._colls = {}
        def __getattr__(self, name):
            if name.startswith("_"):
                return super().__getattribute__(name)
            if name not in self._colls:
                self._colls[name] = FakeCollection(name)
            return self._colls[name]

    db = FakeDB()

    seed_styles(db)
    seed_character_specs(db)
    seed_reference_image_bank(db)
    seed_prompt_templates(db)

    print("\n  Validation (in-memory):")
    print(f"    character_specs:      {db.character_specs.count_documents({})}")
    print(f"    style_library:        {db.style_library.count_documents({})}")
    print(f"    reference_image_bank: {db.reference_image_bank.count_documents({})}")
    print(f"    prompt_templates:     {db.prompt_templates.count_documents({})}")

    # Patch spec_schema to use our fake db for assembly
    import lib.spec_schema as schema
    _orig_get = schema.get_character_spec
    _orig_style = schema.get_style
    _orig_tpl = schema.get_prompt_template

    schema.get_character_spec = lambda db2, cid: db.character_specs.find_one({"_id": cid})
    schema.get_style = lambda db2, sid: db.style_library.find_one({"_id": sid})
    schema.get_prompt_template = lambda db2, tid: db.prompt_templates.find_one({"_id": tid})

    print("\n  Test prompt assembly (in-memory):")
    for char_id, tpl_id in [
        ("cultivation_youth", "flux_humanoid_tpose"),
        ("suanni_lion", "flux_quadruped_side"),
        ("cultivation_youth", "stage2_humanoid_detail"),
        ("suanni_lion", "stage2_quadruped_detail"),
        ("placeholder_male_warrior", "flux_humanoid_tpose"),
        ("placeholder_dire_wolf", "flux_quadruped_side"),
    ]:
        try:
            prompt = schema.assemble_prompt(db, char_id, tpl_id)
            neg = schema.assemble_negative(db, char_id, tpl_id)
            words = len(prompt.split())
            clip_est = int(words * 1.3)
            warn = " CLIP OVERFLOW!" if clip_est > 77 and "flux" not in tpl_id else ""
            print(f"\n    [{char_id}] template={tpl_id}")
            print(f"    Prompt ({words} words, ~{clip_est} CLIP tokens{warn}):")
            print(f"      {prompt}")
            print(f"    Negative: {neg[:120]}...")
        except Exception as e:
            print(f"\n    [{char_id}] template={tpl_id} ERROR: {e}")

    # Restore
    schema.get_character_spec = _orig_get
    schema.get_style = _orig_style
    schema.get_prompt_template = _orig_tpl

    print("\n  DRY RUN complete. All data structures valid.")
    print("  To write to MongoDB: python worker/seed_spec_library.py  (no --dry-run)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data locally without MongoDB")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    print("=" * 60)
    print("  Seed Spec Library — MongoDB")
    print("=" * 60)

    try:
        db = get_db()
        # Quick connectivity test
        db.command("ping")
        print("  [MongoDB] Connected OK")
    except Exception as e:
        print(f"\n  [ERROR] Cannot connect to MongoDB: {e}")
        print("  Make sure the CPU instance (18.207.13.85) is running.")
        print("  Or run with --dry-run to validate locally.\n")
        sys.exit(1)

    seed_styles(db)
    seed_character_specs(db)
    seed_reference_image_bank(db)
    seed_prompt_templates(db)

    # Quick validation
    print("\n  Validation:")
    print(f"    character_specs:      {db.character_specs.count_documents({})}")
    print(f"    style_library:        {db.style_library.count_documents({})}")
    print(f"    reference_image_bank: {db.reference_image_bank.count_documents({})}")
    print(f"    prompt_templates:     {db.prompt_templates.count_documents({})}")

    # Test prompt assembly
    from lib.spec_schema import assemble_prompt, assemble_negative
    print("\n  Test prompt assembly:")
    for char_id, tpl_id in [
        ("cultivation_youth", "flux_humanoid_tpose"),
        ("suanni_lion", "flux_quadruped_side"),
        ("cultivation_youth", "stage2_humanoid_detail"),
    ]:
        prompt = assemble_prompt(db, char_id, tpl_id)
        neg = assemble_negative(db, char_id, tpl_id)
        words = len(prompt.split())
        print(f"\n    [{char_id}] template={tpl_id}")
        print(f"    Prompt ({words} words): {prompt[:150]}...")
        print(f"    Negative: {neg[:100]}...")

    print("\n  Done. Run pipeline_dashboard.py to view in Gradio.")


if __name__ == "__main__":
    main()
