#!/usr/bin/env python3
"""
seed_spec_library.py — Populate MongoDB with character specs, style blocks,
reference image bank, and prompt templates.

Run once (idempotent via upsert):
  python worker/seed_spec_library.py

Validate locally without MongoDB:
  python worker/seed_spec_library.py --dry-run
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


# ══════════════════════════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════════════════════════

def seed_styles(db):
    """
    Style DNA blocks — SAME block used in every generation for consistency.
    Think of these as locked CSS variables for the art pipeline.
    """

    # ── Master game-asset style ───────────────────────────────────────────────
    upsert_style(db, {
        "_id": "default_3d_game",
        "name": "Semi-Realistic 3D Game Asset",
        "positive_tags": (
            "semi-realistic 3D render, game character design, PBR materials, "
            "clean polygon topology, smooth fabric and skin surfaces, "
            "game-ready character asset, soft studio lighting, subtle rim light, "
            "consistent material definition, clear full-body silhouette, "
            "no background, white backdrop"
        ),
        "negative_tags": (
            "anime style, 2D illustration, cel shading, cartoon, manga, "
            "flat art, watercolor, sketch, painterly, photorealistic portrait, "
            "noisy, grainy, harsh shadows, bloom, particle effects, "
            "motion blur, extra limbs, deformed anatomy, text, watermark"
        ),
        "description": "Master style for all 3D game character assets. Semi-realistic, clean PBR, soft lighting.",
        "use_case": "Default for any character type. Overridden by theme-specific styles.",
        "version": 2,
    })

    # ── Eastern fantasy / xianxia cultivation ────────────────────────────────
    upsert_style(db, {
        "_id": "xianxia_cultivation",
        "name": "Xianxia Cultivation — 3D Game",
        "positive_tags": (
            "semi-realistic 3D render, eastern fantasy xianxia aesthetic, "
            "cultivation world atmosphere, Chinese historical fantasy, "
            "game character design, PBR materials, clean polygon topology, "
            "smooth silk and linen fabric surfaces, game-ready asset, "
            "soft studio lighting, subtle rim light, clear full-body silhouette, "
            "no background, white backdrop"
        ),
        "negative_tags": (
            "anime style, 2D illustration, cel shading, cartoon, manga, "
            "flat art, watercolor, sketch, photorealistic, "
            "western armor, medieval european, roman, norse, "
            "dramatic aura glow, energy explosion, particle effects, "
            "noisy, grainy, background clutter, extra limbs, deformed, "
            "text, watermark"
        ),
        "description": "Eastern fantasy xianxia style. Flowing fabric, clean game mesh, NO western aesthetics.",
        "use_case": "cultivation_youth, suanni_lion, all xianxia-theme characters.",
        "version": 2,
    })

    # ── Dark fantasy ──────────────────────────────────────────────────────────
    upsert_style(db, {
        "_id": "dark_fantasy",
        "name": "Dark Fantasy — 3D Game",
        "positive_tags": (
            "semi-realistic 3D render, dark fantasy aesthetic, "
            "moody desaturated palette, game character design, PBR materials, "
            "clean topology, slightly weathered surfaces, game-ready asset, "
            "soft ambient lighting, dramatic rim light, clear silhouette, "
            "no background, white backdrop"
        ),
        "negative_tags": (
            "anime style, 2D illustration, cel shading, cartoon, manga, "
            "flat art, watercolor, sketch, photorealistic, bright colors, "
            "cheerful, eastern fantasy, noisy, grainy, extra limbs, "
            "text, watermark"
        ),
        "description": "Dark fantasy aesthetic. Moody, desaturated, semi-realistic.",
        "use_case": "Dire wolf, dark knight, horror creatures.",
        "version": 2,
    })

    print("  [styles] 3 styles upserted.")


# ══════════════════════════════════════════════════════════════════════════════
#  CHARACTER SPECS
# ══════════════════════════════════════════════════════════════════════════════

def seed_character_specs(db):
    """
    Structured character specs — body/face/hair/clothing as DATA, not prose.
    Prompts are assembled programmatically from these fields.

    Design principle:
      spec + style + template → prompt
      Never hand-write prompts.
    """

    # ══════════════════════════════════════════════════════════════════════════
    #  CHARACTER 1 — Wei Liang, Outer Disciple (Donghua Xianxia Youth)
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Inspired by: Tang San (Soul Land), Xiao Yan (Battle Through the Heavens),
    # Yun Che (Against the Gods), Lin Feng (Peerless Martial God)
    # Archetype: mortal-born outer disciple, young, sharp-featured, humble clothing
    # No aura, no glow, no sect favor — just a young man with potential
    # ══════════════════════════════════════════════════════════════════════════
    upsert_character_spec(db, {
        "_id": "cultivation_youth",
        "display_name": "Wei Liang — Outer Disciple",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        # ── Physical body ──────────────────────────────────────────────────
        "body": {
            "gender": "male",
            "build": "lean wiry athletic",
            "age_range": "17-19",
            "height": "172cm",
            "skin": "fair east asian complexion, slightly sun-touched",
        },

        # ── Face ───────────────────────────────────────────────────────────
        # Based on typical donghua hero: sharp but not intimidating,
        # calm with latent intensity — think Tang San or Xiao Yan in early arcs
        "face": {
            "structure": "oval face with defined cheekbones",
            "jaw": "slightly sharp clean jaw",
            "nose": "straight medium nose",
            "expression": "calm composed, slight focused intensity",
            "eyes": "dark black almond-shaped eyes, sharp focused gaze",
            "eyebrows": "straight dark brows, not heavy",
            "lips": "neutral, closed",
        },

        # ── Hair ───────────────────────────────────────────────────────────
        # Standard donghua hero: long black hair, partially tied
        # Wooden hairpin = simple, not wealthy
        "hair": {
            "style": "long black hair, upper half tied into loose topknot",
            "color": "jet black",
            "length": "mid-back length when loose",
            "texture": "straight, slightly coarse",
            "loose_strands": "a few loose strands framing the face at temples",
            "accessories": ["plain undecorated wooden hairpin securing topknot"],
        },

        # ── Clothing ───────────────────────────────────────────────────────
        # Outer disciple = low rank. Plain cloth, minimal ornamentation.
        # NOT silk, NOT brocade, NOT glowing spiritual items.
        # Inspired by: early-arc Tang San, pre-breakthrough Xiao Yan
        "clothing": {
            "primary": "ash-gray linen hanfu outer robe with wide crossover collar",
            "inner": "off-white linen inner robe, visible at collar and cuffs",
            "belt": "braided hemp rope belt cinched at waist",
            "details": [
                "plain weave matte fabric, no pattern",
                "slightly worn and faded from washing",
                "wide sleeves cinched with cloth tie at wrist",
                "ankle-length hemline, slight fraying at cuffs",
            ],
            "accessories": [
                "faded outer-disciple sect emblem patch on left chest",
                "small worn brown leather herb-gathering pouch at left hip",
                "simple dark cloth shoes, worn at the soles",
            ],
        },

        # ── Color palette (locked for consistency) ─────────────────────────
        "color_palette": {
            "outer_robe": "ash gray #8A8880",
            "inner_robe": "off-white ivory #F0EBE0",
            "belt": "natural hemp tan #A89070",
            "hair": "jet black #1A1A1A",
            "skin": "fair warm ivory #F5DFC0",
            "sect_patch_accent": "faded sage green #7A9478",
            "shoe": "dark charcoal gray #3A3A38",
        },

        # ── Generation settings ────────────────────────────────────────────
        "pose_default": "T-pose",
        "view_default": "front",

        "flux_generation": {
            "width": 768,
            "height": 1024,
            "steps": 4,
            "guidance_scale": 0.0,
            "model": "FLUX.1-schnell",
        },
        "sd_stage1": {
            "denoise": 0.20,
            "cfg": 7.5,
            "controlnet": "openpose + canny dual",
            "openpose_weight": 0.85,
            "canny_weight": 0.55,
        },
        "sd_stage2": {
            "denoise": 0.35,
            "cfg": 7.0,
            "controlnet": "openpose + canny dual",
        },

        # ── Reference images (for IP-Adapter / img2img conditioning) ──────
        "reference_ids": [
            "ref_human_male_tpose_front",
            "ref_human_male_side",
            "ref_human_male_back",
        ],

        "notes": (
            "Outer disciple archetype. Mortal-born, no sect favor, no spiritual energy visible. "
            "NO glowing aura. NO elaborate embroidery. NO silk or brocade. "
            "Simple worn daily clothing. Heroic potential hidden under humble exterior. "
            "Donghua refs: Tang San (Soul Land), Xiao Yan (BTTH), early-arc Lin Feng."
        ),
        "lore": "Wei Liang, age 17. Swept floors for 2 years before outer-disciple trial. Carried no talent spiritual root — or so they thought.",
    })

    # ══════════════════════════════════════════════════════════════════════════
    #  CHARACTER 2 — Suanni, Sacred Dragon-Lion (Fantasy Beast)
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Based on: Suanni (狻猊) — 5th son of the dragon, lion-form, fire element
    # Hybrid with Pixiu (貔貅) — one horn (male), small feathered wings
    # Result: noble divine lion with horn, small wings, jade scale patches, flame mane
    # Visual refs: Chinese incense burner Suanni statues, Pixiu amulets, Foo Dog guardian lions
    # ══════════════════════════════════════════════════════════════════════════
    upsert_character_spec(db, {
        "_id": "suanni_lion",
        "display_name": "Suanni — Sacred Dragon-Lion",
        "category": "fantastical_animal",
        "subcategory": "quadruped",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        # ── Body ───────────────────────────────────────────────────────────
        "body": {
            "species": "suanni divine lion beast",
            "build": "large powerful feline body, deep broad chest",
            "fur": "deep amber-gold short dense coat, sleek except mane",
            "size": "large, shoulder height of a war horse",
            "legs": "four thick muscular legs with large cushioned paws",
            "tail_base": "thick at root",
        },

        # ── Face features (fed into face_desc for templates that use it) ──
        "face": {
            "eyes": "large golden amber eyes with vertical dragon-slit pupils",
            "horn": "single spiraling bone-white horn rising from forehead, 25cm",
            "whiskers": "six long white dragon whiskers extending from muzzle",
            "muzzle": "broad lion muzzle, slightly flattened",
            "expression": "noble and serene, mouth closed, dignified gaze",
            "ears": "rounded lion ears with amber fur inner lining",
        },

        # ── Distinguishing features (fed into creature_desc) ──────────────
        # These are the visually distinctive elements that separate Suanni
        # from a mundane lion. KEY for Flux prompt.
        "distinguishing_features": {
            "mane": "thick flame-shaped mane, rust-amber color, tips curled upward like small flames",
            "wings": "two compact feathered wings on upper back, folded closed naturally, primary feathers gold-tipped white",
            "chest_scales": "overlapping jade-tinted dragon scales on chest and both shoulder joints",
            "knee_scales": "small protective jade scale clusters at knee joints",
            "spine_markings": "row of cool blue-white bioluminescent dots running along spine ridge",
            "tail": "thick tail with large amber-gold tasseled tuft at tip",
            "rear_hooves": "cloven rear hooves instead of paws, showing dragon lineage",
            "front_claws": "dark onyx slightly curved retracted claws on front paws",
            "horn_base": "small runic diamond-shaped mark where horn meets skull",
        },

        # ── Color palette (locked for consistency) ─────────────────────────
        "color_palette": {
            "body_fur": "deep amber gold #C48B3A",
            "mane": "rust orange-amber #B85C2A",
            "wing_feathers": "dusty ivory with gold tips #F0E8D0 / #D4A840",
            "scales_chest": "jade teal #5B8C7A",
            "spine_marks": "cool blue-white #A8D0E8",
            "horn": "bone white #F0ECD8",
            "eyes": "molten amber #E8A820",
            "claws": "dark onyx #2A2828",
            "hooves": "dark charcoal brown #4A3A30",
        },

        # ── Generation settings ────────────────────────────────────────────
        "pose_default": "neutral standing",
        "view_default": "side profile",

        "flux_generation": {
            "width": 1024,
            "height": 768,
            "steps": 4,
            "guidance_scale": 0.0,
            "model": "FLUX.1-schnell",
            "note": "Landscape. Side profile shows wings + horn + mane + tail cleanly.",
        },
        "sd_stage1": {
            "denoise": 0.20,
            "cfg": 7.5,
            "controlnet": "canny only",
            "canny_weight": 0.70,
            "note": "Quadruped — no skeleton ref available. Canny-only for edge guidance.",
        },
        "sd_stage2": {
            "denoise": 0.35,
            "cfg": 7.0,
            "controlnet": "canny only",
            "token_warning": "Keep assembled prompt under 77 CLIP tokens for SD1.5",
        },

        # ── Reference images ───────────────────────────────────────────────
        "reference_ids": [
            "ref_feline_standing_side",
            "ref_feline_standing_front",
        ],

        "notes": (
            "Suanni (狻猊) + Pixiu hybrid design. "
            "Suanni: 5th son of dragon, lion form, fire/smoke association, serene noble temperament. "
            "Pixiu: one horn (male), feathered wings, fierce guardian energy. "
            "Wings are SMALL and FOLDED for clean silhouette in side-view generation. "
            "Use side profile for Flux (shows horn, wing, mane, tail cleanly). "
            "Game asset: jade scales add color variety and break up plain fur surface."
        ),
        "lore": "Ancient guardian beast that roamed the cultivation world before the Qi Depletion. Drawn to incense smoke and the scent of cultivation pills.",
    })

    # ── PLACEHOLDERS ──────────────────────────────────────────────────────────
    # These will be fleshed out in future iterations

    upsert_character_spec(db, {
        "_id": "placeholder_male_warrior",
        "display_name": "Male Warrior (Placeholder)",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "generic_fantasy",
        "style_ref": "default_3d_game",
        "biome_ids": [],
        "body": {"gender": "male", "build": "athletic muscular", "age_range": "25-35", "height": "185cm"},
        "face": {"jaw": "strong square jaw", "expression": "determined fierce", "eyes": "intense dark", "scar": "small scar on left cheek"},
        "hair": {"style": "short cropped", "color": "dark brown", "accessories": []},
        "clothing": {
            "primary": "leather armor vest with metal pauldrons",
            "details": ["reinforced leather vambraces", "wide combat belt"],
            "accessories": ["sword sheath on back", "travel cloak"],
        },
        "pose_default": "T-pose", "view_default": "front",
        "flux_generation": {"width": 768, "height": 1024, "steps": 4, "guidance_scale": 0.0},
        "notes": "Generic warrior placeholder. Customize per biome.",
    })

    upsert_character_spec(db, {
        "_id": "placeholder_female_mage",
        "display_name": "Female Mage (Placeholder)",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "generic_fantasy",
        "style_ref": "default_3d_game",
        "biome_ids": [],
        "body": {"gender": "female", "build": "slender", "age_range": "20-28", "height": "165cm"},
        "face": {"expression": "calm focused", "eyes": "glowing pale blue", "features": "delicate sharp"},
        "hair": {"style": "long flowing", "color": "silver white", "accessories": ["thin silver circlet"]},
        "clothing": {
            "primary": "deep indigo hooded robe with silver rune trim",
            "details": ["wide flowing sleeves", "sash belt with rune clasp"],
            "accessories": ["staff holster on back", "small potion pouch at hip"],
        },
        "pose_default": "T-pose", "view_default": "front",
        "flux_generation": {"width": 768, "height": 1024, "steps": 4, "guidance_scale": 0.0},
        "notes": "Generic mage placeholder.",
    })

    upsert_character_spec(db, {
        "_id": "placeholder_dire_wolf",
        "display_name": "Dire Wolf (Placeholder)",
        "category": "fantastical_animal",
        "subcategory": "quadruped",
        "theme": "dark_fantasy",
        "style_ref": "dark_fantasy",
        "biome_ids": [],
        "body": {"species": "dire wolf", "build": "large powerful canine", "fur": "dark gray-black thick fur", "size": "large"},
        "face": {"eyes": "piercing ice-blue eyes", "fangs": "elongated white fangs", "expression": "alert predatory"},
        "distinguishing_features": {
            "mane": "bristled dark mane along spine and neck",
            "tail": "thick bushy tail",
            "paws": "oversized paws with dark hooked claws",
        },
        "pose_default": "neutral standing", "view_default": "side profile",
        "flux_generation": {"width": 1024, "height": 768, "steps": 4, "guidance_scale": 0.0},
        "notes": "Quadruped placeholder. Canny-only ControlNet.",
    })

    upsert_character_spec(db, {
        "_id": "placeholder_phoenix",
        "display_name": "Phoenix (Placeholder)",
        "category": "fantastical_animal",
        "subcategory": "avian",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": [],
        "body": {"species": "phoenix firebird", "build": "elegant large avian", "plumage": "gradient crimson-to-gold feathers", "size": "large"},
        "face": {"eyes": "bright golden eyes", "crest": "flame-shaped head crest", "beak": "sharp curved golden beak"},
        "distinguishing_features": {
            "wings": "wide spread wings with trailing flame-tipped feathers",
            "tail": "long flowing tail feathers with ember glow at tips",
        },
        "pose_default": "perched neutral", "view_default": "side profile",
        "flux_generation": {"width": 1024, "height": 768, "steps": 4, "guidance_scale": 0.0},
        "notes": "Avian placeholder. No skeleton ref — Canny only.",
    })

    print("  [character_specs] 6 specs upserted (2 detailed + 4 placeholders).")


# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCE IMAGE BANK
# ══════════════════════════════════════════════════════════════════════════════

def seed_reference_image_bank(db):
    """
    Reference image bank — grouped by character type.
    S3 keys are placeholders. Upload real reference images to fill them.
    Used for: IP-Adapter conditioning, img2img reference, comparison.
    """

    refs = [
        # ── Humans — Male Bipedal ──────────────────────────────────────────
        {
            "_id": "ref_human_male_tpose_front",
            "group": "humans",
            "subgroup": "male_bipedal",
            "pose": "T-pose",
            "view": "front",
            "description": "Male humanoid T-pose front view. Arms extended horizontal, legs straight, feet apart, neutral expression. Used as init_image for male bipedal generation.",
            "s3_key": "references/humans/male_tpose_front.png",
            "s3_url": f"{S3_BASE}/references/humans/male_tpose_front.png",
            "tags": ["humanoid", "male", "T-pose", "front", "bipedal", "reference"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Stage 1 init_image for male bipedal characters. Provides pose skeleton.",
        },
        {
            "_id": "ref_human_female_tpose_front",
            "group": "humans",
            "subgroup": "female_bipedal",
            "pose": "T-pose",
            "view": "front",
            "description": "Female humanoid T-pose front view. Arms extended, legs straight, neutral.",
            "s3_key": "references/humans/female_tpose_front.png",
            "s3_url": f"{S3_BASE}/references/humans/female_tpose_front.png",
            "tags": ["humanoid", "female", "T-pose", "front", "bipedal", "reference"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Stage 1 init_image for female bipedal characters.",
        },
        {
            "_id": "ref_human_male_side",
            "group": "humans",
            "subgroup": "male_bipedal",
            "pose": "neutral standing",
            "view": "side",
            "description": "Male humanoid neutral standing side view. Left or right profile, arms relaxed.",
            "s3_key": "references/humans/male_side.png",
            "s3_url": f"{S3_BASE}/references/humans/male_side.png",
            "tags": ["humanoid", "male", "standing", "side", "bipedal", "reference", "multiview"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Stage 3 multi-view: side generation init. Use Flux concept as base, denoise 0.4-0.5.",
        },
        {
            "_id": "ref_human_male_back",
            "group": "humans",
            "subgroup": "male_bipedal",
            "pose": "neutral standing",
            "view": "back",
            "description": "Male humanoid neutral standing back view. Arms relaxed at sides.",
            "s3_key": "references/humans/male_back.png",
            "s3_url": f"{S3_BASE}/references/humans/male_back.png",
            "tags": ["humanoid", "male", "standing", "back", "bipedal", "reference", "multiview"],
            "metadata": {"width": 768, "height": 1024, "format": "png", "source": "placeholder"},
            "usage_notes": "Stage 3 multi-view: back generation. Use Flux concept, denoise 0.4-0.5.",
        },

        # ── Fantastical Animals — Feline Quadruped ─────────────────────────
        {
            "_id": "ref_feline_standing_side",
            "group": "fantastical_animals",
            "subgroup": "feline_quadruped",
            "pose": "neutral standing",
            "view": "side profile",
            "description": "Large feline quadruped. Neutral standing, strict side profile, all four legs planted flat. Head level. Used for lion/tiger creature generation.",
            "s3_key": "references/fantastical_animals/feline_standing_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/feline_standing_side.png",
            "tags": ["quadruped", "feline", "lion", "tiger", "standing", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Canny-only reference for Suanni lion. Landscape 1024x768. Shows silhouette cleanly.",
        },
        {
            "_id": "ref_feline_standing_front",
            "group": "fantastical_animals",
            "subgroup": "feline_quadruped",
            "pose": "neutral standing",
            "view": "front",
            "description": "Large feline quadruped. Neutral standing, front-facing view.",
            "s3_key": "references/fantastical_animals/feline_standing_front.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/feline_standing_front.png",
            "tags": ["quadruped", "feline", "lion", "standing", "front", "reference"],
            "metadata": {"width": 768, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Front view for feline quads. Square orientation.",
        },

        # ── Fantastical Animals — Canine ───────────────────────────────────
        {
            "_id": "ref_canine_standing_side",
            "group": "fantastical_animals",
            "subgroup": "canine_quadruped",
            "pose": "neutral standing",
            "view": "side profile",
            "description": "Large canine / wolf quadruped. Neutral standing side profile.",
            "s3_key": "references/fantastical_animals/canine_standing_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/canine_standing_side.png",
            "tags": ["quadruped", "canine", "wolf", "standing", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Side profile ref for wolf/canine quads.",
        },

        # ── Fantastical Animals — Avian ────────────────────────────────────
        {
            "_id": "ref_avian_perched_side",
            "group": "fantastical_animals",
            "subgroup": "avian",
            "pose": "perched neutral",
            "view": "side profile",
            "description": "Large bird. Perched neutral, wings folded, side profile.",
            "s3_key": "references/fantastical_animals/avian_perched_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/avian_perched_side.png",
            "tags": ["avian", "bird", "phoenix", "perched", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Side profile for avian creatures.",
        },

        # ── Fantastical Animals — Serpentine ───────────────────────────────
        {
            "_id": "ref_dragon_eastern_side",
            "group": "fantastical_animals",
            "subgroup": "serpentine",
            "pose": "coiled hover",
            "view": "side profile",
            "description": "Eastern serpentine dragon. Coiled S-shape hovering, side view.",
            "s3_key": "references/fantastical_animals/dragon_eastern_side.png",
            "s3_url": f"{S3_BASE}/references/fantastical_animals/dragon_eastern_side.png",
            "tags": ["serpentine", "dragon", "eastern", "coiled", "side", "reference"],
            "metadata": {"width": 1024, "height": 768, "format": "png", "source": "placeholder"},
            "usage_notes": "Ref for eastern serpentine dragons.",
        },
    ]

    for ref in refs:
        upsert_reference(db, ref)

    print(f"  [reference_image_bank] {len(refs)} references upserted (placeholder S3 keys).")


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

def seed_prompt_templates(db):
    """
    Tested prompt templates with {variable} slots.
    Variables are filled by assemble_prompt() from character spec + style.

    Slots available:
      {body_desc}     — from build_body_desc()
      {face_desc}     — from build_face_desc()
      {hair_desc}     — from _flatten_spec_field(spec, "hair")
      {clothing_desc} — from build_clothing_desc()
      {creature_desc} — from build_creature_desc() [body + distinguishing_features]
      {style_tags}    — from style_library positive_tags
      {extra_tags}    — passed at call time
      {char_name}     — spec display_name
      {theme}         — spec theme
    """

    templates = [

        # ── FLUX: Humanoid T-Pose ─────────────────────────────────────────
        # Best Flux template for bipedal game characters.
        # Includes face + hair for detailed donghua character generation.
        # FLUX uses T5 encoder — no 77-token limit. Keep rich.
        {
            "_id": "flux_humanoid_tpose",
            "name": "Flux — Humanoid T-Pose (Front, Full Detail)",
            "stage": "flux",
            "category": "humanoid",
            "template": (
                "full body character, {body_desc}, "
                "{face_desc}, {hair_desc}, "
                "{clothing_desc}, "
                "T-pose arms extended horizontally legs straight, "
                "front view orthographic projection symmetrical centered, "
                "white background flat diffuse studio lighting full body head to toe, "
                "{style_tags}"
            ),
            "variables": ["body_desc", "face_desc", "hair_desc", "clothing_desc", "style_tags"],
            "negative_template": (
                "deformed anatomy, extra limbs, missing limbs, merged limbs, "
                "text, watermark, background scenery, colored background, "
                "shadows, harsh lighting, blurry, nsfw, split image, collage"
            ),
            "tested": True,
            "notes": (
                "FLUX.1-schnell optimized. T5 encoder — no token limit. "
                "Includes face + hair for detailed character output. "
                "Settings: 768x1024, steps=4, guidance=0.0"
            ),
            "version": 3,
        },

        # ── FLUX: Quadruped Side Profile ──────────────────────────────────
        # Side profile is CRITICAL for quadrupeds:
        # - Shows horn, mane, wings, tail all in one view
        # - Clean silhouette for Trellis 3D input
        # - Landscape orientation (1024x768)
        {
            "_id": "flux_quadruped_side",
            "name": "Flux — Quadruped Side Profile (Full Detail)",
            "stage": "flux",
            "category": "quadruped",
            "template": (
                "full body mythical creature, {creature_desc}, "
                "{face_desc}, "
                "neutral standing pose orthographic strict side profile view, "
                "all four legs planted spine horizontal centered, "
                "white background flat diffuse studio lighting full body head to tail, "
                "{style_tags}"
            ),
            "variables": ["creature_desc", "face_desc", "style_tags"],
            "negative_template": (
                "deformed anatomy, extra limbs, fused limbs, human figure, bipedal, "
                "text, watermark, background scenery, colored background, "
                "shadows, harsh lighting, blurry, nsfw, running, jumping, sitting, crouching"
            ),
            "tested": True,
            "notes": (
                "Side profile shows horn + wing + mane + tail cleanly. "
                "Settings: 1024x768 landscape, steps=4, guidance=0.0. "
                "face_desc used to include horn + eyes in prompt."
            ),
            "version": 3,
        },

        # ── FLUX: Avian Side Profile ──────────────────────────────────────
        {
            "_id": "flux_avian_side",
            "name": "Flux — Avian Side Profile",
            "stage": "flux",
            "category": "avian",
            "template": (
                "full body mythical bird, {creature_desc}, "
                "{face_desc}, "
                "perched neutral pose orthographic side profile view wings folded, "
                "white background flat diffuse studio lighting full body, "
                "{style_tags}"
            ),
            "variables": ["creature_desc", "face_desc", "style_tags"],
            "negative_template": (
                "deformed, extra wings, text, watermark, background, shadows, blurry, human, nsfw"
            ),
            "tested": False,
            "notes": "Avian variant. Wings folded for clean silhouette.",
            "version": 1,
        },

        # ── STAGE 1: Humanoid Pose Lock ───────────────────────────────────
        # MINIMAL prompt — SD only corrects pose geometry.
        # Flux image provides the design. Do NOT describe the character here.
        {
            "_id": "stage1_humanoid_pose",
            "name": "SD1.5 Stage 1 — Humanoid Pose Lock",
            "stage": "stage1",
            "category": "humanoid",
            "template": (
                "full body character, T-pose, arms extended horizontally, "
                "legs straight, front view, orthographic, symmetrical, "
                "white background, flat lighting, clean silhouette"
            ),
            "variables": [],
            "negative_template": (
                "deformed, extra limbs, bent arms, bent knees, text, watermark, "
                "background, shadows, blurry, nsfw, sitting, crouching"
            ),
            "tested": True,
            "notes": (
                "KEEP MINIMAL. SD only corrects pose — Flux provides all design. "
                "denoise: 0.15-0.25. ControlNet: OpenPose 0.85 + Canny 0.55."
            ),
            "version": 2,
        },

        # ── STAGE 1: Quadruped Pose Lock ──────────────────────────────────
        {
            "_id": "stage1_quadruped_pose",
            "name": "SD1.5 Stage 1 — Quadruped Pose Lock",
            "stage": "stage1",
            "category": "quadruped",
            "template": (
                "full body creature, neutral standing, "
                "side profile view, orthographic, "
                "all four legs planted, clean silhouette, white background"
            ),
            "variables": [],
            "negative_template": (
                "deformed, extra limbs, fused limbs, text, watermark, "
                "background, shadows, blurry, human, bipedal, nsfw, "
                "running, jumping, sitting, crouching, flying"
            ),
            "tested": True,
            "notes": (
                "Quadruped Stage 1. Canny-only, no skeleton ref available. "
                "denoise: 0.20. Canny weight: 0.70."
            ),
            "version": 2,
        },

        # ── STAGE 2: Humanoid Detail ──────────────────────────────────────
        # Adds character-specific details on top of corrected pose.
        # CLIP limit: 77 tokens. Keep prompt short.
        {
            "_id": "stage2_humanoid_detail",
            "name": "SD1.5 Stage 2 — Humanoid Detail",
            "stage": "stage2",
            "category": "humanoid",
            "template": (
                "best quality, {style_tags}, "
                "{body_desc}, {clothing_desc}, "
                "front view, white background, clean game asset"
            ),
            "variables": ["style_tags", "body_desc", "clothing_desc"],
            "negative_template": (
                "background, blurry, extra limbs, text, watermark, deformed, nsfw, "
                "silk brocade, glowing aura, heavy ornament, dramatic lighting"
            ),
            "tested": True,
            "notes": (
                "Stage 2 adds character-specific detail. "
                "CLIP 77-token limit — keep style_tags + body_desc + clothing_desc concise. "
                "denoise: 0.35, CFG: 7.0. Face and hair come through from Stage 1 Flux image."
            ),
            "version": 2,
        },

        # ── STAGE 2: Quadruped Detail ─────────────────────────────────────
        {
            "_id": "stage2_quadruped_detail",
            "name": "SD1.5 Stage 2 — Quadruped Detail",
            "stage": "stage2",
            "category": "quadruped",
            "template": (
                "best quality, {style_tags}, "
                "{creature_desc}, "
                "side profile, white background, clean game asset"
            ),
            "variables": ["style_tags", "creature_desc"],
            "negative_template": (
                "background, blurry, extra limbs, text, watermark, deformed, nsfw, "
                "mundane animal, human, sitting, crouching, running"
            ),
            "tested": True,
            "notes": (
                "Quad Stage 2 detail. CLIP 77-token hard limit for SD1.5. "
                "Shorten creature_desc if >40 tokens. denoise: 0.35."
            ),
            "version": 2,
        },

        # ── STAGE 3: Multi-view Side ──────────────────────────────────────
        # Uses FLUX concept image (NOT SD output) as init_image
        {
            "_id": "stage3_side_view",
            "name": "SD1.5 Stage 3 — Side View",
            "stage": "stage3",
            "category": "any",
            "template": (
                "same character, side view, profile view, "
                "{style_tags}, clean silhouette, white background, game asset"
            ),
            "variables": ["style_tags"],
            "negative_template": (
                "deformed, extra limbs, text, watermark, blurry, nsfw, front view, back view"
            ),
            "tested": False,
            "notes": (
                "Multi-view side. IMPORTANT: Use FLUX concept image as init_image, "
                "NOT the SD stage 2 output. denoise: 0.4-0.5."
            ),
            "version": 1,
        },

        # ── STAGE 3: Multi-view Back ──────────────────────────────────────
        {
            "_id": "stage3_back_view",
            "name": "SD1.5 Stage 3 — Back View",
            "stage": "stage3",
            "category": "any",
            "template": (
                "same character, back view, rear view, "
                "{style_tags}, clean silhouette, white background, game asset"
            ),
            "variables": ["style_tags"],
            "negative_template": (
                "deformed, extra limbs, text, watermark, blurry, nsfw, "
                "front view, face visible"
            ),
            "tested": False,
            "notes": (
                "Multi-view back. IMPORTANT: Use FLUX concept image as init_image. "
                "denoise: 0.4-0.5."
            ),
            "version": 1,
        },
    ]

    for tpl in templates:
        upsert_prompt_template(db, tpl)

    print(f"  [prompt_templates] {len(templates)} templates upserted.")


# ══════════════════════════════════════════════════════════════════════════════
#  DRY RUN — validate locally without MongoDB
# ══════════════════════════════════════════════════════════════════════════════

def dry_run():
    print("=" * 70)
    print("  Seed Spec Library — DRY RUN (no MongoDB)")
    print("=" * 70)

    # ── Fake in-memory DB ──────────────────────────────────────────────────
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
        def count_documents(self, filt=None):
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

    print("\n  Document counts (in-memory):")
    print(f"    character_specs:      {db.character_specs.count_documents()}")
    print(f"    style_library:        {db.style_library.count_documents()}")
    print(f"    reference_image_bank: {db.reference_image_bank.count_documents()}")
    print(f"    prompt_templates:     {db.prompt_templates.count_documents()}")

    # Patch spec_schema functions for in-memory access
    import lib.spec_schema as schema
    _orig = (schema.get_character_spec, schema.get_style, schema.get_prompt_template)
    schema.get_character_spec  = lambda db2, cid: db.character_specs.find_one({"_id": cid})
    schema.get_style           = lambda db2, sid: db.style_library.find_one({"_id": sid})
    schema.get_prompt_template = lambda db2, tid: db.prompt_templates.find_one({"_id": tid})

    print()
    print("=" * 70)
    print("  ASSEMBLED FLUX PROMPTS (what will be sent to FLUX.1-schnell)")
    print("=" * 70)

    test_cases = [
        # (char_id,              template_id,               label)
        ("cultivation_youth",    "flux_humanoid_tpose",     "Wei Liang — Flux Front"),
        ("suanni_lion",          "flux_quadruped_side",     "Suanni Lion — Flux Side"),
        ("cultivation_youth",    "stage2_humanoid_detail",  "Wei Liang — SD Stage 2"),
        ("suanni_lion",          "stage2_quadruped_detail", "Suanni Lion — SD Stage 2"),
        ("placeholder_male_warrior", "flux_humanoid_tpose", "Warrior Placeholder — Flux"),
    ]

    for char_id, tpl_id, label in test_cases:
        try:
            prompt = schema.assemble_prompt(db, char_id, tpl_id)
            neg    = schema.assemble_negative(db, char_id, tpl_id)
            words  = len(prompt.split())
            clip   = int(words * 1.3)
            is_flux = "flux" in tpl_id
            limit_warn = ""
            if not is_flux and clip > 77:
                limit_warn = f"  ⚠️  CLIP OVERFLOW — {clip} > 77 tokens! Shorten prompt."
            elif not is_flux:
                limit_warn = f"  ✓ CLIP OK ({clip}/77)"
            else:
                limit_warn = f"  [Flux T5 encoder — no token limit]"

            print(f"\n  ─── {label} ───")
            print(f"  Words: {words} | Est. CLIP: {clip}{limit_warn}")
            print(f"\n  PROMPT:\n  {prompt}")
            print(f"\n  NEGATIVE:\n  {neg[:200]}{'...' if len(neg) > 200 else ''}")
        except Exception as e:
            print(f"\n  [{char_id}] {tpl_id} ERROR: {e}")

    # Restore
    schema.get_character_spec, schema.get_style, schema.get_prompt_template = _orig

    print()
    print("=" * 70)
    print("  DRY RUN complete. All structures valid.")
    print("  To write to MongoDB: python worker/seed_spec_library.py")
    print("  (SSH to CPU instance first: ssh -i us_cpu_key.pem ubuntu@18.207.13.85)")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Seed spec library into MongoDB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data structures locally without MongoDB")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    print("=" * 70)
    print("  Seed Spec Library — MongoDB")
    print("=" * 70)

    try:
        db = get_db()
        db.command("ping")
        print("  [MongoDB] Connected OK")
    except Exception as e:
        print(f"\n  [ERROR] Cannot connect to MongoDB: {e}")
        print("  Make sure you are running this FROM the CPU instance (18.207.13.85).")
        print("  MongoDB listens on localhost:27017 only.")
        print("  Or validate locally: python worker/seed_spec_library.py --dry-run\n")
        sys.exit(1)

    seed_styles(db)
    seed_character_specs(db)
    seed_reference_image_bank(db)
    seed_prompt_templates(db)

    print("\n  Validation:")
    print(f"    character_specs:      {db.character_specs.count_documents({})}")
    print(f"    style_library:        {db.style_library.count_documents({})}")
    print(f"    reference_image_bank: {db.reference_image_bank.count_documents({})}")
    print(f"    prompt_templates:     {db.prompt_templates.count_documents({})}")

    # Show assembled prompts for the two active characters
    from lib.spec_schema import assemble_prompt, assemble_negative
    print("\n  Assembled Flux prompts:")
    for char_id, tpl_id in [
        ("cultivation_youth", "flux_humanoid_tpose"),
        ("suanni_lion",       "flux_quadruped_side"),
    ]:
        prompt = assemble_prompt(db, char_id, tpl_id)
        words  = len(prompt.split())
        print(f"\n  [{char_id}]")
        print(f"  Words: {words}")
        print(f"  {prompt}")

    print("\n  Done. Open http://18.207.13.85:7860 → Pipeline Dashboard tab to view.")


if __name__ == "__main__":
    main()
