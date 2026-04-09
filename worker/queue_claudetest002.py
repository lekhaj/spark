#!/usr/bin/env python3
"""
seed_claudetest002.py  (kept as queue_claudetest002.py for compatibility)
=========================================================================
Biome seeder for 'claudetest002' — creates / updates the biome document
in MongoDB with 2 fully-defined cultivation-world characters.

This script ONLY creates the biome.  To queue tasks use the generic handler:
  python worker/enqueue_generation.py --biome-id claudetest002 [--stage1-only]

Characters:
  1. cultivation_youth   — humanoid, early-stage neutral cultivation disciple
  2. suanni_lion         — quadruped, wuxia lion beast (Suanni-inspired)

Usage:
  python worker/queue_claudetest002.py [--dry-run] [--show-prompts-only]
"""

import argparse
import os
import time

import pymongo
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
BIOME_ID   = "claudetest002"
REDIS_HOST = os.getenv("REDIS_HOST", "18.207.13.85")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MONGO_URI  = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@18.207.13.85:27017")
MONGO_DB   = os.getenv("MONGO_DB", "World_builder")

# ── Character Definitions ─────────────────────────────────────────────────────
#
# Stage 1 prompt strategy  → pure STRUCTURE for ControlNet guidance
#   - T-pose / neutral standing only
#   - white background, flat lighting
#   - NO character-specific details (those come in Stage 2)
#   - Target: 20-30 tokens (+ quality prefix ~5 = 25-35 total)
#
# Stage 2 prompt strategy  → CHARACTER DETAIL for img2img refinement
#   - Specific clothing, markings, palette, style
#   - white background maintained
#   - Target: 45-55 tokens (+ quality prefix ~5 = 50-60 total)
#   - Hard ceiling: 68 tokens (CLIP limit)

CHARACTERS = {

    # ─────────────────────────────────────────────────────────────────────────
    "cultivation_youth": {
        "character_type":    "humanoid",
        "creature_category": "bipedal",

        "description": (
            "Young male outer disciple at early Qi Condensation stage. "
            "Neutral non-Chinese eastern fantasy aesthetic — universal cultivation archetype. "
            "Lean wiry build, 17-20 years old, slightly underfed but determined. "
            "Plain ash-gray coarse linen hanfu robe, crossover collar, straight sleeves, "
            "knee-length hem. Simple rope belt — no jade or metal accessories. "
            "Half-up topknot secured with a plain wooden hairpin, rest of hair loose. "
            "Dark sharp eyes with a calm, quietly determined expression. "
            "Small faded sect insignia patch on left chest (simple stitching). "
            "Worn leather herb pouch at hip. No aura, no glow — raw mortal-born "
            "cultivator at the very beginning of the immortal path."
        ),

        # Stage 1 — STRUCTURE ONLY ("boring but correct")
        # Rule: plain language, ≤20 words, NO weighted syntax, NO lore, NO details
        # CFG 5.5, CN 0.6, Steps 25 (see sd15_image_worker.py)
        # Stage 1 output = anatomy + pose + clean silhouette. Details come in Stage 2.
        "stage1_prompt": (
            "full body young male, thin wiry build, T-pose, arms horizontal, legs straight, "
            "front view, centered, white background, simple loose robe, flat lighting"
        ),
        "stage1_negative": (
            "muscular, bodybuilder, gym body, modern clothing, pants, sportswear, "
            "room, background, shadows, environment, "
            "bad hands, deformed, extra limbs, text, watermark"
        ),

        # Stage 2 — img2img detail pass (~50 tokens with prefix, CLIP-safe ≤68)
        "stage2_prompt": (
            "best quality, masterpiece, "
            "young male cultivator, ash gray linen hanfu robe, crossover collar, "
            "rope belt, half-up topknot wooden hairpin, "
            "lean wiry build, dark sharp eyes, calm determined expression, "
            "faded sect patch chest, outer disciple beginner, "
            "neutral xianxia fantasy, no aura, "
            "front view, white background, clean game asset"
        ),
        "stage2_negative": (
            "background, photorealistic, blurry, extra limbs, text, watermark, deformed, "
            "ugly, nsfw, silk, brocade, embroidery, aura, glow, magic, western armor, "
            "chinese opera, heavy ornament, master robes"
        ),
    },

    # ─────────────────────────────────────────────────────────────────────────
    "suanni_lion": {
        "character_type":    "quadruped",
        "creature_category": "quadruped",

        "description": (
            "Suanni-inspired fantastical lion beast from wuxia mythology. "
            "Large powerful feline body, noticeably larger than a natural lion. "
            "Deep amber-gold fur, flame-shaped rust-orange mane with tips that curl "
            "upward as if smoldering. Single spiraling bone-white horn centered on the "
            "forehead. Partial dragon-scale patches overlaid on chest and knee joints. "
            "Thin blue-white luminous stripe markings tracing the spine and flanks "
            "like circuit lines of spiritual energy. Dragon-style whiskers on the jaw. "
            "Cloven hooves on the rear legs (Qilin influence). Thick tail ending in a "
            "tasseled tip. Golden amber eyes with vertical slit pupils. "
            "Wisps of incense smoke curling around the paws. "
            "Runic diamond mark between the horns. Majestic and serene, not aggressive."
        ),

        # Stage 1 — ControlNet structure pass (~56 tokens, CLIP-safe ≤68)
        # Stage 1 — STRUCTURE ONLY ("boring but correct")
        # Rule: plain language, ≤20 words, side view for quadrupeds
        "stage1_prompt": (
            "full body lion, side view, neutral standing, all four legs on ground, "
            "centered, white background, flat lighting, clean silhouette"
        ),
        "stage1_negative": (
            "running, jumping, dynamic pose, background, shadows, environment, "
            "human, rider, wings, deformed, extra limbs, text, watermark"
        ),

        # Stage 2 — img2img detail pass (~52 tokens with prefix, CLIP-safe ≤68)
        "stage2_prompt": (
            "best quality, masterpiece, "
            "fantastical lion beast xianxia, amber golden fur, "
            "flame-shaped rust-orange mane, bone-white spiral horn forehead, "
            "dragon scales on chest and knees, "
            "blue luminous spine markings, dragon whiskers, "
            "cloven rear hooves, tasseled tail, amber slit-pupil eyes, "
            "incense smoke paws, runic mark, "
            "majestic serene, white background, clean game asset"
        ),
        "stage2_negative": (
            "background, photorealistic, blurry, extra limbs, text, watermark, deformed, "
            "ugly, nsfw, mundane lion, wings, human figure, regular animal"
        ),
    },
}

# ── MongoDB biome document ────────────────────────────────────────────────────

BIOME_DOCUMENT = {
    "_id": BIOME_ID,
    "biome_name":    "Claude Test — Cultivation World",
    "biome_type":    "cultivation_realm",
    "description":   (
        "Test biome for claudetest002. "
        "A neutral cultivation world setting — neither fully Chinese nor fully Western. "
        "Features beginner-level human cultivators and mythical wuxia-inspired beasts."
    ),
    "theme":         "xianxia_neutral",
    "created_at":    time.time(),
    "created_by":    "claude-pipeline",
    "possible_structures": {
        "characters": {
            char_name: {
                "character_type":    char["character_type"],
                "creature_category": char["creature_category"],
                "description":       char["description"],
                "status":            "not_started",
                "generation_stage":  None,
                # ── 2-layer image generation structure ────────────────────────
                "stage1": {
                    "prompt":    char["stage1_prompt"],
                    "negative":  char["stage1_negative"],
                    "status":    None,
                    "image_key": None,
                    "image_url": None,
                },
                "stage2": {
                    "prompt":    char["stage2_prompt"],
                    "negative":  char["stage2_negative"],
                    "status":    None,
                    "image_key": None,
                    "image_url": None,
                },
                # ── Final image references ─────────────────────────────────
                "images": {
                    "base":    None,   # Stage 1 ControlNet output
                    "refined": None,   # Stage 2 img2img output
                    "final":   None,   # Same as refined (pipeline compat)
                },
                "image_url":         None,   # Top-level for Gradio viewer
                # ── 3D model references ────────────────────────────────────
                "model_path":        None,
                "model_url":         None,
                "rigged_model_url":  None,
                "created_at":        time.time(),
            }
            for char_name, char in CHARACTERS.items()
        }
    },
}


def token_estimate(prompt: str) -> int:
    """Rough CLIP token estimate (words * 1.3 average)."""
    return round(len(prompt.split()) * 1.3)


def show_prompts():
    """Print Stage 1 prompts exactly as the worker will send them."""
    print("\n" + "=" * 70)
    print("  STAGE 1 PROMPTS  (ControlNet structure pass)")
    print("  These are sent to SD1.5 + ControlNet as-is")
    print("=" * 70)

    for char_name, char in CHARACTERS.items():
        p = char["stage1_prompt"]
        n = char["stage1_negative"]
        t = token_estimate(p)
        print(f"\n{'─'*60}")
        print(f"  CHARACTER : {char_name}")
        print(f"  TYPE      : {char['character_type']} / {char['creature_category']}")
        print(f"  CONTROLNET: {'T-pose openpose (bipedal)' if char['creature_category'] == 'bipedal' else 'Quad skeleton (quadruped)'}")
        print(f"  TOKENS    : ~{t}  (limit: 68)")
        print(f"\n  POSITIVE  :\n    {p}")
        print(f"\n  NEGATIVE  :\n    {n}")

    print("\n" + "=" * 70)
    print("  STAGE 2 PROMPTS  (img2img detail pass)")
    print("=" * 70)

    for char_name, char in CHARACTERS.items():
        p = char["stage2_prompt"]
        n = char["stage2_negative"]
        t = token_estimate(p)
        over = "  ⚠ OVER LIMIT" if t > 68 else "  ✓ OK"
        print(f"\n{'─'*60}")
        print(f"  CHARACTER : {char_name}")
        print(f"  TOKENS    : ~{t}  (limit: 68){over}")
        print(f"\n  POSITIVE  :\n    {p}")
        print(f"\n  NEGATIVE  :\n    {n}")


def create_biome(db):
    """Insert or replace the biome document in MongoDB."""
    db.biomes.replace_one({"_id": BIOME_ID}, BIOME_DOCUMENT, upsert=True)
    print(f"[MongoDB] Biome '{BIOME_ID}' created/updated.")


def main():
    parser = argparse.ArgumentParser(
        description="Seed the claudetest002 biome in MongoDB."
    )
    parser.add_argument("--dry-run",           action="store_true",
                        help="Show prompts without writing to MongoDB")
    parser.add_argument("--show-prompts-only", action="store_true",
                        help="Print prompts and exit")
    args = parser.parse_args()

    show_prompts()

    if args.show_prompts_only:
        print("\n[INFO] --show-prompts-only: exiting without DB writes.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would create/update biome '{BIOME_ID}' in MongoDB.")
        print("          Re-run without --dry-run to write.")
        return

    client = pymongo.MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    create_biome(db)

    print(f"\n✓  Biome '{BIOME_ID}' seeded in MongoDB.")
    print(f"\n   To queue generation tasks, run:")
    print(f"   python worker/enqueue_generation.py --biome-id {BIOME_ID} --stage1-only")
    print(f"   python worker/enqueue_generation.py --biome-id {BIOME_ID}  # full pipeline")


if __name__ == "__main__":
    main()
