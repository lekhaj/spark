#!/usr/bin/env python3
"""
claudetest002 — Biome creation + task queue script
===================================================
Creates the 'claudetest002' biome in MongoDB with 2 characters,
then queues their SD1.5 generation tasks to Redis.

Characters:
  1. cultivation_youth   — humanoid, early-stage neutral cultivation disciple
  2. suanni_lion         — quadruped, wuxia lion beast (Suanni-inspired)

Run from repo root:
  python worker/queue_claudetest002.py [--dry-run] [--show-prompts-only]
"""

import argparse
import json
import os
import time
import uuid

import pymongo
import redis
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
BIOME_ID   = "claudetest002"
REDIS_HOST = os.getenv("REDIS_HOST", "18.207.13.85")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MONGO_URI  = (os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or "")
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

        # Stage 1 — ControlNet POSE CORRECTION ONLY (very light touch)
        # KEY RULE: SD should not redesign — only correct T-pose and symmetry.
        # Flux image provides the design; SD just nudges with ControlNet.
        # Denoise 0.20 = barely touches the image. Minimal prompt = minimal drift.
        "stage1_prompt": (
            "same character, T-pose, arms extended horizontally, "
            "front view, orthographic, symmetrical, clean silhouette, white background"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw"
        ),

        # Stage 2 — img2img detail pass
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, 3D game character, stylized realism, "
            "young male cultivator, ash gray linen hanfu robe, crossover collar, "
            "rope belt, half-up topknot wooden hairpin, "
            "lean build, dark sharp eyes, calm determined expression, "
            "faded sect patch chest, outer disciple, "
            "front view, white background, clean game asset"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, "
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

        # Stage 1 — ControlNet POSE CORRECTION ONLY (Canny-only for quads)
        # Uses Canny edges extracted from the Flux image — no OpenPose skeleton needed.
        # Flux already has correct standing side-profile; SD just locks the silhouette.
        "stage1_prompt": (
            "same creature, neutral standing, side profile view, "
            "orthographic, all four legs planted, clean silhouette, white background"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, "
            "human, bipedal, nsfw, running, jumping, sitting"
        ),

        # Stage 2 — img2img detail pass
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, 3D game character, stylized realism, "
            "fantastical lion beast xianxia, amber golden fur, "
            "flame-shaped rust-orange mane, bone-white spiral horn forehead, "
            "dragon scales on chest and knees, "
            "blue luminous spine markings, dragon whiskers, "
            "cloven rear hooves, tasseled tail, amber slit-pupil eyes, "
            "majestic serene, white background, clean game asset"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, "
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
    """Upsert the biome document — preserves existing flux_concept fields."""
    existing = db.biomes.find_one({"_id": BIOME_ID}) or {}
    existing_chars = (
        existing.get("possible_structures", {}).get("characters", {})
    )

    # Merge flux_concept data back in before replacing
    for char_name, char_doc in BIOME_DOCUMENT["possible_structures"]["characters"].items():
        ex = existing_chars.get(char_name, {})
        # Preserve flux_concept block if already generated
        if ex.get("flux_concept"):
            char_doc["flux_concept"] = ex["flux_concept"]
        # Preserve flux_concept S3 key in images dict
        flux_key = ex.get("images", {}).get("flux_concept")
        if flux_key:
            char_doc["images"]["flux_concept"] = flux_key
        # Preserve top-level image_url if flux was the last generated image
        if ex.get("flux_concept", {}).get("image_url") and not ex.get("image_url"):
            char_doc["image_url"] = ex["flux_concept"]["image_url"]

    db.biomes.replace_one({"_id": BIOME_ID}, BIOME_DOCUMENT, upsert=True)
    print(f"[MongoDB] Biome '{BIOME_ID}' created/updated (flux_concept preserved).")


def queue_tasks(r):
    """Push SD1.5 tasks to Redis sd15_tasks queue.

    All 4 prompt fields are passed explicitly so sd15_image_worker.py
    uses them directly for the 2-stage pipeline without falling back to
    the default CREATURE_PROMPTS templates.
    """
    for char_name, char in CHARACTERS.items():
        payload = {
            "task_id":          str(uuid.uuid4()),
            "biome_id":         BIOME_ID,
            "character_name":   char_name,
            "character_type":   char["character_type"],
            # ── Stage 1: ControlNet structure pass ──────────────────────────
            "stage1_prompt":    char["stage1_prompt"],
            "stage1_negative":  char["stage1_negative"],
            # ── Stage 2: img2img detail pass ────────────────────────────────
            "stage2_prompt":    char["stage2_prompt"],
            "stage2_negative":  char["stage2_negative"],
            "timestamp":        time.time(),
        }
        r.rpush("sd15_tasks", json.dumps(payload))
        print(f"[Redis] Queued: {char_name} → sd15_tasks  (task_id={payload['task_id'][:8]}...)")

    depth = r.llen("sd15_tasks")
    print(f"[Redis] sd15_tasks depth: {depth}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",           action="store_true",
                        help="Show everything but don't write to MongoDB or Redis")
    parser.add_argument("--show-prompts-only", action="store_true",
                        help="Print prompts and exit without touching DB or queue")
    args = parser.parse_args()

    show_prompts()

    if args.show_prompts_only:
        print("\n[INFO] --show-prompts-only: exiting without DB/queue writes.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would create biome + queue tasks. Pass no flags to execute.")
        return

    # Connect
    client = pymongo.MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    r      = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    create_biome(db)
    queue_tasks(r)

    print(f"\n✓  claudetest002 ready. Pipeline: sd15 → trellis → rig")
    print(f"   Monitor GPU: tail -f /tmp/gpu_workers.log")
    print(f"   Check MongoDB: db.biomes.findOne({{_id: '{BIOME_ID}'}})")


if __name__ == "__main__":
    main()
