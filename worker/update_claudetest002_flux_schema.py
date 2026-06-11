#!/usr/bin/env python3
"""
Update claudetest002 MongoDB biome — Flux concept image schema
==============================================================
Upserts the claudetest002 biome document with the extended schema that
supports the Flux → SD1.5 + ControlNet → Trellis pipeline.

New fields added per character:
  flux_concept   : Flux Schnell concept image output (Stage 0)
  images.flux_concept : S3 key alias for SD1.5 worker init_image
  stage1.cn_canny_weight : Canny/SoftEdge CN weight (0.65)
  stage1.cn_openpose_weight : OpenPose CN weight (0.9)
  stage1.denoise  : 0.3 (SD img2img denoise — much lower than pure text2img)

Full 5-step pipeline:
  Stage 0 — Flux Schnell: concept image (design + anatomy)
  Normalize — crop, center, white bg cleanup
  Stage 1 — SD1.5 + ControlNet (OpenPose 0.9 + Canny 0.65, denoise 0.3): pose lock
  Stage 2 — SD1.5 img2img (denoise 0.35): clothing + detail refinement
  Stage 3 — multi-view (side + back, denoise 0.45–0.55) → Trellis input

Run locally before running flux_concept_generator.py on GPU:
  python worker/update_claudetest002_flux_schema.py
"""

import time

import pymongo

MONGO_URI  = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or ""
MONGO_DB   = "World_builder"
BIOME_ID   = "claudetest002"

client = pymongo.MongoClient(MONGO_URI)
db     = client[MONGO_DB]

# ── Character definitions (descriptions + prompts stored in DB) ───────────────

CHARACTERS = {

    "cultivation_youth": {
        "character_type":    "humanoid",
        "creature_category": "bipedal",
        "description": (
            "Young male outer disciple at early Qi Condensation stage. "
            "Lean wiry build, 17-20 years old. Plain ash-gray linen hanfu robe, "
            "crossover collar, rope belt. Half-up topknot with wooden hairpin. "
            "Dark sharp eyes, calm determined expression. Faded sect patch on chest. "
            "Worn leather herb pouch at hip. No aura, no glow — mortal-born cultivator."
        ),
        # ── Stage 0: Flux Schnell concept prompt ──────────────────────────────
        "flux_prompt": (
            "young male cultivator, T-pose, arms extended horizontally, legs straight, "
            "orthographic front view, symmetrical, centered, "
            "plain gray hanfu robe, rope belt, lean build, topknot hair, "
            "simple clothing, clean silhouette, minimal detail, "
            "game-ready character design, "
            "white background, flat lighting, full body, head to toe"
        ),
        # ── Stage 1: SD1.5 + ControlNet structure lock ─────────────────────────
        "stage1_prompt": (
            "best quality, masterpiece, "
            "young male in plain gray hanfu robe, lean build, T-pose, "
            "arms extended horizontally, legs straight, "
            "front view, full body, white background, flat lighting, centered"
        ),
        "stage1_negative": (
            "background, shadows, complex textures, accessories, jewelry, weapons, "
            "text, watermark, blurry, cropped, extra limbs, deformed, nsfw, "
            "aura, glow, sparkles, magic effects"
        ),
        "stage1_cn_openpose_weight": 0.9,   # locks pose exactly
        "stage1_cn_canny_weight":    0.65,  # locks silhouette
        "stage1_denoise":            0.30,  # low — preserve Flux structure
        "stage1_cfg":                6.5,
        # ── Stage 2: img2img detail pass ──────────────────────────────────────
        "stage2_prompt": (
            "best quality, masterpiece, "
            "young male cultivator, ash gray linen hanfu robe, crossover collar, "
            "rope belt, half-up topknot wooden hairpin, "
            "lean wiry build, dark sharp eyes, calm determined expression, "
            "faded sect patch chest, outer disciple, "
            "front view, white background, clean game asset"
        ),
        "stage2_negative": (
            "background, photorealistic, blurry, extra limbs, text, watermark, deformed, "
            "ugly, nsfw, silk, brocade, embroidery, aura, glow, magic, western armor"
        ),
        "stage2_denoise": 0.35,
        "stage2_cfg":     7.0,
    },

    "suanni_lion": {
        "character_type":    "quadruped",
        "creature_category": "quadruped",
        "description": (
            "Suanni-inspired fantastical lion beast. Large muscular feline body. "
            "Deep amber-gold fur, flame-shaped rust-orange mane with smoldering tips. "
            "Single spiraling bone-white horn on forehead. Dragon-scale patches on chest "
            "and knees. Blue-white luminous spine markings. Dragon whiskers. "
            "Cloven rear hooves. Tasseled tail. Golden slit-pupil eyes. "
            "Incense smoke at paws. Runic mark at horn base. Majestic, not aggressive."
        ),
        "flux_prompt": (
            "fantastical lion beast, mythical creature, "
            "neutral standing pose, orthographic strict side profile view, "
            "all four legs planted, spine horizontal, symmetrical, centered, "
            "amber gold fur, flame-shaped orange mane, single horn on forehead, "
            "dragon scales on chest, luminous spine markings, tasseled tail, "
            "simple clean shapes, minimal detail, "
            "game-ready creature design, "
            "white background, flat lighting, full body, head to tail"
        ),
        "stage1_prompt": (
            "best quality, masterpiece, "
            "four-legged fantastical lion creature, amber fur, mane, single horn, "
            "neutral standing pose, side view, "
            "all four paws flat, spine horizontal, "
            "full body, white background, flat lighting, centered"
        ),
        "stage1_negative": (
            "background, shadows, complex textures, text, watermark, blurry, "
            "cropped, extra limbs, deformed, human, bipedal, nsfw, wings"
        ),
        "stage1_cn_openpose_weight": 0.85,  # animal skeleton openpose
        "stage1_cn_canny_weight":    0.65,
        "stage1_denoise":            0.30,
        "stage1_cfg":                6.5,
        "stage2_prompt": (
            "best quality, masterpiece, "
            "fantastical lion beast xianxia, amber golden fur, "
            "flame-shaped rust-orange mane, bone-white spiral horn forehead, "
            "dragon scales on chest and knees, "
            "blue luminous spine markings, dragon whiskers, "
            "cloven rear hooves, tasseled tail, amber slit-pupil eyes, "
            "majestic serene, white background, clean game asset"
        ),
        "stage2_negative": (
            "background, photorealistic, blurry, extra limbs, text, watermark, deformed, "
            "ugly, nsfw, mundane lion, wings, human figure"
        ),
        "stage2_denoise": 0.35,
        "stage2_cfg":     7.0,
    },
}

# ── Biome document ────────────────────────────────────────────────────────────

BIOME_DOC = {
    "_id":         BIOME_ID,
    "biome_name":  "Claude Test — Cultivation World",
    "biome_type":  "cultivation_realm",
    "description": (
        "Flux → SD1.5 → Trellis experiment biome. "
        "5-stage pipeline: Flux concept → normalize → SD1.5+ControlNet pose lock "
        "→ SD1.5 img2img detail → multi-view → Trellis 3D."
    ),
    "theme":       "xianxia_neutral",
    "created_at":  time.time(),
    "pipeline":    "flux_sd15_trellis",

    "possible_structures": {
        "characters": {
            char_name: {
                "character_type":    c["character_type"],
                "creature_category": c["creature_category"],
                "description":       c["description"],

                # ── Pipeline status ────────────────────────────────────────────
                "status":           "not_started",
                "generation_stage": None,

                # ── Stage 0: Flux concept ──────────────────────────────────────
                "flux_concept": {
                    "status":       "not_started",
                    "prompt":       c["flux_prompt"],
                    "image_key":    None,
                    "image_url":    None,
                    "generated_at": None,
                },

                # ── Stage 1: SD1.5 + ControlNet (pose lock) ───────────────────
                "stage1": {
                    "prompt":              c["stage1_prompt"],
                    "negative":            c["stage1_negative"],
                    "cn_openpose_weight":  c["stage1_cn_openpose_weight"],
                    "cn_canny_weight":     c["stage1_cn_canny_weight"],
                    "denoise":             c["stage1_denoise"],
                    "cfg":                 c["stage1_cfg"],
                    "status":              None,
                    "image_key":           None,
                    "image_url":           None,
                },

                # ── Stage 2: SD1.5 img2img (detail pass) ──────────────────────
                "stage2": {
                    "prompt":    c["stage2_prompt"],
                    "negative":  c["stage2_negative"],
                    "denoise":   c["stage2_denoise"],
                    "cfg":       c["stage2_cfg"],
                    "status":    None,
                    "image_key": None,
                    "image_url": None,
                },

                # ── Stage 3: multi-view (future) ───────────────────────────────
                "stage3": {
                    "status":     None,
                    "side_key":   None,
                    "back_key":   None,
                    "front_key":  None,
                },

                # ── Image references (flat, for TRELLIS + Gradio) ──────────────
                "images": {
                    "flux_concept": None,   # Stage 0 — Flux output S3 key
                    "base":         None,   # Stage 1 — ControlNet output S3 key
                    "refined":      None,   # Stage 2 — img2img output S3 key
                    "final":        None,   # Stage 2 alias (TRELLIS reads this)
                },

                # Top-level for Gradio viewer
                "image_url":        None,

                # ── 3D pipeline (set by TRELLIS / UniRig workers) ──────────────
                "model_path":       None,
                "model_url":        None,
                "rigged_model_url": None,

                "created_at": time.time(),
            }
            for char_name, c in CHARACTERS.items()
        }
    },
}


def main():
    print(f"[MongoDB] Upserting biome '{BIOME_ID}' ...")
    result = db.biomes.replace_one({"_id": BIOME_ID}, BIOME_DOC, upsert=True)
    if result.upserted_id:
        print(f"  → Inserted new biome")
    else:
        print(f"  → Replaced existing biome (matched={result.matched_count})")

    # Verify
    doc = db.biomes.find_one({"_id": BIOME_ID})
    chars = doc["possible_structures"]["characters"]
    print(f"\n  Characters registered:")
    for name, c in chars.items():
        print(f"    {name}: type={c['character_type']}  status={c['status']}")
        print(f"      Stage1 denoise={c['stage1']['denoise']}  "
              f"OpenPose={c['stage1']['cn_openpose_weight']}  "
              f"Canny={c['stage1']['cn_canny_weight']}")

    print(f"\n  Next: python worker/flux_concept_generator.py  (on GPU)")
    client.close()


if __name__ == "__main__":
    main()
