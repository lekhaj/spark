#!/usr/bin/env python3
"""
Flux Schnell Concept Image Generator — claudetest002
=====================================================
Standalone GPU script. Generates high-quality concept images for
claudetest002 characters using FLUX.1-schnell, uploads to S3, and
updates MongoDB with the resulting image keys/URLs.

These images serve as Stage 0 concept references — they will be passed
as init_image to the SD1.5 + ControlNet stage (Stage 1) for structure
refinement.

Usage (on GPU, from /home/ubuntu/worker):
  python flux_concept_generator.py

Requirements (install if missing):
  pip install diffusers>=0.30 transformers>=4.44 accelerate sentencepiece
  pip install bitsandbytes  # optional — for 8-bit T5 to save VRAM
"""

import io
import json
import os
import sys
import time

import boto3
import pymongo
import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
from PIL import Image

# Reduce VRAM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Load environment ──────────────────────────────────────────────────────────
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)

REDIS_HOST  = os.getenv("REDIS_HOST",  "18.207.13.85")
MONGO_URI   = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or ""
MONGO_DB    = os.getenv("MONGO_DB",    "World_builder")
S3_BUCKET   = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
AWS_REGION  = os.getenv("AWS_REGION",  "us-east-1")

BIOME_ID    = "claudetest002"
MODEL_ID    = "black-forest-labs/FLUX.1-schnell"

# ── Character definitions ─────────────────────────────────────────────────────
#
# Flux Schnell is a natural-language model — NO tag-based prompts.
# These are rich descriptive paragraphs tuned for 3D-asset use:
#   - Full body, clear silhouette, white background, flat lighting
#   - Neutral pose appropriate for TRELLIS mesh generation
#   - Enough detail that SD1.5 Stage 1 can use this as an init_image
#
CHARACTERS = {

    # ── Young Cultivator (humanoid, bipedal) ──────────────────────────────────
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

        # Flux Schnell prompt.
        #
        # KEY RULE: Keep it SHORT + STRUCTURAL.
        # Flux tends to over-stylize with long prompts — lighting, pose drift, extra
        # details. Use short cues, repeat orthographic/symmetry terms.
        # SD 1.5 + ControlNet will handle structure lock in Stage 1.
        # Flux job = get DESIGN + ANATOMY right, not photographic perfection.
        #
        "flux_prompt": (
            "young male cultivator, T-pose, arms extended horizontally, legs straight, "
            "orthographic front view, symmetrical, centered, "
            "plain gray hanfu robe, rope belt, lean build, topknot hair, "
            "simple clothing, clean silhouette, minimal detail, "
            "game-ready character design, "
            "white background, flat lighting, full body, head to toe"
        ),

        # S3 destination
        "s3_key": f"images/{BIOME_ID}/cultivation_youth_flux_concept.png",

        # Flux generation parameters
        "width":               768,
        "height":              1024,
        "num_inference_steps": 4,    # Schnell optimized for 1-4 steps
        "guidance_scale":      0.0,  # Schnell is distilled — always 0
    },

    # ── Suanni Lion (quadruped) ───────────────────────────────────────────────
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

        # Flux Schnell prompt — short, orthographic, side-profile for quadruped.
        # Side view is better for quadrupeds: ControlNet reads quad skeleton from side.
        # Keep design tags short so Flux doesn't over-detail.
        #
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

        # S3 destination
        "s3_key": f"images/{BIOME_ID}/suanni_lion_flux_concept.png",

        # Flux generation parameters (landscape for side-profile quad)
        "width":               1024,
        "height":              768,
        "num_inference_steps": 4,
        "guidance_scale":      0.0,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def upload_image_to_s3(image: Image.Image, s3_key: str, s3_client) -> str:
    """Upload PIL image to S3, return public URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=buf,
        ContentType="image/png",
    )
    url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    print(f"  [S3] Uploaded → {url}")
    return url


def save_locally(image: Image.Image, char_name: str):
    """Save a local copy for inspection."""
    out_path = f"/tmp/{char_name}_flux_concept.png"
    image.save(out_path)
    print(f"  [Local] Saved  → {out_path}")
    return out_path


def update_mongodb(db, char_name: str, char_data: dict,
                   s3_key: str, image_url: str, prompt: str):
    """Update the claudetest002 biome document with flux_concept fields."""
    now = time.time()
    update_path = f"possible_structures.characters.{char_name}"
    db.biomes.update_one(
        {"_id": BIOME_ID},
        {"$set": {
            # Flux concept block
            f"{update_path}.flux_concept.status":       "complete",
            f"{update_path}.flux_concept.image_key":    s3_key,
            f"{update_path}.flux_concept.image_url":    image_url,
            f"{update_path}.flux_concept.prompt":       prompt,
            f"{update_path}.flux_concept.generated_at": now,
            # Stage 1 init image — SD1.5 worker reads this
            f"{update_path}.images.flux_concept":       s3_key,
            # Top-level image_url for Gradio viewer
            f"{update_path}.image_url":                 image_url,
        }},
        upsert=False,
    )
    print(f"  [MongoDB] Updated {char_name}.flux_concept")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  Flux Schnell Concept Generator — biome: {BIOME_ID}")
    print("=" * 70)

    # Sanity check GPU
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found. Run on the GPU instance.")
        sys.exit(1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  VRAM: {vram_gb:.1f} GB")

    # Connect to S3 + MongoDB
    s3 = get_s3_client()
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    # Verify biome exists
    biome = db.biomes.find_one({"_id": BIOME_ID})
    if not biome:
        print(f"[WARN] Biome '{BIOME_ID}' not found in MongoDB — will create flux_concept "
              "fields but character docs won't exist. Run update_claudetest002_flux_schema.py first.")

    # ── Load Flux Schnell ─────────────────────────────────────────────────────
    print(f"\n[Flux] Loading {MODEL_ID} ...")
    print("  Using enable_sequential_cpu_offload() — each layer moved to GPU")
    print("  one at a time during forward pass. Fits any VRAM, ~3-5 min/image.")

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    # Sequential offload: moves each sub-module to GPU one at a time.
    # Much more memory-safe than model_cpu_offload (which loads whole components).
    pipe.enable_sequential_cpu_offload()
    # Sliced VAE decode to avoid VRAM spikes on large images
    pipe.vae.enable_slicing()
    print("[Flux] Model ready.\n")

    # ── Generate images ───────────────────────────────────────────────────────
    results = {}

    for char_name, char in CHARACTERS.items():
        print(f"{'─' * 60}")
        print(f"[{char_name}] Generating concept image ...")
        print(f"  Size:  {char['width']} x {char['height']}")
        print(f"  Steps: {char['num_inference_steps']}")
        print(f"  Prompt ({len(char['flux_prompt'].split())} words):")
        print(f"    {char['flux_prompt'][:200]}...")

        t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                prompt=char["flux_prompt"],
                width=char["width"],
                height=char["height"],
                num_inference_steps=char["num_inference_steps"],
                guidance_scale=char["guidance_scale"],
            )
        image = out.images[0]
        elapsed = time.time() - t0
        print(f"  Generation: {elapsed:.1f}s")

        # Save locally
        local_path = save_locally(image, char_name)

        # Upload to S3
        image_url = upload_image_to_s3(image, char["s3_key"], s3)

        # Update MongoDB
        if biome:
            update_mongodb(db, char_name, char, char["s3_key"], image_url,
                           char["flux_prompt"])

        results[char_name] = {
            "local_path": local_path,
            "s3_key":     char["s3_key"],
            "image_url":  image_url,
        }
        print(f"  [OK] {char_name} done in {elapsed:.1f}s\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  COMPLETED — Flux concept images ready")
    print("=" * 70)
    for char_name, r in results.items():
        print(f"\n  {char_name}:")
        print(f"    Local  : {r['local_path']}")
        print(f"    S3 key : {r['s3_key']}")
        print(f"    URL    : {r['image_url']}")

    print(f"\n  Next step: pass flux_concept images as init_image into SD1.5 Stage 1")
    print(f"  Queue     : python worker/queue_claudetest002.py")

    # Write results JSON for easy retrieval
    results_path = "/tmp/flux_concept_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON: {results_path}")

    client.close()


if __name__ == "__main__":
    main()
