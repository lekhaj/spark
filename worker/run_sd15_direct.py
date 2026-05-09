#!/usr/bin/env python3
"""
Direct SD1.5 Stage 1+2 runner — no Redis queue needed.
Reads Flux concept images from MongoDB, runs the full 2-stage pipeline,
uploads refined images to S3, prints public URLs.

Usage (on GPU):
  python worker/run_sd15_direct.py
"""

import io
import os
import sys
import time

import boto3
import numpy as np
import pymongo
import torch
from dotenv import load_dotenv
from PIL import Image, ImageFilter
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI  = os.getenv("MONGO_URI",    "mongodb://kartik:Kartikg421@18.207.13.85:27017")
S3_BUCKET  = os.getenv("AWS_S3_BUCKET","sparkassets-us")
AWS_REGION = os.getenv("AWS_REGION",   "us-east-1")
BIOME_ID   = "bhavesh_batch_001"  # batch id for this generation run

SD15_MODEL_ID          = "Lykon/DreamShaper"
CONTROLNET_OPENPOSE_ID = "lllyasviel/control_v11p_sd15_openpose"
CONTROLNET_CANNY_ID    = "lllyasviel/control_v11p_sd15_canny"

TPOSE_OPENPOSE_PATH  = os.getenv("TPOSE_OPENPOSE_PATH",
                                  "/home/ec2-user/controlnet_refs/tpose_openpose.png")

IMG_SIZE = 512

# Stage 1 — VERY LIGHT TOUCH. SD only corrects pose. Flux provides design.
STAGE1_STEPS       = 20
STAGE1_CFG         = 5.5
STAGE1_STRENGTH    = 0.20   # 0.15–0.25 range
STAGE1_CN_OPENPOSE = 0.85   # bipedal: OpenPose skeleton
STAGE1_CN_CANNY    = 0.55   # bipedal: Canny from Flux
STAGE1_CN_CANNY_QUAD = 0.70 # quad: Canny-only (no skeleton)

# Stage 2 — detail pass
STAGE2_STEPS    = 20
STAGE2_CFG      = 6.5
STAGE2_STRENGTH = 0.22

# ── Characters ────────────────────────────────────────────────────────────────
# Style direction: semi-realistic fantasy, rich PBR textures, stylized realism.
# Inspired by high-quality isometric fantasy game aesthetics.
# Stage 1 — MINIMAL: pose correction only. Stage 2 — FULL visual identity.
#
# 5 HUMANOIDS  +  5 QUADRUPEDS
CHARACTERS = {

    # ══════════════════════════════════════════════════════════════════════════
    #  HUMANOIDS (bipedal)
    # ══════════════════════════════════════════════════════════════════════════

    "human_ranger": {
        "creature_category": "bipedal",
        "stage1_prompt": (
            "full body human female, front-facing T-pose, arms extended horizontally, "
            "legs straight, realistic anatomy, detailed hands, 5 fingers, "
            "natural face, symmetrical eyes, looking forward, auburn hair, medieval fantasy clothing, "
            "fitted tunic, leather belt, tall boots, isolated on pure white background, flat lighting"
        ),
        "stage1_negative": (
            "scenery, background, room, walls, mutated hands, missing fingers, deformed face, "
            "winking, crossed eyes, open mouth, extra limbs, robotic armor, "
            "dynamic pose, cropped body"
        ),
        "stage2_prompt": (
            "full body female ranger, front-facing T-pose, realistic facial features, symmetrical face, "
            "open aligned eyes, intense stare, realistic hands, auburn wavy hair, "
            "blue and brown medieval tunic, leather corset, utility belts and pouches, "
            "dark fitted pants, knee-high leather boots, grounded fantasy RPG style, "
            "isolated on pure white background, studio lighting, no shadow"
        ),
        "stage2_negative": (
            "scenery, environment, room, stone arch, shadow, mutated hands, deformed fingers, "
            "winking, cross-eyed, asymmetrical face, distorted mouth, "
            "messy hair, overdesigned fantasy, cartoon, sketch"
        ),
    },

    "dwarven_knight": {
        "creature_category": "bipedal",
        "stage1_prompt": (
            "humanoid character, T-pose, arms extended horizontally, legs straight, "
            "front view, full body, white background, flat lighting, symmetrical, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw, "
            "sitting, crouching, dynamic pose"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game character, "
            "male dwarf knight, stocky muscular build, full heavy plate armor, "
            "aged silver steel plates, ornate engraved chest piece, red fabric underlayer, "
            "thick brown beard braided with iron rings, stern determined expression, "
            "round shield on arm, battle-worn armor with scratches and dents, "
            "front view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "modern clothes, sci-fi, tall, slender"
        ),
    },

    "human_battlemage": {
        "creature_category": "bipedal",
        "stage1_prompt": (
            "humanoid character, T-pose, arms extended horizontally, legs straight, "
            "front view, full body, white background, flat lighting, symmetrical, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw, "
            "sitting, crouching, dynamic pose"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game character, "
            "male human battlemage, athletic build, layered robes over chainmail, "
            "deep indigo and gold robes, arcane runic symbols embroidered on fabric, "
            "leather armored shoulders, dark short hair, amber glowing eyes, "
            "arcane staff in hand, jeweled ring on finger, high fantasy mage, "
            "front view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "modern clothes, sci-fi"
        ),
    },

    "orc_berserker": {
        "creature_category": "bipedal",
        "stage1_prompt": (
            "humanoid character, T-pose, arms extended horizontally, legs straight, "
            "front view, full body, white background, flat lighting, symmetrical, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw, "
            "sitting, crouching, dynamic pose"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game character, "
            "male orc berserker, massive muscular build, dark green skin, tribal bone armor, "
            "iron pauldrons, fur loincloth, war paint on face and arms, "
            "tusks, red eyes, fierce snarling expression, scars on chest, "
            "heavy axe, spiked bracers, savage warrior, "
            "front view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "human skin, slender build, sci-fi"
        ),
    },

    "undead_knight": {
        "creature_category": "bipedal",
        "stage1_prompt": (
            "humanoid character, T-pose, arms extended horizontally, legs straight, "
            "front view, full body, white background, flat lighting, symmetrical, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw, "
            "sitting, crouching, dynamic pose"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game character, "
            "undead knight, skeletal face with decayed skin, cracked dark plate armor, "
            "rusted black iron pauldrons, tattered black cape, glowing purple eyes in skull, "
            "exposed bone hands gripping longsword, green necrotic energy faintly glowing at joints, "
            "dark fantasy undead warrior, eerie regal posture, "
            "front view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, ugly mess, nsfw, "
            "living human, sci-fi, modern"
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  QUADRUPEDS (four-legged creatures)
    # ══════════════════════════════════════════════════════════════════════════

    "armored_warbear": {
        "creature_category": "quadruped",
        "stage1_prompt": (
            "four-legged bear creature, neutral standing pose, side view, "
            "all four paws flat on ground, spine horizontal, head forward, "
            "full body, white background, flat lighting, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, "
            "human, bipedal, nsfw, running, jumping, sitting, rearing"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game creature, "
            "massive armored war bear, dark brown fur, heavy forged iron plate armor on back and shoulders, "
            "spiked pauldrons, gold rivets on armor plates, battle-scarred snout, "
            "glowing amber eyes, thick muscular legs, iron paw guards, "
            "side view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "human, wings, sitting, sci-fi"
        ),
    },

    "shadow_wolf": {
        "creature_category": "quadruped",
        "stage1_prompt": (
            "four-legged wolf creature, neutral standing pose, side view, "
            "all four paws flat on ground, spine horizontal, tail extended, "
            "full body, white background, flat lighting, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, "
            "human, bipedal, nsfw, running, jumping, sitting"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game creature, "
            "large shadow wolf, sleek jet-black fur with subtle dark blue shimmer, "
            "arcane silver runic markings along spine and legs, "
            "piercing pale blue glowing eyes, long powerful legs, sleek muscular body, "
            "slightly translucent shadow wisps at paws, sharp fangs visible, "
            "side view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "human, wings, sitting, sci-fi"
        ),
    },

    "stone_golem_hound": {
        "creature_category": "quadruped",
        "stage1_prompt": (
            "four-legged dog-like creature, neutral standing pose, side view, "
            "all four paws flat on ground, spine horizontal, "
            "full body, white background, flat lighting, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, "
            "human, bipedal, nsfw, running, jumping, sitting"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game creature, "
            "stone golem hound, large quadruped made of jagged grey granite and dark basalt, "
            "glowing orange magma cracks along the seams, heavy blocky legs, "
            "angular rock head with no visible eyes only orange glow slits, "
            "moss patches on back, ancient carved runes on flanks, "
            "side view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "human, fur, organic flesh, sci-fi"
        ),
    },

    "swamp_basilisk": {
        "creature_category": "quadruped",
        "stage1_prompt": (
            "four-legged lizard creature, neutral standing pose, side view, "
            "all four legs on ground, spine horizontal, tail extended behind, "
            "full body, white background, flat lighting, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, "
            "human, bipedal, nsfw, running, jumping, sitting, wings"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game creature, "
            "swamp basilisk, large four-legged lizard, dark olive green and brown mottled scales, "
            "spiky dorsal ridge from neck to tail, yellow slit-pupil eyes with petrifying gaze, "
            "thick muscular tail, wide splayed clawed feet, dewlap under jaw, "
            "subtle bioluminescent green spots along flanks, "
            "side view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "human, bipedal, wings, fur, sci-fi"
        ),
    },

    "skeletal_steed": {
        "creature_category": "quadruped",
        "stage1_prompt": (
            "four-legged horse creature, neutral standing pose, side view, "
            "all four hooves flat on ground, spine horizontal, head forward, "
            "full body, white background, flat lighting, centered"
        ),
        "stage1_negative": (
            "deformed, extra limbs, text, watermark, background, shadows, blurry, "
            "human, bipedal, nsfw, running, jumping, sitting"
        ),
        "stage2_prompt": (
            "best quality, masterpiece, semi-realistic, stylized realism, 3D game creature, "
            "skeletal undead warhorse, bleached white and grey bones, "
            "tattered black ethereal mane and tail made of dark smoke, "
            "glowing red eyes in hollow skull sockets, iron horseshoes on hooves, "
            "cracked ribs visible, shadowy necrotic energy wisps around legs, "
            "ornate bone saddle on back, dark fantasy, eerie elegant, "
            "side view, white background, clean game asset, PBR materials, rich colors"
        ),
        "stage2_negative": (
            "anime, cartoon, cel shading, 2D illustration, manga, flat art, sketch, "
            "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, "
            "living horse, flesh, fur, sci-fi"
        ),
    },

}  # END CHARACTERS


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_canny(img: Image.Image) -> Image.Image:
    img_r = img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    try:
        import cv2
        arr   = np.array(img_r)
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 80, 180)
        return Image.fromarray(np.stack([edges]*3, axis=-1).astype(np.uint8))
    except ImportError:
        gray  = img_r.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.point(lambda x: 255 if x > 25 else 0)
        return Image.merge("RGB", [edges, edges, edges])


def s3_url(key: str) -> str:
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload(s3, img: Image.Image, key: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf, ContentType="image/png")
    return s3_url(key)


def download_s3_img(s3, key: str) -> Image.Image:
    buf = io.BytesIO()
    s3.download_fileobj(Bucket=S3_BUCKET, Key=key, Fileobj=buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def mongo_update(db, char_name: str, fields: dict):
    """Update character status in MongoDB. Silently skips if MongoDB is unreachable."""
    if db is None:
        return  # MongoDB not connected — skip silently
    try:
        db.biomes.update_one(
            {"_id": BIOME_ID},
            {"$set": {f"possible_structures.characters.{char_name}.{k}": v
                      for k, v in fields.items()}},
            upsert=True,
        )
    except Exception as e:
        print(f"  [MongoDB] Status update skipped (DB unreachable): {e.__class__.__name__}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate one or all characters using SD1.5 + ControlNet."
    )
    parser.add_argument(
        "--char",
        help="Name of the single character to generate (e.g. elven_ranger). "
             "If omitted, ALL characters are generated.",
        default=None,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available character names and exit.",
    )
    args = parser.parse_args()

    # ── --list: just show available names and quit ─────────────────────────────
    if args.list:
        print("\nAvailable characters:")
        for i, name in enumerate(CHARACTERS.keys(), 1):
            cat = CHARACTERS[name]["creature_category"]
            tag = "👤 Humanoid" if cat == "bipedal" else "🐾 Quadruped"
            print(f"  {i:>2}. {name:<25} {tag}")
        print()
        return

    # ── Pick which characters to run ───────────────────────────────────────────
    if args.char:
        if args.char not in CHARACTERS:
            print(f"\n[ERROR] Character '{args.char}' not found.")
            print("Run with --list to see all available characters.")
            sys.exit(1)
        chars_to_run = {args.char: CHARACTERS[args.char]}
    else:
        chars_to_run = CHARACTERS

    print("=" * 70)
    print(f"  SD1.5 Direct Runner — biome: {BIOME_ID}")
    print(f"  Generating: {', '.join(chars_to_run.keys())}")
    print(f"  Stage 1 settings: denoise={STAGE1_STRENGTH}  CFG={STAGE1_CFG}")
    print(f"  Stage 2 settings: denoise={STAGE2_STRENGTH}  CFG={STAGE2_CFG}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found. This script must run on the GPU machine.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU] {gpu_name}  ({vram_gb:.1f} GB VRAM)")

    # ── Connections ───────────────────────────────────────────────────────────
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # MongoDB is optional — used only for status tracking and Flux concept lookup.
    # If it's unreachable the script continues and generates images via txt2img fallback.
    print("[MongoDB] Connecting (5s timeout)...")
    try:
        db = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5_000,   # fail fast — don't block for 30s
            connectTimeoutMS=5_000,
            socketTimeoutMS=5_000,
        )["World_builder"]
        db.command("ping")  # quick test
        print("[MongoDB] Connected ✓")
    except Exception as e:
        print(f"[MongoDB] WARNING: Cannot connect — {e.__class__.__name__}: {e}")
        print("[MongoDB] Continuing WITHOUT MongoDB. Status tracking disabled.")
        print("[MongoDB] Images will still be generated and uploaded to S3. ✓")
        db = None

    # ── Clear leftover VRAM from any crashed previous session ─────────────────
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[GPU] VRAM free before loading: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[Models] Loading ControlNets (OpenPose + Canny)...")
    cn_openpose = ControlNetModel.from_pretrained(CONTROLNET_OPENPOSE_ID, torch_dtype=torch.float16)
    cn_canny    = ControlNetModel.from_pretrained(CONTROLNET_CANNY_ID,    torch_dtype=torch.float16)

    print(f"[Models] Loading SD1.5: {SD15_MODEL_ID}")
    pipe_cn = StableDiffusionControlNetPipeline.from_pretrained(
        SD15_MODEL_ID, controlnet=cn_openpose,
        torch_dtype=torch.float16, safety_checker=None,
    )
    pipe_cn.scheduler = UniPCMultistepScheduler.from_config(pipe_cn.scheduler.config)
    try:
        pipe_cn.enable_xformers_memory_efficient_attention()
        print("[Models] xformers memory efficient attention enabled.")
    except Exception:
        pipe_cn.enable_attention_slicing()
        print("[Models] xformers unavailable, using attention slicing fallback.")
    pipe_cn.to("cuda")
    cn_canny.to("cuda")   # move to CUDA before building shared pipelines

    # Shared-weight pipelines — controlnet list must be set at init, not swapped after
    base_components = dict(pipe_cn.components)

    # Bipedal: OpenPose + Canny dual
    biped_components = {**base_components, "controlnet": [cn_openpose, cn_canny]}
    pipe_biped_i2i = StableDiffusionControlNetImg2ImgPipeline(**biped_components)
    pipe_biped_i2i.scheduler = UniPCMultistepScheduler.from_config(pipe_biped_i2i.scheduler.config)

    # Quadruped: Canny-only (correct standing pose from Flux, no broken skeleton)
    quad_components = {**base_components, "controlnet": cn_canny}
    pipe_quad_i2i = StableDiffusionControlNetImg2ImgPipeline(**quad_components)
    pipe_quad_i2i.scheduler = UniPCMultistepScheduler.from_config(pipe_quad_i2i.scheduler.config)

    # Stage 2: plain img2img (no controlnet)
    i2i_components = {k: v for k, v in base_components.items() if k != "controlnet"}
    pipe_i2i = StableDiffusionImg2ImgPipeline(**i2i_components)
    pipe_i2i.scheduler = UniPCMultistepScheduler.from_config(pipe_i2i.scheduler.config)

    print("[Models] All loaded — UNet shared.\n")

    # OpenPose T-pose skeleton reference
    if os.path.exists(TPOSE_OPENPOSE_PATH):
        openpose_ref = Image.open(TPOSE_OPENPOSE_PATH).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        print(f"[OpenPose] T-pose reference loaded: {TPOSE_OPENPOSE_PATH}")
    else:
        openpose_ref = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)  # black = no skeleton
        print(f"[OpenPose] WARNING: T-pose reference NOT found at {TPOSE_OPENPOSE_PATH}")
        print("[OpenPose] Humanoid Stage 1 will use a blank skeleton — pose quality may be lower.")
        print("[OpenPose] To fix: place tpose_openpose.png in /home/ec2-user/controlnet_refs/")

    results = {}

    for char_name, char in chars_to_run.items():
        is_quad = char["creature_category"] == "quadruped"
        type_label = "🐾 Quadruped" if is_quad else "👤 Humanoid"

        print()
        print("═" * 70)
        print(f"  CHARACTER: {char_name}   ({type_label})")
        print("═" * 70)
        print()
        print("  ┌─ WHAT IS HAPPENING ──────────────────────────────────────────┐")
        print(f"  │ We are generating a 2D character image in 2 stages.          │")
        print(f"  │ Type: {type_label:<55}│")
        print("  └──────────────────────────────────────────────────────────────┘")

        # ── Step 1: Check if a Flux concept image exists ──────────────────────
        print()
        print("  [STEP 1/5] Checking MongoDB for a Flux concept image...")
        print("  → Flux is a high-quality AI (like DALL-E 3) that creates the")
        print("    initial concept art. If it exists, SD1.5 uses it as a base.")

        flux_img = None
        if db is not None:
            try:
                biome_doc = db.biomes.find_one({"_id": BIOME_ID})
                char_doc  = (biome_doc or {}).get("possible_structures", {}).get("characters", {}).get(char_name, {})
                flux_key  = (char_doc.get("flux_concept") or {}).get("image_key") or char_doc.get("images", {}).get("flux_concept")
            except Exception as e:
                print(f"  [MongoDB] Flux lookup failed: {e.__class__.__name__} — using txt2img fallback.")
                flux_key = None
        else:
            print("  [MongoDB] Skipped (DB offline) — using txt2img fallback.")
            flux_key = None

        if flux_key:
            print(f"  ✓ Found Flux concept in S3: {flux_key}")
            print("  → Downloading it to use as the starting image...")
            flux_img = download_s3_img(s3, flux_key)
            print(f"  ✓ Downloaded. Size: {flux_img.size[0]}×{flux_img.size[1]} px")
        else:
            print("  ⚠ No Flux concept found.")
            print("  → Will generate from scratch using OpenPose skeleton + text prompt.")

        # ── Stage 1: Pose correction ──────────────────────────────────────────
        print()
        print("  [STEP 2/5] STAGE 1 — Pose Correction")
        print("  ┌──────────────────────────────────────────────────────────────┐")
        if is_quad:
            print("  │ QUADRUPED MODE:                                              │")
            print("  │  • Uses CANNY ControlNet only                                │")
            print("  │  • Canny = edge detection map (like a pencil outline)        │")
            print("  │  • Tells AI: keep this silhouette, 4 legs on ground          │")
            print("  │  • Denoise=0.20 → AI only changes 20% of the image           │")
            print("  │    (very light touch, preserves Flux's design)               │")
        else:
            print("  │ HUMANOID (BIPEDAL) MODE:                                     │")
            print("  │  • Uses OPENPOSE + CANNY ControlNet (dual)                   │")
            print("  │  • OpenPose = skeleton map (shows where joints should be)    │")
            print("  │  • Canny = edge outline from the Flux image                  │")
            print("  │  • Together they LOCK the T-pose (arms out, legs straight)   │")
            print("  │  • Denoise=0.20 → AI only changes 20% of the image           │")
            print("  │    (very light touch, preserves Flux's design)               │")
        print("  └──────────────────────────────────────────────────────────────┘")
        print(f"  Prompt: {char['stage1_prompt'][:80]}...")

        s1_pos = char["stage1_prompt"]
        s1_neg = char["stage1_negative"]
        mongo_update(db, char_name, {"status": "generating", "stage1.status": "generating"})
        t0 = time.time()
        print()
        print("  ⏳ Running Stage 1 AI inference... (this takes ~30-60 seconds)")

        if flux_img is not None:
            init_img  = flux_img.resize((IMG_SIZE, IMG_SIZE))
            canny_img = extract_canny(flux_img)

            if is_quad:
                with torch.no_grad():
                    result = pipe_quad_i2i(
                        prompt=s1_pos, negative_prompt=s1_neg,
                        image=init_img, control_image=canny_img,
                        strength=STAGE1_STRENGTH, num_inference_steps=STAGE1_STEPS,
                        guidance_scale=STAGE1_CFG,
                        controlnet_conditioning_scale=STAGE1_CN_CANNY_QUAD,
                        width=IMG_SIZE, height=IMG_SIZE,
                    )
            else:
                with torch.no_grad():
                    result = pipe_biped_i2i(
                        prompt=s1_pos, negative_prompt=s1_neg,
                        image=init_img, control_image=[openpose_ref, canny_img],
                        strength=STAGE1_STRENGTH, num_inference_steps=STAGE1_STEPS,
                        guidance_scale=STAGE1_CFG,
                        controlnet_conditioning_scale=[STAGE1_CN_OPENPOSE, STAGE1_CN_CANNY],
                        width=IMG_SIZE, height=IMG_SIZE,
                    )
        else:
            with torch.no_grad():
                result = pipe_cn(
                    prompt=s1_pos, negative_prompt=s1_neg,
                    image=openpose_ref,
                    num_inference_steps=STAGE1_STEPS, guidance_scale=STAGE1_CFG,
                    controlnet_conditioning_scale=0.85,
                    width=IMG_SIZE, height=IMG_SIZE,
                )

        stage1_time = time.time() - t0
        base_img = result.images[0]
        base_key = f"images/{BIOME_ID}/{char_name}_base_v2.png"

        # ── Step 3: Upload Stage 1 to S3 ──────────────────────────────────────
        print(f"  ✓ Stage 1 done in {stage1_time:.1f}s")
        print()
        print("  [STEP 3/5] Uploading Stage 1 image to S3 (cloud storage)...")
        print("  → S3 is like Google Drive for the project. All images live there.")
        base_url = upload(s3, base_img, base_key)
        print(f"  ✓ Uploaded! Public URL:")
        print(f"    {base_url}")
        mongo_update(db, char_name, {"stage1.status": "complete", "stage1.image_key": base_key,
                                      "stage1.image_url": base_url, "images.base": base_key})
        torch.cuda.empty_cache()

        # ── Stage 2: Detail pass ──────────────────────────────────────────────
        print()
        print("  [STEP 4/5] STAGE 2 — Detail Pass")
        print("  ┌──────────────────────────────────────────────────────────────┐")
        print("  │ • Takes the Stage 1 image as input                           │")
        print("  │ • NO ControlNet this time — AI has full creative freedom      │")
        print("  │ • Denoise=0.35 → changes 35% of the image                    │")
        print("  │ • Adds: clothing details, textures, colors, materials         │")
        print("  │ • The Stage 2 prompt has ALL the visual identity info         │")
        print("  │ • Result: polished, detailed, game-ready character            │")
        print("  └──────────────────────────────────────────────────────────────┘")
        print(f"  Prompt (first 100 chars): {char['stage2_prompt'][:100]}...")
        print()
        print("  ⏳ Running Stage 2 AI inference... (this takes ~30-60 seconds)")

        s2_pos = char["stage2_prompt"]
        s2_neg = char["stage2_negative"]
        t0 = time.time()
        mongo_update(db, char_name, {"stage2.status": "generating"})

        with torch.no_grad():
            result = pipe_i2i(
                prompt=s2_pos, negative_prompt=s2_neg,
                image=base_img.resize((IMG_SIZE, IMG_SIZE)),
                strength=STAGE2_STRENGTH, num_inference_steps=STAGE2_STEPS,
                guidance_scale=STAGE2_CFG,
            )

        stage2_time = time.time() - t0
        refined_img = result.images[0]
        refined_key = f"images/{BIOME_ID}/{char_name}_refined_v2.png"

        # ── Step 5: Upload Stage 2 to S3 ──────────────────────────────────────
        print(f"  ✓ Stage 2 done in {stage2_time:.1f}s")
        print()
        print("  [STEP 5/5] Uploading final image to S3...")
        refined_url = upload(s3, refined_img, refined_key)
        print(f"  ✓ Uploaded! Final character URL:")
        print(f"    {refined_url}")

        mongo_update(db, char_name, {
            "generation_stage": "image_complete",
            "status":           "image_complete",
            "stage2.status":    "complete",
            "stage2.image_key": refined_key,
            "stage2.image_url": refined_url,
            "images.refined":   refined_key,
            "images.final":     refined_key,
            "image_url":        refined_url,
        })
        torch.cuda.empty_cache()

        results[char_name] = {"base": base_url, "refined": refined_url}

        print()
        print(f"  ✅ {char_name} COMPLETE!")
        print(f"     Stage 1 (pose only): {base_url}")
        print(f"     Stage 2 (final):     {refined_url}")
        print(f"     Total time: {stage1_time + stage2_time:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for char_name, r in results.items():
        print(f"\n  {char_name}:")
        print(f"    Stage 1 (base)   : {r['base']}")
        print(f"    Stage 2 (refined): {r['refined']}")


if __name__ == "__main__":
    main()
