#!/usr/bin/env python3
"""
SD First -> Flux Second Pipeline Experiment
===========================================
This is the new architectural approach.
Step 1: SD1.5 (txt2img) + OpenPose ControlNet generates the PERFECT T-pose structure.
Step 2: Flux.1-schnell (img2img) takes that perfect structure and upgrades the graphics,
        clothing details, faces, and hands.

Usage (on GPU):
  python sd_first_flux_second.py
"""

import io
import os
import sys
import time
import argparse

import boto3
import pymongo
import torch
from dotenv import load_dotenv
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# For the Flux Img2Img pass
try:
    from diffusers import FluxImg2ImgPipeline
except ImportError:
    print("[ERROR] Please upgrade diffusers to use FluxImg2ImgPipeline:")
    print("        pip install -U diffusers")
    sys.exit(1)

# ── Environment Setup ────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MONGO_URI  = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@18.207.13.85:27017")
MONGO_DB   = os.getenv("MONGO_DB", "World_builder")
S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BIOME_ID   = "bhavesh_sd_flux_001"

SD15_MODEL_ID          = "Lykon/DreamShaper"
CONTROLNET_OPENPOSE_ID = "lllyasviel/control_v11p_sd15_openpose"
FLUX_MODEL_ID          = "black-forest-labs/FLUX.1-schnell"

TPOSE_OPENPOSE_PATH = os.getenv("TPOSE_OPENPOSE_PATH", "/home/ec2-user/controlnet_refs/tpose_openpose.png")

IMG_SIZE = 768

# ── Character Config ─────────────────────────────────────────────────────────
CHARACTERS = {
    "human_ranger": {
        # SD1.5: Fix the structure. Stop it from leaning forward, and stop shoes on hands.
        "sd_prompt": (
            "full body female, standing perfectly upright, straight spine, orthographic front view, "
            "looking directly at camera, perfect anatomical T-pose, both arms extended horizontally, "
            "straight legs, hands wearing tactical gloves, feet wearing simple boots, "
            "neat tight bun hairstyle, no loose strands, clean hair silhouette, "
            "highly detailed symmetric realistic human face, natural skin texture, "
            "simple zip jacket, tight pants, pure blank white background, flat lighting"
        ),
        "sd_neg": (
            "leaning forward, bent spine, looking down, perspective, angled camera, "
            "boots on hands, shoes on hands, extra shoes, "
            "messy hair, long hair, flowing hair, wild hair, windswept hair, "
            "anime, doll, cartoon, big eyes, chibi, plastic texture, "
            "arms raised, arms up, arms bent, holding object, holding bar, holding pole, "
            "cape, cloak, skirt, dress, robe, flowing cloth, "
            "loose hair, open mouth, fused fingers, extra limbs, "
            "cropped body, cut off, missing feet, missing legs, "
            "background, cityscape, city, buildings, urban, construction, "
            "machinery, pipes, scenery, environment, sky, shadow, grey background, "
            "spiral, swirl, pattern, watermark, logo, symbol, gradient, text, design"
        ),
        
        "flux_prompt": (
            "high quality 3d game asset, beautiful female ranger, standing perfectly upright, "
            "orthographic front view, T-pose, horizontal arms, tactical zip jacket, combat pants, "
            "feet wearing knee-high boots, hands wearing detailed tactical gloves, "
            "neat tight bun auburn hair, clean hair silhouette, photorealistic human face, "
            "highly detailed symmetric facial features, natural skin texture, professional studio photography, "
            "realistic skin, real life photography style, pure blank white background, flat studio lighting"
        ),
    }
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

def upload_image_to_s3(image: Image.Image, s3_key: str, s3_client) -> str:
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

def mongo_update(db, char_name: str, fields: dict):
    if db is None:
        return
    try:
        db.biomes.update_one(
            {"_id": BIOME_ID},
            {"$set": {f"possible_structures.characters.{char_name}.{k}": v for k, v in fields.items()}},
            upsert=True,
        )
        print("  [MongoDB] Database updated successfully.")
    except Exception as e:
        print(f"  [MongoDB] Update skipped (DB unreachable): {e}")

# ── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", type=str, help="Character to run (e.g. human_ranger)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found. Must run on the GPU instance.")
        return

    char_name = args.char or "human_ranger"
    if char_name not in CHARACTERS:
        print(f"[ERROR] Character {char_name} not found.")
        return
        
    cfg = CHARACTERS[char_name]
    s3_client = get_s3_client()
    
    # Connect to Mongo
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = mongo_client[MONGO_DB]
        mongo_client.admin.command('ping')
    except Exception:
        db = None
        print("  [WARN] MongoDB not connected.")

    print("=" * 70)
    print(f"  SD First -> Flux Second Pipeline")
    print(f"  Character: {char_name}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 1: SD1.5 Txt2Img for Structure
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading SD1.5 + OpenPose ControlNet...")
    if not os.path.exists(TPOSE_OPENPOSE_PATH):
        print(f"[ERROR] OpenPose image not found: {TPOSE_OPENPOSE_PATH}")
        return

    openpose_img = Image.open(TPOSE_OPENPOSE_PATH).convert("RGB")
    openpose_img = openpose_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_OPENPOSE_ID, torch_dtype=torch.float16
    )
    sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD15_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.to("cuda")
    sd_pipe.enable_model_cpu_offload() # Save VRAM

    print("  [SD1.5] Generating perfect T-pose structure...")
    with torch.no_grad():
        sd_out = sd_pipe(
            prompt=cfg["sd_prompt"],
            negative_prompt=cfg["sd_neg"],
            image=openpose_img,
            controlnet_conditioning_scale=1.0, # 100% lock to skeleton
            num_inference_steps=25,
            guidance_scale=7.5,
            height=IMG_SIZE,
            width=IMG_SIZE,
        ).images[0]

    os.makedirs("/tmp", exist_ok=True)
    sd_out_path = f"/tmp/{char_name}_stage1_sd15.png"
    sd_out.save(sd_out_path)
    print(f"  ✓ SD1.5 done. Structure saved to {sd_out_path}")

    # CLEANUP SD1.5 completely from GPU Memory to make room for Flux
    print("  [VRAM] Unloading SD1.5 from memory...")
    del sd_pipe
    del controlnet
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE 2: Flux Img2Img for Graphics Upgrade
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Loading Flux.1-schnell Img2Img Pipeline...")
    flux_pipe = FluxImg2ImgPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16
    )
    flux_pipe.enable_sequential_cpu_offload() # Very important for VRAM!
    flux_pipe.vae.enable_slicing()

    print("  [Flux] Upgrading graphics and fixing details...")
    with torch.inference_mode():
        flux_out = flux_pipe(
            prompt=cfg["flux_prompt"],
            image=sd_out,
            strength=0.65, # 65% Flux, 35% SD base
            num_inference_steps=4, # Schnell optimized for exactly 4 steps
            guidance_scale=0.0,    # Schnell requires 0.0
        ).images[0]

    flux_out_path = f"/tmp/{char_name}_stage2_flux.png"
    flux_out.save(flux_out_path)
    print(f"  ✓ Flux done. Final image saved to {flux_out_path}")

    # CLEANUP Flux
    print("  [VRAM] Unloading Flux from memory...")
    del flux_pipe
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE 3: Upload & Database
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Uploading images to S3 and updating database...")
    s3_key_sd   = f"images/{BIOME_ID}/{char_name}_sd_structure.png"
    s3_key_flux = f"images/{BIOME_ID}/{char_name}_flux_final.png"

    url_sd   = upload_image_to_s3(sd_out, s3_key_sd, s3_client)
    url_flux = upload_image_to_s3(flux_out, s3_key_flux, s3_client)

    mongo_update(db, char_name, {
        "status": "complete",
        "sd_structure_url": url_sd,
        "image_url": url_flux,
        "updated_at": time.time()
    })

    print("\n" + "="*70)
    print("  SUCCESS! New Pipeline Complete.")
    print("="*70)
    print(f"  Structure (SD): {url_sd}")
    print(f"  Final (Flux)  : {url_flux}")

if __name__ == "__main__":
    main()
