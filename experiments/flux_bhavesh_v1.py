#!/usr/bin/env python3
"""
Flux Concept Generator (Sandbox)
Generates the base 'human_ranger' concept using FLUX.1-schnell.
Enforces pure white background and humanistic face from the source.
"""

import io
import os
import sys
import time

import boto3
import pymongo
import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)

MONGO_URI   = os.getenv("MONGO_URI",   "mongodb://kartik:Kartikg421@18.207.13.85:27017")
S3_BUCKET   = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
AWS_REGION  = os.getenv("AWS_REGION",  "us-east-1")

BIOME_ID    = "bhavesh_batch_001"
MODEL_ID    = "black-forest-labs/FLUX.1-schnell"

CHARACTERS = {
    "human_ranger": {
        "flux_prompt": (
            "female ranger, orthographic front view, perfect T-pose, arms stretched horizontally, "
            "simple black gloves, boots, tight bun auburn hair, photorealistic human face, "
            "tactical zip jacket, combat pants, "
            "isolated on pure solid white background, no shadow, flat studio lighting, full body"
        ),
        "s3_key": f"images/{BIOME_ID}/human_ranger_flux_concept_v2.png",
        "width": 768,
        "height": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
    }
}

def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

def upload_image_to_s3(image, s3_key, s3_client):
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
    return url, s3_key

def update_mongodb(db, char_name, s3_key, image_url, prompt):
    now = time.time()
    update_path = f"possible_structures.characters.{char_name}"
    db.biomes.update_one(
        {"_id": BIOME_ID},
        {"$set": {
            f"{update_path}.flux_concept.status":       "complete",
            f"{update_path}.flux_concept.image_key":    s3_key,
            f"{update_path}.flux_concept.image_url":    image_url,
            f"{update_path}.flux_concept.prompt":       prompt,
            f"{update_path}.flux_concept.generated_at": now,
            f"{update_path}.images.flux_concept":       s3_key,
        }},
        upsert=True,
    )
    print(f"  [MongoDB] Updated {char_name}.flux_concept")

def main():
    print("=" * 70)
    print(f"  Flux Concept Generator (Sandbox) — biome: {BIOME_ID}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found.")
        sys.exit(1)

    print("[Models] Loading Flux.1-schnell in bfloat16 with STRICT VRAM management...")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload() # Extremely important to prevent OOM
    pipe.vae.enable_slicing()

    s3 = get_s3_client()
    try:
        db = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)["World_builder"]
        db.command("ping")
        print("[MongoDB] Connected ✓")
    except Exception as e:
        print(f"[MongoDB] WARNING: Cannot connect — {e}")
        db = None

    for char_name, char in CHARACTERS.items():
        print(f"\n  Generating concept for: {char_name}")
        prompt = char["flux_prompt"]
        
        t0 = time.time()
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                width=char["width"],
                height=char["height"],
                num_inference_steps=char["num_inference_steps"],
                guidance_scale=char["guidance_scale"],
            ).images[0]
        print(f"  ✓ Generated in {time.time()-t0:.1f}s")
        
        url, key = upload_image_to_s3(image, char["s3_key"], s3)
        if db:
            update_mongodb(db, char_name, key, url, prompt)

    print("\n  ✅ DONE. You can now run sd15_bhavesh_v1.py")

if __name__ == "__main__":
    main()
