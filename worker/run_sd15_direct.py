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
BIOME_ID   = "claudetest002"

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
STAGE2_CFG      = 7.0
STAGE2_STRENGTH = 0.35

# ── Characters ────────────────────────────────────────────────────────────────
# Minimal Stage 1 prompt — "same character, correct the pose only"
# Detailed Stage 2 prompt — clothing, identity, material
CHARACTERS = {
    "cultivation_youth": {
        "creature_category": "bipedal",
        "stage1_prompt":  "same character, T-pose, arms extended horizontally, front view, orthographic, symmetrical, clean silhouette, white background",
        "stage1_negative": "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw",
        "stage2_prompt":  "best quality, masterpiece, semi-realistic, 3D game character, young male cultivator, ash gray linen hanfu robe, crossover collar, rope belt, half-up topknot, wooden hairpin, lean build, dark sharp eyes, calm expression, faded sect patch chest, front view, white background, clean game asset",
        "stage2_negative": "anime, cartoon, cel shading, 2D illustration, manga, background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, silk, brocade, aura, glow",
    },
    "suanni_lion": {
        "creature_category": "quadruped",
        "stage1_prompt":  "same creature, neutral standing, side profile view, orthographic, all four legs planted, clean silhouette, white background",
        "stage1_negative": "deformed, extra limbs, text, watermark, background, shadows, blurry, human, bipedal, nsfw, running, jumping, sitting",
        "stage2_prompt":  "best quality, masterpiece, semi-realistic, 3D game character, fantastical lion beast xianxia, amber golden fur, flame-shaped rust-orange mane, bone-white spiral horn forehead, dragon scales on chest and knees, blue luminous spine markings, dragon whiskers, cloven rear hooves, tasseled tail, amber slit-pupil eyes, majestic serene, white background, clean game asset",
        "stage2_negative": "anime, cartoon, cel shading, 2D illustration, manga, background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw, mundane lion, wings, human, sitting",
    },
}


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
    db.biomes.update_one(
        {"_id": BIOME_ID},
        {"$set": {f"possible_structures.characters.{char_name}.{k}": v
                  for k, v in fields.items()}},
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  SD1.5 Direct Runner — biome: {BIOME_ID}")
    print(f"  Stage 1: denoise={STAGE1_STRENGTH}  CFG={STAGE1_CFG}  (very light touch)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU."); sys.exit(1)
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # ── Connections ───────────────────────────────────────────────────────────
    s3 = boto3.client("s3", region_name=AWS_REGION)
    db = pymongo.MongoClient(MONGO_URI)["World_builder"]

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
    pipe_cn.enable_xformers_memory_efficient_attention()
    pipe_cn.to("cuda")

    # Shared-weight pipelines
    cn_i2i_components = dict(pipe_cn.components)
    cn_i2i_components["controlnet"] = [cn_openpose, cn_canny]
    pipe_cn_i2i = StableDiffusionControlNetImg2ImgPipeline(**cn_i2i_components)
    pipe_cn_i2i.scheduler = UniPCMultistepScheduler.from_config(pipe_cn_i2i.scheduler.config)

    i2i_components = {k: v for k, v in pipe_cn.components.items() if k != "controlnet"}
    pipe_i2i = StableDiffusionImg2ImgPipeline(**i2i_components)
    pipe_i2i.scheduler = UniPCMultistepScheduler.from_config(pipe_i2i.scheduler.config)

    print("[Models] All loaded — UNet shared.\n")

    # OpenPose T-pose skeleton reference
    openpose_ref = (
        Image.open(TPOSE_OPENPOSE_PATH).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        if os.path.exists(TPOSE_OPENPOSE_PATH)
        else Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
    )

    results = {}

    for char_name, char in CHARACTERS.items():
        print(f"{'─'*60}")
        print(f"[{char_name}]  category={char['creature_category']}")
        is_quad = char["creature_category"] == "quadruped"

        # ── Fetch Flux concept from MongoDB/S3 ───────────────────────────────
        flux_img = None
        biome_doc  = db.biomes.find_one({"_id": BIOME_ID})
        char_doc   = (biome_doc or {}).get("possible_structures", {}).get("characters", {}).get(char_name, {})
        flux_key   = (char_doc.get("flux_concept") or {}).get("image_key") or char_doc.get("images", {}).get("flux_concept")

        if flux_key:
            print(f"  [Flux] Downloading concept: {flux_key}")
            flux_img = download_s3_img(s3, flux_key)
            print(f"  [Flux] Loaded {flux_img.size}")
        else:
            print(f"  [WARN] No Flux concept in MongoDB — Stage 1 will fallback to txt2img+CN")

        # ── Stage 1: pose correction ──────────────────────────────────────────
        s1_pos = char["stage1_prompt"]
        s1_neg = char["stage1_negative"]
        print(f"  [Stage 1] prompt: {s1_pos}")
        print(f"  [Stage 1] denoise={STAGE1_STRENGTH}  CFG={STAGE1_CFG}")

        mongo_update(db, char_name, {"status": "generating", "stage1.status": "generating"})
        t0 = time.time()

        if flux_img is not None:
            init_img  = flux_img.resize((IMG_SIZE, IMG_SIZE))
            canny_img = extract_canny(flux_img)

            if is_quad:
                # Canny-only — Flux already captured correct standing pose
                print(f"  [Stage 1] Quad: Canny-only  CN={STAGE1_CN_CANNY_QUAD}")
                pipe_cn_i2i.controlnet = [cn_canny]
                with torch.no_grad():
                    result = pipe_cn_i2i(
                        prompt=s1_pos, negative_prompt=s1_neg,
                        image=init_img, control_image=canny_img,
                        strength=STAGE1_STRENGTH, num_inference_steps=STAGE1_STEPS,
                        guidance_scale=STAGE1_CFG,
                        controlnet_conditioning_scale=STAGE1_CN_CANNY_QUAD,
                        width=IMG_SIZE, height=IMG_SIZE,
                    )
            else:
                # OpenPose + Canny dual
                print(f"  [Stage 1] Bipedal: OpenPose={STAGE1_CN_OPENPOSE} + Canny={STAGE1_CN_CANNY}")
                pipe_cn_i2i.controlnet = [cn_openpose, cn_canny]
                with torch.no_grad():
                    result = pipe_cn_i2i(
                        prompt=s1_pos, negative_prompt=s1_neg,
                        image=init_img, control_image=[openpose_ref, canny_img],
                        strength=STAGE1_STRENGTH, num_inference_steps=STAGE1_STEPS,
                        guidance_scale=STAGE1_CFG,
                        controlnet_conditioning_scale=[STAGE1_CN_OPENPOSE, STAGE1_CN_CANNY],
                        width=IMG_SIZE, height=IMG_SIZE,
                    )
        else:
            # Fallback: txt2img + OpenPose only
            print(f"  [Stage 1] Fallback txt2img + OpenPose  CN=0.85")
            pipe_cn.controlnet = cn_openpose
            with torch.no_grad():
                result = pipe_cn(
                    prompt=s1_pos, negative_prompt=s1_neg,
                    image=openpose_ref,
                    num_inference_steps=STAGE1_STEPS, guidance_scale=STAGE1_CFG,
                    controlnet_conditioning_scale=0.85,
                    width=IMG_SIZE, height=IMG_SIZE,
                )

        base_img = result.images[0]
        base_key = f"images/{BIOME_ID}/{char_name}_base_v2.png"
        base_url = upload(s3, base_img, base_key)
        print(f"  [Stage 1] {time.time()-t0:.1f}s  → {base_url}")
        mongo_update(db, char_name, {"stage1.status": "complete", "stage1.image_key": base_key,
                                      "stage1.image_url": base_url, "images.base": base_key})
        torch.cuda.empty_cache()

        # ── Stage 2: detail pass ──────────────────────────────────────────────
        s2_pos = char["stage2_prompt"]
        s2_neg = char["stage2_negative"]
        print(f"  [Stage 2] denoise={STAGE2_STRENGTH}  CFG={STAGE2_CFG}")
        t0 = time.time()
        mongo_update(db, char_name, {"stage2.status": "generating"})

        with torch.no_grad():
            result = pipe_i2i(
                prompt=s2_pos, negative_prompt=s2_neg,
                image=base_img.resize((IMG_SIZE, IMG_SIZE)),
                strength=STAGE2_STRENGTH, num_inference_steps=STAGE2_STEPS,
                guidance_scale=STAGE2_CFG,
            )
        refined_img = result.images[0]
        refined_key = f"images/{BIOME_ID}/{char_name}_refined_v2.png"
        refined_url = upload(s3, refined_img, refined_key)
        print(f"  [Stage 2] {time.time()-t0:.1f}s  → {refined_url}")
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
        print(f"  [OK] {char_name} complete.\n")

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
