#!/usr/bin/env python3
"""
run_bhavesh_test.py — FULL PIPELINE SANDBOX TEST
=================================================
Runs on GPU (spark_l4 / bhavesh-dev branch only).

Mirrors the PRODUCTION pipeline exactly:
  Stage 0 → Flux generates concept character image   (flux_bhavesh_v1.py)
  Stage 1 → SD1.5 converts it to T-pose              (sd_model_bhavesh_v1.py)

Uploads BOTH images to S3 so you can compare:
  - What Flux generated (the concept, any pose)
  - What SD1.5 produced (same character, T-pose)

DO NOT RUN WITH PRODUCTION FILES — sandbox only.
"""

import io
import os
import sys
import torch
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.flux_bhavesh_v1 import load_flux, generate_concept
from experiments.sd_model_bhavesh_v1 import load_sd, run_stage1

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
AWS_REGION = os.getenv("AWS_REGION",    "us-east-1")

# Fixed S3 keys — always overwrite same file (no space waste)
S3_KEY_FLUX = "images/bhavesh_experiments/flux_concept.png"     # Flux output
S3_KEY_SD   = "images/bhavesh_experiments/tpose_result_v3.png"  # SD1.5 T-pose output

# SD1.5 prompt — describes what we want but identity/outfit comes from Flux image
SD_PROMPT = (
    "full body female ranger, T-pose, arms extended horizontally, "
    "legs straight, head to toe, feet visible, "
    "pure white background, flat lighting, no shadows"
)
SD_NEGATIVE = (
    "cropped, cut off, missing feet, missing legs, "
    "bent arms, raised arms, dynamic pose, "
    "background pattern, mandala, circle, geometric, watermark, "
    "grey background, shadow, gradient, floor, "
    "anime, cartoon, deformed, extra limbs, mutated"
)


# ── S3 Upload helper ──────────────────────────────────────────────────────────
def upload_to_s3(img, s3_key: str) -> str:
    import boto3
    from PIL import Image
    s3 = boto3.client("s3", region_name=AWS_REGION)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf, ContentType="image/png")
    url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    print(f"  → {url}")
    return url


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BHAVESH PIPELINE TEST — Flux → SD1.5 T-Pose (mirrors production)")
    print("=" * 70)
    print()
    print("  Stage 0: Flux generates concept character")
    print("  Stage 1: SD1.5 converts concept to T-pose")
    print()

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found. Run this on the GPU instance (spark_l4).")
        return

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB\n")

    # ── STAGE 0: Flux Concept Generation ─────────────────────────────────────
    print("─" * 70)
    print("STAGE 0 — Flux Concept Generation")
    print("─" * 70)

    flux_pipe = load_flux()
    flux_image = generate_concept(flux_pipe)

    print("[Stage 0] Uploading Flux concept to S3...")
    flux_url = upload_to_s3(flux_image, S3_KEY_FLUX)

    # CRITICAL: Unload Flux completely before loading SD1.5
    # Both models together would exceed L4 VRAM (24GB)
    print("\n[VRAM] Unloading Flux to free GPU memory for SD1.5...")
    del flux_pipe
    torch.cuda.empty_cache()
    print("[VRAM] Flux unloaded. GPU memory freed.\n")

    # ── STAGE 1: SD1.5 T-Pose Conversion ──────────────────────────────────────
    print("─" * 70)
    print("STAGE 1 — SD1.5 T-Pose Conversion")
    print("─" * 70)
    print("Sandbox changes vs production:")
    print("  ip_adapter_weight : 0.45 → 0.25  (less A-pose bleed)")
    print("  canny_weight      : 0.20 → 0.0   (disabled — fights T-pose on humanoids)")
    print()

    sd_pipes = load_sd("Lykon/DreamShaper")

    print("[Stage 1] Running T-pose conversion...")
    print("  init_img        = Flux concept image  ← real character as base")
    print("  ip_adapter_img  = Flux concept image  ← same image for identity lock")
    print("  openpose_ref    = T-pose skeleton      ← forces arms horizontal\n")

    tpose_image = run_stage1(
        pipes=sd_pipes,
        init_img=flux_image,          # Flux image as the base to modify
        prompt=SD_PROMPT,
        negative=SD_NEGATIVE,
        ip_adapter_image=flux_image,  # Same Flux image for identity preservation
    )

    print("\n[Stage 1] Uploading T-pose result to S3...")
    sd_url = upload_to_s3(tpose_image, S3_KEY_SD)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  [Stage 0] Flux concept (any pose):")
    print(f"    {flux_url}")
    print(f"\n  [Stage 1] T-pose result (same character, T-pose):")
    print(f"    {sd_url}")
    print()
    print("  Open both URLs to compare — outfit/face should be consistent,")
    print("  but pose should change from Flux's natural pose → perfect T-pose.")
    print()


if __name__ == "__main__":
    main()
