#!/usr/bin/env python3
"""
run_bhavesh_test.py — SANDBOX PIPELINE TEST
============================================
Branch : bhavesh-dev  |  GPU : spark_l4

Mirrors the PRODUCTION pipeline exactly as requested, in 3 stages:

  [STAGE 0] Flux Concept Generation
            Runs exact same 768x1024 generation as server's flux_concept_generator.py

  [STAGE 1] Normalize to 512
            Takes the 768x1024 image, resizes/normalizes it to 512x512
            exactly as the server does before passing to SD1.5.

  [STAGE 2] SD1.5 T-Pose Conversion (OUR IMPROVEMENTS HERE)
            This is the T-pose conversion stage. 
            We apply our sandbox fixes here:
              - ip_adapter_weight: 0.45 -> 0.25
              - canny_weight: 0.20 -> 0.0
              - passing Flux image to IP-Adapter to preserve identity.

DO NOT RUN WITH PRODUCTION FILES — sandbox experiments only.
"""

import io
import logging
import os
import sys
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("BhaveshTest")

# ── Sandbox model imports ─────────────────────────────────────────────────────
from experiments.flux_bhavesh_v1 import load_flux_production, run_flux_production
from experiments.sd_model_bhavesh_v1 import load_sd, run_stage1

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
for _p in [
    os.path.join(os.path.dirname(__file__), "..", ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env.gpu"),
]:
    if os.path.exists(_p):
        load_dotenv(_p)
        break

S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
AWS_REGION = os.getenv("AWS_REGION",    "us-east-1")

S3_KEY_STAGE0 = "images/bhavesh_experiments/stage0_flux_768x1024.png"
S3_KEY_STAGE1 = "images/bhavesh_experiments/stage1_normalized_512.png"
S3_KEY_STAGE2 = "images/bhavesh_experiments/stage2_tpose_result.png"

# SD1.5 prompt — describe POSE + BACKGROUND only.
SD_PROMPT = (
    "full body character, T-pose, arms extended horizontally, "
    "legs straight and together, head to toe, feet fully visible, "
    "pure white background, flat lighting, no shadows"
)
SD_NEGATIVE = (
    "cropped body, cut off, missing feet, missing legs, "
    "bent arms, raised arms, dynamic pose, "
    "background pattern, mandala, geometric pattern, watermark, "
    "grey background, dark background, shadows, gradient, floor, "
    "anime, cartoon, deformed limbs, extra limbs, mutated hands"
)


# ── S3 upload helper ──────────────────────────────────────────────────────────
def upload_to_s3(img, s3_key: str) -> str:
    """Upload PIL image to S3, return public URL."""
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=buf,
        ContentType="image/png",
    )
    url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    return url


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("BHAVESH PIPELINE TEST  |  Stage 0 -> Stage 1 -> Stage 2")
    print("=" * 70)

    # GPU check
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Must run on spark_l4 GPU instance.")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 0 — Flux Concept Generation
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 0 — Flux Concept Generation (Exact Server Match)")
    print("─" * 70)
    print(f"  Model  : black-forest-labs/FLUX.1-schnell")
    print(f"  Size   : 768 x 1024")
    
    flux_pipe  = load_flux_production()
    stage0_img = run_flux_production(flux_pipe)

    print("\n[Stage 0] Uploading Flux concept to S3...")
    stage0_url = upload_to_s3(stage0_img, S3_KEY_STAGE0)
    print(f"  → {stage0_url}\n")

    print("[VRAM] Unloading Flux...")
    del flux_pipe
    torch.cuda.empty_cache()
    print("[VRAM] Freed.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Normalize to 512
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 1 — Normalize to 512 (Exact Server Match)")
    print("─" * 70)
    print(f"  Action : Resizing 768x1024 image to 512x512 for SD1.5")
    
    # Resize with high quality LANCZOS filter
    stage1_img = stage0_img.resize((512, 512), Image.LANCZOS).convert("RGB")
    
    print("\n[Stage 1] Uploading Normalized image to S3...")
    stage1_url = upload_to_s3(stage1_img, S3_KEY_STAGE1)
    print(f"  → {stage1_url}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2 — SD1.5 T-Pose Conversion
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 2 — SD1.5 T-Pose Conversion (OUR IMPROVEMENTS HERE)")
    print("─" * 70)
    print("  Our experiment fixes to make T-pose better:")
    print("    1. ip_adapter_weight : 0.45 → 0.25  (less A-pose bleed)")
    print("    2. canny_weight      : 0.20 → 0.0   (disabled — fights T-pose)")
    print("    3. ip_adapter_image  : now uses Stage 1 image for identity lock\n")

    sd_pipes   = load_sd("Lykon/DreamShaper")
    stage2_img = run_stage1(
        pipes=sd_pipes,
        init_img=stage1_img,           # Stage 1 image as base
        prompt=SD_PROMPT,
        negative=SD_NEGATIVE,
        ip_adapter_image=stage1_img,   # Stage 1 image as IP-Adapter ref
    )

    print("\n[Stage 2] Uploading T-pose result to S3...")
    stage2_url = upload_to_s3(stage2_img, S3_KEY_STAGE2)
    print(f"  → {stage2_url}")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL 3 STAGES COMPLETE")
    print("=" * 70)
    print(f"\n  [STAGE 0] Flux concept (768x1024):")
    print(f"    {stage0_url}")
    print(f"\n  [STAGE 1] Normalized (512x512):")
    print(f"    {stage1_url}")
    print(f"\n  [STAGE 2] T-pose result (512x512):")
    print(f"    {stage2_url}")
    print()


if __name__ == "__main__":
    main()
