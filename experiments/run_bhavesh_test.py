#!/usr/bin/env python3
"""
run_bhavesh_test.py — SANDBOX PIPELINE TEST
============================================
Branch : bhavesh-dev  |  GPU : spark_l4

Mirrors the PRODUCTION pipeline exactly, with ONE improvement in Stage 1:

  Stage 0  Flux concept generation
           ─────────────────────────────────────────────────────────────────
           Model   : black-forest-labs/FLUX.1-schnell
           Loader  : flux_model_bhavesh_v1.py  ← EXACT copy of models/flux_model.py
           Prompt  : cultivation_youth from flux_concept_generator.py (verbatim)
           Size    : 512 × 512  (production MAX_SIZE cap)
           Params  : steps=4, guidance=0.0  (same as production)

  Stage 1  SD1.5 T-pose conversion
           ─────────────────────────────────────────────────────────────────
           Model   : Lykon/DreamShaper
           Loader  : sd_model_bhavesh_v1.py  ← sandbox copy of models/sd_model.py
           Changes vs production:
             ip_adapter_weight : 0.45 → 0.25  (less A-pose bleed)
             canny_weight      : 0.20 → 0.0   (disabled — fabric edges fight T-pose)
           OUR IMPROVEMENT:
             ip_adapter_image = flux_image     ← production passes None here
             This passes the Flux concept image to IP-Adapter for identity lock.

Uploads to fixed S3 keys (always overwrite — no space waste):
  flux_concept.png     → Stage 0 output  (Flux, natural pose)
  tpose_result_v3.png  → Stage 1 output  (SD1.5, T-pose)

Compare both URLs to verify:
  - Outfit / face consistent between the two  (IP-Adapter identity lock)
  - Pose changes from Flux's natural pose → clean T-pose

DO NOT RUN WITH PRODUCTION FILES — sandbox experiments only.
"""

import io
import logging
import os
import sys
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("BhaveshTest")

# ── Sandbox model imports ─────────────────────────────────────────────────────
from experiments.flux_model_bhavesh_v1 import load_flux, run_flux   # exact production copy
from experiments.sd_model_bhavesh_v1 import load_sd, run_stage1     # sandbox improvements

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

# Fixed S3 keys — always overwrite same files (no space waste)
S3_KEY_FLUX = "images/bhavesh_experiments/flux_concept.png"
S3_KEY_SD   = "images/bhavesh_experiments/tpose_result_v3.png"

# ── Character prompt ──────────────────────────────────────────────────────────
# Copied VERBATIM from flux_concept_generator.py → cultivation_youth → flux_prompt
FLUX_PROMPT = (
    "young male cultivator, T-pose, arms extended horizontally, legs straight, "
    "orthographic front view, symmetrical, centered, "
    "plain gray hanfu robe, rope belt, lean build, topknot hair, "
    "simple clothing, clean silhouette, minimal detail, "
    "game-ready character design, "
    "white background, flat lighting, full body, head to toe"
)

# SD1.5 prompt — describe POSE + BACKGROUND only.
# Outfit/identity comes from the Flux image passed via ip_adapter_image.
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

# Flux generation params — same as production (512×512 MAX_SIZE cap)
FLUX_PARAMS = {
    "width":          512,
    "height":         512,
    "steps":          4,
    "guidance_scale": 0.0,
}


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
    print("BHAVESH TEST  |  Flux (production) → SD1.5 T-Pose (sandbox fix)")
    print("=" * 70)

    # GPU check
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Must run on spark_l4 GPU instance.")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 0 — Flux Concept Generation (identical to production)
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 0 — Flux  [IDENTICAL to production models/flux_model.py]")
    print("─" * 70)
    print(f"  Model  : black-forest-labs/FLUX.1-schnell")
    print(f"  Size   : {FLUX_PARAMS['width']} x {FLUX_PARAMS['height']}  (production 512 cap)")
    print(f"  Steps  : {FLUX_PARAMS['steps']}")
    print(f"  Prompt : {FLUX_PROMPT[:80]}...\n")

    flux_pipe  = load_flux("black-forest-labs/FLUX.1-schnell")
    flux_image = run_flux(flux_pipe, FLUX_PROMPT, FLUX_PARAMS)

    print("[Stage 0] Uploading Flux concept to S3...")
    flux_url = upload_to_s3(flux_image, S3_KEY_FLUX)
    print(f"  → {flux_url}\n")

    # Unload Flux completely — L4 cannot hold both Flux + SD1.5 in VRAM
    print("[VRAM] Unloading Flux...")
    del flux_pipe
    torch.cuda.empty_cache()
    print("[VRAM] Freed.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1 — SD1.5 T-Pose Conversion (sandbox improvements)
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 1 — SD1.5  [SANDBOX — 2 param changes + IP-Adapter improvement]")
    print("─" * 70)
    print("  Sandbox changes vs production:")
    print("    ip_adapter_weight : 0.45 → 0.25  (less A-pose bleed)")
    print("    canny_weight      : 0.20 → 0.0   (disabled — fights T-pose)")
    print("  Our experiment:")
    print("    ip_adapter_image  : None → flux_image  (identity lock improvement)")
    print("    (Production _handle_sd_stage1 does not pass ip_adapter_image)\n")

    sd_pipes    = load_sd("Lykon/DreamShaper")
    tpose_image = run_stage1(
        pipes=sd_pipes,
        init_img=flux_image,           # same as production: Flux image as the base
        prompt=SD_PROMPT,
        negative=SD_NEGATIVE,
        ip_adapter_image=flux_image,   # OUR IMPROVEMENT: also pass as IP-Adapter ref
    )

    print("\n[Stage 1] Uploading T-pose result to S3...")
    sd_url = upload_to_s3(tpose_image, S3_KEY_SD)
    print(f"  → {sd_url}")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\n  Stage 0  Flux concept (natural pose):")
    print(f"    {flux_url}")
    print(f"\n  Stage 1  T-pose result:")
    print(f"    {sd_url}")
    print()
    print("  What to check:")
    print("    1. Does Stage 1 preserve the outfit + face from Stage 0? (IP-Adapter)")
    print("    2. Are the arms perfectly horizontal? (OpenPose skeleton)")
    print("    3. Is the background clean white?")
    print("    4. Is the full body visible head to toe?")
    print()


if __name__ == "__main__":
    main()
