#!/usr/bin/env python3
"""
run_bhavesh_test.py — SANDBOX PIPELINE TEST
============================================
Branch : bhavesh-dev  |  GPU : spark_l4

3-stage pipeline:
  STAGE 0 : Flux concept generation (768x1024) — EXACT server match
  STAGE 1 : Normalize to 512x512             — EXACT server match
  STAGE 2 : SD1.5 T-pose conversion          — OUR IMPROVEMENTS

SKIP_FLUX_STAGES = True
  → Skips Stage 0 and Stage 1 and loads already-generated images from S3.
  → Use this to save time on re-runs (Flux takes 3-5 minutes).
  → Set to False only when you need to regenerate the Flux concept.
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

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONTROL FLAG
#
#  True  → SKIP Stage 0 (Flux) and Stage 1 (normalize).
#          Load the already-generated images directly from S3.
#          USE THIS for all re-runs to save 3-5 min of Flux time.
#
#  False → Run Stage 0 + Stage 1 fresh (only needed if you want a new
#          Flux concept image).
# ─────────────────────────────────────────────────────────────────────────────
SKIP_FLUX_STAGES = True   # ← Change to False only to regenerate the Flux image

# Fixed S3 keys
S3_KEY_STAGE0 = "images/bhavesh_experiments/stage0_flux_768x1024.png"
S3_KEY_STAGE1 = "images/bhavesh_experiments/stage1_normalized_512.png"
S3_KEY_STAGE2 = "images/bhavesh_experiments/stage2_tpose_result.png"

# ── SD1.5 prompts ─────────────────────────────────────────────────────────────
# Stage 2 prompt: describe POSE + BACKGROUND only.
# Outfit/identity comes from ip_adapter_image (the normalized Flux image).
SD_PROMPT = (
    "full body character, T-pose, arms extended horizontally, "
    "legs straight and together, head to toe, feet fully visible, "
    "pure white background, flat lighting, no shadows"
)
# UPDATED NEGATIVES: now explicitly target the ghost-hand hallucination
SD_NEGATIVE = (
    # Ghost hand / extra limb terms (new — target the specific hallucination)
    "extra hands, multiple hands, ghost hands, floating hands, "
    "hands on chest, hands on torso, arms in front of body, "
    "four arms, overlapping arms, duplicate hands, extra arms, "
    # Original negatives
    "cropped body, cut off, missing feet, missing legs, "
    "bent arms, raised arms, dynamic pose, "
    "background pattern, mandala, geometric pattern, watermark, "
    "grey background, dark background, shadows, gradient, floor, "
    "anime, cartoon, deformed limbs, extra limbs, mutated hands"
)


# ── S3 helpers ────────────────────────────────────────────────────────────────
def upload_to_s3(img, s3_key: str) -> str:
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf, ContentType="image/png")
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


def download_from_s3(s3_key: str) -> Image.Image:
    import boto3
    s3  = boto3.client("s3", region_name=AWS_REGION)
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("BHAVESH PIPELINE TEST  |  Stage 0 → Stage 1 → Stage 2")
    print(f"  SKIP_FLUX_STAGES = {SKIP_FLUX_STAGES}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Must run on spark_l4.")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 0 + 1 — Either generate fresh or load from S3
    # ──────────────────────────────────────────────────────────────────────────
    if SKIP_FLUX_STAGES:
        print("─" * 70)
        print("STAGE 0+1 — SKIPPED (loading from S3)")
        print("─" * 70)
        print(f"  Loading Stage 1 (normalized 512x512) from S3...")
        print(f"  Key: {S3_KEY_STAGE1}")
        stage1_img = download_from_s3(S3_KEY_STAGE1)
        stage0_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{S3_KEY_STAGE0}"
        stage1_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{S3_KEY_STAGE1}"
        print(f"  ✓ Loaded. Size: {stage1_img.size}\n")

    else:
        # Import Flux only when needed (slow to load)
        from experiments.flux_bhavesh_v1 import load_flux_production, run_flux_production

        print("─" * 70)
        print("STAGE 0 — Flux Concept Generation (Exact Server Match)")
        print("─" * 70)
        flux_pipe  = load_flux_production()
        stage0_img = run_flux_production(flux_pipe)

        print("[Stage 0] Uploading to S3...")
        stage0_url = upload_to_s3(stage0_img, S3_KEY_STAGE0)
        print(f"  → {stage0_url}\n")

        print("[VRAM] Unloading Flux...")
        del flux_pipe
        torch.cuda.empty_cache()
        print("[VRAM] Freed.\n")

        print("─" * 70)
        print("STAGE 1 — Normalize to 512 (Exact Server Match)")
        print("─" * 70)
        stage1_img = stage0_img.resize((512, 512), Image.LANCZOS).convert("RGB")
        stage1_url = upload_to_s3(stage1_img, S3_KEY_STAGE1)
        print(f"  → {stage1_url}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2 — SD1.5 T-Pose Conversion (OUR IMPROVEMENTS)
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 2 — SD1.5 T-Pose Conversion (Ghost Hand Fix Applied)")
    print("─" * 70)
    print("  Changes to fix ghost hands:")
    print("    openpose_weight : 1.30 → 1.70  (skeleton dominates more strongly)")
    print("    canny_weight    : 0.0  → 0.05  (tiny structural anchor vs floating ghosts)")
    print("    ip_adapter_weight: 0.25 → 0.10 (stops sleeve shape encoding as arm)")
    print("    negatives       : + explicit ghost-hand terms\n")

    sd_pipes   = load_sd("Lykon/DreamShaper")
    stage2_img = run_stage1(
        pipes=sd_pipes,
        init_img=stage1_img,
        prompt=SD_PROMPT,
        negative=SD_NEGATIVE,
        ip_adapter_image=stage1_img,
    )

    print("\n[Stage 2] Uploading to S3...")
    stage2_url = upload_to_s3(stage2_img, S3_KEY_STAGE2)
    print(f"  → {stage2_url}")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETE")
    print("=" * 70)
    print(f"\n  [STAGE 0] Flux concept (768x1024):")
    print(f"    {stage0_url}")
    print(f"\n  [STAGE 1] Normalized (512x512):")
    print(f"    {stage1_url}")
    print(f"\n  [STAGE 2] T-pose result:")
    print(f"    {stage2_url}")
    print()
    print("  If ghost hands are still present:")
    print("    → Open sd_model_bhavesh_v1.py and try ip_adapter_image=None next")


if __name__ == "__main__":
    main()
