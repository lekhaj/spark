#!/usr/bin/env python3
"""
run_shakker_test.py — SHAKKER LABS CONTROLNET TEST RUNNER
==========================================================
Branch : bhavesh-dev  |  GPU : spark_l4

Tests the Shakker Labs Flux ControlNet pipeline as a full
replacement for the old SD1.5 (DreamShaper) T-pose stage.

PIPELINE:
  This script ONLY runs Stage 2 (the pose conversion).
  It reuses the Stage 1 concept image that was already
  generated and saved to S3 by run_bhavesh_test.py.

⚙️  CHANGE THESE TWO FLAGS TO CONTROL THE RUN:

  CHARACTER_NAME  → which character to test
                    "cultivation_youth"  — humanoid (gray hanfu)
                    "iron_soldier"       — humanoid (plate armor)
                    "lion_mount"         — quadruped (animal)

  SHAKKER_STEPS   → how many denoising steps to use
                    20 = fast, good quality
                    28 = slower, excellent quality

COMPARISON GOAL:
  After this runs, compare the S3 URLs:
  - Old SD1.5 result : *_stage2_tpose.png
  - New Shakker result: *_shakker_tpose.png
  If Shakker is clearly better → SD1.5 pipeline is retired.
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
logger = logging.getLogger("ShakkerTest")

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

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️  CHANGE THESE FLAGS TO CONTROL THE RUN
# ═════════════════════════════════════════════════════════════════════════════

CHARACTER_NAME = "cultivation_youth"   # ← "cultivation_youth", "iron_soldier", or "lion_mount"
SHAKKER_STEPS  = 20                    # ← 20 = fast & good. 28 = excellent.

# ═════════════════════════════════════════════════════════════════════════════

# ── S3 keys ───────────────────────────────────────────────────────────────────
# Input: uses the Stage 1 image already on S3 from run_bhavesh_test.py
# Output: saves to a NEW key so it never overwrites the SD1.5 result
S3_KEY_INPUT  = f"images/bhavesh_experiments/{CHARACTER_NAME}_stage1_norm512.png"
S3_KEY_OUTPUT = f"images/bhavesh_experiments/{CHARACTER_NAME}_shakker_tpose.png"

# ── Prompts ───────────────────────────────────────────────────────────────────
# IMPORTANT: Flux does NOT need negative prompts.
# Keep prompts simple and under 77 tokens (Flux can handle more,
# but simple prompts actually work better with ControlNet).

HUMANOID_PROMPT = (
    "full body T-pose, arms extended horizontally, legs straight, "
    "feet visible, white background, flat lighting"
)

ANIMAL_PROMPT = (
    "3/4 side view, neutral standing pose, all four paws on ground, "
    "white background, flat lighting, full body"
)

ANIMAL_CHARACTERS = ["lion_mount"]

PROMPT = ANIMAL_PROMPT if CHARACTER_NAME in ANIMAL_CHARACTERS else HUMANOID_PROMPT

# ── S3 helpers ────────────────────────────────────────────────────────────────
def upload_to_s3(img: Image.Image, s3_key: str) -> str:
    import boto3
    s3  = boto3.client("s3", region_name=AWS_REGION)
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
    print(f"SHAKKER LABS TEST  |  Character: {CHARACTER_NAME}")
    print(f"  Steps   : {SHAKKER_STEPS}")
    print(f"  Prompt  : {PROMPT}")
    print(f"  Input   : {S3_KEY_INPUT}")
    print(f"  Output  : {S3_KEY_OUTPUT}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Must run on spark_l4.")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB\n")

    # ── Load concept image from S3 ─────────────────────────────────────────────
    print("─" * 70)
    print(f"Loading Stage 1 concept image from S3...")
    print("─" * 70)
    try:
        concept_img = download_from_s3(S3_KEY_INPUT)
        print(f"  ✓ Loaded  size={concept_img.size}\n")
    except Exception as e:
        print(f"[ERROR] Could not load concept from S3: {e}")
        print(f"  → Make sure you ran run_bhavesh_test.py first for '{CHARACTER_NAME}'")
        return

    # ── Load Shakker pipeline ──────────────────────────────────────────────────
    print("─" * 70)
    print("Loading Shakker Labs Flux ControlNet pipeline...")
    print("─" * 70)
    from shakker_model_bhavesh import load_shakker, run_shakker
    pipes = load_shakker()

    # ── Run Shakker stage 2 ────────────────────────────────────────────────────
    print("─" * 70)
    print(f"SHAKKER STAGE — Pose Generation: {CHARACTER_NAME}")
    print("─" * 70)

    category = "quadruped" if CHARACTER_NAME in ANIMAL_CHARACTERS else "humanoid"
    size     = (512, 512)
    print(f"  Category : {category}")
    print(f"  Size     : {size[0]}x{size[1]}\n")

    result_img = run_shakker(
        pipes=pipes,
        concept_img=concept_img,
        prompt=PROMPT,
        num_inference_steps=SHAKKER_STEPS,
        width=size[0],
        height=size[1],
    )

    # ── Upload result ──────────────────────────────────────────────────────────
    print("\n[Shakker] Uploading result to S3...")
    result_url = upload_to_s3(result_img, S3_KEY_OUTPUT)
    print(f"  → {result_url}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DONE — {CHARACTER_NAME}")
    print("=" * 70)
    input_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{S3_KEY_INPUT}"
    print(f"\n  [INPUT  — Flux concept ] {input_url}")
    print(f"  [OUTPUT — Shakker posed] {result_url}")
    print()
    print("  COMPARISON CHECKLIST:")
    print("  - Ghost hands gone?      ✅ → Shakker is the winner")
    print("  - Correct T-pose?        ✅ → Shakker is the winner")
    print("  - Identity preserved?    ✅ → Shakker is the winner")
    print("  - Any hallucinations?    ❌ → Tune controlnet_scale up (0.7 → 0.9)")
    print()
    print("  SD1.5 result for comparison:")
    sd15_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/images/bhavesh_experiments/{CHARACTER_NAME}_stage2_tpose.png"
    print(f"  [SD1.5 result] {sd15_url}")


if __name__ == "__main__":
    main()
