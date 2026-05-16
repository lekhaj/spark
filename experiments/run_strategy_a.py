#!/usr/bin/env python3
"""
run_strategy_a.py — SHAKKER LABS STRATEGY A RUNNER
====================================================
Branch : bhavesh-dev  |  GPU : spark_l4

STRATEGY A PIPELINE:
  Text Prompt + OpenPose Skeleton
          ↓
  Shakker Labs FLUX.1-dev-ControlNet-Union-Pro (mode=4, POSE)
          ↓
  High-Quality T-Pose Character Image
          ↓
  Normalize to 512x512
          ↓
  Upload to S3 → Ready for Trellis 3D

WHY NO SD1.5:
  SD1.5 (2022) hallucinated a random woman when given our hanfu character.
  Strategy A goes directly from text → T-pose using Flux (2024 architecture).
  Same model that understands the character is the one posing it.
  No identity gap. No ghost hands. Clean output for Trellis.

SWITCHING TO STRATEGY B:
  If you need to preserve a specific concept image's face/clothing:
  1. Run: python experiments/run_bhavesh_test.py  (generates concept → S3)
  2. Edit run_shakker_test.py → change control_mode to CANNY_MODE
  Strategy B is ready to go at any time without changing this file.

⚙️  CHANGE THESE FLAGS:
  CHARACTER_NAME → which character to generate
  SHAKKER_STEPS  → quality (20=fast, 28=excellent)
"""

import io
import logging
import os
import sys

import torch
from PIL import Image

# ── Bulletproof path setup ────────────────────────────────────────────────────
# Works whether you run from spark/ or spark/experiments/
_here = os.path.abspath(os.path.dirname(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)                              # spark/experiments/
_root = os.path.abspath(os.path.join(_here, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)                              # spark/

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("StrategyA")

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
for _p in [
    os.path.join(_root, ".env"),
    os.path.join(_root, ".env.gpu"),
]:
    if os.path.exists(_p):
        load_dotenv(_p)
        break

S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
AWS_REGION = os.getenv("AWS_REGION",    "us-east-1")

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️  CHANGE THESE FLAGS TO CONTROL THE RUN
# ═════════════════════════════════════════════════════════════════════════════

CHARACTER_NAME = "red_hair_adventurer" # ← New female character test
SHAKKER_STEPS  = 4                   # ← 4 steps required for FLUX.1-schnell

# ═════════════════════════════════════════════════════════════════════════════

# ── S3 output key ─────────────────────────────────────────────────────────────
# Saves to a UNIQUE key so it never overwrites Stage 0 or SD1.5 results
S3_KEY_OUTPUT = f"images/bhavesh_experiments/{CHARACTER_NAME}_strategyA_tpose.png"

# ── Character definitions ─────────────────────────────────────────────────────
# Strategy A does NOT use a concept image. It uses these text prompts directly.
# Keep under 77 tokens. No negative prompts — Flux doesn't support them.
CHARACTERS = {
    "red_hair_adventurer": {
        "prompt": (
            "young female adventurer, T-pose, arms extended horizontally, "
            "long red hair, blue peasant blouse with red trim, brown leather corset, "
            "dark pants, clean design, white background, flat studio lighting, full body, feet visible"
        ),
        "width":  512,
        "height": 512,
        "skeleton_size": (512, 512),
    },
    "hooded_assassin": {
        "prompt": (
            "young male assassin, T-pose, arms extended horizontally, "
            "wearing a simple dark hood, white and brown leather tunic armor, "
            "gauntlets, tall brown boots, clean design, no complex headwear, "
            "white background, flat studio lighting, full body, feet visible"
        ),
        "width":  512,
        "height": 512,
        "skeleton_size": (512, 512),
    },
    "cultivation_youth": {
        "prompt": (
            "young male cultivator, T-pose, arms extended horizontally, "
            "plain gray hanfu robe, rope belt, topknot hair, lean build, "
            "white background, flat studio lighting, full body, feet visible"
        ),
        "width":  512,
        "height": 512,
        "skeleton_size": (512, 512),   # OpenPose skeleton size (must match output)
    },
    "iron_soldier": {
        "prompt": (
            "young male soldier, T-pose, arms extended horizontally, "
            "dark iron plate armor, pauldrons, gauntlets, short dark hair, "
            "white background, flat studio lighting, full body, feet visible"
        ),
        "width":  512,
        "height": 512,
        "skeleton_size": (512, 512),
    },
    "lion_mount": {
        "prompt": (
            "majestic lion, neutral symmetrical standing pose, orthographic 3/4 side profile, "
            "four paws flat on ground, legs slightly bent, stiff horizontal tail extending straight backward, "
            "mouth slightly open, clean design, white background, flat studio lighting, full body"
        ),
        "width":  512,
        "height": 512,
        "skeleton_size": (512, 512),
        "is_animal": True,             # ← bypasses humanoid ControlNet
    },
    "lion_mount_hack": {
        "prompt": (
            "majestic lion, neutral symmetrical standing pose, orthographic 3/4 side profile, "
            "four paws flat on ground, legs slightly bent, tail held high in the air, "
            "panting heavily, clean design, white background, flat studio lighting, full body"
        ),
        "width":  512,
        "height": 512,
        "skeleton_size": (512, 512),
        "is_animal": True,
    },
}

# ── S3 helper ─────────────────────────────────────────────────────────────────
def upload_to_s3(img: Image.Image, s3_key: str) -> str:
    import boto3
    s3  = boto3.client("s3", region_name=AWS_REGION)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf, ContentType="image/png")
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Validate character ─────────────────────────────────────────────────────
    if CHARACTER_NAME not in CHARACTERS:
        print(f"[ERROR] Unknown character: '{CHARACTER_NAME}'")
        print(f"  Valid options: {list(CHARACTERS.keys())}")
        return

    cfg = CHARACTERS[CHARACTER_NAME]

    print("=" * 70)
    print(f"STRATEGY A — Direct T-Pose  |  Character: {CHARACTER_NAME}")
    print(f"  Steps   : {SHAKKER_STEPS}")
    print(f"  Prompt  : {cfg['prompt']}")
    print(f"  Output  : {S3_KEY_OUTPUT}")
    print("=" * 70)

    # ── GPU check ─────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. This script must run on spark_l4 GPU.")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB\n")

    # ── Build control image ───────────────────────────────────────────────────
    from shakker_model_bhavesh import (
        load_shakker, run_shakker, POSE_MODE, CANNY_MODE, DEFAULT_CTRL_SCALE
    )

    is_animal = cfg.get("is_animal", False)
    use_canny = cfg.get("use_canny_mode", False)
    
    control_mode = CANNY_MODE if use_canny else POSE_MODE
    ctrl_scale   = DEFAULT_CTRL_SCALE

    print("─" * 70)
    if is_animal:
        # Quadruped test: OpenPose is human-only, so we completely disable ControlNet
        # by sending a blank image and setting the scale to 0.0.
        # This lets pure Flux generate the perfect orthographic side profile from text alone.
        print("Quadruped detected — Disabling ControlNet. Using pure text-to-image Flux!")
        print("─" * 70)
        control_img = Image.new("RGB", (cfg["width"], cfg["height"]), (0, 0, 0))
        control_mode = POSE_MODE
        ctrl_scale = 0.0
        print(f"  ✓ Blank control image created  size={control_img.size}\n")
    elif use_canny:
        # Strategy B fallback
        print("Strategy B detected — loading concept image from S3 for Canny mode...")
        print("─" * 70)
        import boto3
        s3_concept_key = f"images/bhavesh_experiments/{CHARACTER_NAME}_stage1_norm512.png"
        s3  = boto3.client("s3", region_name=AWS_REGION)
        buf = io.BytesIO()
        try:
            s3.download_fileobj(S3_BUCKET, s3_concept_key, buf)
            buf.seek(0)
            control_img = Image.open(buf).convert("RGB")
            print(f"  ✓ Loaded concept  size={control_img.size}\n")
        except Exception as e:
            print(f"[ERROR] Could not load concept from S3: {e}")
            return
    else:
        # For humanoids: generate the OpenPose T-pose skeleton programmatically
        print("Generating T-pose OpenPose skeleton (Strategy A)...")
        print("─" * 70)
        from openpose_humanoid import generate_tpose_skeleton
        w, h = cfg["skeleton_size"]
        control_img = generate_tpose_skeleton(width=w, height=h)
        print(f"  ✓ Skeleton generated  size={control_img.size}\n")

    # ── Load Shakker pipeline ──────────────────────────────────────────────────
    print("─" * 70)
    mode_name = "CANNY (Strategy B)" if use_canny else "POSE mode=4 (Strategy A)"
    print(f"Loading Shakker Labs FLUX ControlNet ({mode_name})...")
    print("─" * 70)
    pipes = load_shakker()

    # ── Run inference ──────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"Generating T-pose character: {CHARACTER_NAME}")
    print("─" * 70)

    result_img = run_shakker(
        pipes=pipes,
        control_image=control_img,
        prompt=cfg["prompt"],
        control_mode=control_mode,
        controlnet_scale=ctrl_scale,   # ← Passes 0.0 for animals, 0.65 for humans
        num_inference_steps=SHAKKER_STEPS,
        width=cfg["width"],
        height=cfg["height"],
        seed=404,                      # ← changed seed so it draws a fresh layout
    )

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    print("\n[S3] Uploading result...")
    result_url = upload_to_s3(result_img, S3_KEY_OUTPUT)
    print(f"  → {result_url}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DONE — {CHARACTER_NAME}")
    print("=" * 70)
    print(f"\n  [RESULT] {result_url}")
    print()
    print("  QUALITY CHECKLIST:")
    print("  - T-pose arms horizontal?        ✅ → Strategy A works")
    print("  - Correct character clothing?    ✅ → Prompt is strong enough")
    print("  - White background?              ✅ → Ready for Trellis")
    print("  - Ghost hands or extra limbs?    ❌ → Raise controlnet_scale (0.65 → 0.8)")
    print("  - Wrong gender / wrong clothes?  ❌ → Make the prompt more specific")
    print()
    print("  Next step: Feed this image to Trellis for 3D generation!")


if __name__ == "__main__":
    main()
