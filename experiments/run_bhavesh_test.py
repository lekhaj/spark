#!/usr/bin/env python3
"""
run_bhavesh_test.py — SANDBOX PIPELINE TEST
============================================
Branch : bhavesh-dev  |  GPU : spark_l4

3-stage pipeline:
  STAGE 0 : Flux concept generation (768x1024) — EXACT server match
  STAGE 1 : Normalize to 512x512              — EXACT server match
  STAGE 2 : SD1.5 T-pose conversion           — OUR IMPROVEMENTS

⚙️  TWO FLAGS TO CONTROL THE RUN:

  CHARACTER_NAME      → which character to test
                        "cultivation_youth"  — original (gray hanfu, wide sleeves)
                        "iron_soldier"       — NEW (fitted plate armor, no wide sleeves)
                        Purpose: prove whether ghost hands are hanfu-specific or general

  SKIP_FLUX_STAGES    → True  = load Stage 0+1 from S3 (fast re-run, saves 5 min)
                        False = generate fresh Flux concept (first run for a character)

  RULE: First time running a character → SKIP_FLUX_STAGES = False
        All re-runs of same character  → SKIP_FLUX_STAGES = True
"""

import io
import logging
import os
import sys
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
# Force the experiments/ directory into sys.path so Python always finds modules
# regardless of which directory you run the script from.
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
logger = logging.getLogger("BhaveshTest")

# ── Sandbox model imports ─────────────────────────────────────────────────────
# Import directly — _here (experiments/) is guaranteed to be in sys.path above
from sd_model_bhavesh_v1 import load_sd, run_stage1

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
# ⚙️  CHANGE THESE TWO FLAGS TO CONTROL THE RUN
# ═════════════════════════════════════════════════════════════════════════════

CHARACTER_NAME   = "cultivation_youth"     # ← "cultivation_youth", "iron_soldier", or "lion_mount"
SKIP_FLUX_STAGES = False            # ← False to regenerate Flux with new 3/4 pose prompt

# ═════════════════════════════════════════════════════════════════════════════

# S3 keys are character-specific so runs never overwrite each other
S3_KEY_STAGE0 = f"images/bhavesh_experiments/{CHARACTER_NAME}_stage0_flux.png"
S3_KEY_STAGE1 = f"images/bhavesh_experiments/{CHARACTER_NAME}_stage1_norm512.png"
S3_KEY_STAGE2 = f"images/bhavesh_experiments/{CHARACTER_NAME}_stage2_tpose.png"

# ── SD1.5 prompts ─────────────────────────────────────────────────────────────
# We now dynamically select prompts based on character type
HUMANOID_PROMPT = (
    "full body, T-pose, arms extended horizontally, legs straight, "
    "feet fully visible, white background, flat lighting"
)
HUMANOID_NEGATIVE = (
    "extra hands, ghost hands, four arms, overlapping arms, duplicate hands, "
    "cropped, missing feet, bent arms, dynamic pose, "
    "pattern, shadows, floor, cartoon, deformed limbs, mutated hands"
)

ANIMAL_PROMPT = (
    "3/4 side view, neutral standing pose, all four paws on ground, "
    "white background, flat lighting, full body"
)
ANIMAL_NEGATIVE = (
    "walking, trotting, running, leg raised, floating leg, "
    "front view, missing legs, deformed legs, sitting, lying down"
)

if CHARACTER_NAME == "lion_mount":
    SD_PROMPT   = ANIMAL_PROMPT
    SD_NEGATIVE = ANIMAL_NEGATIVE
    SD_CATEGORY = "quadruped"
    SD_CANNY_WEIGHT = 1.0   # Quadrupeds need full Canny strength to hold their shape
    SD_IP_ADAPTER = 0.6     # Normal IP-Adapter strength for animals
else:
    SD_PROMPT   = HUMANOID_PROMPT
    SD_NEGATIVE = HUMANOID_NEGATIVE
    SD_CATEGORY = "humanoid"
    SD_CANNY_WEIGHT = 0.05  # Lowered specifically to fix ghost hands
    SD_IP_ADAPTER = 0.10    # Lowered specifically to stop sleeve bleed


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
    print(f"BHAVESH PIPELINE TEST  |  Character: {CHARACTER_NAME}")
    print(f"  SKIP_FLUX_STAGES = {SKIP_FLUX_STAGES}")
    print(f"  S3 Stage0 : {S3_KEY_STAGE0}")
    print(f"  S3 Stage1 : {S3_KEY_STAGE1}")
    print(f"  S3 Stage2 : {S3_KEY_STAGE2}")
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
        print(f"  Loading: {S3_KEY_STAGE1}")
        stage1_img = download_from_s3(S3_KEY_STAGE1)
        stage0_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{S3_KEY_STAGE0}"
        stage1_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{S3_KEY_STAGE1}"
        print(f"  ✓ Loaded. Size: {stage1_img.size}\n")

    else:
        from experiments.flux_bhavesh_v1 import load_flux_production, run_flux_production

        print("─" * 70)
        print(f"STAGE 0 — Flux Concept: {CHARACTER_NAME}")
        print("─" * 70)
        flux_pipe  = load_flux_production()
        stage0_img = run_flux_production(flux_pipe, character_name=CHARACTER_NAME)

        print("\n[Stage 0] Uploading to S3...")
        stage0_url = upload_to_s3(stage0_img, S3_KEY_STAGE0)
        print(f"  → {stage0_url}\n")

        print("[VRAM] Unloading Flux...")
        del flux_pipe
        torch.cuda.empty_cache()
        print("[VRAM] Freed.\n")

        print("─" * 70)
        print("STAGE 1 — Normalize to 512")
        print("─" * 70)
        stage1_img = stage0_img.resize((512, 512), Image.LANCZOS).convert("RGB")
        stage1_url = upload_to_s3(stage1_img, S3_KEY_STAGE1)
        print(f"  → {stage1_url}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2 — SD1.5 T-Pose Conversion
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"STAGE 2 — T-Pose Conversion: {CHARACTER_NAME}")
    print("─" * 70)
    print("  Active params: openpose=1.70 | canny=0.05 | ip_adapter=0.10\n")

    sd_pipes   = load_sd("Lykon/DreamShaper")
    stage2_img = run_stage1(
        pipes=sd_pipes,
        init_img=stage1_img,
        prompt=SD_PROMPT,
        negative=SD_NEGATIVE,
        ip_adapter_image=stage1_img,
        params={
            "category": SD_CATEGORY,
            "canny_weight": SD_CANNY_WEIGHT,
            "ip_adapter_weight": SD_IP_ADAPTER
        }
    )

    print("\n[Stage 2] Uploading to S3...")
    stage2_url = upload_to_s3(stage2_img, S3_KEY_STAGE2)
    print(f"  → {stage2_url}")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DONE — {CHARACTER_NAME}")
    print("=" * 70)
    print(f"\n  [STAGE 0] Flux concept (768x1024):\n    {stage0_url}")
    print(f"\n  [STAGE 1] Normalized (512x512):\n    {stage1_url}")
    print(f"\n  [STAGE 2] T-pose result:\n    {stage2_url}")
    print()
    print("  NEXT STEPS:")
    print("  - Compare Stage 1 (original) vs Stage 2 (T-pose) visually")
    print("  - Ghost hands gone? ✅  Identity same? ✅  → Parameters are working")
    print("  - Ghost hands gone? ✅  Identity wrong? ❌ → SD1.5 structural limit, move to Shakker")
    print("  - To rerun Stage 2 only: set SKIP_FLUX_STAGES = True and run again")


if __name__ == "__main__":
    main()
