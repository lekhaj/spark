#!/usr/bin/env python3
"""
run_bhavesh_test.py — FULL PIPELINE SANDBOX TEST
=================================================
Branch : bhavesh-dev  |  Runs on : GPU (spark_l4)

Mirrors the PRODUCTION pipeline exactly:
  Stage 0 → Flux generates concept image          [flux_bhavesh_v1.py]
             IDENTICAL to production flux_concept_generator.py
  Stage 1 → SD1.5 converts concept to T-pose      [sd_model_bhavesh_v1.py]
             SANDBOX — two param changes vs production:
               ip_adapter_weight : 0.45 → 0.25
               canny_weight      : 0.20 → 0.0

Uploads BOTH images to fixed S3 keys (overwrites — no space waste):
  flux_concept.png      — what Flux produced (character, natural pose)
  tpose_result_v3.png   — same character converted to T-pose by SD1.5

DO NOT RUN WITH PRODUCTION FILES — sandbox only.
"""

import io
import os
import sys
import torch

# ── Path setup — must come before any local imports ───────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.flux_bhavesh_v1 import load_flux, generate_concept
from experiments.sd_model_bhavesh_v1 import load_sd, run_stage1

# ── Environment — load .env for S3 credentials ────────────────────────────────
from dotenv import load_dotenv

_env_paths = [
    os.path.join(os.path.dirname(__file__), "..", ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env.gpu"),
]
for _p in _env_paths:
    if os.path.exists(_p):
        load_dotenv(_p)
        break

S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
AWS_REGION = os.getenv("AWS_REGION",    "us-east-1")

# Fixed S3 keys — always overwrite same file (no space waste per run)
S3_KEY_FLUX = "images/bhavesh_experiments/flux_concept.png"
S3_KEY_SD   = "images/bhavesh_experiments/tpose_result_v3.png"

# SD1.5 prompt — outfit/identity comes from the Flux image via IP-Adapter.
# Keep this brief and focused on POSE + BACKGROUND only.
SD_PROMPT = (
    "full body character, T-pose, arms extended horizontally, "
    "legs straight and together, head to toe, feet fully visible, "
    "pure white background, flat lighting, no shadows"
)
SD_NEGATIVE = (
    "cropped body, cut off at legs, missing feet, missing legs, "
    "bent arms, raised arms, dynamic pose, fighting pose, "
    "background pattern, mandala, mandala circle, geometric pattern, "
    "watermark, logo, text, gradient background, grey background, "
    "dark background, shadows, floor reflection, "
    "anime, cartoon, 2d, deformed limbs, extra limbs, mutated"
)


# ── S3 Upload helper ──────────────────────────────────────────────────────────
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
    print(f"  → {url}")
    return url


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BHAVESH PIPELINE TEST  |  Flux (production) → SD1.5 T-pose (sandbox)")
    print("=" * 70)
    print()

    # GPU check
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU found. Run on the GPU instance (spark_l4).")
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM: {vram_gb:.1f} GB\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 0 — Flux Concept Generation (IDENTICAL to production)
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 0 — Flux Concept Generation  [production parameters]")
    print("─" * 70)

    flux_pipe   = load_flux()
    flux_image  = generate_concept(flux_pipe)

    print("[Stage 0] Uploading Flux concept to S3...")
    flux_url = upload_to_s3(flux_image, S3_KEY_FLUX)

    # CRITICAL: delete Flux fully before loading SD1.5
    # Both together would exceed L4 VRAM budget (24 GB)
    print("\n[VRAM] Unloading Flux to free GPU memory...")
    del flux_pipe
    torch.cuda.empty_cache()
    print("[VRAM] Freed. Loading SD1.5 next.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1 — SD1.5 T-Pose Conversion (SANDBOX — two param changes)
    # ──────────────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("STAGE 1 — SD1.5 T-Pose Conversion  [sandbox params]")
    print("─" * 70)
    print("  Sandbox changes vs production sd_model.py:")
    print("    ip_adapter_weight : 0.45 → 0.25  (less A-pose bleed from Flux image)")
    print("    canny_weight      : 0.20 → 0.0   (disabled — fabric edges fight T-pose)")
    print()
    print("  Inputs to SD1.5:")
    print("    init_img       = Flux concept image  (real character as base to modify)")
    print("    ip_adapter_img = Flux concept image  (same image for identity/outfit lock)")
    print("    openpose_ref   = T-pose skeleton      (forces arms to be horizontal)\n")

    sd_pipes = load_sd("Lykon/DreamShaper")

    tpose_image = run_stage1(
        pipes=sd_pipes,
        init_img=flux_image,            # real character from Flux — not a blank square
        prompt=SD_PROMPT,
        negative=SD_NEGATIVE,
        ip_adapter_image=flux_image,    # same real character for identity preservation
    )

    print("\n[Stage 1] Uploading T-pose result to S3...")
    sd_url = upload_to_s3(tpose_image, S3_KEY_SD)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Stage 0 — Flux concept (character, natural pose):")
    print(f"    {flux_url}")
    print(f"\n  Stage 1 — T-pose result (same character, T-pose):")
    print(f"    {sd_url}")
    print()
    print("  Compare both URLs:")
    print("  → Outfit + face should look consistent (IP-Adapter preserved identity)")
    print("  → Pose should change from Flux's natural pose → clean T-pose")
    print()


if __name__ == "__main__":
    main()
