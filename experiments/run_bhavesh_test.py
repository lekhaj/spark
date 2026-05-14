#!/usr/bin/env python3
"""
run_bhavesh_test.py — SANDBOX TEST RUNNER
==========================================
Runs on GPU (spark_l4).
Imports from sd_model_bhavesh_v1.py (our sandbox AI model file).
Uploads result to S3 and prints public URL.

FIXES in this version:
  - Init image changed from GREY to WHITE → fixes grey background + full body
  - Prompt tightened for full body framing
"""

import io
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.sd_model_bhavesh_v1 import load_sd, run_stage1


def main():
    print("=" * 60)
    print("BHAVESH EXPERIMENT RUNNER — T-POSE FIX v2")
    print("=" * 60)
    print("Sandbox params vs production:")
    print("  ip_adapter_weight : 0.45 → 0.25  (less A-pose bleed)")
    print("  canny_weight      : 0.20 → 0.0   (disabled for humanoids)")
    print("Init image          : WHITE (fixes grey background + full body)")
    print()

    # 1. Load models
    print("[1/4] Loading SD1.5 + ControlNet + IP-Adapter onto GPU...")
    pipes = load_sd("Lykon/DreamShaper")
    print("Models loaded!\n")

    # WHITE init image keeps background white during generation
    # ip_image = None because we have no real Flux character in this test.
    # If we pass the white square as IP-Adapter reference, the model treats
    # "white" as the character's identity → generates white clothes on white bg.
    print("[2/4] Creating WHITE init image (no IP-Adapter in test mode)...")
    init_img = Image.new("RGB", (512, 512), (255, 255, 255))  # white bg
    ip_image = None  # ← KEY FIX: disables IP-Adapter (no real Flux image in test)

    prompt = (
        "full body female ranger, T-pose, arms extended horizontally, "
        "legs straight, head to toe, feet visible, "
        "black zip jacket, black gloves, black pants, boots, "
        "pure white background, flat lighting, no shadows"
    )
    negative = (
        "cropped, cut off, missing feet, missing legs, "
        "bent arms, raised arms, dynamic pose, "
        "grey background, shadow, gradient, floor, "
        "anime, cartoon, deformed, extra limbs"
    )

    # 3. Generate
    print("[3/4] Running generation with experimental weights...")
    result_img = run_stage1(
        pipes=pipes,
        init_img=init_img,
        prompt=prompt,
        negative=negative,
        ip_adapter_image=ip_image,
    )

    # 4. Upload to S3
    print("\n[4/4] Uploading to S3...")
    try:
        import boto3
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

        S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
        AWS_REGION = os.getenv("AWS_REGION",    "us-east-1")
        S3_KEY     = "images/bhavesh_experiments/test_tpose_result_v2.png"

        s3 = boto3.client("s3", region_name=AWS_REGION)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)
        s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf, ContentType="image/png")

        url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{S3_KEY}"
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f"\nOpen in browser to see result:")
        print(f"  {url}\n")

    except Exception as e:
        out_path = os.path.join(os.path.dirname(__file__), "test_tpose_result_v2.png")
        result_img.save(out_path)
        print(f"\nS3 upload failed: {e}")
        print(f"Saved locally: {out_path}")


if __name__ == "__main__":
    main()
