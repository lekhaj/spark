#!/usr/bin/env python3
"""
Test script for the new SD1.5 T-pose parameters (bhavesh-dev branch).
Sandbox file — uses sd_model_bhavesh_v1.py, NOT the production sd_model.py
Uploads result to S3 and prints a public URL (same as old sd15_bhavesh_v1.py)
"""

import io
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.sd_model_bhavesh_v1 import load_sd, run_stage1


def main():
    print("=" * 60)
    print("BHAVESH EXPERIMENT RUNNER — T-POSE FIX")
    print("=" * 60)
    print("Changes vs production:")
    print("  ip_adapter_weight : 0.45 -> 0.25  (less A-pose bleed)")
    print("  canny_weight      : 0.20 -> 0.0   (disabled for humanoids)")
    print()

    # 1. Load models
    print("[1/4] Loading SD1.5 + ControlNet + IP-Adapter onto GPU...")
    pipes = load_sd("Lykon/DreamShaper")
    print("Models loaded!\n")

    # 2. Mock Flux input (grey square — no real Flux image in sandbox test)
    print("[2/4] Creating mock Flux input image...")
    init_img = Image.new("RGB", (512, 512), (128, 128, 128))
    ip_image = init_img.copy()

    prompt = (
        "female ranger, full body, head to toe, completely showing feet, "
        "perfect T-pose, arms horizontally straight, simple black gloves, "
        "black zip jacket, realistic human face, "
        "fully simple white background, purely solid white, nothing in background"
    )
    negative = (
        "cropped body, out of frame, missing legs, missing feet, "
        "gradient, fog, dark shadows at bottom, floor, "
        "arms raised, bent arms, dark background, grey background, "
        "bare hands, exposed fingers, anime, cartoon"
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

    # 4. Upload to S3 — same bucket/pattern as old scripts
    print("\n[4/4] Uploading to S3...")
    try:
        import boto3
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

        S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
        AWS_REGION = os.getenv("AWS_REGION",    "us-east-1")
        S3_KEY     = "images/bhavesh_experiments/test_tpose_result_v1.png"

        s3 = boto3.client("s3", region_name=AWS_REGION)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)
        s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buf, ContentType="image/png")

        url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{S3_KEY}"
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f"\nOpen this URL in your browser to see the result:")
        print(f"  {url}\n")

    except Exception as e:
        # S3 failed — fallback to local save
        out_path = os.path.join(os.path.dirname(__file__), "test_tpose_result.png")
        result_img.save(out_path)
        print(f"\nS3 upload failed: {e}")
        print(f"Saved locally at: {out_path}")
        print("Use SCP to download it to your laptop.")


if __name__ == "__main__":
    main()
