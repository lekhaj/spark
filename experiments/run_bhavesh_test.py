#!/usr/bin/env python3
"""
Test script for the new SD1.5 T-pose parameters (bhavesh-dev branch).
This script uses our isolated sandbox model file: sd_model_bhavesh_v1.py
"""

import os
import sys
from PIL import Image

# Import our sandbox model file (not the production one)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.sd_model_bhavesh_v1 import load_sd, run_stage1

def main():
    print("=" * 60)
    print("🧪 BHAVESH EXPERIMENT RUNNER — T-POSE FIX")
    print("=" * 60)
    print("This script uses the new production architecture but points to")
    print("our sandbox file (sd_model_bhavesh_v1.py) with the new weights:\n")
    print("  ip_adapter_weight: 0.25 (was 0.45)")
    print("  canny_weight:      0.0  (was 0.20)\n")

    # 1. Load the pipelines
    print("[1/3] Loading SD1.5 models onto GPU...")
    # Using dreamshaper as the base model, same as production
    pipes = load_sd("Lykon/DreamShaper")
    print("Models loaded successfully!\n")

    # 2. Create a dummy "Flux" image (an A-pose grey character) for testing
    # In a real run, this would be downloaded from S3
    print("[2/3] Creating mock Flux input image (grey background)...")
    init_img = Image.new("RGB", (512, 512), (128, 128, 128))
    
    # We use the same image for the IP-Adapter reference
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

    # 3. Run the generation!
    print("\n[3/3] Running generation with experimental weights...")
    result_img = run_stage1(
        pipes=pipes,
        init_img=init_img,
        prompt=prompt,
        negative=negative,
        ip_adapter_image=ip_image
    )

    # Save the result
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_tpose_result.png"))
    result_img.save(out_path)
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)
    print(f"Result saved to: {out_path}")
    print("Download this image and check if the arms are horizontally straight!")

if __name__ == "__main__":
    main()
