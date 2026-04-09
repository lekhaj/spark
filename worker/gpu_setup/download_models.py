#!/usr/bin/env python3
"""
download_models.py — One-time model download to local disk
===========================================================
Run this ONCE on the GPU after first setup (or after wiping the instance).
Downloads all models needed for the SD1.5 image generation pipeline to
/home/ec2-user/models/ so the worker never contacts HuggingFace at runtime.

Usage:
  python worker/gpu_setup/download_models.py
  python worker/gpu_setup/download_models.py --models-root /home/ec2-user/models

After this script completes, workers load entirely from local disk.
No HuggingFace token or internet access needed at inference time.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

# ── Models to download ────────────────────────────────────────────────────────
MODELS = [
    {
        "name":    "DreamShaper (SD1.5 base)",
        "repo_id": "Lykon/DreamShaper",
        "subdir":  "dreamshaper",
        "ignore":  ["*.ckpt", "*.safetensors.bak"],
    },
    {
        "name":    "ControlNet OpenPose SD1.5",
        "repo_id": "lllyasviel/control_v11p_sd15_openpose",
        "subdir":  "controlnet_openpose",
        "ignore":  [],
    },
]


def download_all(models_root: str):
    os.makedirs(models_root, exist_ok=True)
    print(f"\nModels root: {models_root}")
    print("=" * 60)

    for m in MODELS:
        dest = os.path.join(models_root, m["subdir"])
        print(f"\n[{m['name']}]")
        print(f"  repo  : {m['repo_id']}")
        print(f"  dest  : {dest}")

        if os.path.isdir(dest) and os.listdir(dest):
            print(f"  status: already exists — skipping (delete dir to re-download)")
            continue

        print(f"  status: downloading...")
        try:
            snapshot_download(
                repo_id=m["repo_id"],
                local_dir=dest,
                ignore_patterns=m["ignore"],
            )
            print(f"  status: done ✓")
        except Exception as e:
            print(f"  status: FAILED — {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All models downloaded.")
    print(f"\nSet in .env or environment:")
    print(f"  MODELS_ROOT={models_root}")
    print(f"  SD15_MODEL_PATH={os.path.join(models_root, 'dreamshaper')}")
    print(f"  CONTROLNET_OPENPOSE_PATH={os.path.join(models_root, 'controlnet_openpose')}")


def main():
    parser = argparse.ArgumentParser(description="Download SD1.5 models to local disk.")
    parser.add_argument(
        "--models-root",
        default=os.getenv("MODELS_ROOT", "/home/ec2-user/models"),
        help="Directory to store all models (default: /home/ec2-user/models)",
    )
    args = parser.parse_args()
    download_all(args.models_root)


if __name__ == "__main__":
    main()
