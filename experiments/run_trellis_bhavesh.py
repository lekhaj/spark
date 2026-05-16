#!/usr/bin/env python3
"""
run_trellis_bhavesh.py — MODULAR TRELLIS.2 PIPELINE
===================================================
A standalone, production-grade 3D mesh generator for Trellis.
Takes the T-pose image generated from Strategy A, removes the background,
and feeds it into the Trellis.2-4B O-Voxel model.

How to use:
    python experiments/run_trellis_bhavesh.py --image_path "https://s3-url-to-your-image.png"
"""

import os
import sys
import time
import argparse
import tempfile
import requests
import io
import torch
from PIL import Image

# ── 1. CONFIGURE TRELLIS REPO ───────────────────────────────────────────────
TRELLIS_REPO_PATH = os.getenv("TRELLIS_REPO_PATH", os.path.expanduser("~/trellis"))
if TRELLIS_REPO_PATH not in sys.path:
    sys.path.insert(0, TRELLIS_REPO_PATH)

try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel
except ImportError:
    print("❌ ERROR: Trellis is not installed or the repo path is wrong.")
    print("Please run: bash worker/gpu_setup/install_trellis.sh")
    sys.exit(1)

try:
    import rembg
except ImportError:
    print("❌ ERROR: rembg is not installed. Run: pip install rembg[gpu]")
    sys.exit(1)

# ── 2. AWS S3 HELPER (Reused from Strategy A) ───────────────────────────────
from dotenv import load_dotenv
load_dotenv(".env")
load_dotenv(".env.gpu", override=True)

def upload_to_s3(file_path: str, s3_key: str) -> str:
    import boto3
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET not set in .env")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
    )
    s3.upload_file(file_path, bucket, s3_key, ExtraArgs={"ContentType": "model/gltf-binary"})
    return f"https://{bucket}.s3.{os.getenv('S3_REGION', 'us-east-1')}.amazonaws.com/{s3_key}"

# ── 3. MAIN PIPELINE ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Local path or S3 URL of the T-pose image")
    parser.add_argument("--character_name", type=str, default="trellis_test_char", help="Name of character for S3")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 BHAVESH TRELLIS.2 PIPELINE (Standalone Modular Sandbox)")
    print("="*70)

    # 1. Load Image
    print(f"\n[1/5] Loading input image from: {args.image_path}")
    if args.image_path.startswith("http"):
        resp = requests.get(args.image_path)
        image = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    else:
        image = Image.open(args.image_path).convert("RGBA")

    # 2. Remove Background
    print("[2/5] Removing background using rembg (u2net)...")
    bg_remover = rembg.new_session("u2net")
    clean_image = rembg.remove(image, session=bg_remover)

    # 3. Format for Trellis (White Background, Pad to Square, Resize to 512)
    print("[3/5] Formatting image for Trellis.2 (padding to square, 512x512)...")
    
    # First, paste onto a white background
    bg = Image.new("RGB", clean_image.size, (255, 255, 255))
    bg.paste(clean_image, mask=clean_image.split()[3])
    
    # Next, pad the image to a perfect square to prevent squishing the tall 512x768 images
    w, h = bg.size
    max_dim = max(w, h)
    square_bg = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    square_bg.paste(bg, (offset_x, offset_y))
    
    # Finally, resize to the 512x512 Trellis requirement
    formatted_image = square_bg.resize((512, 512), Image.Resampling.LANCZOS)

    # 4. Load Trellis & Generate 3D
    print(f"\n[4/5] Loading Trellis.2-4B into VRAM (Takes ~15 seconds)...")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    
    print("      Running 3D Mesh Generation (This will take 1-3 minutes)...")
    t0 = time.time()
    with torch.no_grad():
        mesh = pipe.run(formatted_image)[0]
    
    print(f"      Generation finished in {time.time()-t0:.1f} seconds! Exporting GLB...")
    
    # 5. Export to GLB with PBR Materials
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,   # Max 1M triangles for game assets
        texture_size=1024,           # 1024x1024 PBR textures
        remesh=True,                 # Clean topology for Auto-Rig Pro
    )

    # Save and Upload
    tmp_dir = tempfile.mkdtemp()
    glb_path = os.path.join(tmp_dir, f"{args.character_name}.glb")
    glb.export(glb_path, extension_webp=True)
    
    print(f"\n[5/5] Uploading GLB to S3...")
    s3_key = f"images/bhavesh_experiments/{args.character_name}_trellis.glb"
    glb_url = upload_to_s3(glb_path, s3_key)

    print("\n" + "="*70)
    print(f"✅ TRELLIS DONE! 3D Model available at:")
    print(f"   {glb_url}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
