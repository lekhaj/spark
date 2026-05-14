"""
flux_bhavesh_v1.py — SANDBOX EXPERIMENT COPY
=============================================
Original file : worker/flux_concept_generator.py
Branch        : bhavesh-dev

PURPOSE: Stage 0 concept generation — IDENTICAL to production.
  - Same model        : black-forest-labs/FLUX.1-schnell
  - Same loading      : enable_sequential_cpu_offload + vae.enable_slicing
  - Same character    : cultivation_youth (exact prompt + parameters from production)
  - Same image size   : 768 x 1024
  - Same dtype        : torch.bfloat16

ONLY difference from production:
  - No MongoDB write   (no update_mongodb call)
  - No S3 upload       (run_bhavesh_test.py handles that)
  - Exposes load_flux() + generate_concept() as importable functions

DO NOT MERGE TO MAIN.
"""

import os
import torch
from PIL import Image

# Same as production flux_concept_generator.py
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# ── Exact production character definition (from flux_concept_generator.py) ────
# Copied verbatim — NO changes to prompt or parameters.
CHARACTER = {
    "name": "cultivation_youth",
    "character_type":    "humanoid",
    "creature_category": "bipedal",

    # Exact prompt from production flux_concept_generator.py
    "flux_prompt": (
        "young male cultivator, T-pose, arms extended horizontally, legs straight, "
        "orthographic front view, symmetrical, centered, "
        "plain gray hanfu robe, rope belt, lean build, topknot hair, "
        "simple clothing, clean silhouette, minimal detail, "
        "game-ready character design, "
        "white background, flat lighting, full body, head to toe"
    ),

    # Exact parameters from production
    "width":               768,
    "height":              1024,
    "num_inference_steps": 4,    # Schnell optimized for 1-4 steps
    "guidance_scale":      0.0,  # Schnell is distilled — always 0
}


def load_flux():
    """
    Load FLUX.1-schnell onto GPU.
    Identical to the loading block in production flux_concept_generator.main().
    """
    from diffusers import FluxPipeline

    print(f"[Flux] Loading {FLUX_MODEL_ID} ...")
    print("  torch_dtype          : bfloat16")
    print("  enable_sequential_cpu_offload : ON  (same as production)")
    print("  vae.enable_slicing            : ON  (same as production)")

    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    # Sequential offload: moves each sub-module to GPU one at a time.
    # Same strategy as production — fits any VRAM.
    pipe.enable_sequential_cpu_offload()
    # Sliced VAE decode prevents VRAM spikes on large (768×1024) images.
    pipe.vae.enable_slicing()

    print("[Flux] Pipeline ready.\n")
    return pipe


def generate_concept(pipe) -> Image.Image:
    """
    Run Flux inference.
    Identical to the generation block inside production flux_concept_generator.main().
    Returns the PIL Image (caller handles upload/delete).
    """
    cfg = CHARACTER
    print(f"[Flux] Generating: {cfg['name']}")
    print(f"  Size  : {cfg['width']} x {cfg['height']}")
    print(f"  Steps : {cfg['num_inference_steps']}")
    print(f"  Prompt: {cfg['flux_prompt'][:120]}...")

    with torch.inference_mode():
        out = pipe(
            prompt=cfg["flux_prompt"],
            width=cfg["width"],
            height=cfg["height"],
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=cfg["guidance_scale"],
        )

    image = out.images[0]
    print(f"[Flux] Done. Image size: {image.size}\n")
    return image
