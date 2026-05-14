"""
flux_bhavesh_v1.py — SANDBOX EXPERIMENT COPY
=============================================
Original file : worker/flux_concept_generator.py
Branch        : bhavesh-dev
Purpose       : Generate a Flux concept image for a test "female_ranger"
                character. Mirrors what the production flux_concept_generator
                does, but with NO MongoDB, NO Redis — just generates and returns
                the image so run_bhavesh_test.py can pass it to SD1.5.

DO NOT MERGE THIS FILE TO MAIN.
"""

import os
import torch
from PIL import Image

# Reduce VRAM fragmentation (same as production)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# ── Test character definition ─────────────────────────────────────────────────
# Mirrors the structure in flux_concept_generator.py CHARACTERS dict.
# Short, structural Flux prompt — same rules as production:
#   Keep it SHORT + STRUCTURAL.
#   Flux job = get DESIGN + ANATOMY right.
#   SD1.5 + ControlNet will handle T-pose lock in Stage 1.
CHARACTER = {
    "name": "female_ranger",
    "flux_prompt": (
        "female ranger character, full body, front view, centered, "
        "symmetrical, standing natural pose, "
        "crimson red zip tactical jacket, black combat pants, dark boots, black gloves, "
        "auburn hair, realistic human face, neutral expression, "
        "game-ready character design, "
        "white background, flat lighting, head to toe"
    ),
    "width":               512,
    "height":              768,    # taller canvas for full body
    "num_inference_steps": 4,      # Schnell optimized for 1-4 steps
    "guidance_scale":      0.0,    # Schnell is distilled — always 0.0
}


def load_flux() -> object:
    """
    Load FLUX.1-schnell pipeline onto GPU using sequential CPU offload.
    Same loading strategy as production flux_concept_generator.py.
    Returns the loaded pipeline.
    """
    from diffusers import FluxPipeline

    print(f"[Flux] Loading {FLUX_MODEL_ID} ...")
    print("  Using enable_sequential_cpu_offload() — same as production.")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()  # moves each sub-module to GPU one at a time
    pipe.vae.enable_slicing()             # prevents VRAM spikes on large images
    print("[Flux] Pipeline ready.\n")
    return pipe


def generate_concept(pipe) -> Image.Image:
    """
    Run Flux inference for the test character.
    Returns the generated PIL Image.
    Mirrors the generation loop in production flux_concept_generator.main().
    """
    cfg = CHARACTER
    print(f"[Flux] Generating concept for: {cfg['name']}")
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
    print(f"[Flux] Concept image generated. Size: {image.size}\n")
    return image
