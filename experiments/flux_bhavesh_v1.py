#!/usr/bin/env python3
"""
flux_bhavesh_v1.py — EXACT PRODUCTION FLUX CONCEPT GENERATOR
=============================================================
Branch  : bhavesh-dev

Exact extraction of the Flux generation logic from
`worker/flux_concept_generator.py`.

Generates at 768 x 1024 — same as the real server.

CHARACTERS available:
  - cultivation_youth  : original test character (gray hanfu, wide sleeves)
  - iron_soldier       : NEW test — fitted plate armor, tight clothes (no wide sleeves)
                         Purpose: check if ghost-hand issue was hanfu-specific or general

Change CHARACTER_NAME in run_bhavesh_test.py to switch.
"""

import os
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Character definitions ─────────────────────────────────────────────────────

CHARACTERS = {

    # Original test character — gray hanfu, WIDE sleeves
    # This is what was causing the ghost hand issue
    "cultivation_youth": {
        "flux_prompt": (
            "young male cultivator, T-pose, arms extended horizontally, legs straight, "
            "orthographic front view, symmetrical, centered, "
            "plain gray hanfu robe, rope belt, lean build, topknot hair, "
            "simple clothing, clean silhouette, minimal detail, "
            "game-ready character design, "
            "white background, flat lighting, full body, head to toe"
        ),
        "width":               768,
        "height":              1024,
        "num_inference_steps": 4,
        "guidance_scale":      0.0,
    },

    # NEW test character — fitted plate armor, NO wide sleeves
    # Purpose: Test if ghost-hand/identity-loss is hanfu-specific or general SD1.5 issue
    "iron_soldier": {
        "flux_prompt": (
            "young male soldier, T-pose, arms extended horizontally, legs straight, "
            "orthographic front view, symmetrical, centered, "
            "fitted dark iron plate armor, pauldrons, chest plate, gauntlets, "
            "dark red short cape, sturdy boots, short dark hair, determined expression, "
            "tight-fitting armor with no loose fabric, clean silhouette, "
            "game-ready character design, "
            "white background, flat lighting, full body, head to toe"
        ),
        "width":               768,
        "height":              1024,
        "num_inference_steps": 4,
        "guidance_scale":      0.0,
    },

}

# ── Model functions ───────────────────────────────────────────────────────────

def load_flux_production():
    """Exact loading block from worker/flux_concept_generator.py"""
    from diffusers import FluxPipeline
    print("[Flux] Loading black-forest-labs/FLUX.1-schnell ...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    return pipe


def run_flux_production(pipe, character_name: str = "cultivation_youth") -> Image.Image:
    """
    Exact inference block from worker/flux_concept_generator.py.
    Pass character_name to switch between test characters.
    """
    if character_name not in CHARACTERS:
        raise ValueError(
            f"Unknown character: '{character_name}'. "
            f"Available: {list(CHARACTERS.keys())}"
        )
    c = CHARACTERS[character_name]
    print(f"[Flux] Character   : {character_name}")
    print(f"[Flux] Size        : {c['width']}x{c['height']}")
    print(f"[Flux] Prompt      : {c['flux_prompt'][:80]}...")
    with torch.inference_mode():
        out = pipe(
            prompt=c["flux_prompt"],
            width=c["width"],
            height=c["height"],
            num_inference_steps=c["num_inference_steps"],
            guidance_scale=c["guidance_scale"],
        )
    return out.images[0]
