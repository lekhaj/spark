#!/usr/bin/env python3
"""
flux_bhavesh_v1.py — EXACT PRODUCTION FLUX CONCEPT GENERATOR
=============================================================
Branch  : bhavesh-dev

This is an exact extraction of the Flux generation logic from
`worker/flux_concept_generator.py`. 

It generates at 768 x 1024 (NO 512 cap) just like the real server.
"""

import os
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Exact character dict from worker/flux_concept_generator.py
CHAR_CULTIVATION_YOUTH = {
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
}

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

def run_flux_production(pipe) -> Image.Image:
    """Exact inference block from worker/flux_concept_generator.py"""
    c = CHAR_CULTIVATION_YOUTH
    print(f"[Flux] Generating size {c['width']}x{c['height']}...")
    with torch.inference_mode():
        out = pipe(
            prompt=c["flux_prompt"],
            width=c["width"],
            height=c["height"],
            num_inference_steps=c["num_inference_steps"],
            guidance_scale=c["guidance_scale"],
        )
    return out.images[0]
