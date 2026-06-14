#!/usr/bin/env python3
"""
hunyuan3d_infer.py — standalone inference entry-point for Hunyuan3D-2.0
========================================================================

Called as a subprocess by hunyuan3d_model.py inside the `hunyuan3d` conda env.
Never imported by the spark worker directly — it runs under a different Python
that has hy3dgen, custom_rasterizer, and DifferentiableRenderer installed.

Usage:
    python hunyuan3d_infer.py \\
        --image   /tmp/input.png   \\
        --output  /tmp/output.glb  \\
        [--seed 42]                \\
        [--steps 50]               \\
        [--guidance-scale 7.5]     \\
        [--texture-resolution 2048]\\
        [--remove-bg]
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hunyuan3D-2.0 inference")
    p.add_argument("--image",               required=True,  help="Input image path")
    p.add_argument("--output",              required=True,  help="Output GLB path")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--steps",               type=int,   default=50,   help="Shape diffusion steps")
    p.add_argument("--guidance-scale",      type=float, default=7.5)
    p.add_argument("--texture-resolution",  type=int,   default=2048)
    p.add_argument("--remove-bg",           action="store_true", help="Strip background via rembg")
    return p.parse_args()


def _remove_bg(image):
    try:
        import rembg
        session = rembg.new_session("u2net")
        return rembg.remove(image.convert("RGBA"), session=session)
    except ImportError:
        _log("rembg not installed — background not removed")
        return image.convert("RGBA")


def _log(msg: str) -> None:
    print(f"[hunyuan3d_infer] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.image):
        _log(f"ERROR: input not found: {args.image}")
        sys.exit(1)

    import torch
    from PIL import Image

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        _log("WARNING: CUDA not available — inference will be very slow")

    _log(f"loading image: {args.image}")
    image = Image.open(args.image)
    if args.remove_bg:
        _log("removing background")
        image = _remove_bg(image)
    else:
        image = image.convert("RGBA")

    # ── Shape generation ──────────────────────────────────────────────────────
    _log("loading shape pipeline (Hunyuan3DDiTFlowMatchingPipeline)")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    model_id    = os.getenv("HUNYUAN3D_MODEL_ID", "tencent/Hunyuan3D-2")
    shape_pipe  = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id)
    if torch.cuda.is_available():
        # .to() is in-place and returns None on this pipeline — never reassign.
        shape_pipe.to("cuda")

    _log(f"generating shape  steps={args.steps}  guidance={args.guidance_scale}  seed={args.seed}")
    generator = None
    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.no_grad():
        shape_results = shape_pipe(
            image,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )

    mesh = shape_results[0] if isinstance(shape_results, (list, tuple)) else shape_results
    _log("shape generation done")

    # Free shape pipeline VRAM before loading paint pipeline
    del shape_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Texture painting ──────────────────────────────────────────────────────
    _log(f"loading paint pipeline  texture_resolution={args.texture_resolution}")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(model_id)
    if torch.cuda.is_available() and hasattr(paint_pipe, "to"):
        # .to() may be in-place and return None — call without reassigning.
        paint_pipe.to("cuda")

    with torch.no_grad():
        # texture_resolution kwarg is supported in Hunyuan3D-2; pass as keyword
        # and catch TypeError in case an older hy3dgen version lacks it.
        try:
            textured_mesh = paint_pipe(
                mesh,
                image=image,
                texture_resolution=args.texture_resolution,
            )
        except TypeError:
            _log("paint pipeline does not accept texture_resolution — running without it")
            textured_mesh = paint_pipe(mesh, image=image)

    _log("texture painting done")

    # ── Export GLB ────────────────────────────────────────────────────────────
    _log(f"exporting GLB: {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Handle trimesh.Trimesh, custom mesh objects, or anything with .export()
    if hasattr(textured_mesh, "export"):
        textured_mesh.export(args.output)
    else:
        # Fallback: assume trimesh Scene or list
        import trimesh
        if isinstance(textured_mesh, (list, tuple)):
            textured_mesh = textured_mesh[0]
        if isinstance(textured_mesh, trimesh.Scene):
            textured_mesh.export(args.output)
        else:
            raise RuntimeError(
                f"Unexpected paint pipeline output type: {type(textured_mesh)}"
            )

    size_kb = os.path.getsize(args.output) // 1024
    _log(f"done  size={size_kb} KB")


if __name__ == "__main__":
    main()
