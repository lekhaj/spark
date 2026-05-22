#!/usr/bin/env python3
"""
compare_3d.py — A/B/C comparison harness: TRELLIS vs Pixal3D vs Hunyuan3D-2.0
===============================================================================

Runs all three image→3D models on the same input image and writes a Markdown
results table with: wall time, output file size, and triangle count.

Run this in the spark conda env on the GPU instance (or locally against a
remote image if the subprocess env vars point to the right paths).

Usage:
    python tools/compare_3d.py --image /path/to/character.png [--out ./cmp_out]

Env vars respected (same as production):
    TRELLIS_REPO_PATH, PIXAL3D_PYTHON, PIXAL3D_REPO, PIXAL3D_ATTN,
    HUNYUAN3D_PYTHON, HUNYUAN3D_REPO, HUNYUAN3D_TIMEOUT_S

Output:
    <out>/trellis.glb
    <out>/pixal3d.glb
    <out>/hunyuan3d.glb
    <out>/results.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from typing import NamedTuple

# Make worker root importable when run directly from repo root.
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER_ROOT = os.path.join(_REPO_ROOT, "worker")
for _p in [_REPO_ROOT, _WORKER_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class ModelResult(NamedTuple):
    name:        str
    wall_s:      float
    size_kb:     int
    tri_count:   int
    error:       str | None


def count_triangles(glb_bytes: bytes) -> int:
    try:
        import trimesh
        scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
        total = 0
        for geom in scene.geometry.values():
            total += len(geom.faces)
        return total
    except Exception:
        return -1


def run_model(name: str, fn, *args, **kwargs) -> tuple[bytes | None, ModelResult]:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0    = time.perf_counter()
    error = None
    glb   = None
    try:
        glb = fn(*args, **kwargs)
    except Exception as exc:
        error = str(exc)
        print(f"  ERROR: {exc}", file=sys.stderr)
    wall_s   = time.perf_counter() - t0
    size_kb  = len(glb) // 1024 if glb else 0
    tri_cnt  = count_triangles(glb) if glb else -1
    result   = ModelResult(name, wall_s, size_kb, tri_cnt, error)
    if error:
        print(f"  {name}: FAILED in {wall_s:.1f}s  error={error[:80]}")
    else:
        print(f"  {name}: done in {wall_s:.1f}s  {size_kb} KB  {tri_cnt:,} tris")
    return glb, result


def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="3D model A/B/C comparison")
    ap.add_argument("--image",   required=True, help="Input image (PNG/JPG)")
    ap.add_argument("--out",     default=None,  help="Output directory (default: ./compare_<ts>)")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--skip",    nargs="*", default=[], help="Models to skip: trellis pixal3d hunyuan3d")
    ap.add_argument("--trellis-texture", type=int, default=1024)
    ap.add_argument("--pixal3d-texture", type=int, default=2048)
    ap.add_argument("--hunyuan3d-texture", type=int, default=2048)
    ap.add_argument("--hunyuan3d-steps",   type=int, default=50)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Input not found: {args.image}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or f"./compare_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    import shutil
    shutil.copy(args.image, os.path.join(out_dir, "input.png"))

    image   = _load_image(args.image)
    results = []
    skip    = set(args.skip)

    # ── TRELLIS (in-process) ──────────────────────────────────────────────────
    if "trellis" not in skip:
        try:
            from models.trellis_model import load_trellis, run_trellis
            pipes = load_trellis()
            glb, r = run_model(
                "TRELLIS",
                run_trellis,
                pipes, image,
                {"texture_size": args.trellis_texture, "simplify": 0.95},
            )
            if glb:
                with open(os.path.join(out_dir, "trellis.glb"), "wb") as f:
                    f.write(glb)
            results.append(r)
            # Free VRAM before next model
            import torch
            del pipes
            torch.cuda.empty_cache()
        except Exception as exc:
            results.append(ModelResult("TRELLIS", 0, 0, -1, str(exc)))
            print(f"TRELLIS load failed: {exc}", file=sys.stderr)

    # ── Pixal3D (subprocess) ──────────────────────────────────────────────────
    if "pixal3d" not in skip:
        try:
            from models.pixal3d_runner import run_pixal3d
            glb, r = run_model(
                "Pixal3D",
                run_pixal3d,
                image,
                {"seed": args.seed, "texture_size": args.pixal3d_texture},
            )
            if glb:
                with open(os.path.join(out_dir, "pixal3d.glb"), "wb") as f:
                    f.write(glb)
            results.append(r)
        except Exception as exc:
            results.append(ModelResult("Pixal3D", 0, 0, -1, str(exc)))
            print(f"Pixal3D load failed: {exc}", file=sys.stderr)

    # ── Hunyuan3D (subprocess) ────────────────────────────────────────────────
    if "hunyuan3d" not in skip:
        try:
            from models.hunyuan3d_model import run_hunyuan3d
            glb, r = run_model(
                "Hunyuan3D-2",
                run_hunyuan3d,
                image,
                {
                    "seed":                args.seed,
                    "steps":               args.hunyuan3d_steps,
                    "texture_resolution":  args.hunyuan3d_texture,
                },
            )
            if glb:
                with open(os.path.join(out_dir, "hunyuan3d.glb"), "wb") as f:
                    f.write(glb)
            results.append(r)
        except Exception as exc:
            results.append(ModelResult("Hunyuan3D-2", 0, 0, -1, str(exc)))
            print(f"Hunyuan3D load failed: {exc}", file=sys.stderr)

    # ── Results table ─────────────────────────────────────────────────────────
    md = _render_table(results, args)
    table_path = os.path.join(out_dir, "results.md")
    with open(table_path, "w") as f:
        f.write(md)

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump([r._asdict() for r in results], f, indent=2)

    print(f"\n{'='*60}")
    print(md)
    print(f"\nOutputs: {out_dir}/")


def _render_table(results: list[ModelResult], args) -> str:
    lines = [
        "# 3D Model Comparison",
        "",
        f"Input params: seed={args.seed}",
        "",
        "| Model | Status | Wall time | GLB size | Triangles |",
        "|-------|--------|-----------|----------|-----------|",
    ]
    for r in results:
        status   = "ERROR" if r.error else "OK"
        wall     = f"{r.wall_s:.1f}s"
        size     = f"{r.size_kb} KB" if r.size_kb else "—"
        tris     = f"{r.tri_count:,}" if r.tri_count > 0 else "—"
        lines.append(f"| {r.name} | {status} | {wall} | {size} | {tris} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Triangle counts include all LODs if the exporter bakes them.")
    lines.append("- Wall time includes model load on first run (trellis: in-process; pixal3d/hunyuan3d: subprocess warm-up).")
    lines.append("- VRAM peak is not measured here — use `nvidia-smi dmon` externally if needed.")
    if any(r.error for r in results):
        lines.append("")
        lines.append("## Errors")
        for r in results:
            if r.error:
                lines.append(f"- **{r.name}**: `{r.error}`")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
