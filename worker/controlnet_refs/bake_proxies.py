"""
bake_proxies.py — render every (morphology, view) PNG into this directory.

The PNGs are committed to git so workers don't have to re-render them at
runtime — they read the bundled file the same way the existing
``apose_canonical.png`` is consumed.

Run from anywhere::

    python worker/controlnet_refs/bake_proxies.py [--size 1024]

For ant variant of B7 (arthropod), additionally writes B7_arthropod_ant.png
so the worker can switch between spider (default) and ant by view name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from proxy_generators import (
    MORPHOLOGIES,
    MORPHOLOGY_VIEWS,
    proxy_filename,
    render_arthropod,
    render_proxy,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for morph in MORPHOLOGIES:
        if morph == "B1_humanoid":
            # Handled by generate_apose.py — skip.
            continue
        for view in MORPHOLOGY_VIEWS[morph]:
            img = render_proxy(morph, view=view, size=args.size)
            out = args.out_dir / proxy_filename(morph, view)
            img.save(out)
            print(f"wrote {out}  ({args.size}x{args.size})")

    # Ant variant of B7 — same renderer, leg_count=6, segmented body
    ant_img = render_arthropod(size=args.size, view="primary",
                               leg_count=6, segmented_body=True)
    out = args.out_dir / "B7_arthropod_ant.png"
    ant_img.save(out)
    print(f"wrote {out}  ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
