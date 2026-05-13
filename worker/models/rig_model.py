"""
rig_model.py — Auto-Rig Pro rigging via Blender headless subprocess
====================================================================

Public API
----------
    run_rig(input_glb_path, output_glb_path, params)  -> None

No GPU required — this is a pure CPU subprocess call to Blender with the
Auto-Rig Pro add-on.  The caller passes absolute paths for input + output GLBs.

Param defaults
--------------
    char_type (str)  "humanoid"  — rigging skeleton preset:
                                    "humanoid"  — bipedal ARP rig
                                    "quadruped" — four-leg ARP rig
                                    "bird"      — avian rig
                                    "fish"      — fish rig

Notes
-----
- Reads BLENDER_PATH and ARP_SCRIPT_PATH from env at import time.
- Raises RuntimeError if Blender exits non-zero or output file not created.
- BLENDER_TIMEOUT_SEC defaults to 600 (10 minutes).
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger("models.rig")

# ── Config (from env) ─────────────────────────────────────────────────────────

BLENDER_PATH        = os.getenv("BLENDER_PATH", os.path.expanduser("~/blender/blender"))
_THIS_DIR           = os.path.dirname(os.path.abspath(__file__))
# blender_scripts/ lives one level above models/
ARP_SCRIPT_PATH     = os.getenv(
    "ARP_SCRIPT_PATH",
    os.path.join(os.path.dirname(_THIS_DIR), "blender_scripts", "auto_rig_smart.py"),
)
BLENDER_TIMEOUT_SEC = int(os.getenv("BLENDER_TIMEOUT_SEC", str(10 * 60)))

# ── Char-type normalisation ───────────────────────────────────────────────────

_CHAR_TYPE_MAP: dict[str, str] = {
    "humanoid": "humanoid", "bipedal": "humanoid", "human": "humanoid",
    "quadruped": "quadruped", "quad": "quadruped", "animal": "quadruped",
    "bird": "bird",
    "fish": "fish",
}

# ── Param defaults ────────────────────────────────────────────────────────────

PARAM_DEFAULTS: dict = {
    "char_type": "humanoid",
}


# ── Run ───────────────────────────────────────────────────────────────────────

def run_rig(
    input_glb_path:  str,
    output_glb_path: str,
    params:          dict | None = None,
) -> None:
    """
    Call Blender headlessly to rig *input_glb_path* using Auto-Rig Pro.

    The rigged GLB is written to *output_glb_path*.  Raises on failure.

    Args:
        input_glb_path:  Absolute path to the source (unrigged) GLB file.
        output_glb_path: Absolute path for the output (rigged) GLB file.
        params:          Override dict — any subset of PARAM_DEFAULTS keys:
                           char_type (str) — "humanoid" | "quadruped" |
                                             "bird" | "fish"

    Raises:
        FileNotFoundError: If BLENDER_PATH binary is not found.
        RuntimeError:      If Blender exits non-zero or output not created.
    """
    p = {**PARAM_DEFAULTS, **(params or {})}

    raw_type  = str(p["char_type"])
    char_type = _CHAR_TYPE_MAP.get(raw_type.lower(), "humanoid")

    if not os.path.isfile(BLENDER_PATH):
        raise FileNotFoundError(
            f"Blender binary not found at {BLENDER_PATH!r}. "
            "Set BLENDER_PATH env var or install Blender."
        )

    logger.info(
        f"[rig] run  char_type={char_type}  "
        f"input={os.path.basename(input_glb_path)}  "
        f"timeout={BLENDER_TIMEOUT_SEC}s"
    )

    cmd = [
        BLENDER_PATH, "--background",
        "--python", ARP_SCRIPT_PATH,
        "--",
        "--input",          input_glb_path,
        "--output",         output_glb_path,
        "--character_type", char_type,
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BLENDER_TIMEOUT_SEC,
    )

    # Surface relevant log lines
    blender_out = proc.stdout + proc.stderr
    for line in blender_out.splitlines():
        if any(tag in line for tag in ("[RIG]", "[ARP]", "Error", "error", "WARNING", "RIGGING")):
            logger.info(f"  blender: {line}")

    if proc.returncode != 0:
        tail = "\n".join(blender_out.splitlines()[-20:])
        raise RuntimeError(
            f"Blender exited {proc.returncode}.\nLast 20 lines:\n{tail}"
        )

    if not os.path.exists(output_glb_path):
        # Dump full blender log to help diagnose — script exceptions don't fail exit code
        tail = "\n".join(blender_out.splitlines()[-40:])
        raise RuntimeError(
            f"Blender exited OK but output GLB not found: {output_glb_path}\n"
            f"Last 40 lines of Blender output:\n{tail}"
        )

    size_mb = os.path.getsize(output_glb_path) / 1e6
    logger.info(f"[rig] done  rigged GLB {size_mb:.2f} MB")
