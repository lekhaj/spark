"""
rig_model.py — Auto-Rig Pro rigging via Blender virtual-desktop subprocess
===========================================================================

Public API
----------
    run_rig(input_glb_path, output_glb_path, params)  -> None

No GPU required — this is a pure CPU subprocess call to Blender with the
Auto-Rig Pro add-on.  The caller passes absolute paths for input + output GLBs.

Startup model (Option B)
------------------------
Blender is launched inside a throwaway virtual X desktop:

    blender_desktop_run.sh  →  Xvfb + fluxbox WM  →  blender --python ...

The shell wrapper starts Xvfb (virtual framebuffer) and fluxbox (window
manager) before Blender opens.  With a real WM managing the window, Blender's
default workspace loads fully — VIEW_3D area, regions, space_data — so ARP's
view3d operators work without any context hacks.  The wrapper kills Xvfb and
fluxbox when Blender exits (or crashes).

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
- Requires: xvfb (Xvfb binary), fluxbox — installed once via apt.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger("models.rig")

# ── Config (from env) ─────────────────────────────────────────────────────────

BLENDER_PATH        = os.getenv("BLENDER_PATH", os.path.expanduser("~/blender/blender"))
_THIS_DIR            = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR         = os.path.join(os.path.dirname(_THIS_DIR), "blender_scripts")
# blender_scripts/ lives one level above models/
ARP_SCRIPT_PATH      = os.getenv(
    "ARP_SCRIPT_PATH",
    os.path.join(_SCRIPTS_DIR, "auto_rig_smart.py"),
)
DESKTOP_WRAPPER_PATH = os.getenv(
    "DESKTOP_WRAPPER_PATH",
    os.path.join(_SCRIPTS_DIR, "blender_desktop_run.sh"),
)
BLENDER_TIMEOUT_SEC  = int(os.getenv("BLENDER_TIMEOUT_SEC", str(10 * 60)))

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
    output_fbx_path: str | None = None,
) -> dict:
    """
    Call Blender headlessly to rig *input_glb_path* using Auto-Rig Pro and
    export BOTH a clean deform-only GLB (*output_glb_path*) and, when
    *output_fbx_path* is given, an engine-ready ARP FBX.

    Returns a dict ``{"glb": path, "fbx": path|None, "rig_status": "auto"|"manual"}``.

    Rigging failure is **non-fatal**: the headless script falls back to exporting
    the unrigged mesh and prints ``[RIG_RESULT] rig_status=manual``. We only
    raise if no GLB is produced at all.

    Args:
        input_glb_path:  Absolute path to the source (unrigged) GLB file.
        output_glb_path: Absolute path for the output (rigged) GLB file.
        params:          Override dict — char_type ("humanoid"|"quadruped"|
                         "bird"|"fish") and morphology (B1_humanoid..B7_arthropod).
        output_fbx_path: Absolute path for the engine FBX (optional).

    Raises:
        FileNotFoundError: If BLENDER_PATH binary is not found.
        RuntimeError:      If Blender produced no GLB at all.
    """
    p = {**PARAM_DEFAULTS, **(params or {})}

    raw_type   = str(p["char_type"])
    char_type  = _CHAR_TYPE_MAP.get(raw_type.lower(), "humanoid")
    morphology = str(p.get("morphology") or "B1_humanoid")

    if not os.path.isfile(BLENDER_PATH):
        raise FileNotFoundError(
            f"Blender binary not found at {BLENDER_PATH!r}. "
            "Set BLENDER_PATH env var or install Blender."
        )
    if not os.path.isfile(DESKTOP_WRAPPER_PATH):
        raise FileNotFoundError(
            f"Desktop wrapper script not found at {DESKTOP_WRAPPER_PATH!r}."
        )

    logger.info(
        f"[rig] run  char_type={char_type}  morphology={morphology}  "
        f"input={os.path.basename(input_glb_path)}  "
        f"fbx={'yes' if output_fbx_path else 'no'}  "
        f"timeout={BLENDER_TIMEOUT_SEC}s"
    )

    # Launch Blender inside a throwaway virtual desktop (Xvfb + fluxbox).
    # The wrapper script handles: start Xvfb → start fluxbox WM → run Blender
    # → kill both on exit.  With a real WM, Blender's VIEW_3D area is fully
    # initialised so ARP's view3d.* operators poll correctly without hacks.
    cmd = [
        "bash", DESKTOP_WRAPPER_PATH,
        BLENDER_PATH,
        "--python", ARP_SCRIPT_PATH,
        "--",
        "--input",          input_glb_path,
        "--output",         output_glb_path,
        "--character_type", char_type,
        "--morphology",     morphology,
    ]
    if output_fbx_path:
        cmd += ["--output_fbx", output_fbx_path]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BLENDER_TIMEOUT_SEC,
    )

    # Surface relevant log lines + parse the rig-result marker.
    blender_out = proc.stdout + proc.stderr
    rig_status = "auto"
    for line in blender_out.splitlines():
        if "[RIG_RESULT]" in line and "rig_status=manual" in line:
            rig_status = "manual"
        if any(tag in line for tag in ("[RIG]", "[ARP]", "[RIG_RESULT]",
                                       "Error", "error", "WARNING", "RIGGING")):
            logger.info(f"  blender: {line}")

    glb_ok = os.path.exists(output_glb_path) and os.path.getsize(output_glb_path) > 0
    fbx_ok = bool(output_fbx_path) and os.path.exists(output_fbx_path) \
        and os.path.getsize(output_fbx_path) > 0

    # Check output file FIRST: Blender can SIGSEGV during Python finalisation
    # (e.g. double quit_blender in atexit) even after a fully successful export.
    # If the GLB is present and non-empty we treat that as success regardless of
    # exit code, and just log a warning about the non-zero code.
    if glb_ok:
        if proc.returncode != 0:
            logger.warning(
                f"[rig] Blender exited {proc.returncode} but output GLB exists "
                f"— treating as success (likely post-export crash in atexit/GC)"
            )
        size_mb = os.path.getsize(output_glb_path) / 1e6
        logger.info(
            f"[rig] done  rig_status={rig_status}  GLB {size_mb:.2f} MB"
            f"{'  + FBX' if fbx_ok else ''}"
        )
        return {
            "glb": output_glb_path,
            "fbx": output_fbx_path if fbx_ok else None,
            "rig_status": rig_status,
        }

    # Output not found — now the exit code matters for the error message
    if proc.returncode != 0:
        tail = "\n".join(blender_out.splitlines()[-20:])
        raise RuntimeError(
            f"Blender exited {proc.returncode}.\nLast 20 lines:\n{tail}"
        )

    # Exited 0 but no output — script bug or silent failure
    tail = "\n".join(blender_out.splitlines()[-40:])
    raise RuntimeError(
        f"Blender exited OK but output GLB not found: {output_glb_path}\n"
        f"Last 40 lines of Blender output:\n{tail}"
    )
