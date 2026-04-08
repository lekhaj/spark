#!/usr/bin/env python3
"""
Auto-Rig Pro Rigging Worker
============================
Reads from queue : rig_model    (output of TRELLIS 3D worker)
Writes to queue  : None         (final stage — updates MongoDB only)

Rigging backend: Auto-Rig Pro 3.76.22 via Blender 4.2 (headless)

All character types use Auto-Rig Pro Smart:
  humanoid / bipedal  → ARP Smart (humanoid mode)
  quadruped           → ARP Smart (quadruped mode)
  bird / other        → ARP Smart (best-effort)

Pipeline per character:
  1. Download GLB from S3
  2. Run Blender headless with auto_rig_smart.py
     → ARP Smart detects body parts → builds armature → binds skin
  3. Export rigged GLB
  4. Upload rigged GLB to S3
  5. Update MongoDB  status = rig_complete
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import logging

from .base_worker import BaseWorker

logger = logging.getLogger("RigWorker")

# ── Queue ─────────────────────────────────────────────────────────────────────
INPUT_QUEUE = "rig_model"

# ── Blender + ARP config ──────────────────────────────────────────────────────
BLENDER_PATH = os.getenv(
    "BLENDER_PATH",
    os.path.expanduser("~/blender/blender"),
)

# Path to the Blender rigging script (relative to repo root)
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_THIS_DIR)
ARP_SCRIPT  = os.path.join(_REPO_ROOT, "blender_scripts", "auto_rig_smart.py")

BLENDER_TIMEOUT_SEC = int(os.getenv("BLENDER_TIMEOUT_SEC", str(10 * 60)))  # 10 min

# ARP character type mapping from pipeline task character_type values
_CHAR_TYPE_MAP = {
    "humanoid":  "humanoid",
    "bipedal":   "humanoid",
    "human":     "humanoid",
    "quadruped": "quadruped",
    "quad":      "quadruped",
    "animal":    "quadruped",
    "bird":      "bird",
    "fish":      "fish",
}


def _map_character_type(raw_type: str) -> str:
    return _CHAR_TYPE_MAP.get((raw_type or "humanoid").lower(), "humanoid")


class RigWorker(BaseWorker):
    """Auto-Rig Pro rigging worker — converts bare GLB → fully rigged GLB."""

    worker_name = "RigWorker"
    input_queue = INPUT_QUEUE

    def __init__(self):
        super().__init__()

    # ── Model Loading (no ML models — just validate Blender + ARP) ───────────

    def load_models(self):
        logger.info("Verifying Blender + Auto-Rig Pro installation...")

        if not os.path.isfile(BLENDER_PATH):
            raise FileNotFoundError(
                f"Blender not found at {BLENDER_PATH}. "
                "Run: bash worker/gpu_setup/install_blender_arp.sh"
            )

        if not os.path.isfile(ARP_SCRIPT):
            raise FileNotFoundError(
                f"ARP Blender script not found at {ARP_SCRIPT}"
            )

        # Quick Blender sanity check
        result = subprocess.run(
            [BLENDER_PATH, "--version"],
            capture_output=True, text=True, timeout=30,
        )
        version_line = (result.stdout + result.stderr).strip().splitlines()[0]
        logger.info(f"Blender OK: {version_line}")
        logger.info(f"ARP script: {ARP_SCRIPT}")

    # ── Core rigging ──────────────────────────────────────────────────────────

    def _run_blender_rig(
        self,
        input_glb: str,
        output_glb: str,
        character_type: str,
    ) -> None:
        """
        Invoke Blender headless with the ARP Smart script.
        Raises RuntimeError if Blender exits non-zero or output not produced.
        """
        arp_type = _map_character_type(character_type)

        cmd = [
            BLENDER_PATH,
            "--background",
            "--python", ARP_SCRIPT,
            "--",                    # separator: args after this go to the script
            "--input",  input_glb,
            "--output", output_glb,
            "--character_type", arp_type,
        ]

        logger.info(f"Running Blender ARP: type={arp_type}")
        logger.debug("CMD: " + " ".join(cmd))

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BLENDER_TIMEOUT_SEC,
        )

        # Always log Blender output for debugging
        blender_output = proc.stdout + proc.stderr
        for line in blender_output.splitlines():
            if any(tag in line for tag in ("[ARP]", "Error", "error", "WARNING", "RIGGING")):
                logger.info(f"  blender: {line}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"Blender exited {proc.returncode}.\n"
                f"Last 20 lines:\n" +
                "\n".join(blender_output.splitlines()[-20:])
            )

        if not os.path.exists(output_glb):
            raise RuntimeError(
                f"Blender exited OK but output not found: {output_glb}\n"
                "Check Blender log above."
            )

        size_mb = os.path.getsize(output_glb) / 1e6
        logger.info(f"Rigged GLB ready: {output_glb} ({size_mb:.2f} MB)")

    # ── Task Processing ───────────────────────────────────────────────────────

    def process_task(self, task: dict, r, db) -> None:
        biome_id   = task["biome_id"]
        char_name  = task["character_name"]
        char_type  = task.get("character_type", "humanoid")
        model_key  = task["model_path"]   # S3 key from TRELLIS worker

        self.mongo_update(biome_id, char_name, {
            "status":           "rigging",
            "generation_stage": "auto_rig_pro",
        })

        with tempfile.TemporaryDirectory() as tmp:
            # ── Step 1: Download GLB from S3 ─────────────────────────────────
            input_glb  = os.path.join(tmp, f"{char_name}_raw.glb")
            output_glb = os.path.join(tmp, f"{char_name}_rigged.glb")

            logger.info(f"[{char_name}] Downloading: {model_key}")
            self.download_file(model_key, input_glb)

            if not os.path.exists(input_glb):
                raise RuntimeError(f"Download failed: {model_key}")

            logger.info(f"[{char_name}] Downloaded {os.path.getsize(input_glb)/1e6:.2f} MB")

            # ── Step 2: Auto-Rig Pro Smart rigging ───────────────────────────
            logger.info(f"[{char_name}] Rigging via ARP Smart (type={char_type})...")
            t0 = time.time()
            self._run_blender_rig(input_glb, output_glb, char_type)
            elapsed = time.time() - t0
            logger.info(f"[{char_name}] Rigging done in {elapsed:.1f}s")

            # ── Step 3: Upload rigged GLB to S3 ──────────────────────────────
            rigged_key = f"rigged/{biome_id}/{char_name}_rigged.glb"
            self.upload_file(output_glb, rigged_key, "model/gltf-binary")
            logger.info(f"[{char_name}] Uploaded → s3://{self.cfg.S3_BUCKET}/{rigged_key}")

        # ── Step 4: Update MongoDB ────────────────────────────────────────────
        self.mongo_update(biome_id, char_name, {
            "status":              "rig_complete",
            "generation_stage":    "complete",
            "rigged_model_url":    f"https://{self.cfg.S3_BUCKET}.s3.{self.cfg.S3_REGION}.amazonaws.com/{rigged_key}",
            "rigged_s3_key":       rigged_key,
            "rigged_timestamp":    time.time(),
            "rig_method":          "auto_rig_pro",
            "character_type_used": char_type,
        })

        logger.info(f"[{char_name}] Complete. Rigged model: {rigged_key}")
