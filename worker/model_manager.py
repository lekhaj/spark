"""
model_manager.py — GPU model lifecycle manager
===============================================

ModelManager owns lazy loading and VRAM eviction for all model families.
The worker calls manager.ensure(family) before running a stage — the manager
loads the requested family and evicts any incompatible ones first.

Usage
-----
    from model_manager import ModelManager

    mgr = ModelManager()

    # Before a Flux stage:
    mgr.ensure("flux")
    img = run_flux(mgr.flux_pipe, prompt, params)

    # Before an SD stage:
    mgr.ensure("sd")
    img = run_stage1(mgr.sd_pipes, init_img, prompt, negative, params)

    # Before a TRELLIS stage:
    mgr.ensure("trellis")
    glb_bytes = run_trellis(mgr.trellis_pipes, front_img, params)

    # Rig is CPU-only — no ensure() needed, just call run_rig() directly.

VRAM eviction rules
-------------------
    flux    → evicts sd + trellis before loading
    sd      → evicts flux + trellis before loading
    trellis → evicts flux + sd before loading
    rig     → no GPU, no eviction

Config (from env)
-----------------
    FLUX_MODEL_ID        — default: "black-forest-labs/FLUX.1-schnell"
    SD_MODEL_ID          — default: "Lykon/DreamShaper"
    TPOSE_OPENPOSE_PATH  — path to T-pose skeleton image for SD stage 1
    TRELLIS_MODEL_ID     — default: "microsoft/TRELLIS.2-4B"
    TRELLIS_REPO_PATH    — path to cloned TRELLIS repo
    REMOVE_BG            — "true"/"false" (default: "true")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("ModelManager")

# ── Env config ────────────────────────────────────────────────────────────────

FLUX_MODEL_ID       = os.getenv("FLUX_MODEL_ID",    "black-forest-labs/FLUX.1-schnell")
SD_MODEL_ID         = os.getenv("SD_MODEL_ID",      "Lykon/DreamShaper")
TPOSE_OPENPOSE_PATH = os.getenv("TPOSE_OPENPOSE_PATH", "")
TRELLIS_MODEL_ID    = os.getenv("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")
TRELLIS_REPO_PATH   = os.getenv("TRELLIS_REPO_PATH", os.path.expanduser("~/trellis"))
REMOVE_BG           = os.getenv("REMOVE_BG", "true").lower() == "true"

# ── VRAM eviction table ───────────────────────────────────────────────────────
# Keys: family being loaded.  Values: set of families to evict first.
EVICT_BEFORE: dict[str, set[str]] = {
    "flux":    {"sd", "trellis"},
    "sd":      {"flux", "trellis"},
    "trellis": {"flux", "sd"},
    "rig":     set(),  # CPU-only — no eviction needed
}

VALID_FAMILIES = frozenset(EVICT_BEFORE.keys())


class ModelManager:
    """
    Owns lazy loading and VRAM eviction for Flux, SD, TRELLIS, and Rig models.

    All pipelines are loaded on first use (ensure) and stay in memory until
    an incompatible family is needed — at which point eviction moves them to
    CPU and empties the CUDA cache before the new family is loaded.
    """

    def __init__(self):
        self._flux_pipe:     Optional[object] = None   # diffusers.FluxPipeline
        self._sd_pipes:      Optional[object] = None   # models.sd_model.SDPipes
        self._trellis_pipes: Optional[object] = None   # models.trellis_model.TrellisPipes
        # rig has no persistent pipeline — Blender subprocess per call

    # ── Public: ensure a model family is loaded ───────────────────────────────

    def ensure(self, family: str) -> None:
        """
        Guarantee that *family* is loaded onto CUDA.

        Evicts all incompatible model families first (VRAM eviction table),
        then loads the requested family if not already present.

        Args:
            family: One of "flux", "sd", "trellis", "rig".

        Raises:
            ValueError: If *family* is not recognised.
        """
        if family not in VALID_FAMILIES:
            raise ValueError(
                f"Unknown model family: {family!r}. "
                f"Must be one of: {sorted(VALID_FAMILIES)}"
            )

        if family == "rig":
            return  # CPU-only — nothing to load

        # Evict incompatible families first
        for victim in EVICT_BEFORE.get(family, set()):
            self.evict(victim)

        # Load if not already in memory
        loader = {
            "flux":    self._load_flux,
            "sd":      self._load_sd,
            "trellis": self._load_trellis,
        }[family]
        loader()

    # ── Public: evict a family from VRAM ─────────────────────────────────────

    def evict(self, family: str) -> None:
        """
        Move *family*'s pipelines to CPU and free CUDA cache.

        Safe to call even if the family is not loaded (no-op).

        Args:
            family: One of "flux", "sd", "trellis", "rig".
        """
        if family == "flux":
            self._evict_flux()
        elif family == "sd":
            self._evict_sd()
        elif family == "trellis":
            self._evict_trellis()
        # "rig" is a no-op

    # ── Public: pipeline accessors ────────────────────────────────────────────

    @property
    def flux_pipe(self):
        """Return the loaded FluxPipeline (call ensure("flux") first)."""
        if self._flux_pipe is None:
            raise RuntimeError("Flux pipeline not loaded. Call ensure('flux') first.")
        return self._flux_pipe

    @property
    def sd_pipes(self):
        """Return the loaded SDPipes (call ensure("sd") first)."""
        if self._sd_pipes is None:
            raise RuntimeError("SD pipelines not loaded. Call ensure('sd') first.")
        return self._sd_pipes

    @property
    def trellis_pipes(self):
        """Return the loaded TrellisPipes (call ensure("trellis") first)."""
        if self._trellis_pipes is None:
            raise RuntimeError("TRELLIS pipeline not loaded. Call ensure('trellis') first.")
        return self._trellis_pipes

    # ── Private: loaders ─────────────────────────────────────────────────────

    def _load_flux(self) -> None:
        if self._flux_pipe is not None:
            return
        from models.flux_model import load_flux
        self._flux_pipe = load_flux(FLUX_MODEL_ID)

    def _load_sd(self) -> None:
        if self._sd_pipes is not None:
            return
        from models.sd_model import load_sd
        self._sd_pipes = load_sd(SD_MODEL_ID, openpose_ref_path=TPOSE_OPENPOSE_PATH)

    def _load_trellis(self) -> None:
        if self._trellis_pipes is not None:
            # Already loaded — make sure it's on GPU
            try:
                self._trellis_pipes.pipe.to("cuda")
            except Exception:
                pass
            return
        from models.trellis_model import load_trellis
        self._trellis_pipes = load_trellis(
            model_id=TRELLIS_MODEL_ID,
            repo_path=TRELLIS_REPO_PATH,
            remove_bg=REMOVE_BG,
        )

    # ── Private: evictors ────────────────────────────────────────────────────

    def _evict_flux(self) -> None:
        """Flux uses sequential CPU offload — no persistent VRAM state. Just clear cache."""
        torch.cuda.empty_cache()

    def _evict_sd(self) -> None:
        if self._sd_pipes is None:
            return
        pipes = self._sd_pipes
        for obj in (
            pipes.pipe_biped,
            pipes.pipe_quad,
            pipes.pipe_i2i,
        ):
            if obj is not None:
                try:
                    obj.to("cpu")
                except Exception:
                    pass
        torch.cuda.empty_cache()
        logger.info("[evict] SD pipelines moved to CPU")

    def _evict_trellis(self) -> None:
        if self._trellis_pipes is None:
            return
        try:
            self._trellis_pipes.pipe.to("cpu")
        except Exception:
            try:
                self._trellis_pipes.pipe.cpu()
            except Exception:
                pass
        torch.cuda.empty_cache()
        logger.info("[evict] TRELLIS pipeline moved to CPU")

    # ── Info ─────────────────────────────────────────────────────────────────

    def loaded_families(self) -> list[str]:
        """Return names of currently loaded model families."""
        loaded = []
        if self._flux_pipe    is not None: loaded.append("flux")
        if self._sd_pipes     is not None: loaded.append("sd")
        if self._trellis_pipes is not None: loaded.append("trellis")
        return loaded

    def vram_summary(self) -> str:
        """Return a short VRAM usage string (requires CUDA)."""
        if not torch.cuda.is_available():
            return "CUDA not available"
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        return (
            f"VRAM {allocated:.1f}/{total:.1f} GB allocated  "
            f"({reserved:.1f} GB reserved)  "
            f"loaded={self.loaded_families()}"
        )
