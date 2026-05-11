"""
model_manager.py — GPU model lifecycle manager
===============================================

Owns lazy loading and VRAM eviction for all model families.
Model families are registered dynamically — the class never needs editing
when a new model is added.

Usage
-----
    from model_manager import ModelManager

    mgr = ModelManager()

    # ── Built-in families ─────────────────────────────────────────────────────
    mgr.ensure("flux")
    img = run_flux(mgr.get("flux"), prompt, params)

    mgr.ensure("sd")
    img = run_stage1(mgr.get("sd"), init_img, prompt, neg, params)

    mgr.ensure("trellis")
    glb = run_trellis(mgr.get("trellis"), img, params)

    # "rig" is CPU-only — no ensure() needed

    # ── Adding a new model family (e.g. SDXL) ────────────────────────────────
    from models.sdxl_model import load_sdxl
    mgr.register(
        name    = "sdxl",
        loader  = lambda: load_sdxl(os.getenv("SDXL_MODEL_ID", "...")),
        evicts  = {"flux", "trellis"},   # families that must be off VRAM first
    )
    mgr.ensure("sdxl")
    img = run_sdxl(mgr.get("sdxl"), prompt, params)

Built-in EVICT_BEFORE rules
----------------------------
    flux    — evicts: sd, trellis
    sd      — evicts: flux, trellis
    trellis — evicts: flux, sd
    rig     — CPU-only, no eviction

Config (from env)
-----------------
    FLUX_MODEL_ID       default: "black-forest-labs/FLUX.1-schnell"
    SD_MODEL_ID         default: "Lykon/DreamShaper"
    TPOSE_OPENPOSE_PATH default: ""
    TRELLIS_MODEL_ID    default: "JeffreyXiang/TRELLIS-image-large"
    TRELLIS_REPO_PATH   default: "~/trellis"
    REMOVE_BG           default: "true"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger("ModelManager")

# ── Env config ────────────────────────────────────────────────────────────────

FLUX_MODEL_ID       = os.getenv("FLUX_MODEL_ID",    "black-forest-labs/FLUX.1-schnell")
SD_MODEL_ID         = os.getenv("SD_MODEL_ID",      "Lykon/DreamShaper")
TPOSE_OPENPOSE_PATH = os.getenv("TPOSE_OPENPOSE_PATH", "")
TRELLIS_MODEL_ID    = os.getenv("TRELLIS_MODEL_ID", "JeffreyXiang/TRELLIS-image-large")
TRELLIS_REPO_PATH   = os.getenv("TRELLIS_REPO_PATH", os.path.expanduser("~/trellis"))
REMOVE_BG           = os.getenv("REMOVE_BG", "true").lower() == "true"


class ModelManager:
    """
    Dynamic GPU model registry.

    Each family has:
      - a loader callable (called once, result cached)
      - a set of families that must be evicted from VRAM before it loads
      - a cached pipeline object (or None if not yet loaded)

    Eviction moves the pipeline object to CPU and empties the CUDA cache.
    """

    def __init__(self):
        # { name -> {"loader": fn, "evicts": set, "pipeline": obj | None} }
        self._registry: dict[str, dict] = {}

        # Register all built-in model families
        self._register_builtins()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        name:   str,
        loader: Callable[[], Any],
        evicts: set[str] | None = None,
    ) -> None:
        """
        Register a model family.

        Can be called at any time — before or after the manager is used.
        If *name* is already registered the loader and eviction set are updated
        (useful for swapping a model at runtime).

        Args:
            name:   Unique family identifier, e.g. "sdxl", "animatediff".
            loader: Zero-argument callable that returns the loaded pipeline.
                    Called once; result is cached.  Example:
                        lambda: load_sdxl(os.getenv("SDXL_MODEL_ID"))
            evicts: Set of family names that must be evicted from VRAM before
                    this family loads.  Defaults to evicting everything else.
        """
        if name in self._registry and self._registry[name]["pipeline"] is not None:
            logger.warning(
                f"[register] {name!r} is already loaded — "
                "new loader takes effect on next ensure() call after eviction."
            )

        self._registry[name] = {
            "loader":   loader,
            "evicts":   evicts if evicts is not None else set(),
            "pipeline": self._registry.get(name, {}).get("pipeline"),  # keep if already loaded
        }
        logger.debug(f"[register] family={name!r}  evicts={evicts}")

    def _register_builtins(self) -> None:
        """Register Flux, SD, TRELLIS, and Rig with their eviction rules."""

        self.register(
            name   = "flux",
            loader = self._load_flux,
            evicts = {"sd", "trellis"},
        )
        self.register(
            name   = "sd",
            loader = self._load_sd,
            evicts = {"flux", "trellis"},
        )
        self.register(
            name   = "trellis",
            loader = self._load_trellis,
            evicts = {"flux", "sd"},
        )
        self.register(
            name   = "rig",
            loader = lambda: None,   # CPU-only — pipeline object not needed
            evicts = set(),
        )

    # ── ensure / evict / get ──────────────────────────────────────────────────

    def ensure(self, family: str) -> None:
        """
        Guarantee *family* is loaded on CUDA.

        1. Evicts all families listed in this family's ``evicts`` set.
        2. Calls the loader if the pipeline is not yet cached.

        Args:
            family: Registered family name.

        Raises:
            KeyError: If *family* was never registered.
        """
        if family not in self._registry:
            raise KeyError(
                f"Unknown model family: {family!r}. "
                f"Registered: {sorted(self._registry)}"
            )

        entry = self._registry[family]

        if entry["pipeline"] is not None:
            # Already loaded — move back to CUDA if it was evicted
            self._resume_cuda(family, entry["pipeline"])
            return

        # Evict incompatible families first
        for victim in entry["evicts"]:
            self.evict(victim)

        logger.info(f"[ensure] Loading family: {family!r}  {self.vram_summary()}")
        entry["pipeline"] = entry["loader"]()
        logger.info(f"[ensure] {family!r} ready  {self.vram_summary()}")

    def evict(self, family: str) -> None:
        """
        Move *family*'s pipeline to CPU and free CUDA cache.

        Safe to call even if *family* is not loaded or not registered.

        Args:
            family: Family name to evict.
        """
        entry = self._registry.get(family)
        if entry is None or entry["pipeline"] is None:
            return

        pipeline = entry["pipeline"]
        self._move_to_cpu(pipeline)
        torch.cuda.empty_cache()
        logger.info(f"[evict] {family!r} moved to CPU  {self.vram_summary()}")

    def get(self, family: str) -> Any:
        """
        Return the cached pipeline for *family*.

        Call ensure(family) first — raises RuntimeError if not loaded.

        Args:
            family: Registered family name.

        Returns:
            The pipeline object returned by the loader (type depends on family).
        """
        entry = self._registry.get(family)
        if entry is None:
            raise KeyError(f"Unknown family: {family!r}")
        if entry["pipeline"] is None:
            raise RuntimeError(
                f"Family {family!r} not loaded. Call ensure({family!r}) first."
            )
        return entry["pipeline"]

    # ── Convenience aliases (backwards-compat with old property names) ────────

    @property
    def flux_pipe(self):
        return self.get("flux")

    @property
    def sd_pipes(self):
        return self.get("sd")

    @property
    def trellis_pipes(self):
        return self.get("trellis")

    # ── Built-in loaders ──────────────────────────────────────────────────────

    def _load_flux(self):
        from models.flux_model import load_flux
        return load_flux(FLUX_MODEL_ID)

    def _load_sd(self):
        from models.sd_model import load_sd
        return load_sd(SD_MODEL_ID, openpose_ref_path=TPOSE_OPENPOSE_PATH)

    def _load_trellis(self):
        from models.trellis_model import load_trellis
        return load_trellis(
            model_id  = TRELLIS_MODEL_ID,
            repo_path = TRELLIS_REPO_PATH,
            remove_bg = REMOVE_BG,
        )

    # ── VRAM helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _move_to_cpu(pipeline: Any) -> None:
        """Best-effort move of any pipeline object to CPU."""
        if pipeline is None:
            return
        # Try common diffusers patterns
        if hasattr(pipeline, "to"):
            try:
                pipeline.to("cpu")
                return
            except Exception:
                pass
        if hasattr(pipeline, "cpu"):
            try:
                pipeline.cpu()
                return
            except Exception:
                pass
        # Dataclass (e.g. SDPipes, TrellisPipes) — move each sub-pipeline
        if hasattr(pipeline, "__dataclass_fields__"):
            for field_name in pipeline.__dataclass_fields__:
                sub = getattr(pipeline, field_name, None)
                if sub is not None and hasattr(sub, "to"):
                    try:
                        sub.to("cpu")
                    except Exception:
                        pass

    @staticmethod
    def _resume_cuda(family: str, pipeline: Any) -> None:
        """Best-effort move of an already-loaded pipeline back to CUDA."""
        if pipeline is None or family == "rig":
            return
        # Flux uses sequential CPU offload — no explicit .to("cuda") needed
        if family == "flux":
            return
        if hasattr(pipeline, "to"):
            try:
                pipeline.to("cuda")
                return
            except Exception:
                pass
        if hasattr(pipeline, "cuda"):
            try:
                pipeline.cuda()
                return
            except Exception:
                pass
        if hasattr(pipeline, "__dataclass_fields__"):
            for field_name in pipeline.__dataclass_fields__:
                sub = getattr(pipeline, field_name, None)
                if sub is not None and hasattr(sub, "to"):
                    try:
                        sub.to("cuda")
                    except Exception:
                        pass

    # ── Info ──────────────────────────────────────────────────────────────────

    def loaded_families(self) -> list[str]:
        """Return names of families that have a cached pipeline."""
        return [
            name for name, entry in self._registry.items()
            if entry["pipeline"] is not None
        ]

    def registered_families(self) -> list[str]:
        """Return all registered family names."""
        return sorted(self._registry)

    def vram_summary(self) -> str:
        """Short VRAM usage string."""
        if not torch.cuda.is_available():
            return "CUDA not available"
        alloc = torch.cuda.memory_allocated() / 1e9
        rsvd  = torch.cuda.memory_reserved()  / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return (
            f"VRAM {alloc:.1f}/{total:.1f}GB alloc "
            f"({rsvd:.1f}GB reserved) "
            f"loaded={self.loaded_families()}"
        )
