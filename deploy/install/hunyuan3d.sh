#!/usr/bin/env bash
# install/hunyuan3d.sh — Hunyuan3D-2.0 in isolated conda env (hunyuan3d).
#
# This stage is expensive to install cold (~30-60 min for compiled CUDA
# extensions). After first successful boot, AMI rebake is recommended so
# subsequent spot boots skip compilation entirely.
#
# Bumping spec.yaml stage_defs.hunyuan3d.version forces a re-install.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

HY_REPO="${HUNYUAN3D_REPO:-$HOME/Hunyuan3D-2}"
HY_ENV="hunyuan3d"
HY_MODEL_ID="${HUNYUAN3D_MODEL_ID:-tencent/Hunyuan3D-2}"

# ── Conda env ────────────────────────────────────────────────────────────────
if [[ ! -d "$CONDA_BASE/envs/$HY_ENV" ]]; then
    log "creating conda env $HY_ENV (python 3.10)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda create -n "$HY_ENV" python=3.10 -y
fi

activate_env "$HY_ENV"

# ── PyTorch cu124 ─────────────────────────────────────────────────────────────
pip_install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Clone source tree ────────────────────────────────────────────────────────
if [[ ! -d "$HY_REPO" ]]; then
    log "cloning Hunyuan3D-2 → $HY_REPO"
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$HY_REPO"
else
    log "Hunyuan3D-2 repo exists at $HY_REPO — pulling latest"
    git -C "$HY_REPO" pull --ff-only || log "  git pull failed (pinned commit?) — continuing"
fi

cd "$HY_REPO"

# ── Core deps ─────────────────────────────────────────────────────────────────
pip_install transformers==4.46.0 diffusers==0.30.0 safetensors==0.4.4 \
    accelerate huggingface-hub
pip_install trimesh scipy einops pillow rembg[gpu] requests

# ── Hunyuan3D requirements ────────────────────────────────────────────────────
if [[ -f requirements.txt ]]; then
    log "installing requirements.txt"
    # Exclude torch pins (already installed above) and direct git deps that
    # need special handling to avoid version conflicts.
    grep -v "^torch" requirements.txt \
        | grep -v "^torchvision" \
        | grep -v "^torchaudio" \
        | pip install --quiet -r /dev/stdin 2>&1 \
        | tail -5 | while read -r line; do log "  pip: $line"; done || \
        log "  some requirements skipped — continuing"
fi

# ── Install hy3dgen package from repo root ────────────────────────────────────
if [[ -f setup.py ]] || [[ -f pyproject.toml ]]; then
    log "installing hy3dgen package from repo"
    pip_install -e . --no-build-isolation
fi

# ── Custom rasterizer (CUDA extension) ───────────────────────────────────────
# This takes 10-20 min on first build. Result is cached in conda env.
if [[ -d hy3dpaint/custom_rasterizer ]]; then
    if ! python -c "import custom_rasterizer" 2>/dev/null; then
        log "building custom_rasterizer CUDA extension (takes ~15 min)"
        cd hy3dpaint/custom_rasterizer
        pip_install -e . --no-build-isolation
        cd "$HY_REPO"
    else
        log "custom_rasterizer already built — skipping"
    fi
fi

# ── DifferentiableRenderer ────────────────────────────────────────────────────
if [[ -d hy3dpaint/DifferentiableRenderer ]]; then
    if [[ -f hy3dpaint/DifferentiableRenderer/compile_mesh_painter.sh ]]; then
        if ! python -c "import mesh_painter" 2>/dev/null; then
            log "compiling DifferentiableRenderer (takes ~5 min)"
            cd hy3dpaint/DifferentiableRenderer
            bash compile_mesh_painter.sh 2>&1 | tail -5 | while read -r l; do log "  compile: $l"; done
            cd "$HY_REPO"
        else
            log "DifferentiableRenderer already compiled — skipping"
        fi
    fi
fi

# ── RealESRGAN upscaler weights ───────────────────────────────────────────────
ESRGAN_CKPT="$HY_REPO/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
if [[ ! -f "$ESRGAN_CKPT" ]]; then
    log "downloading RealESRGAN_x4plus.pth"
    mkdir -p "$(dirname "$ESRGAN_CKPT")"
    curl -fsSL \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -o "$ESRGAN_CKPT" \
        || die "RealESRGAN download failed"
else
    log "RealESRGAN weights already present — skipping"
fi

# ── HF model weights ──────────────────────────────────────────────────────────
hf_pull "$HY_MODEL_ID"

log "hunyuan3d install done"
