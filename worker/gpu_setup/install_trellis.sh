#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# install_trellis.sh — Install TRELLIS.2 into the spark conda env
# ════════════════════════════════════════════════════════════════════════════
#
# TRELLIS.2 (Microsoft) — 4B param image-to-3D with:
#   • O-Voxel representation (better topology than original TRELLIS)
#   • Full PBR materials (base color, roughness, metallic)
#   • Cleaner mesh → better Auto-Rig Pro binding
#   • Requires 24 GB VRAM — exact fit for L4 at 512³
#
# Model: microsoft/TRELLIS.2-4B (~15 GB, downloaded on first run)
#
# Usage:
#   bash worker/gpu_setup/install_trellis.sh 2>&1 | tee /tmp/trellis_install.log
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CONDA_DIR=/home/ec2-user/miniconda3
PIP="$CONDA_DIR/envs/spark/bin/pip"
PYTHON="$CONDA_DIR/envs/spark/bin/python3"
TRELLIS_DIR=/home/ec2-user/trellis          # repo cloned here (TRELLIS_REPO_PATH)
EXT_DIR=/tmp/trellis2_extensions             # temp build area

log()  { echo ""; echo "════ $* ════"; echo ""; }
ok()   { echo "  [OK] $*"; }
warn() { echo "  [WARN] $*"; }

# ── 1. Clone TRELLIS.2 repo ───────────────────────────────────────────────────
log "STEP 1: Clone TRELLIS.2 repository"
if [ -d "$TRELLIS_DIR/.git" ]; then
  REMOTE=$(cd "$TRELLIS_DIR" && git remote get-url origin 2>/dev/null || true)
  if echo "$REMOTE" | grep -q "TRELLIS.2"; then
    echo "TRELLIS.2 already cloned at $TRELLIS_DIR — pulling latest"
    cd "$TRELLIS_DIR" && git pull --recurse-submodules || warn "git pull had warnings"
  else
    echo "Found OLD TRELLIS repo at $TRELLIS_DIR — replacing with TRELLIS.2"
    rm -rf "$TRELLIS_DIR"
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
    ok "TRELLIS.2 cloned to $TRELLIS_DIR"
  fi
else
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
  ok "TRELLIS.2 cloned to $TRELLIS_DIR"
fi

# ── 2. Install TRELLIS.2 package (editable) ───────────────────────────────────
log "STEP 2: Install trellis2 package into spark env"
cd "$TRELLIS_DIR"
$PIP install -e . --no-build-isolation 2>&1 | tail -5
ok "trellis2 package installed"

# ── 3. Core Python dependencies ───────────────────────────────────────────────
log "STEP 3: Core dependencies"
$PIP install \
  imageio \
  imageio-ffmpeg \
  tqdm \
  easydict \
  ninja \
  trimesh \
  transformers \
  scipy \
  lpips \
  zstandard \
  kornia \
  timm \
  2>&1 | tail -3
ok "Core deps installed"

# utils3d (pinned commit, required by TRELLIS.2)
$PIP install \
  "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
  2>&1 | tail -3
ok "utils3d installed"

# ── 4. Build TRELLIS.2 C++ extensions ────────────────────────────────────────
log "STEP 4: Build TRELLIS.2 extensions (o_voxel, CuMesh, FlexGEMM)"
mkdir -p "$EXT_DIR"

build_ext() {
  local name="$1"
  local src="$TRELLIS_DIR/extensions/$name"
  if [ -d "$src" ]; then
    cp -r "$src" "$EXT_DIR/$name"
    cd "$EXT_DIR/$name"
    $PIP install --no-build-isolation . 2>&1 | tail -3 && \
      ok "$name installed" || warn "$name failed"
    cd "$TRELLIS_DIR"
  else
    warn "Extension not found: $src"
  fi
}

# o_voxel — required for GLB export with PBR materials
build_ext "o_voxel"

# CuMesh — mesh operations on GPU
build_ext "cumesh"

# FlexGEMM — faster matrix ops
build_ext "flexgemm"

# ── 5. nvdiffrast (differentiable rasterization — texture baking) ─────────────
log "STEP 5: nvdiffrast"
if [ -d "$EXT_DIR/nvdiffrast" ]; then
  echo "nvdiffrast already built"
else
  cd "$EXT_DIR"
  git clone https://github.com/NVlabs/nvdiffrast.git
  cd nvdiffrast
  $PIP install --no-build-isolation . 2>&1 | tail -3 && \
    ok "nvdiffrast installed" || warn "nvdiffrast failed — texture baking may fail"
  cd "$TRELLIS_DIR"
fi

# ── 6. opencv (headless — no X11) ─────────────────────────────────────────────
log "STEP 6: opencv-python-headless"
$PIP install "opencv-python-headless<4.10" 2>&1 | tail -2 && ok "opencv installed" || \
  warn "opencv already present"

# ── 7. rembg (background removal for cleaner TRELLIS.2 input) ─────────────────
log "STEP 7: rembg background remover"
$PIP install rembg 2>&1 | tail -3 && ok "rembg installed" || warn "rembg already present"

# ── 8. Update .env ────────────────────────────────────────────────────────────
log "STEP 8: Update .env with TRELLIS.2 config"
ENV_FILE=/home/ec2-user/spark/.env

# Remove any old TRELLIS entries first, then append fresh ones
if grep -q 'TRELLIS_REPO_PATH\|TRELLIS_MODEL_ID' "$ENV_FILE" 2>/dev/null; then
  echo "Updating existing TRELLIS entries in .env..."
  sed -i '/^TRELLIS_REPO_PATH/d'   "$ENV_FILE"
  sed -i '/^TRELLIS_MODEL_ID/d'    "$ENV_FILE"
  sed -i '/^TRELLIS_STEPS/d'       "$ENV_FILE"
  sed -i '/^TRELLIS_CFG/d'         "$ENV_FILE"
  sed -i '/^TRELLIS_SIMPLIFY/d'    "$ENV_FILE"
  sed -i '/^TRELLIS_TEXTURE/d'     "$ENV_FILE"
  sed -i '/^TRELLIS_DECIMATION/d'  "$ENV_FILE"
  sed -i '/^TRELLIS_REMESH/d'      "$ENV_FILE"
fi

cat >> "$ENV_FILE" << 'ENVEOF'

# TRELLIS.2 3D generation config
TRELLIS_REPO_PATH=/home/ec2-user/trellis
TRELLIS_MODEL_ID=microsoft/TRELLIS.2-4B
TRELLIS_TEXTURE_SIZE=1024
TRELLIS_DECIMATION=1000000
TRELLIS_REMESH=true
ENVEOF
ok ".env updated with TRELLIS.2 config"

# ── 9. Verify import ──────────────────────────────────────────────────────────
log "STEP 9: Verify trellis2 import"
export PYTHONPATH="$TRELLIS_DIR:${PYTHONPATH:-}"
$PYTHON - << 'PYEOF'
import sys
sys.path.insert(0, '/home/ec2-user/trellis')
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print("  [OK] trellis2 import: Trellis2ImageTo3DPipeline found")
except ImportError as e:
    print(f"  [FAIL] trellis2 import failed: {e}")
try:
    import o_voxel
    print("  [OK] o_voxel import OK")
except ImportError as e:
    print(f"  [WARN] o_voxel not available: {e}")
PYEOF

# ── 10. HuggingFace model pre-cache (optional, recommended) ──────────────────
log "STEP 10: Pre-cache TRELLIS.2-4B model weights from HuggingFace"
echo "  This downloads ~15 GB. Press Ctrl+C to skip (will download on first task)."
$PYTHON - << 'PYEOF'
import sys
sys.path.insert(0, '/home/ec2-user/trellis')
import os
os.environ.setdefault('ATTN_BACKEND', 'xformers')
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download('microsoft/TRELLIS.2-4B')
    print(f"  [OK] Model cached at: {path}")
    # Write actual snapshot path to .env for reliable loading
    env_path = '/home/ec2-user/spark/.env'
    with open(env_path) as f:
        content = f.read()
    if path not in content:
        with open(env_path, 'a') as f:
            f.write(f'\n# TRELLIS.2 local snapshot path (avoids HF auth on load)\n')
            f.write(f'TRELLIS_MODEL_ID={path}\n')
        print(f"  [OK] .env updated with local snapshot path")
except Exception as e:
    print(f"  [WARN] Pre-cache failed: {e}")
    print("  Model will download on first task run.")
PYEOF

echo ""
echo "════════════════════════════════════════════════════════"
echo "  TRELLIS.2 install complete!"
echo ""
echo "  Model  : microsoft/TRELLIS.2-4B"
echo "  Repo   : $TRELLIS_DIR"
echo "  VRAM   : ~20-22 GB at 512³ (L4 safe)"
echo ""
echo "  Features vs original TRELLIS:"
echo "    ✓ PBR materials (roughness/metallic)"
echo "    ✓ Better mesh topology for rigging"
echo "    ✓ WebP-compressed textures in GLB"
echo ""
echo "  Start workers:"
echo "    bash worker/gpu_setup/start_workers.sh"
echo "════════════════════════════════════════════════════════"
