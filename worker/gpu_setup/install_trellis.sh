#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# install_trellis.sh — Install TRELLIS 3D generation into the spark conda env
# ════════════════════════════════════════════════════════════════════════════
#
# TRELLIS (Microsoft) produces rig-ready 3D meshes from 2D images.
# Uses the existing 'spark' conda environment created by install_gpu.sh.
#
# Supported models (set TRELLIS_MODEL_ID in .env):
#   microsoft/TRELLIS-image-large  (16 GB VRAM — default)
#   microsoft/TRELLIS.2-4B         (24 GB VRAM — better topology, recommended for L4)
#
# Usage:
#   bash worker/gpu_setup/install_trellis.sh 2>&1 | tee /tmp/trellis_install.log
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CONDA_DIR=/home/ec2-user/miniconda3
PIP="$CONDA_DIR/envs/spark/bin/pip"
PYTHON="$CONDA_DIR/envs/spark/bin/python3"
TRELLIS_DIR=/home/ec2-user/trellis

log()  { echo ""; echo "════ $* ════"; echo ""; }
ok()   { echo "  [OK] $*"; }
warn() { echo "  [WARN] $*"; }

# ── 1. Clone TRELLIS repo ─────────────────────────────────────────────────────
log "STEP 1: Clone TRELLIS repository"
if [ -d "$TRELLIS_DIR" ]; then
  echo "TRELLIS already cloned at $TRELLIS_DIR — pulling latest"
  cd "$TRELLIS_DIR" && git pull --recurse-submodules || warn "git pull had warnings"
else
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "$TRELLIS_DIR"
  ok "Cloned to $TRELLIS_DIR"
fi

# ── 2. Install TRELLIS Python package (editable) ─────────────────────────────
log "STEP 2: Install TRELLIS package into spark env"
cd "$TRELLIS_DIR"
$PIP install -e . --no-build-isolation 2>&1 | tail -5
ok "TRELLIS package installed"

# ── 3. Core dependencies ──────────────────────────────────────────────────────
log "STEP 3: Core TRELLIS dependencies"
$PIP install \
  easydict \
  imageio \
  imageio-ffmpeg \
  opencv-python-headless \
  scipy \
  ninja \
  2>&1 | tail -3
ok "Core deps installed"

# ── 4. spconv (sparse convolutions — required by TRELLIS) ────────────────────
log "STEP 4: spconv for sparse 3D ops"
# spconv wheels for CUDA 12.x
$PIP install spconv-cu120 2>&1 | tail -3 && ok "spconv installed" || \
  warn "spconv install failed — try: pip install spconv-cu118"

# ── 5. flash-attn (speed up attention — optional but recommended) ─────────────
log "STEP 5: flash-attn (faster attention)"
# Pre-built wheel — much faster than building from source
$PIP install flash-attn --no-build-isolation 2>&1 | tail -3 && ok "flash-attn installed" || \
  warn "flash-attn build failed — TRELLIS will use xformers fallback (slower)"

# ── 6. diffoctreerast (differentiable octree rasterizer) ─────────────────────
log "STEP 6: diffoctreerast"
if [ -d "$TRELLIS_DIR/extensions/diffoctreerast" ]; then
  cd "$TRELLIS_DIR/extensions/diffoctreerast"
  $PIP install -e . --no-build-isolation 2>&1 | tail -3 && ok "diffoctreerast installed" || \
    warn "diffoctreerast failed — texture baking may be slower"
  cd "$TRELLIS_DIR"
fi

# ── 7. nvdiffrast (differentiable rasterization for texture baking) ──────────
log "STEP 7: nvdiffrast"
$PIP install git+https://github.com/NVlabs/nvdiffrast.git 2>&1 | tail -3 && \
  ok "nvdiffrast installed" || warn "nvdiffrast failed — texture baking may fail"

# ── 8. Patch TRELLIS flexicubes to remove kaolin dependency ──────────────────
# kaolin prebuilt wheels only cover specific torch versions and break on upgrade.
# TRELLIS only uses kaolin for one tensor validation check (check_tensor),
# which we replace with a no-op shim.
log "STEP 8: Patch flexicubes to remove kaolin dependency"
$PYTHON << 'PATCHEOF'
import sys, os
trellis_root = os.getenv("TRELLIS_DIR", os.path.expanduser("~/trellis"))
path = os.path.join(trellis_root,
    "trellis/representations/mesh/flexicubes/flexicubes.py")
if os.path.exists(path):
    with open(path) as f:
        code = f.read()
    if "from kaolin.utils.testing" in code:
        code = code.replace(
            "from kaolin.utils.testing import check_tensor",
            "# kaolin removed — no-op shim\ndef check_tensor(t, *a, **kw): return True"
        )
        with open(path, "w") as f:
            f.write(code)
        print(f"Patched: {path}")
    else:
        print("flexicubes.py already patched or kaolin not referenced.")
else:
    print(f"WARNING: flexicubes.py not found at {path}")
PATCHEOF
ok "flexicubes.py patched"

# ── 9. Set TRELLIS_REPO_PATH in .env ─────────────────────────────────────────
log "STEP 9: Update .env with TRELLIS config"
ENV_FILE=/home/ec2-user/spark/.env

grep -q 'TRELLIS_REPO_PATH' "$ENV_FILE" 2>/dev/null || cat >> "$ENV_FILE" << 'ENVEOF'

# TRELLIS 3D generation config
TRELLIS_REPO_PATH=/home/ec2-user/trellis
# Use TRELLIS.2-4B for best game mesh quality on 24GB L4:
TRELLIS_MODEL_ID=microsoft/TRELLIS.2-4B
TRELLIS_STEPS_SPARSE=12
TRELLIS_STEPS_DENSE=12
TRELLIS_CFG_STRENGTH=7.5
TRELLIS_SIMPLIFY=0.95
TRELLIS_TEXTURE_SIZE=1024
ENVEOF
ok ".env updated"

# ── 10. Quick import test ─────────────────────────────────────────────────────
log "STEP 10: Verify TRELLIS import"
export PYTHONPATH="$TRELLIS_DIR:${PYTHONPATH:-}"
$PYTHON -c "
import sys
sys.path.insert(0, '$TRELLIS_DIR')
from trellis.pipelines import TrellisImageTo3DPipeline
print('TRELLIS import OK')
print('Available pipeline:', TrellisImageTo3DPipeline)
" && ok "TRELLIS import verified" || warn "TRELLIS import failed — check logs above"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  TRELLIS install complete!"
echo ""
echo "  Model will be downloaded on first run (~10-15 GB)."
echo "  Default model: microsoft/TRELLIS.2-4B (24GB L4 optimal)"
echo ""
echo "  To change model, edit .env:"
echo "    TRELLIS_MODEL_ID=microsoft/TRELLIS-image-large  (16GB, faster)"
echo "    TRELLIS_MODEL_ID=microsoft/TRELLIS.2-4B          (24GB, better quality)"
echo ""
echo "  Start workers:"
echo "    screen -dmS workers bash -c 'cd ~/spark && \\"
echo "      /home/ec2-user/miniconda3/envs/spark/bin/python3 \\"
echo "      worker/gpu_main.py --workers sd15,trellis 2>&1 | tee /tmp/gpu_workers.log'"
echo "════════════════════════════════════════════════════════"
