#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# install_gpu.sh — Full GPU Setup for Spark Pipeline on Amazon Linux 2023
# ════════════════════════════════════════════════════════════════════════════
# Installs:
#   • NVIDIA drivers + CUDA 12.4
#   • Python 3.11 via Miniconda
#   • PyTorch 2.3 (CUDA 12.1 wheel — compatible with CUDA 12.4)
#   • diffusers, transformers, xformers (SD1.5 + ControlNet)
#   • controlnet-aux (OpenPose + AnimalPose preprocessors)
#   • Hunyuan3D-2.1 (TRELLIS 3D backend)
#   • rembg (background removal)
#   • boto3, redis, pymongo, python-dotenv, Pillow
#
# Usage (run as ec2-user on L4):
#   bash install_gpu.sh 2>&1 | tee /tmp/gpu_install.log
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail   # no -e: let dnf conflicts be non-fatal
export DEBIAN_FRONTEND=noninteractive

log() { echo ""; echo "════ $* ════"; echo ""; }
ok()  { echo "  [OK] $*"; }
warn(){ echo "  [WARN] $*"; }

# ── 1. System packages ────────────────────────────────────────────────────────
log "STEP 1: System packages"
sudo dnf update -y 2>/dev/null || warn "dnf update had warnings (ok)"

# Install packages individually so one conflict doesn't block the rest
for pkg in gcc gcc-c++ make cmake git wget \
  "kernel-devel-$(uname -r)" "kernel-headers-$(uname -r)" \
  python3-devel python3-pip \
  libGL libGLU mesa-libGL-devel \
  libXrender libXext libSM \
  screen htop tmux unzip; do
  sudo dnf install -y "$pkg" 2>/dev/null && ok "$pkg" || warn "skipped: $pkg"
done

# ── 2. NVIDIA Driver + CUDA 12.4 ─────────────────────────────────────────────
log "STEP 2: NVIDIA Driver + CUDA 12.4"
if nvidia-smi &>/dev/null; then
  echo "NVIDIA driver already installed — skipping"
else
  # Add CUDA repo for Amazon Linux 2023
  sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
  sudo dnf clean expire-cache
  sudo dnf module install -y nvidia-driver:latest-dkms
  sudo dnf install -y cuda-toolkit-12-4

  # Add CUDA to PATH permanently
  cat >> /home/ec2-user/.bashrc << 'EOF'
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
EOF
  source /home/ec2-user/.bashrc
  echo "CUDA 12.4 installed. You may need to reboot for driver to load."
fi

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# ── 3. Miniconda (Python 3.11) ────────────────────────────────────────────────
log "STEP 3: Miniconda Python 3.11"
CONDA_DIR=/home/ec2-user/miniconda3
if [ -d "$CONDA_DIR" ]; then
  echo "Miniconda already installed at $CONDA_DIR"
else
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm /tmp/miniconda.sh
fi

export PATH="$CONDA_DIR/bin:$PATH"
conda init bash 2>/dev/null || true

# Create spark env with Python 3.11
if conda info --envs | grep -q '^spark '; then
  echo "Conda env 'spark' already exists"
else
  conda create -y -n spark python=3.11
fi

# Activate for this script
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate spark

echo "Python: $(python --version)"

# Add activation to .bashrc
grep -q 'conda activate spark' /home/ec2-user/.bashrc || \
  echo "conda activate spark" >> /home/ec2-user/.bashrc

# ── 4. PyTorch 2.3 + CUDA 12.1 ───────────────────────────────────────────────
log "STEP 4: PyTorch 2.3 (CUDA 12.1 wheel)"
pip install --upgrade pip
pip install \
  torch==2.3.0 \
  torchvision==0.18.0 \
  torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
           print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ── 5. xFormers ───────────────────────────────────────────────────────────────
log "STEP 5: xFormers memory-efficient attention"
pip install xformers==0.0.26post1 --index-url https://download.pytorch.org/whl/cu121

# ── 6. Diffusers + Transformers (SD1.5 + ControlNet) ─────────────────────────
log "STEP 6: Diffusers / Transformers / Accelerate"
pip install \
  diffusers==0.27.2 \
  transformers==4.40.2 \
  accelerate==0.30.1 \
  safetensors==0.4.3 \
  omegaconf

# ── 7. ControlNet-Aux (OpenPose + AnimalPose preprocessors) ──────────────────
log "STEP 7: controlnet-aux"
pip install controlnet-aux==0.0.7

# ── 8. Hunyuan3D-2.1 (TRELLIS 3D backend) ────────────────────────────────────
log "STEP 8: Hunyuan3D-2.1"
# Install mesh/geometry libs
pip install \
  trimesh==4.3.2 \
  pymeshlab==2023.12.post2 \
  einops \
  open3d \
  scikit-image

# Install Hunyuan3D from GitHub
pip install git+https://github.com/tencent/Hunyuan3D-2.git || \
  echo "WARNING: Hunyuan3D install may require manual setup — check docs"

# ── 9. Background removal ─────────────────────────────────────────────────────
log "STEP 9: rembg background removal"
pip install rembg[gpu]==2.0.56

# ── 10. Infrastructure + Utility ─────────────────────────────────────────────
log "STEP 10: AWS / Redis / MongoDB / utility libs"
pip install \
  boto3==1.34.100 \
  redis==5.0.4 \
  pymongo==4.7.2 \
  python-dotenv==1.0.1 \
  Pillow==10.3.0 \
  requests \
  tqdm \
  psutil

# ── 11. Verify installation ───────────────────────────────────────────────────
log "STEP 11: Verification"
python - << 'PYEOF'
import sys
print(f"Python: {sys.version}")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
import diffusers; print(f"diffusers: {diffusers.__version__}")
import transformers; print(f"transformers: {transformers.__version__}")
import boto3; print(f"boto3: {boto3.__version__}")
import redis; print(f"redis: {redis.__version__}")
import pymongo; print(f"pymongo: {pymongo.__version__}")
print("All imports OK")
PYEOF

# ── 12. Create controlnet_refs dir ───────────────────────────────────────────
log "STEP 12: Create directories"
mkdir -p /home/ec2-user/controlnet_refs
mkdir -p /home/ec2-user/spark/worker/workers

echo ""
echo "════════════════════════════════════════════════════════"
echo "  GPU setup complete!"
echo ""
echo "  Next steps:"
echo "  1. If CUDA driver was just installed: sudo reboot"
echo "  2. After reboot: source ~/.bashrc"
echo "  3. Generate ControlNet skeletons:"
echo "     python /home/ec2-user/spark/worker/prepare_controlnet_refs.py"
echo "  4. Start workers:"
echo "     screen -dmS workers python /home/ec2-user/spark/worker/gpu_main.py"
echo "     screen -r workers"
echo "════════════════════════════════════════════════════════"
