#!/usr/bin/env bash
# install/pixal3d.sh — Pixal3D in isolated conda env (`pixal3d`).
# Bumping spec.yaml stage_defs.pixal3d.version forces a re-install.
# See memory/pixal3d_setup.md for the full why.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

# Conda env (create only if missing — AMI usually has it)
if [[ ! -d "$CONDA_BASE/envs/pixal3d" ]]; then
    log "creating conda env pixal3d (python 3.10)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda create -n pixal3d python=3.10 -y
fi

activate_env pixal3d

# Torch (cu124) + Pixal3D deps
pip_install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

PIXAL_REPO="${PIXAL3D_REPO:-$HOME/Pixal3D}"
[[ -d "$PIXAL_REPO" ]] || die "Pixal3D source tree missing at $PIXAL_REPO — needs to be in the AMI"

cd "$PIXAL_REPO"
pip_install -r requirements-hfdemo.txt --no-build-isolation
pip_install "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl" --no-build-isolation
TMPDIR=/tmp/piptmp pip install --quiet --cache-dir /tmp/pipcache flash-attn==2.7.4.post1 --no-build-isolation \
    || die "flash-attn install failed"

# HF models
hf_pull "TencentARC/Pixal3D"
hf_pull "ZhengPeng7/BiRefNet"

# Patches — applied idempotently
INFER_PY="$PIXAL_REPO/inference.py"
if ! grep -q "PIXAL3D_LOW_VRAM" "$INFER_PY" 2>/dev/null; then
    log "applying inference.py patch (low_vram + attn backend env-driven)"
    python3 - "$INFER_PY" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
if 'os.environ.setdefault("ATTN_BACKEND"' not in s:
    s = s.replace(
        'import os',
        'import os\nos.environ.setdefault("ATTN_BACKEND", os.environ.get("PIXAL3D_ATTN", "flash_attn"))',
        1)
s = re.sub(r'pipeline\.low_vram\s*=\s*True',
           'pipeline.low_vram = os.environ.get("PIXAL3D_LOW_VRAM", "1") == "1"', s)
p.write_text(s)
PY
fi

# Pipeline JSON patch (briaai/RMBG-2.0 gated → ZhengPeng7/BiRefNet ungated)
PIPELINE_JSON="$(ls "$HF_HOME"/hub/models--TencentARC--Pixal3D/snapshots/*/pipeline.json 2>/dev/null | head -1 || true)"
if [[ -n "$PIPELINE_JSON" ]] && grep -q "briaai/RMBG-2.0" "$PIPELINE_JSON"; then
    log "patching pipeline.json: briaai/RMBG-2.0 → ZhengPeng7/BiRefNet"
    sed -i 's|briaai/RMBG-2.0|ZhengPeng7/BiRefNet|g' "$PIPELINE_JSON"
fi

log "pixal3d install done"
