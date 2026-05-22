#!/usr/bin/env bash
# install/trellis.sh — TRELLIS image→3D weights.
# TRELLIS source tree at ~/trellis is assumed AMI-baked (vendor wheels for
# cumesh/o_voxel/flex_gemm/nvdiffrec are too painful to rebuild every boot).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

activate_env spark

[[ -d "$HOME/trellis" ]] || die "~/trellis source tree missing — needs to be in the AMI"

hf_pull "JeffreyXiang/TRELLIS-image-large"

log "trellis install done"
