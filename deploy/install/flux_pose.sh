#!/usr/bin/env bash
# install/flux_pose.sh — FLUX.1-dev + Shakker-Labs ControlNet-Union-Pro-2.0.
# Adds controlnet-aux for OpenPose/Depth/SoftEdge extractors used by flux_pose.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

activate_env spark

# pip deps (idempotent — pip skips already-satisfied)
pip_install "diffusers>=0.30" "controlnet-aux>=0.0.7"

# Gated model — must be logged in (huggingface-cli login OR HF_TOKEN env var).
# Check via the python API rather than a hardcoded path because the token
# location varies by huggingface_hub version (~/.huggingface/token old,
# ~/.cache/huggingface/token new).
python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1 \
    || die "huggingface not logged in — run 'huggingface-cli login' (need access to gated FLUX.1-dev)"

hf_pull "black-forest-labs/FLUX.1-dev"
hf_pull "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

log "flux_pose install done"
