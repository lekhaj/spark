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

# Gated model — must have ~/.huggingface/token with FLUX.1-dev access
[[ -f "$HOME/.huggingface/token" ]] || die "huggingface token missing at ~/.huggingface/token (required for gated FLUX.1-dev)"

hf_pull "black-forest-labs/FLUX.1-dev"
hf_pull "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

log "flux_pose install done"
