#!/usr/bin/env bash
# install/sd_tpose.sh — SD1.5 + IP-Adapter + OpenPose/Canny ControlNet.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

activate_env spark

hf_pull "Lykon/DreamShaper"
hf_pull "lllyasviel/control_v11p_sd15_openpose"
hf_pull "lllyasviel/control_v11p_sd15_canny"
hf_pull "h94/IP-Adapter"
hf_pull "lllyasviel/Annotators"

log "sd_tpose install done"
