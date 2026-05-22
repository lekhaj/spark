#!/usr/bin/env bash
# install/flux.sh — FLUX.1-schnell. Lives in the `spark` conda env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

activate_env spark
hf_pull "black-forest-labs/FLUX.1-schnell"
log "flux install done"
