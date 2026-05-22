#!/usr/bin/env bash
# deploy/lib/common.sh — shared helpers for bootstrap + per-stage install scripts.
# Source me; don't execute me.

set -euo pipefail

# --- Paths ---
: "${SPARK_ROOT:=/home/ec2-user/spark}"
: "${SPARK_STATE_DIR:=$HOME/.spark}"
: "${SPARK_SENTINEL_DIR:=$SPARK_STATE_DIR/installed}"
: "${SPARK_LOG_DIR:=$SPARK_STATE_DIR/log}"
: "${HF_HOME:=$HOME/.cache/huggingface}"
: "${CONDA_BASE:=$HOME/miniconda3}"

mkdir -p "$SPARK_STATE_DIR" "$SPARK_SENTINEL_DIR" "$SPARK_LOG_DIR"

# --- Logging ---
log()  { echo "[$(date -u +%H:%M:%S)] [$STAGE_NAME] $*" | tee -a "$SPARK_LOG_DIR/bootstrap.log" >&2; }
die()  { log "FATAL: $*"; exit 1; }

# --- Conda env activation (safe in scripts) ---
activate_env() {
    local env_name="$1"
    [[ -d "$CONDA_BASE/envs/$env_name" ]] || die "conda env '$env_name' not found at $CONDA_BASE/envs/$env_name"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$env_name"
    log "conda env active: $(which python) ($(python --version 2>&1))"
}

# --- Sentinel: skip work that's already done ---
sentinel_path() {
    local stage="$1" version="$2"
    echo "$SPARK_SENTINEL_DIR/${stage}.v${version}"
}

sentinel_ok() {
    local stage="$1" version="$2"
    [[ -f "$(sentinel_path "$stage" "$version")" ]]
}

sentinel_write() {
    local stage="$1" version="$2"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(git -C "$SPARK_ROOT" rev-parse HEAD 2>/dev/null || echo no-git)" \
        > "$(sentinel_path "$stage" "$version")"
}

# --- HF model download (idempotent) ---
# Skips download entirely if the model already exists in the HF cache. This
# matters because snapshot_download can spill 100s of GB into xet temp space
# even when the final cache size is small — disastrous on a 256GB root volume.
#
# To force re-download: rm -rf ~/.cache/huggingface/hub/models--<org>--<name>
hf_pull() {
    local model_id="$1"
    local cache="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${model_id//\//--}"

    if [[ -d "$cache/snapshots" ]] && [[ -n "$(ls -A "$cache/snapshots" 2>/dev/null)" ]]; then
        local size; size="$(du -sh "$cache" 2>/dev/null | cut -f1)"
        log "HF pull: $model_id  → already cached (${size:-?})"
        return 0
    fi

    log "HF pull: $model_id  (downloading)"
    HF_HUB_DISABLE_PROGRESS_BARS=1 python - "$model_id" <<'PY' 2>&1 | tail -3 | while read -r line; do log "  hf: $line"; done
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=sys.argv[1],
    # Skip duplicate format files to keep disk usage sane. Most diffusers
    # pipelines work from safetensors; .bin is redundant. msgpack/ot are
    # JAX/Flax weights we never use.
    ignore_patterns=["*.bin", "*.msgpack", "*.ot", "*.h5", "*.onnx"],
)
print(f"cached at {path}")
PY
    [[ -d "$cache/snapshots" ]] || die "HF download failed: $model_id (no $cache/snapshots)"
}

# --- pip install (idempotent — pip already skips satisfied) ---
pip_install() {
    log "pip install: $*"
    pip install --quiet "$@" 2>&1 | tail -3 | while read -r line; do log "  pip: $line"; done
}

# --- Redis status publish (best-effort; OK if Redis unreachable) ---
status_publish() {
    local key="$1" value="$2"
    if [[ -n "${REDIS_HOST:-}" ]] && command -v redis-cli >/dev/null 2>&1; then
        redis-cli -h "$REDIS_HOST" -p "${REDIS_PORT:-6379}" \
            ${REDIS_PASSWORD:+-a "$REDIS_PASSWORD"} \
            SET "$key" "$value" EX 86400 >/dev/null 2>&1 || true
    fi
}
