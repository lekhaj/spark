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
# Uses `huggingface-cli download` which skips files already in HF_HOME cache.
hf_pull() {
    local model_id="$1"
    log "HF pull: $model_id"
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
        huggingface-cli download "$model_id" \
        --quiet \
        2>&1 | tail -5 | while read -r line; do log "  hf: $line"; done || \
        die "HF download failed: $model_id"
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
