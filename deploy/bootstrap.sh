#!/usr/bin/env bash
# =============================================================================
# deploy/bootstrap.sh — fresh-boot installer for the L40S spot.
#
# Runs as a oneshot systemd unit before manual_gen_worker.service. Idempotent:
# completes in ~5s when nothing has changed; pulls + installs the delta when
# spec.yaml has been updated.
#
# Hard rule: the only writes outside ~/.spark, HF cache, and /etc/systemd/system
# are inside the git-managed working tree, which gets reset to origin/main at
# the start of every run. No drift, ever.
#
# Manual usage:
#   sudo systemctl restart spark-bootstrap.service
#   journalctl -u spark-bootstrap -f
# =============================================================================

set -euo pipefail

# --- Hardcoded role (single-GPU deployment — L40S spot only) ---
export STAGE_NAME="bootstrap"
export SPARK_ROOT="${SPARK_ROOT:-/home/ec2-user/spark}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

_imds_token="$(curl -sS -m 2 -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || echo)"
INSTANCE_ID="$(curl -sS -m 2 -H "X-aws-ec2-metadata-token: $_imds_token" 'http://169.254.169.254/latest/meta-data/instance-id' 2>/dev/null || echo unknown)"
[[ -n "$INSTANCE_ID" ]] || INSTANCE_ID="unknown"
STATUS_KEY="bootstrap:status:${INSTANCE_ID}"
COMMIT_KEY="bootstrap:commit:${INSTANCE_ID}"

trap 'status_publish "$STATUS_KEY" "failed:line_${LINENO}"; log "bootstrap aborted"; exit 1' ERR

status_publish "$STATUS_KEY" "starting"
log "===== bootstrap starting (instance=$INSTANCE_ID) ====="

# --- Step 1: git update (hard rule — repo is source of truth) ---
if [[ -f "$SPARK_STATE_DIR/pin_branch" ]]; then
    log "pin_branch marker exists → skipping git update"
else
    log "git fetch + hard reset to origin/main"
    cd "$SPARK_ROOT"
    git fetch --quiet origin main
    git reset --hard --quiet origin/main
    log "now at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
fi
status_publish "$COMMIT_KEY" "$(git -C "$SPARK_ROOT" rev-parse HEAD)"

# --- Step 2: parse spec.yaml ---
SPEC="$SPARK_ROOT/deploy/spec.yaml"
[[ -f "$SPEC" ]] || die "spec.yaml missing at $SPEC"

# Use python (always present in spark conda env) to read YAML — avoids needing yq.
PYTHON_BIN="$CONDA_BASE/envs/spark/bin/python"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

read_spec() {
    "$PYTHON_BIN" - "$SPEC" <<'PY'
import sys, yaml, json
spec = yaml.safe_load(open(sys.argv[1]))
print(json.dumps(spec))
PY
}

SPEC_JSON="$(read_spec)"

get_stages() {
    echo "$SPEC_JSON" | "$PYTHON_BIN" -c "import sys,json; print(' '.join(json.load(sys.stdin)['stages']))"
}

get_stage_def() {
    local stage="$1" key="$2"
    echo "$SPEC_JSON" | "$PYTHON_BIN" -c "
import sys, json
d = json.load(sys.stdin)['stage_defs'].get('$stage', {})
v = d.get('$key', '')
if isinstance(v, list): print('\n'.join(str(x) for x in v))
elif isinstance(v, dict): print('\n'.join(f'{k}={v}' for k,v in v.items()))
else: print(v)
"
}

# --- Step 3: run each stage installer ---
STAGES="$(get_stages)"
log "stages to process: $STAGES"

for stage in $STAGES; do
    version="$(get_stage_def "$stage" version)"
    : "${version:=1}"

    if sentinel_ok "$stage" "$version"; then
        log "✓ $stage v$version already installed — skipping"
        continue
    fi

    log "→ installing $stage v$version"
    status_publish "$STATUS_KEY" "installing:$stage"

    installer="$SPARK_ROOT/deploy/install/${stage}.sh"
    [[ -x "$installer" ]] || die "no installer at $installer"

    STAGE_NAME="$stage" \
    SPEC_JSON="$SPEC_JSON" \
    bash "$installer"

    sentinel_write "$stage" "$version"
    log "✓ $stage v$version installed"
done

# --- Step 4: reconcile systemd units ---
log "reconciling systemd unit files"
UNITS_REPO="$SPARK_ROOT/deploy/systemd"
UNITS_SYS="/etc/systemd/system"
CHANGED=0

for unit in "$UNITS_REPO"/*.service; do
    [[ -e "$unit" ]] || continue
    name="$(basename "$unit")"
    target="$UNITS_SYS/$name"
    if ! sudo -n cmp -s "$unit" "$target" 2>/dev/null; then
        sudo -n install -m 644 "$unit" "$target"
        log "  installed/updated: $name"
        CHANGED=1
    fi
done
# Drop-ins (per-service overrides)
if [[ -d "$UNITS_REPO/dropins" ]]; then
    for dropin_dir in "$UNITS_REPO/dropins"/*/; do
        # Directories under dropins/ are named exactly like the systemd target,
        # e.g. dropins/manual_gen_worker.service.d/ → /etc/systemd/system/manual_gen_worker.service.d/
        svc="$(basename "$dropin_dir")"
        target_dir="$UNITS_SYS/${svc}"
        sudo -n mkdir -p "$target_dir"
        for f in "$dropin_dir"*.conf; do
            [[ -e "$f" ]] || continue
            if ! sudo -n cmp -s "$f" "$target_dir/$(basename "$f")" 2>/dev/null; then
                sudo -n install -m 644 "$f" "$target_dir/"
                log "  installed/updated dropin: ${svc}.d/$(basename "$f")"
                CHANGED=1
            fi
        done
    done
fi
if [[ $CHANGED -eq 1 ]]; then
    sudo -n systemctl daemon-reload
    log "systemctl daemon-reload done"
fi

# --- Step 5: ready ---
status_publish "$STATUS_KEY" "ready"
log "===== bootstrap complete ====="
exit 0
