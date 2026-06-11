#!/usr/bin/env bash
# switch_gpu.sh — move the single roaming models root volume between two
# SAME-AZ g6e instances, carry the Elastic IP along, and roll back if the
# target won't start. Use for opportunistic spot<->on-demand in the live AZ.
#
# PRECONDITION: both instances are in the SAME AZ as the roaming volume.
# (Cross-AZ requires relocate_models_az.sh first — EBS is AZ-locked.)
#
# Usage:
#   switch_gpu.sh --to <target-instance-id> [--yes]
#
# It discovers which instance currently holds the roaming volume (the source),
# stops it, detaches the root, attaches to the target, sets DOT=false, starts
# the target, and re-associates the EIP. If the target fails to reach running,
# it ROLLS BACK to the source.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ROAMING_VOL="${ROAMING_VOL:-vol-0c7bb27c340f01b05}"
EIP_ALLOC="${EIP_ALLOC:-eipalloc-0db12aa4d8be94e92}"
ROOT_DEV="${ROOT_DEV:-/dev/xvda}"
START_TIMEOUT="${START_TIMEOUT:-300}"

TARGET=""; ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --to) TARGET="$2"; shift 2;;
    --yes) ASSUME_YES=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -n "$TARGET" ]] || { echo "ERROR: --to <instance-id> required" >&2; exit 2; }

aws() { command aws --region "$REGION" "$@"; }
log() { echo "[switch_gpu] $*"; }

vol_state() { aws ec2 describe-volumes --volume-ids "$ROAMING_VOL" \
  --query 'Volumes[0].{State:State,Attached:Attachments[0].InstanceId,AZ:AvailabilityZone}' --output json; }
inst_state() { aws ec2 describe-instances --instance-ids "$1" \
  --query 'Reservations[0].Instances[0].State.Name' --output text; }
inst_az() { aws ec2 describe-instances --instance-ids "$1" \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text; }

wait_state() { # instance target-state timeout
  local id="$1" want="$2" t="$3" s
  for ((i=0;i<t;i+=5)); do s=$(inst_state "$id"); echo "  $id => $s"; [[ "$s" == "$want" ]] && return 0; sleep 5; done
  return 1
}

set_dot_false() { # instance
  aws ec2 modify-instance-attribute --instance-id "$1" \
    --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"DeleteOnTermination\":false}}]"
}

VOLAZ=$(vol_state | python3 -c 'import sys,json;print(json.load(sys.stdin)["AZ"])')
SRC=$(vol_state | python3 -c 'import sys,json;print(json.load(sys.stdin).get("Attached") or "")')
TGTAZ=$(inst_az "$TARGET")

log "roaming volume $ROAMING_VOL in $VOLAZ, currently on: ${SRC:-<detached>}"
log "target $TARGET in $TGTAZ"
[[ "$VOLAZ" == "$TGTAZ" ]] || { echo "ERROR: target AZ $TGTAZ != volume AZ $VOLAZ. Run relocate_models_az.sh first." >&2; exit 1; }
[[ "$SRC" == "$TARGET" ]] && { log "volume already on target. Ensuring DOT=false + EIP."; set_dot_false "$TARGET" || true; aws ec2 associate-address --allocation-id "$EIP_ALLOC" --instance-id "$TARGET" >/dev/null; exit 0; }

if [[ "$ASSUME_YES" != 1 ]]; then
  echo; echo "PLAN: stop ${SRC:-<none>} -> detach $ROAMING_VOL -> attach to $TARGET ($ROOT_DEV) -> start $TARGET -> move EIP $EIP_ALLOC"; echo "Re-run with --yes to execute."; exit 0
fi

# ── stop source & detach ──────────────────────────────────────────────────
if [[ -n "$SRC" ]]; then
  log "stopping source $SRC"; aws ec2 stop-instances --instance-ids "$SRC" >/dev/null
  wait_state "$SRC" stopped 180 || { echo "source did not stop" >&2; exit 1; }
  log "detaching $ROAMING_VOL from $SRC"; aws ec2 detach-volume --volume-id "$ROAMING_VOL" >/dev/null
  aws ec2 wait volume-available --volume-ids "$ROAMING_VOL"
fi

# target must be stopped and have its root detached (we expect a bare target)
if [[ "$(inst_state "$TARGET")" != "stopped" ]]; then
  log "stopping target $TARGET"; aws ec2 stop-instances --instance-ids "$TARGET" >/dev/null
  wait_state "$TARGET" stopped 180 || { echo "target did not stop" >&2; exit 1; }
fi

attach_and_start() { # instance ; returns 0 if running
  local id="$1"
  log "attaching $ROAMING_VOL -> $id $ROOT_DEV"
  aws ec2 attach-volume --volume-id "$ROAMING_VOL" --instance-id "$id" --device "$ROOT_DEV" >/dev/null
  aws ec2 wait volume-in-use --volume-ids "$ROAMING_VOL"
  set_dot_false "$id" || true
  log "starting $id"
  if ! aws ec2 start-instances --instance-ids "$id" >/dev/null 2>/tmp/switch_start_err; then
    log "start-instances failed: $(cat /tmp/switch_start_err)"; return 1
  fi
  wait_state "$id" running "$START_TIMEOUT" || return 1
}

if attach_and_start "$TARGET"; then
  log "associating EIP $EIP_ALLOC -> $TARGET"
  aws ec2 associate-address --allocation-id "$EIP_ALLOC" --instance-id "$TARGET" >/dev/null
  log "DONE. Active GPU is now $TARGET. Update CPU AWS_GPU_INSTANCE_ID + restart fastapi."
else
  log "TARGET FAILED TO START — rolling back"
  aws ec2 detach-volume --volume-id "$ROAMING_VOL" >/dev/null || true
  aws ec2 wait volume-available --volume-ids "$ROAMING_VOL" || true
  if [[ -n "$SRC" ]]; then
    attach_and_start "$SRC" && aws ec2 associate-address --allocation-id "$EIP_ALLOC" --instance-id "$SRC" >/dev/null
    log "rolled back to $SRC."
  fi
  exit 1
fi
