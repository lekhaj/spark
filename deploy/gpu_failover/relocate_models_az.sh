#!/usr/bin/env bash
# relocate_models_az.sh — move the models root volume to another AZ.
#
# EBS is AZ-locked; the only legal cross-AZ move is snapshot -> new volume in
# the target AZ. Use this when the current AZ can't supply a g6e (capacity),
# or as a deliberate migration. RARE operation.
#
# Usage:
#   relocate_models_az.sh \
#       --src-vol vol-xxxx --target-instance i-yyyy [--target-az us-east-1b] \
#       [--root-dev /dev/xvda] [--keep-old-root] [--yes]
#
# Flow (reversible until you delete the old volume yourself):
#   1. snapshot src-vol (transient, tagged)
#   2. wait completed
#   3. create new gp3 volume from snapshot in target AZ
#   4. stop target-instance, detach its current root (KEPT unless --delete-old-root)
#   5. attach new volume as root, set DeleteOnTermination=false
#   6. (caller) start + verify + move EIP + repoint CPU, THEN delete old vol/snap
#
# This script intentionally does NOT start the instance, move the EIP, or delete
# anything — those are gated, verify-first steps done by the operator/runbook.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ROOT_DEV="/dev/xvda"
SRC_VOL=""; TARGET=""; TARGET_AZ=""; ASSUME_YES=0; KEEP_OLD_ROOT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-vol) SRC_VOL="$2"; shift 2;;
    --target-instance) TARGET="$2"; shift 2;;
    --target-az) TARGET_AZ="$2"; shift 2;;
    --root-dev) ROOT_DEV="$2"; shift 2;;
    --keep-old-root) KEEP_OLD_ROOT=1; shift;;
    --yes) ASSUME_YES=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -n "$SRC_VOL" && -n "$TARGET" ]] || { echo "ERROR: --src-vol and --target-instance required" >&2; exit 2; }

aws() { command aws --region "$REGION" "$@"; }
log() { echo "[relocate] $*"; }

[[ -n "$TARGET_AZ" ]] || TARGET_AZ=$(aws ec2 describe-instances --instance-ids "$TARGET" \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)
SRC_SIZE=$(aws ec2 describe-volumes --volume-ids "$SRC_VOL" --query 'Volumes[0].Size' --output text)

log "src vol $SRC_VOL (${SRC_SIZE}GiB) -> new volume in $TARGET_AZ -> root of $TARGET ($ROOT_DEV)"
if [[ "$ASSUME_YES" != 1 ]]; then
  echo "Re-run with --yes to execute (non-destructive; keeps old root vol)."; exit 0
fi

# 1-2. snapshot + wait
log "snapshotting $SRC_VOL ..."
SNAP=$(aws ec2 create-snapshot --volume-id "$SRC_VOL" \
  --description "relocate $SRC_VOL -> $TARGET_AZ" \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=spark-models-relocate},{Key=Purpose,Value=az-relocation-transient}]' \
  --query SnapshotId --output text)
log "snapshot $SNAP — waiting for completion (can take a while)"
aws ec2 wait snapshot-completed --snapshot-ids "$SNAP"
log "snapshot completed"

# 3. new volume in target AZ
NEWVOL=$(aws ec2 create-volume --availability-zone "$TARGET_AZ" --snapshot-id "$SNAP" \
  --volume-type gp3 \
  --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=spark_models_${TARGET_AZ}},{Key=Project,Value=spark-gpu}]" \
  --query VolumeId --output text)
log "created $NEWVOL in $TARGET_AZ — waiting available"
aws ec2 wait volume-available --volume-ids "$NEWVOL"

# 4. stop target, detach old root
if [[ "$(aws ec2 describe-instances --instance-ids "$TARGET" --query 'Reservations[0].Instances[0].State.Name' --output text)" != "stopped" ]]; then
  log "stopping $TARGET"; aws ec2 stop-instances --instance-ids "$TARGET" >/dev/null
  aws ec2 wait instance-stopped --instance-ids "$TARGET"
fi
OLD_ROOT=$(aws ec2 describe-instances --instance-ids "$TARGET" \
  --query "Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName=='$ROOT_DEV'].Ebs.VolumeId | [0]" --output text)
if [[ -n "$OLD_ROOT" && "$OLD_ROOT" != "None" ]]; then
  log "detaching old root $OLD_ROOT from $TARGET (kept; delete later if desired)"
  aws ec2 detach-volume --volume-id "$OLD_ROOT" >/dev/null
  aws ec2 wait volume-available --volume-ids "$OLD_ROOT"
fi

# 5. attach new root + DOT=false
log "attaching $NEWVOL as $ROOT_DEV on $TARGET"
aws ec2 attach-volume --volume-id "$NEWVOL" --instance-id "$TARGET" --device "$ROOT_DEV" >/dev/null
aws ec2 wait volume-in-use --volume-ids "$NEWVOL"
aws ec2 modify-instance-attribute --instance-id "$TARGET" \
  --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"DeleteOnTermination\":false}}]"

cat <<EOF
[relocate] DONE (reversible).
  new root volume : $NEWVOL  ($TARGET_AZ, DOT=false)
  transient snap  : $SNAP    (delete after verify)
  old root (kept) : ${OLD_ROOT:-<none>}
Next (operator, verify-first):
  aws ec2 start-instances --instance-ids $TARGET
  aws ec2 associate-address --allocation-id <eip> --instance-id $TARGET
  ssh + confirm models/services, run a test job
  THEN delete old src vol ($SRC_VOL) and snapshot ($SNAP).
EOF
