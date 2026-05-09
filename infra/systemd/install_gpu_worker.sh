#!/usr/bin/env bash
# install_gpu_worker.sh — install + enable spark-gpu-worker on the GPU.
# Run ON the GPU instance (ec2-user@3.215.211.192) from a fresh git pull.
#
#   cd /home/ec2-user/spark && bash infra/systemd/install_gpu_worker.sh
#
# Idempotent — safe to re-run after changes to the unit file.
set -euo pipefail

UNIT=spark-gpu-worker.service
SRC="$(cd "$(dirname "$0")" && pwd)/${UNIT}"
DST="/etc/systemd/system/${UNIT}"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: unit file not found at $SRC" >&2
    exit 1
fi

# Ensure log file exists with correct ownership before the unit runs as ec2-user.
sudo touch /var/log/spark-gpu-worker.log
sudo chown ec2-user:ec2-user /var/log/spark-gpu-worker.log

sudo install -m 0644 "$SRC" "$DST"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT"

# Stop any pre-existing screen/manual run before we take over.
screen -S workers -X quit 2>/dev/null || true
rm -f /tmp/gpu_main.lock

sudo systemctl restart "$UNIT"
sleep 3
sudo systemctl --no-pager status "$UNIT" | head -20
