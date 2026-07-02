#!/bin/bash
# apply_units.sh — install the current systemd units onto a g7e GPU box and
# reload. Run ON the box (ec2-user) after `git pull`, or via ssh. Idempotent.
#
#   ssh -i us_cpu_key.pem ec2-user@<gpu-ip> \
#     'cd /home/ec2-user/spark && git pull && bash deploy/gpu/apply_units.sh'
#
# Fixes shipped here: worker + prewarm no longer gate on network-online.target
# (AL2023 stop→start boot hang) and the worker uses Restart=always.
set -euo pipefail
SRC=/home/ec2-user/spark
sudo cp "$SRC/deploy/systemd/manual_gen_worker.service" /etc/systemd/system/manual_gen_worker.service
sudo cp "$SRC/worker/gpu_setup/spark-prewarm.service"    /etc/systemd/system/spark-prewarm.service
sudo systemctl daemon-reload
sudo systemctl enable spark-prewarm.service manual_gen_worker.service
echo "units installed + enabled. Restarting worker…"
sudo systemctl restart manual_gen_worker.service
sleep 3
systemctl is-active manual_gen_worker.service
echo "Tip: verify boot behavior with a reboot test: sudo reboot, then re-ssh and check is-active."
