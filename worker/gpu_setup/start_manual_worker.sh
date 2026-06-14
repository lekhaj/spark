#!/bin/bash
# start_manual_worker.sh — systemd ExecStart for the g7e manual-gen worker.
# Sources the tracked .env.gpu + gitignored .env.secrets, puts the TRELLIS.2
# repo on PYTHONPATH, then execs the worker in the `spark` conda env. Kept as a
# wrapper (not EnvironmentFile=) because the .env files use bash quoting that
# systemd's EnvironmentFile parser does not handle identically.
set -euo pipefail

SPARK_ROOT=/home/ec2-user/spark
set -a
# shellcheck disable=SC1091
source "$SPARK_ROOT/.env.gpu"
source "$SPARK_ROOT/.env.secrets"
set +a
export PYTHONPATH=/home/ec2-user/trellis

cd "$SPARK_ROOT"
exec /home/ec2-user/miniconda3/envs/spark/bin/python worker/run_manual_worker.py
