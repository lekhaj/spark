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

# Wait (bounded) for Redis on the CPU's VPC IP before starting — on a stop→start
# boot the network may lag. If it never comes up in time we still exec; systemd
# Restart=always will relaunch. Avoids noisy crash-loops at boot.
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
for i in $(seq 1 30); do
  if timeout 2 bash -c "</dev/tcp/${REDIS_HOST}/${REDIS_PORT}" 2>/dev/null; then
    echo "start_manual_worker: Redis ${REDIS_HOST}:${REDIS_PORT} reachable (attempt $i)"
    break
  fi
  echo "start_manual_worker: waiting for Redis ${REDIS_HOST}:${REDIS_PORT} (attempt $i)…"
  sleep 4
done

cd "$SPARK_ROOT"
exec /home/ec2-user/miniconda3/envs/spark/bin/python worker/run_manual_worker.py
