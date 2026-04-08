#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# start_workers.sh — Start all GPU workers on the Spark L4 instance
# ════════════════════════════════════════════════════════════════════════════
# Usage:
#   bash worker/gpu_setup/start_workers.sh            # all workers
#   bash worker/gpu_setup/start_workers.sh sd15       # image only
#   bash worker/gpu_setup/start_workers.sh sd15,trellis
# ════════════════════════════════════════════════════════════════════════════
WORKERS=${1:-"sd15,trellis,rig"}

SPARK_DIR=/home/ec2-user/spark
TRELLIS_DIR=/home/ec2-user/trellis
PYTHON=/home/ec2-user/miniconda3/envs/spark/bin/python3
LOG=/tmp/gpu_workers.log

# Kill existing session
screen -S workers -X quit 2>/dev/null && echo "Stopped old workers session" || true
sleep 1

# Remove stale lock
rm -f /tmp/gpu_main.lock

echo "Starting workers: $WORKERS"
echo "Log: $LOG"

screen -dmS workers bash -c "
  cd $SPARK_DIR
  export PYTHONPATH=$TRELLIS_DIR:\${PYTHONPATH:-}
  set -a; source $SPARK_DIR/.env; set +a
  $PYTHON worker/gpu_main.py --workers $WORKERS 2>&1 | tee $LOG
"

sleep 3
screen -list
echo ""
echo "Workers started. Monitor with:"
echo "  tail -f $LOG"
echo "  screen -r workers"
