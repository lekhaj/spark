#!/bin/bash
# CPU EC2 Instance Worker Startup Script
# This script starts a Celery worker configured for CPU-intensive tasks

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "Starting CPU worker from: $SRC_DIR"

# Change to the source directory
cd "$SRC_DIR"

# Set environment variables for CPU instance
export REDIS_BROKER_URL="${REDIS_BROKER_URL:-redis://13.203.200.155:6379/0}"
export REDIS_RESULT_BACKEND="${REDIS_RESULT_BACKEND:-redis://13.203.200.155:6379/0}"
export USE_CELERY=True
export WORKER_TYPE=cpu

# CPU worker configuration
export CPU_WORKER_QUEUES="cpu_tasks,infrastructure"
export WORKER_HOSTNAME="cpu-worker@$(hostname)"

echo "Environment Configuration:"
echo "  REDIS_BROKER_URL: $REDIS_BROKER_URL"
echo "  REDIS_RESULT_BACKEND: $REDIS_RESULT_BACKEND"
echo "  WORKER_TYPE: $WORKER_TYPE"
echo "  WORKER_QUEUES: $CPU_WORKER_QUEUES"
echo "  WORKER_HOSTNAME: $WORKER_HOSTNAME"

# Check if Redis is accessible
echo "Testing Redis connection..."
python3 -c "
import redis
try:
    r = redis.Redis.from_url('$REDIS_BROKER_URL')
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    exit(1)
"

# Start CPU worker handling CPU tasks and infrastructure
echo "Starting Celery CPU worker..."
celery -A tasks worker \
    --loglevel=info \
    --queues="$CPU_WORKER_QUEUES" \
    --hostname="$WORKER_HOSTNAME" \
    --concurrency=4 \
    --pool=threads \
    --events \
    --without-gossip \
    --without-mingle \
    --without-heartbeat
