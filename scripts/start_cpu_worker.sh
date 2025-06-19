#!/bin/bash
# CPU EC2 Instance Worker Startup Script
# This script starts a Celery worker configured for CPU-intensive tasks

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "Starting CPU worker from: $SRC_DIR"

# Activate conda environment
echo "Activating conda environment txt23d..."
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate txt23d

# Change to the source directory
cd "$SRC_DIR"

# Load environment from .env.cpu if it exists
ENV_FILE="$PROJECT_ROOT/.env.cpu"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
fi

# Set environment variables for CPU instance (with fallbacks)
export REDIS_BROKER_URL="${REDIS_BROKER_URL:-redis://35.154.102.169:6379/0}"
export REDIS_RESULT_BACKEND="${REDIS_RESULT_BACKEND:-redis://35.154.102.169:6379/0}"
export REDIS_WRITE_URL="${REDIS_WRITE_URL:-$REDIS_BROKER_URL}"
export REDIS_READ_URL="${REDIS_READ_URL:-$REDIS_BROKER_URL}"
export USE_CELERY=True
export WORKER_TYPE=cpu
export GPU_SPOT_INSTANCE_IP="${GPU_SPOT_INSTANCE_IP:-35.154.102.169}"

# CPU worker configuration
export CPU_WORKER_QUEUES="cpu_tasks,infrastructure"
export WORKER_HOSTNAME="cpu-worker@$(hostname)"

echo "Environment Configuration:"
echo "  REDIS_BROKER_URL: $REDIS_BROKER_URL"
echo "  REDIS_RESULT_BACKEND: $REDIS_RESULT_BACKEND"
echo "  REDIS_WRITE_URL: $REDIS_WRITE_URL"
echo "  REDIS_READ_URL: $REDIS_READ_URL"
echo "  WORKER_TYPE: $WORKER_TYPE"
echo "  GPU_SPOT_INSTANCE_IP: $GPU_SPOT_INSTANCE_IP"
echo "  WORKER_QUEUES: $CPU_WORKER_QUEUES"
echo "  WORKER_HOSTNAME: $WORKER_HOSTNAME"

# Check if Redis is accessible and test read/write configuration
echo "Testing Redis configuration..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from config import REDIS_CONFIG
    
    print(f'Worker Type: {REDIS_CONFIG.worker_type}')
    print(f'Write URL: {REDIS_CONFIG.write_url}')
    print(f'Read URL: {REDIS_CONFIG.read_url}')
    
    # Test connections
    results = REDIS_CONFIG.test_connection()
    
    write_ok = results.get('write', {}).get('success', False)
    read_ok = results.get('read', {}).get('success', False)
    
    if write_ok and read_ok:
        print('✅ All Redis connections successful')
    else:
        if not write_ok:
            print(f'❌ Write connection failed: {results.get(\"write\", {}).get(\"error\")}')
        if not read_ok:
            print(f'❌ Read connection failed: {results.get(\"read\", {}).get(\"error\")}')
        exit(1)
        
except Exception as e:
    print(f'❌ Redis configuration test failed: {e}')
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
