#!/bin/bash
# GPU Spot Instance Worker Startup Script
# This script starts Redis server and Celery worker configured for GPU tasks

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "Starting GPU spot instance worker from: $SRC_DIR"

# Change to the source directory
cd "$SRC_DIR"

# Set environment variables for GPU spot instance
export REDIS_BROKER_URL="${REDIS_BROKER_URL:-redis://127.0.0.1:6379/0}"
export REDIS_RESULT_BACKEND="${REDIS_RESULT_BACKEND:-redis://127.0.0.1:6379/0}"
export USE_CELERY=True
export WORKER_TYPE=gpu
export HUNYUAN3D_DEVICE=cuda

# GPU worker configuration
export GPU_WORKER_QUEUES="gpu_tasks"
export WORKER_HOSTNAME="gpu-worker@$(hostname)"

# Spot instance specific settings
export AWS_GPU_IS_SPOT_INSTANCE=True
export SPOT_INSTANCE_HANDLING_ENABLED=True

echo "Environment Configuration:"
echo "  REDIS_BROKER_URL: $REDIS_BROKER_URL"
echo "  REDIS_RESULT_BACKEND: $REDIS_RESULT_BACKEND"
echo "  WORKER_TYPE: $WORKER_TYPE"
echo "  WORKER_QUEUES: $GPU_WORKER_QUEUES"
echo "  WORKER_HOSTNAME: $WORKER_HOSTNAME"
echo "  HUNYUAN3D_DEVICE: $HUNYUAN3D_DEVICE"
echo "  AWS_GPU_IS_SPOT_INSTANCE: $AWS_GPU_IS_SPOT_INSTANCE"

# Check if Redis is installed, install if not
if ! command -v redis-server &> /dev/null; then
    echo "Redis not found. Installing Redis..."
    sudo apt update
    sudo apt install -y redis-server
fi

# Start Redis server (since this instance acts as broker)
echo "Starting Redis server..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
sleep 5

# Test Redis connection
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

# Check CUDA availability for GPU tasks
echo "Checking CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available - GPU tasks may fail')
"

# Start GPU worker handling only GPU tasks
echo "Starting Celery GPU worker..."
celery -A tasks worker \
    --loglevel=info \
    --queues="$GPU_WORKER_QUEUES" \
    --hostname="$WORKER_HOSTNAME" \
    --concurrency=1 \
    --pool=solo \
    --events \
    --without-gossip \
    --without-mingle \
    --without-heartbeat
