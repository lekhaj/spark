#!/bin/bash
# GPU Spot Instance Worker Startup Script
# This script starts Redis server and Celery worker configured for GPU tasks

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "Starting GPU spot instance worker from: $SRC_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "❌ Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please run the deployment script first: ./deploy_to_gpu.sh"
    exit 1
fi

# Change to the source directory
cd "$SRC_DIR"

# Load environment from .env.gpu if it exists
ENV_FILE="$PROJECT_ROOT/.env.gpu"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)
fi

# Set environment variables for GPU spot instance (with fallbacks)
export REDIS_BROKER_URL="${REDIS_BROKER_URL:-redis://127.0.0.1:6379/0}"
export REDIS_RESULT_BACKEND="${REDIS_RESULT_BACKEND:-redis://127.0.0.1:6379/0}"
export REDIS_WRITE_URL="${REDIS_WRITE_URL:-$REDIS_BROKER_URL}"
export REDIS_READ_URL="${REDIS_READ_URL:-$REDIS_BROKER_URL}"
export USE_CELERY=True
export WORKER_TYPE=gpu
export GPU_SPOT_INSTANCE_IP="${GPU_SPOT_INSTANCE_IP:-127.0.0.1}"
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
echo "  REDIS_WRITE_URL: $REDIS_WRITE_URL"
echo "  REDIS_READ_URL: $REDIS_READ_URL"
echo "  WORKER_TYPE: $WORKER_TYPE"
echo "  GPU_SPOT_INSTANCE_IP: $GPU_SPOT_INSTANCE_IP"
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

# Test Redis configuration
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
