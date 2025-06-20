#!/bin/bash
# Quick setup and start GPU worker for Hunyuan3D-2.1

echo "🚀 Quick GPU Worker Setup and Start (Hunyuan3D-2.1)"
echo "=================================================="

# Navigate to project directory (script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Hunyuan3D-2.1 directory
if [ ! -d "Hunyuan3D-2.1" ]; then
    echo "📥 Hunyuan3D-2.1 not found. Cloning repository..."
    git clone https://github.com/Tencent/Hunyuan3D-2.1.git
    if [ $? -ne 0 ]; then
        echo "❌ Failed to clone Hunyuan3D-2.1. Please check your internet connection."
        exit 1
    fi
    echo "✅ Hunyuan3D-2.1 cloned successfully"
fi

# Install dependencies if venv doesn't exist or is missing packages
if [ ! -d "venv" ] || [ ! -f "venv/bin/celery" ]; then
    echo "📦 Installing dependencies..."
    ./install_hunyuan3d_deps.sh
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Load environment variables
echo "📋 Loading environment variables..."
if [ -f ".env.gpu" ]; then
    set -a
    source .env.gpu
    set +a
    echo "✅ Environment variables loaded"
else
    echo "❌ .env.gpu file not found!"
    echo "   Creating basic .env.gpu file..."
    cat > .env.gpu << EOF
# GPU Instance Environment Configuration
USE_CELERY=True
WORKER_TYPE=gpu
GPU_SPOT_INSTANCE_IP=127.0.0.1

# Redis Configuration
REDIS_BROKER_URL=redis://127.0.0.1:6379/0
REDIS_RESULT_BACKEND=redis://127.0.0.1:6379/0

# Hunyuan3D-2.1 Configuration
HUNYUAN3D_MODEL_PATH=tencent/Hunyuan3D-2.1
HUNYUAN3D_SUBFOLDER=hunyuan3d-dit-v2-1
HUNYUAN3D_DEVICE=cuda
HUNYUAN3D_ENABLE_FLASHVDM=True
HUNYUAN3D_COMPILE=False
HUNYUAN3D_LOW_VRAM_MODE=True

# GPU Worker Configuration
GPU_WORKER_QUEUES=gpu_tasks
EOF
    echo "✅ Created basic .env.gpu file"
fi

# Start Redis if not running
echo "🔧 Checking Redis service..."
if ! pgrep -x "redis-server" > /dev/null; then
    echo "🚀 Starting Redis server..."
    sudo systemctl start redis-server || redis-server --daemonize yes
    sleep 2
    echo "✅ Redis server started"
else
    echo "✅ Redis server already running"
fi

# Test Redis connection
echo "🧪 Testing Redis connection..."
python3 -c "
import redis
try:
    r = redis.Redis.from_url('redis://127.0.0.1:6379/0')
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    exit(1)
"

# Test Hunyuan3D-2.1 setup
echo "🧪 Testing Hunyuan3D-2.1 setup..."
python3 test_hunyuan3d_setup.py

# Check if test passed
if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starting GPU worker..."
    cd src
    
    # Kill any existing Celery workers
    pkill -f "celery.*worker" || true
    sleep 2
    
    # Start Celery worker with proper configuration for 2.1
    exec celery -A tasks worker \
        --loglevel=info \
        --queues=gpu_tasks \
        --hostname=gpu-worker-2.1@$(hostname) \
        --concurrency=1 \
        --pool=solo \
        --events \
        --without-gossip \
        --without-mingle \
        --without-heartbeat \
        --max-tasks-per-child=1
else
    echo "❌ Setup test failed. Please check the errors above."
    echo ""
    echo "Common solutions:"
    echo "  1. Install CUDA drivers: sudo apt install nvidia-driver-535"
    echo "  2. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo "  3. Check GPU memory: nvidia-smi"
    echo "  4. Restart the script: ./quick_start_gpu_worker.sh"
    exit 1
fi
