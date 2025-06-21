#!/bin/bash
# Quick setup and start GPU worker for Hunyuan3D-2.1 + SDXL Turbo
# This script:
# 1. Sets up the environment and dependencies
# 2. Tests both Hunyuan3D-2.1 and SDXL Turbo installations
# 3. Starts a Celery worker that handles both 3D generation and image generation tasks
# 4. Optimized for 15-20GB VRAM with memory management

echo "🚀 Quick GPU Worker Setup and Start (Hunyuan3D-2.1 + SDXL Turbo)"
echo "================================================================="
echo "📋 This script will test and launch both:"
echo "   • Hunyuan3D-2.1 for 3D model generation" 
echo "   • SDXL Turbo for fast image generation"
echo ""

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

# SDXL Turbo Configuration
SDXL_MODEL_PATH=stabilityai/sdxl-turbo
SDXL_DEVICE=cuda
SDXL_ENABLE_CPU_OFFLOAD=True
SDXL_ENABLE_ATTENTION_SLICING=True
SDXL_ENABLE_SEQUENTIAL_CPU_OFFLOAD=True
SDXL_MEMORY_EFFICIENT=True
DEFAULT_TEXT_MODEL=sdxl-turbo
DEFAULT_GRID_MODEL=sdxl-turbo

# GPU Worker Configuration
GPU_WORKER_QUEUES=gpu_tasks,sdxl_tasks,image_generation
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

# Test SDXL Turbo setup
echo "🧪 Testing SDXL Turbo setup..."
python3 test_sdxl_turbo_setup.py

if [ $? -ne 0 ]; then
    echo "❌ SDXL Turbo setup test failed. Please check the errors above."
    exit 1
fi

# Additional test: Check if src modules can be imported
echo "🧪 Testing src module imports..."
python3 -c "
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
try:
    import s3_manager
    print('✅ s3_manager imported successfully')
except Exception as e:
    print(f'❌ s3_manager import failed: {e}')
try:
    import db_helper
    print('✅ db_helper imported successfully')
except Exception as e:
    print(f'❌ db_helper import failed: {e}')
try:
    import config
    print('✅ config imported successfully')
except Exception as e:
    print(f'❌ config import failed: {e}')
try:
    import sdxl_turbo_worker
    print('✅ sdxl_turbo_worker imported successfully')
except Exception as e:
    print(f'❌ sdxl_turbo_worker import failed: {e}')
try:
    from tasks import generate_image_sdxl_turbo, batch_generate_images_sdxl_turbo
    print('✅ SDXL Turbo tasks imported successfully')
except Exception as e:
    print(f'❌ SDXL Turbo tasks import failed: {e}')
"

# Check if test passed
if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starting GPU worker - loading Hunyuan3D-2.1 and SDXL Turbo models to GPU VRAM..."
    
    # Set GPU optimization environment variables for direct GPU loading
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,expandable_segments:True
    export HUNYUAN3D_LOW_VRAM_MODE=False
    export CUDA_LAUNCH_BLOCKING=0
    
    # SDXL Turbo optimization settings
    export SDXL_DEVICE=cuda
    export SDXL_ENABLE_CPU_OFFLOAD=True
    export SDXL_ENABLE_ATTENTION_SLICING=True
    export SDXL_ENABLE_SEQUENTIAL_CPU_OFFLOAD=True
    export DEFAULT_TEXT_MODEL=sdxl-turbo
    export DEFAULT_GRID_MODEL=sdxl-turbo
    
    # Set up Python path for module imports (include both src and project root)
    export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR:$PYTHONPATH"
    
    # Clear GPU memory cache before starting
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('🧹 GPU cache cleared')"
    
    # Show GPU memory status
    echo "📊 GPU memory status before starting worker:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    
    cd src
    
    # Kill any existing Celery workers
    pkill -f "celery.*worker" || true
    sleep 2
    
    echo "🎮 Loading both Hunyuan3D-2.1 and SDXL Turbo models to GPU VRAM..."
    echo "📍 Python path: $PYTHONPATH"
    echo "📍 Working directory: $(pwd)"
    echo "🎯 Worker will handle both 3D generation and image generation tasks"
    
    # Final test of imports before starting Celery worker
    echo "🔍 Final import check..."
    python3 -c "
try:
    import s3_manager, db_helper, config, sdxl_turbo_worker
    from tasks import generate_image_sdxl_turbo
    print('✅ All required modules imported successfully')
    print('✅ SDXL Turbo worker and tasks ready')
except Exception as e:
    print(f'❌ Import error: {e}')
    import sys
    print(f'Python path: {sys.path}')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Module import check failed. Cannot start worker."
        exit 1
    fi
    
    # Start Celery worker with GPU-focused configuration for both Hunyuan3D and SDXL Turbo
    exec celery -A tasks worker \
        --loglevel=info \
        --queues=gpu_tasks,sdxl_tasks,image_generation \
        --hostname=gpu-worker-hunyuan3d-sdxl@$(hostname) \
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
    echo "  3. Install Diffusers: pip install diffusers transformers accelerate"
    echo "  4. Check GPU memory: nvidia-smi"
    echo "  5. Ensure both Hunyuan3D-2.1 and SDXL Turbo dependencies are installed"
    echo "  6. Restart the script: ./quick_start_gpu_worker.sh"
    exit 1
fi
