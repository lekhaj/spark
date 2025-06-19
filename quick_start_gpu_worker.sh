#!/bin/bash
# Quick setup and start GPU worker

echo "🚀 Quick GPU Worker Setup and Start"
echo "===================================="

# Navigate to project directory (script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
    exit 1
fi

# Test setup
echo "🧪 Testing setup..."
python3 test_hunyuan3d_setup.py

# Check if test passed
if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starting GPU worker..."
    cd src
    
    # Start Celery worker with proper configuration
    celery -A tasks worker \
        --loglevel=info \
        --queues=gpu_tasks \
        --hostname=gpu-worker@$(hostname) \
        --concurrency=1 \
        --pool=solo \
        --events \
        --without-gossip \
        --without-mingle \
        --without-heartbeat
else
    echo "❌ Setup test failed. Please check the errors above."
    exit 1
fi
