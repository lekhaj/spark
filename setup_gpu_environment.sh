#!/bin/bash
# setup_gpu_environment.sh - Complete GPU environment setup for Hunyuan3D-2.1
# This script runs all necessary setup scripts in sequence

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if script exists and is executable
check_script() {
    if [ -f "$1" ]; then
        chmod +x "$1"
        return 0
    else
        return 1
    fi
}
echo ""
echo "🔧 Step 2: Installing basic dependencies..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing ML frameworks..."
pip install transformers>=4.21.0
pip install diffusers>=0.20.0
pip install accelerate

echo "Installing processing libraries..."
pip install trimesh
pip install pillow
pip install numpy
pip install scipy

echo "Installing task queue libraries..."
pip install celery redis python-dotenv

echo "✅ Basic dependencies installed"

# Step 3: Clone and setup Hunyuan3D
echo ""
echo "🎯 Step 3: Setting up Hunyuan3D..."

HUNYUAN_DIR="$PROJECT_ROOT/Hunyuan3D-2"
if [ ! -d "$HUNYUAN_DIR" ]; then
    echo "Cloning Hunyuan3D repository..."
    cd "$PROJECT_ROOT"
    git clone https://github.com/Tencent/Hunyuan3D-1.git Hunyuan3D-2
    echo "✅ Hunyuan3D cloned"
else
    echo "✅ Hunyuan3D directory already exists"
fi

# Install Hunyuan3D requirements if they exist
if [ -f "$HUNYUAN_DIR/requirements.txt" ]; then
    echo "Installing Hunyuan3D requirements..."
    pip install -r "$HUNYUAN_DIR/requirements.txt"
    echo "✅ Hunyuan3D requirements installed"
fi

# Step 4: Setup environment configuration
echo ""
echo "⚙️  Step 4: Configuring environment..."

# Ensure .env.gpu exists with proper configuration
if [ ! -f "$PROJECT_ROOT/.env.gpu" ]; then
    echo "Creating .env.gpu configuration..."
    cat > "$PROJECT_ROOT/.env.gpu" << 'EOF'
# GPU Spot Instance Environment Configuration
USE_CELERY=True
WORKER_TYPE=gpu
GPU_SPOT_INSTANCE_IP=127.0.0.1

# Redis Configuration
REDIS_BROKER_URL=redis://127.0.0.1:6379/0
REDIS_RESULT_BACKEND=redis://127.0.0.1:6379/0
REDIS_WRITE_URL=redis://127.0.0.1:6379/0
REDIS_READ_URL=redis://127.0.0.1:6379/0

# GPU Configuration
HUNYUAN3D_DEVICE=cuda
HUNYUAN3D_LOW_VRAM_MODE=True
HUNYUAN3D_COMPILE=False

# Spot Instance Configuration
AWS_GPU_IS_SPOT_INSTANCE=True
SPOT_INSTANCE_HANDLING_ENABLED=True
AWS_REGION=ap-south-1

# Worker Configuration
GPU_WORKER_QUEUES=gpu_tasks

# 3D Generation Settings
HUNYUAN3D_STEPS=30
HUNYUAN3D_GUIDANCE_SCALE=7.5
HUNYUAN3D_OCTREE_RESOLUTION=256
HUNYUAN3D_NUM_CHUNKS=200000
HUNYUAN3D_REMOVE_BACKGROUND=True
HUNYUAN3D_ENABLE_FLASHVDM=True

# Application Settings
OUTPUT_DIR=./generated_assets
OUTPUT_IMAGES_DIR=./generated_assets/images
OUTPUT_3D_ASSETS_DIR=./generated_assets/3d_assets
EOF
    echo "✅ Environment configuration created"
else
    echo "✅ Environment configuration already exists"
fi

# Step 5: Start Redis if not running
echo ""
echo "🔴 Step 5: Starting Redis server..."
if ! systemctl is-active --quiet redis-server; then
    echo "Starting Redis server..."
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
    echo "✅ Redis server started"
else
    echo "✅ Redis server already running"
fi

# Step 6: Test Redis connection
echo ""
echo "🧪 Step 6: Testing Redis connection..."
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis connection successful"
else
    echo "❌ Redis connection failed"
    exit 1
fi

# Step 7: Create output directories
echo ""
echo "📁 Step 7: Creating output directories..."
mkdir -p "$PROJECT_ROOT/generated_assets"
mkdir -p "$PROJECT_ROOT/generated_assets/images" 
mkdir -p "$PROJECT_ROOT/generated_assets/3d_assets"
echo "✅ Output directories created"

# Step 8: Run setup test
echo ""
echo "🧪 Step 8: Running setup test..."
cd "$PROJECT_ROOT"
python3 test_hunyuan3d_setup.py

echo ""
echo "🎉 GPU environment setup completed!"
echo ""
echo "Next steps:"
echo "1. To start the GPU worker manually:"
echo "   cd $PROJECT_ROOT && ./scripts/start_gpu_worker.sh"
echo ""
echo "2. To start the GPU worker as a service:"
echo "   sudo systemctl start celery-gpu-worker"
echo ""
echo "3. To test the setup:"
echo "   python3 test_hunyuan3d_setup.py"
