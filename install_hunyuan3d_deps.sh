#!/bin/bash
# install_hunyuan3d_deps.sh - Install Hunyuan3D dependencies for GPU worker

set -e

echo "🚀 Installing Hunyuan3D dependencies for GPU worker..."

# Make sure we're in the project directory
cd /home/ubuntu/Shashwat/spark

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📈 Upgrading pip..."
pip install --upgrade pip

echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "🤖 Installing Transformers and Diffusers..."
pip install transformers>=4.21.0
pip install diffusers>=0.20.0
pip install accelerate

echo "📐 Installing 3D processing libraries..."
pip install trimesh
pip install open3d-python

echo "🖼️ Installing image processing libraries..."
pip install pillow
pip install opencv-python

echo "🔢 Installing numerical libraries..."
pip install numpy
pip install scipy

echo "⚡ Installing performance libraries..."
pip install xformers --index-url https://download.pytorch.org/whl/cu118

echo "📋 Installing other dependencies..."
pip install celery redis python-dotenv

echo "🎯 Installing from requirements.txt if it exists..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "✅ Dependencies installation completed!"
echo ""
echo "🧪 Running test script..."
python3 test_hunyuan3d_setup.py

echo ""
echo "🏁 Setup completed! You can now start the GPU worker with:"
echo "source venv/bin/activate && cd src && celery -A tasks worker --loglevel=info --queues=gpu_tasks --hostname=gpu-worker@\$(hostname) --concurrency=1 --pool=solo"
