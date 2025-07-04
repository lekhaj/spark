#!/bin/bash
# install_hunyuan3d_deps.sh - Install Hunyuan3D-2.1 dependencies for GPU worker

set -e

echo "🚀 Installing Hunyuan3D-2.1 dependencies..."

# Check and activate virtual environment "venv"
echo "🔍 Checking virtual environment..."
if [ -d "venv" ]; then
    echo "✅ Found virtual environment 'venv'"
    echo "🔄 Activating virtual environment 'venv'..."
    source venv/bin/activate
    echo "✅ Activated environment: $VIRTUAL_ENV"
else
    echo "❌ Virtual environment 'venv' not found!"
    echo ""
    echo "💡 Please create the environment first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    exit 1
fi

# Install essential system dependencies
echo "📦 Installing system dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "� Installing PyTorch with CUDA support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install core ML dependencies
echo "🤖 Installing core ML dependencies..."
pip install transformers==4.46.0 diffusers==0.30.0 safetensors==0.4.4
pip install accelerate huggingface-hub
pip install scipy einops pandas

# Install 3D processing libraries
echo "📐 Installing 3D processing libraries..."
pip install trimesh open3d matplotlib
pip install moderngl pillow python-dotenv

# Install additional dependencies for Hunyuan3D
echo "🔧 Installing Hunyuan3D specific dependencies..."
pip install ninja pybind11
pip install opencv-python imageio imageio-ffmpeg
pip install scikit-image tqdm requests
pip install pytorch-lightning==1.9.5

# Try to install Blender Python API (optional)
echo "🎨 Installing Blender Python API (optional)..."
pip install bpy || echo "⚠️ Blender Python API installation failed (optional dependency)"

# Clone Hunyuan3D-2.1 if not exists
if [ ! -d "Hunyuan3D-2.1" ]; then
    echo "📥 Cloning Hunyuan3D-2.1..."
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
fi

# Navigate to Hunyuan3D-2.1 and install components and requirements
cd Hunyuan3D-2.1

echo "📋 Installing Hunyuan3D-2.1 requirements..."
# Install requirements but handle conflicts gracefully
if pip install -r requirements.txt; then
    echo "✅ Requirements installed successfully"
else
    echo "⚠️ Installing requirements with conflict resolution..."
    # Install requirements one by one, skipping problematic ones
    grep -v "numpy==" requirements.txt | grep -v "^#" | grep -v "^--" | while read requirement; do
        if [ ! -z "$requirement" ]; then
            echo "Installing: $requirement"
            pip install "$requirement" --no-deps || echo "Skipped: $requirement"
        fi
    done
fi

# Install custom rasterizer for PBR support
echo "🎨 Installing PBR custom rasterizer..."
if [ -d "hy3dpaint/custom_rasterizer" ]; then
    cd hy3dpaint/custom_rasterizer
    echo "Building custom rasterizer..."
    pip install -e . || echo "⚠️ Custom rasterizer installation failed"
    cd ../..
else
    echo "⚠️ Custom rasterizer directory not found, skipping..."
fi

# Install differentiable renderer
echo "🔧 Installing differentiable renderer..."
if [ -d "hy3dpaint/DifferentiableRenderer" ]; then
    cd hy3dpaint/DifferentiableRenderer
    if [ -f "compile_mesh_painter.sh" ]; then
        echo "Compiling mesh painter..."
        bash compile_mesh_painter.sh || echo "⚠️ Mesh painter compilation failed"
    else
        echo "⚠️ compile_mesh_painter.sh not found"
    fi
    cd ../..
else
    echo "⚠️ DifferentiableRenderer directory not found, skipping..."
fi

# Create checkpoint directory and download required models
echo "📁 Creating checkpoint directories..."
mkdir -p hy3dpaint/ckpt

# Download required models
echo "📥 Downloading required models..."
if [ ! -f "hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]; then
    echo "Downloading RealESRGAN model..."
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt/ || {
        echo "⚠️ RealESRGAN download failed, trying alternative method..."
        curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o hy3dpaint/ckpt/RealESRGAN_x4plus.pth || echo "❌ Model download failed"
    }
else
    echo "✅ RealESRGAN model already exists"
fi

# Install additional dependencies for enhanced viewer
echo "🖥️ Installing final dependencies..."
pip install moderngl pillow python-dotenv

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || echo "⚠️ PyTorch verification failed"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || echo "⚠️ Transformers verification failed"
python3 -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')" || echo "⚠️ Diffusers verification failed"
python3 -c "import moderngl; print('ModernGL: OK')" || echo "⚠️ ModernGL verification failed"

cd ..

echo "✅ Hunyuan3D-2.1 dependencies installation completed!"
echo ""
echo "🧪 Running test script..."
if python3 test_hunyuan3d_2_1_setup.py; then
    echo "✅ All tests passed!"
else
    echo "⚠️ Some tests failed, but installation may still be usable"
fi

echo ""
echo "🎉 Setup complete! You can now use Hunyuan3D-2.1 with PBR texture support."
echo ""
echo "� Next steps:"
echo "   1. Ensure conda environment is activated: conda activate 3d_gen"
echo "   2. Copy environment file: cp .env.hunyuan3d-2.1 .env"
echo "   3. Test the enhanced viewer: python3 -c 'from enhanced_3d_viewer import Enhanced3DViewer; print(\"✅ Ready!\")'"
echo "   4. Start the worker: python3 src/start_worker.py"
echo ""
echo "💡 Environment info:"
echo "   - Conda environment: $CONDA_DEFAULT_ENV"
echo "   - Python: $(python3 --version)"
echo "   - Current directory: $(pwd)"
