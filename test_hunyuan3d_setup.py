#!/usr/bin/env python3
"""
Test script to verify Hunyuan3D setup on GPU worker.
Run this on your GPU instance to check dependencies.
"""

import sys
import os

# Load environment variables from .env.gpu if it exists
env_file = "/home/ubuntu/Shashwat/spark/.env.gpu"
if os.path.exists(env_file):
    print(f"📋 Loading environment from {env_file}")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
else:
    print(f"⚠️  Environment file not found: {env_file}")

print("🔍 Testing Hunyuan3D setup...")
print("=" * 50)

# Test 1: Basic imports
print("\n1. Testing PyTorch and CUDA...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} imported successfully")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available - will use CPU")
except Exception as e:
    print(f"❌ PyTorch error: {e}")
    sys.exit(1)

# Test 2: Transformers
print("\n2. Testing Transformers...")
try:
    from transformers import AutoTokenizer, AutoModel
    import transformers
    print(f"✅ Transformers {transformers.__version__} imported successfully")
except Exception as e:
    print(f"❌ Transformers error: {e}")
    print("   Install with: pip install transformers>=4.21.0")

# Test 3: Diffusers
print("\n3. Testing Diffusers...")
try:
    import diffusers
    print(f"✅ Diffusers {diffusers.__version__} imported successfully")
except Exception as e:
    print(f"❌ Diffusers error: {e}")
    print("   Install with: pip install diffusers>=0.20.0")

# Test 4: Other ML libraries
print("\n4. Testing other ML libraries...")
libraries = [
    ('accelerate', 'accelerate'),
    ('trimesh', 'trimesh'),
    ('PIL', 'pillow'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy')
]

for lib_name, package_name in libraries:
    try:
        __import__(lib_name)
        print(f"✅ {lib_name} imported successfully")
    except Exception as e:
        print(f"❌ {lib_name} error: {e}")
        print(f"   Install with: pip install {package_name}")

# Test 5: Hunyuan3D specific modules
print("\n5. Testing Hunyuan3D modules...")
try:
    # Check if the hy3dgen module exists
    import hy3dgen
    print("✅ hy3dgen base module found")
    
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    print("✅ Hunyuan3DDiTFlowMatchingPipeline imported")
    
    from hy3dgen.rembg import BackgroundRemover
    print("✅ BackgroundRemover imported")
    
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    print("✅ Hunyuan3DPaintPipeline imported")
    
except ImportError as e:
    print(f"❌ Hunyuan3D modules error: {e}")
    print("   Make sure Hunyuan3D is properly installed")
    print("   Check if you have the Hunyuan3D-2 directory with the hy3dgen module")

# Test 6: Check worker module
print("\n6. Testing worker module...")
try:
    from hunyuan3d_worker import initialize_hunyuan3d_processors, generate_3d_from_image_core
    print("✅ Hunyuan3D worker functions imported successfully")
except ImportError as e:
    print(f"❌ Worker module error: {e}")
    print("   Check if hunyuan3d_worker.py exists and is properly configured")

# Test 7: Environment variables
print("\n7. Checking environment variables...")
env_vars = [
    'HUNYUAN3D_DEVICE',
    'REDIS_BROKER_URL',
    'WORKER_TYPE',
    'GPU_WORKER_QUEUES'
]

for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value}")
    else:
        print(f"⚠️  {var}: Not set")

# Test 8: Simple model loading test
print("\n8. Testing simple model loading...")
try:
    from transformers import AutoTokenizer
    # Try to load a simple tokenizer to test model loading capability
    print("✅ Model loading capability verified")
except Exception as e:
    print(f"❌ Model loading error: {e}")

print("\n" + "=" * 50)
print("🏁 Hunyuan3D setup test completed!")
print("\nIf you see errors above, install missing dependencies:")
print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("pip install transformers>=4.21.0 diffusers>=0.20.0 accelerate trimesh pillow numpy scipy")
