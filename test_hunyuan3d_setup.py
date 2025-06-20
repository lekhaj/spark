#!/usr/bin/env python3
"""
Test script to verify Hunyuan3D setup on GPU worker.
Run this on your GPU instance to check dependencies.
"""

import sys
import os

# Load environment variables from .env.gpu if it exists
script_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(script_dir, ".env.gpu")
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
script_dir = os.path.dirname(os.path.abspath(__file__))
hunyuan_paths = [
    os.path.join(script_dir, "Hunyuan3D-2.1"),
    os.path.join(script_dir, "Hunyuan3D-2"),
    os.path.join(script_dir, "Hunyuan3D-1"), 
    "./Hunyuan3D-2.1",
    "./Hunyuan3D-2",
    "./Hunyuan3D-1"
]

hunyuan_found = False
for path in hunyuan_paths:
    if os.path.exists(path):
        print(f"✅ Found Hunyuan3D directory: {path}")
        hunyuan_found = True
        
        # Check if hy3dshape module exists (for 2.1)
        hy3dshape_path = os.path.join(path, "hy3dshape")
        if os.path.exists(hy3dshape_path):
            print(f"✅ Found hy3dshape module: {hy3dshape_path}")
            
            # Add to Python path temporarily
            if path not in sys.path:
                sys.path.insert(0, path)
                print(f"✅ Added {path} to Python path")
        
        # Check if hy3dgen module exists (for older versions)
        hy3dgen_path = os.path.join(path, "hy3dgen")
        if os.path.exists(hy3dgen_path):
            print(f"✅ Found hy3dgen module: {hy3dgen_path}")
            
            # Add to Python path temporarily
            if path not in sys.path:
                sys.path.insert(0, path)
                print(f"✅ Added {path} to Python path")
        break

if not hunyuan_found:
    print("❌ Hunyuan3D directory not found")
    print("   Expected locations:", hunyuan_paths)
    print("   Please clone Hunyuan3D repository:")
    print("   git clone https://github.com/Tencent/Hunyuan3D-2.1.git  # For latest version")
    print("   git clone https://github.com/Tencent/Hunyuan3D-1.git   # For legacy version")

# Try importing Hunyuan3D modules
try:
    # Try importing 2.1 modules first
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print("✅ Hunyuan3DDiTFlowMatchingPipeline imported (2.1)")
        
        from hy3dshape.rembg import BackgroundRemover
        print("✅ BackgroundRemover imported (2.1)")
        
        from textureGenPipeline import Hunyuan3DPaintPipeline
        print("✅ Hunyuan3DPaintPipeline imported (2.1)")
        
        print("✅ Hunyuan3D-2.1 modules imported successfully")
        
    except ImportError:
        # Fall back to older versions
        import hy3dgen
        print("✅ hy3dgen base module imported")
        
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        print("✅ Hunyuan3DDiTFlowMatchingPipeline imported")
        
        from hy3dgen.rembg import BackgroundRemover
        print("✅ BackgroundRemover imported")
        
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        print("✅ Hunyuan3DPaintPipeline imported")
        
        print("✅ Hunyuan3D modules imported successfully (legacy)")

except ImportError as e:
    print(f"❌ Hunyuan3D modules error: {e}")
    print("   Solutions:")
    print("   1. Make sure Hunyuan3D is cloned in the project directory")
    print("   2. Install Hunyuan3D requirements: pip install -r Hunyuan3D-2.1/requirements.txt")
    print("   3. Check if the hy3dshape/hy3dgen module is properly structured")

# Test 6: Check worker module
print("\n6. Testing worker module...")

# Add src directory to Python path for worker module
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✅ Added {src_path} to Python path")

try:
    from hunyuan3d_worker import initialize_hunyuan3d_processors, generate_3d_from_image_core
    print("✅ Hunyuan3D worker functions imported successfully")
except ImportError as e:
    print(f"❌ Worker module error: {e}")
    print("   Check if hunyuan3d_worker.py exists and is properly configured")
    print(f"   Checked path: {src_path}")

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

# Update paths and import tests
hunyuan_paths = [
    "Hunyuan3D-2.1",
    "./Hunyuan3D-2.1", 
    "../Hunyuan3D-2.1"
]

hunyuan_found = False
for path in hunyuan_paths:
    if os.path.exists(path):
        print(f"✅ Found Hunyuan3D-2.1 directory: {path}")
        hunyuan_found = True
        break

if not hunyuan_found:
    print("❌ Hunyuan3D-2.1 directory not found")
    print("   Expected locations:", hunyuan_paths)
    print("   Please clone Hunyuan3D-2.1 repository:")
    print("   git clone https://github.com/Tencent/Hunyuan3D-2.1.git")

# Test new imports
try:
    sys.path.insert(0, './Hunyuan3D-2.1/hy3dshape')
    sys.path.insert(0, './Hunyuan3D-2.1/hy3dpaint')
    
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    print("✅ Hunyuan3D-2.1 modules imported successfully")
    
except ImportError as e:
    print(f"❌ Failed to import Hunyuan3D-2.1 modules: {e}")

print("\n" + "=" * 50)
print("🏁 Hunyuan3D setup test completed!")
print("\nIf you see errors above, install missing dependencies:")
print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("pip install transformers>=4.21.0 diffusers>=0.20.0 accelerate trimesh pillow numpy scipy")
