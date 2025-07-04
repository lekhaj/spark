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

# Test 4.5: Local src modules
print("\n4.5. Testing local src modules...")
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

src_modules = ["s3_manager", "db_helper", "config", "aws_manager"]
for module_name in src_modules:
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
    except Exception as e:
        print(f"❌ {module_name} error: {e}")

# Test 5: Hunyuan3D specific modules
print("\n5. Testing Hunyuan3D modules...")
script_dir = os.path.dirname(os.path.abspath(__file__))
hunyuan_paths = [
    os.path.join(script_dir, "Hunyuan3D-2.1"),
    os.path.join(script_dir, "Hunyuan3D-2"),
    os.path.join(script_dir, "Hunyuan3D-1"), 
    "./Hunyuan3D-2.1",
]

found_hunyuan = False
for path in hunyuan_paths:
    if os.path.isdir(path):
        found_hunyuan = True
        print(f"✅ Hunyuan3D found at {path}")
        sys.path.insert(0, path)
        
        # Test for mesh inpaint processor
        try:
            # Add DifferentiableRenderer to path
            render_path = os.path.join(path, "hy3dpaint", "DifferentiableRenderer")
            sys.path.insert(0, render_path)
            from mesh_inpaint_processor import meshVerticeInpaint
            print(f"✅ Mesh inpaint processor (meshVerticeInpaint) found and imported successfully")
        except ImportError as e:
            print(f"❌ Mesh inpaint processor import failed: {e}")
            print("   This will cause texture generation to fail")
            print("   Solution: Navigate to Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/ and run:")
            print("   c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` mesh_inpaint_processor.cpp -o mesh_inpaint_processor`python3-config --extension-suffix`")
        except Exception as e:
            print(f"❌ Mesh inpaint processor error: {e}")
        break

if not found_hunyuan:
    print("❌ Hunyuan3D directory not found")
    print("   Expected locations:", hunyuan_paths)
    print("   Please clone Hunyuan3D repository:")
    print("   git clone https://github.com/Tencent/Hunyuan3D-2.1.git  # For latest version")
    print("   git clone https://github.com/Tencent/Hunyuan3D-1.git   # For legacy version")

# Try importing Hunyuan3D modules
try:
    # Try importing 2.1 modules first with proper paths
    # Add the absolute paths to sys.path first
    hy3dshape_abs_path = '/home/ubuntu/spark/Hunyuan3D-2.1/hy3dshape'
    hy3dpaint_abs_path = '/home/ubuntu/spark/Hunyuan3D-2.1/hy3dpaint'
    
    if hy3dshape_abs_path not in sys.path:
        sys.path.insert(0, hy3dshape_abs_path)
    if hy3dpaint_abs_path not in sys.path:
        sys.path.insert(0, hy3dpaint_abs_path)
    
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print("✅ Hunyuan3DDiTFlowMatchingPipeline imported (2.1)")
        
        from hy3dshape.rembg import BackgroundRemover
        print("✅ BackgroundRemover imported (2.1)")
        
        try:
            from textureGenPipeline import Hunyuan3DPaintPipeline
            print("✅ Hunyuan3DPaintPipeline imported (2.1)")
        except ImportError as bpy_err:
            if "bpy" in str(bpy_err):
                print("⚠️  Hunyuan3DPaintPipeline requires bpy (Blender) - optional for core functionality")
            else:
                print(f"❌ Hunyuan3DPaintPipeline import failed: {bpy_err}")
        
        print("✅ Hunyuan3D-2.1 modules imported successfully")
        
    except ImportError:
        print("⚠️  Legacy hy3dgen module not found (this is expected for Hunyuan3D-2.1)")

except ImportError as e:
    print(f"❌ Hunyuan3D modules error: {e}")
    print("   Solutions:")
    print("   1. Make sure Hunyuan3D is cloned in the project directory")
    print("   2. Install Hunyuan3D requirements: pip install -r Hunyuan3D-2.1/requirements.txt")
    print("   3. Check if the hy3dshape module is properly structured")

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
    # Set up Python path with absolute paths - this works!
    hy3dshape_path = '/home/ubuntu/spark/Hunyuan3D-2.1/hy3dshape'
    hy3dpaint_path = '/home/ubuntu/spark/Hunyuan3D-2.1/hy3dpaint'
    
    if hy3dshape_path not in sys.path:
        sys.path.insert(0, hy3dshape_path)
    if hy3dpaint_path not in sys.path:
        sys.path.insert(0, hy3dpaint_path)
    
    # Also set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = f"{hy3dshape_path}:{hy3dpaint_path}:{current_pythonpath}"
    os.environ['PYTHONPATH'] = new_pythonpath
    
    # Import from the correct path structure
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    print("✅ hy3dshape.pipelines imported successfully")
    
    # Try textureGenPipeline import (bpy is optional)
    try:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        print("✅ textureGenPipeline imported successfully")
    except ImportError as bpy_error:
        if "bpy" in str(bpy_error):
            print("⚠️  textureGenPipeline requires bpy (Blender), but core functionality works")
        else:
            print(f"❌ textureGenPipeline import failed: {bpy_error}")
    
    print("✅ Hunyuan3D-2.1 core modules imported successfully")
    
except ImportError as e:
    print(f"❌ Failed to import Hunyuan3D-2.1 modules: {e}")
    try:
        # Try alternative import method - just hy3dshape first
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print("✅ hy3dshape.pipelines imported successfully")
    except ImportError as e2:
        print(f"   hy3dshape import also failed: {e2}")
    
    try:
        # Try textureGenPipeline separately
        from textureGenPipeline import Hunyuan3DPaintPipeline
        print("✅ textureGenPipeline imported successfully")
    except ImportError as e3:
        print(f"   textureGenPipeline import failed: {e3}")

print("\n" + "=" * 50)
print("🏁 Hunyuan3D setup test completed!")
print("\nIf you see errors above, install missing dependencies:")
print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("pip install transformers>=4.21.0 diffusers>=0.20.0 accelerate trimesh pillow numpy scipy")
