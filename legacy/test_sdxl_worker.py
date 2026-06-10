#!/usr/bin/env python3
"""
SDXL Turbo Setup Test Script
Tests SDXL Turbo installation, model loading, and image generation
"""

import os
import sys
import torch
import tempfile

def test_cuda_availability():
    """Test CUDA availability"""
    print("🔍 Testing CUDA availability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"✅ CUDA available with {device_count} device(s)")
    print(f"📍 Current device: {current_device} ({device_name})")
    
    # Check VRAM
    try:
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_device)
        reserved_memory = torch.cuda.memory_reserved(current_device)
        
        total_gb = total_memory / 1024**3
        allocated_gb = allocated_memory / 1024**3
        reserved_gb = reserved_memory / 1024**3
        
        print(f"📊 VRAM - Total: {total_gb:.1f}GB, Allocated: {allocated_gb:.1f}GB, Reserved: {reserved_gb:.1f}GB")
        
        if total_gb < 8:
            print("⚠️ Warning: Less than 8GB VRAM detected. SDXL Turbo may run slowly.")
            
        return True
        
    except Exception as e:
        print(f"⚠️ Could not check VRAM details: {e}")
        return True
    
def test_dependencies():
    """Test required dependencies"""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        'diffusers',
        'transformers', 
        'accelerate',
        'torch',
        'torchvision',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_sdxl_worker():
    """Test SDXL worker import and initialization"""
    print("\n🔍 Testing SDXL worker...")
    
    # Add src directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from sdxl_turbo_worker import SDXLTurboWorker
        print("✅ SDXL worker imported successfully")
        
        # Initialize worker
        worker = SDXLTurboWorker()
        print("✅ SDXL worker initialized successfully")
        
        # Test health check
        health = worker.health_check()
        print(f"✅ Health check: {health}")
        
        return worker
        
    except Exception as e:
        print(f"❌ SDXL worker import/init failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_loading(worker):
    """Test SDXL model loading"""
    print("\n🔍 Testing SDXL model loading...")
    
    if not worker:
        print("❌ No worker available for testing")
        return False
    
    try:
        print("📥 Loading SDXL Turbo model...")
        success = worker.load_model()
        
        if success:
            print("✅ SDXL Turbo model loaded successfully")
            memory_usage = worker.get_memory_usage()
            print(f"📊 Memory usage after loading: {memory_usage}")
            return True
        else:
            print("❌ Failed to load SDXL Turbo model")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_generation(worker):
    """Test image generation"""
    print("\n🔍 Testing image generation...")
    
    if not worker or not worker.model_loaded:
        print("❌ Model not loaded, skipping generation test")
        return False
    
    try:
        print("🎨 Generating test image...")
        
        test_prompt = "a beautiful mountain landscape with clear blue sky"
        
        # Generate image
        image, metadata = worker.generate_image(
            prompt=test_prompt,
            num_inference_steps=1,  # Very fast for testing
            width=512,
            height=512
        )
        
        if image is not None:
            print("✅ Image generated successfully")
            print(f"📊 Generation metadata: {metadata}")
            
            # Save test image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f.name)
                print(f"💾 Test image saved to: {f.name}")
            
            return True
        else:
            print("❌ Image generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_celery_tasks():
    """Test SDXL Celery tasks"""
    print("\n🔍 Testing SDXL Celery tasks...")
    
    # Add src directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from tasks import app
        print("✅ Celery app imported successfully")
        
        # Check if SDXL tasks are registered
        task_names = list(app.tasks.keys())
        sdxl_tasks = [name for name in task_names if 'sdxl' in name.lower()]
        
        if sdxl_tasks:
            print(f"✅ Found SDXL tasks: {sdxl_tasks}")
        else:
            print("⚠️ No SDXL-specific tasks found in Celery app")
        
        # Test basic task registration
        if 'generate_image_sdxl_turbo' in task_names:
            print("✅ SDXL Turbo generation task registered")
        
        if 'batch_generate_images_sdxl_turbo' in task_names:
            print("✅ SDXL Turbo batch generation task registered")
        
        # No direct DB update logic in test_sdxl_worker.py, so just ensure batch_generate_images_sdxl_turbo is called with correct arguments in any test
        # Example usage for batch_generate_images_sdxl_turbo:
        # from tasks import batch_generate_images_sdxl_turbo
        # prompts_list = [
        #     {"prompt": "A fantasy castle", "doc_id": "...", "update_collection": "biomes", "category_key": "castle", "item_key": "main"},
        #     {"prompt": "A wooden hut", "doc_id": "...", "update_collection": "biomes", "category_key": "hut", "item_key": "side"}
        # ]
        # result = batch_generate_images_sdxl_turbo(prompts_list, batch_settings={"width": 512, "height": 512})
        # print(result)
        
        return True
        
    except Exception as e:
        print(f"❌ Celery tasks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_models(worker):
    """Clean up test models"""
    print("\n🧹 Cleaning up test models...")
    
    if worker:
        worker.unload_model()
        print("✅ SDXL model unloaded")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ CUDA cache cleared")

def main():
    """Run all SDXL Turbo tests"""
    print("🚀 SDXL Turbo Setup Test")
    print("========================")
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: CUDA
    if test_cuda_availability():
        tests_passed += 1
    
    # Test 2: Dependencies
    if test_dependencies():
        tests_passed += 1
    
    # Test 3: Worker
    worker = test_sdxl_worker()
    if worker:
        tests_passed += 1
    
    # Test 4: Model Loading
    model_loaded = False
    if worker:
        model_loaded = test_model_loading(worker)
        if model_loaded:
            tests_passed += 1
    
    # Test 5: Image Generation
    if model_loaded:
        if test_image_generation(worker):
            tests_passed += 1
    
    # Test 6: Celery Tasks
    if test_celery_tasks():
        tests_passed += 1
    
    # Cleanup
    cleanup_test_models(worker)
    
    # Results
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! SDXL Turbo is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
