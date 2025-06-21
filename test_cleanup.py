#!/usr/bin/env python3
"""
Quick validation script to test that the cleaned up codebase works correctly.
Tests that text-to-3D functionality has been properly removed while keeping image-to-3D.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        import config
        print("✅ config module imports successfully")
    except Exception as e:
        print(f"❌ config import failed: {e}")
        return False
    
    try:
        import tasks
        print("✅ tasks module imports successfully")
    except Exception as e:
        print(f"❌ tasks import failed: {e}")
        return False
    
    try:
        import merged_gradio_app
        print("✅ merged_gradio_app imports successfully")
    except Exception as e:
        print(f"❌ merged_gradio_app import failed: {e}")
        return False
    
    return True

def test_task_configuration():
    """Test that the task configuration has been cleaned up properly."""
    print("\nTesting task configuration...")
    
    from config import CELERY_TASK_ROUTES
    
    # Check that image-to-3D task still exists
    if 'generate_3d_model_from_image' in CELERY_TASK_ROUTES:
        print("✅ generate_3d_model_from_image task route exists")
    else:
        print("❌ generate_3d_model_from_image task route missing")
        return False
    
    # Check that text-to-3D task has been removed
    if 'generate_3d_model_from_prompt' not in CELERY_TASK_ROUTES:
        print("✅ generate_3d_model_from_prompt task route successfully removed")
    else:
        print("❌ generate_3d_model_from_prompt task route still exists")
        return False
    
    return True

def test_available_tasks():
    """Test that the correct tasks are available in the tasks module."""
    print("\nTesting available tasks...")
    
    import tasks
    
    # Check that image-to-3D task exists
    if hasattr(tasks, 'generate_3d_model_from_image'):
        print("✅ generate_3d_model_from_image task function exists")
    else:
        print("❌ generate_3d_model_from_image task function missing")
        return False
    
    # Check that text-to-3D task has been removed
    if not hasattr(tasks, 'generate_3d_model_from_prompt'):
        print("✅ generate_3d_model_from_prompt task function successfully removed")
    else:
        print("❌ generate_3d_model_from_prompt task function still exists")
        return False
    
    return True

def test_gradio_app_functions():
    """Test that the Gradio app has the correct functions."""
    print("\nTesting Gradio app functions...")
    
    import merged_gradio_app
    
    # Check that image-to-3D function exists
    if hasattr(merged_gradio_app, 'submit_3d_from_image_task'):
        print("✅ submit_3d_from_image_task function exists")
    else:
        print("❌ submit_3d_from_image_task function missing")
        return False
    
    # Check that text-to-3D function has been removed
    if not hasattr(merged_gradio_app, 'submit_3d_from_prompt_task'):
        print("✅ submit_3d_from_prompt_task function successfully removed")
    else:
        print("❌ submit_3d_from_prompt_task function still exists")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧹 Testing cleaned up codebase...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_task_configuration,
        test_available_tasks,
        test_gradio_app_functions
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The cleanup was successful.")
        print("\nPipeline now supports:")
        print("  • Prompt/Grid → Image ✅")
        print("  • Image → 3D ✅")
        print("  • Text → 3D ❌ (removed)")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
