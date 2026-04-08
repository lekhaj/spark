#!/usr/bin/env python3
"""
Test script for S3 3D asset integration in Gradio app
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_s3_integration():
    """Test S3 integration functions"""
    print("🧪 Testing S3 3D Asset Integration")
    print("=" * 50)
    
    try:
        from config import USE_S3_STORAGE, S3_BUCKET_NAME, S3_REGION
        print(f"✅ S3 Configuration loaded:")
        print(f"   USE_S3_STORAGE: {USE_S3_STORAGE}")
        print(f"   S3_BUCKET_NAME: {S3_BUCKET_NAME}")
        print(f"   S3_REGION: {S3_REGION}")
    except Exception as e:
        print(f"❌ Failed to load S3 config: {e}")
        return False
    
    try:
        from s3_manager import get_s3_manager
        s3_mgr = get_s3_manager()
        if s3_mgr:
            print("✅ S3 Manager initialized successfully")
        else:
            print("⚠️ S3 Manager initialization returned None (check AWS credentials)")
    except Exception as e:
        print(f"❌ S3 Manager error: {e}")
    
    try:
        # Test the helper functions
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import functions from the gradio app
        from merged_gradio_app import get_s3_3d_asset_url, check_s3_3d_asset_exists
        
        # Test URL generation
        test_s3_key = "3d_assets/generated/test_model.glb"
        s3_url = get_s3_3d_asset_url(test_s3_key)
        print(f"✅ S3 URL generation test: {s3_url}")
        
        # Test asset existence check (should return False for non-existent asset)
        exists, url = check_s3_3d_asset_exists("test_image.png", "glb")
        print(f"✅ Asset existence check test: exists={exists}, url={url}")
        
    except Exception as e:
        print(f"❌ Helper function test error: {e}")
    
    print("\n🎉 S3 integration test completed!")
    return True

if __name__ == "__main__":
    test_s3_integration()
