#!/usr/bin/env python3
"""
Example: 3D Generation from S3 Images

This example demonstrates how to generate 3D models from images stored in AWS S3.
The system automatically detects S3 URLs and downloads images temporarily for processing.

Usage:
    python s3_image_example.py

Requirements:
    - AWS credentials configured (via AWS CLI, environment variables, or IAM roles)
    - boto3 library installed
    - S3 bucket with accessible images
"""

import os
import sys
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hunyuan3d_worker import generate_3d_from_image_core, is_s3_url, initialize_hunyuan3d_processors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example S3 URLs - replace with your actual S3 URLs
EXAMPLE_S3_URLS = [
    "s3://my-bucket/images/example1.jpg",
    "https://my-bucket.s3.amazonaws.com/images/example2.png", 
    "https://s3.amazonaws.com/my-bucket/images/example3.jpg"
]

def test_s3_url_detection():
    """Test S3 URL detection functionality."""
    logger.info("🔍 Testing S3 URL detection...")
    
    test_cases = [
        ("local_image.jpg", False),
        ("/path/to/local/image.png", False),
        ("s3://my-bucket/image.jpg", True),
        ("https://my-bucket.s3.amazonaws.com/image.png", True),
        ("https://s3.amazonaws.com/my-bucket/image.jpg", True),
        ("https://example.com/image.jpg", False),
    ]
    
    for url, expected in test_cases:
        result = is_s3_url(url)
        status = "✅" if result == expected else "❌"
        logger.info(f"{status} {url} -> {result} (expected: {expected})")

def generate_3d_from_s3_example():
    """Example of generating 3D models from S3 images."""
    logger.info("🎯 3D Generation from S3 Images Example")
    
    # First test URL detection
    test_s3_url_detection()
    
    # Initialize processors (this would normally happen in the worker)
    logger.info("🔧 Initializing Hunyuan3D processors...")
    if not initialize_hunyuan3d_processors():
        logger.error("❌ Failed to initialize processors. Make sure Hunyuan3D is properly installed.")
        return
    
    # Example with S3 URL
    s3_image_url = "s3://your-bucket/images/example.jpg"  # Replace with your S3 URL
    
    logger.info(f"🖼️  Processing S3 image: {s3_image_url}")
    
    try:
        result = generate_3d_from_image_core(
            image_path=s3_image_url,
            with_texture=False,
            output_format='glb',
            progress_callback=lambda progress, status: logger.info(f"📊 Progress: {progress}% - {status}")
        )
        
        if result.get('status') == 'success':
            logger.info(f"🎉 Success! 3D model generated:")
            logger.info(f"   Model Path: {result.get('model_path')}")
            logger.info(f"   Model ID: {result.get('model_id')}")
            logger.info(f"   Processing Time: {result.get('generation_stats', {}).get('generation_time', 'N/A')}s")
        else:
            logger.error(f"❌ Failed: {result.get('message')}")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def main():
    """Main function."""
    logger.info("🚀 S3 Image to 3D Generation Example")
    
    # Check if we're in a proper environment
    if not os.path.exists('../src/hunyuan3d_worker.py'):
        logger.error("❌ This script should be run from the examples/ directory")
        return
    
    # Set up environment variables for testing (optional)
    os.environ.setdefault('S3_TEMP_DIR', '/tmp/s3_images')
    os.environ.setdefault('S3_DOWNLOAD_TIMEOUT', '60')
    
    generate_3d_from_s3_example()

if __name__ == "__main__":
    main()
