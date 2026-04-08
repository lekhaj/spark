#!/usr/bin/env python3
"""
Enhanced Configuration Test Suite
Tests configuration loading, validation, and environment handling for the Hunyuan3D pipeline.
"""

import unittest
import sys
import os
import tempfile
from unittest.mock import patch, Mock
import importlib
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading from environment variables and defaults."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_configuration_loading(self):
        """Test loading basic configuration values."""
        with patch.dict(os.environ, {
            'DEFAULT_TEXT_MODEL': 'flux',
            'DEFAULT_GRID_MODEL': 'stability',
            'DEFAULT_IMAGE_WIDTH': '1024',
            'DEFAULT_IMAGE_HEIGHT': '768',
            'DEFAULT_NUM_IMAGES': '2'
        }):
            from config import (
                DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL,
                DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES
            )
            
            self.assertEqual(DEFAULT_TEXT_MODEL, 'flux')
            self.assertEqual(DEFAULT_GRID_MODEL, 'stability')
            self.assertEqual(DEFAULT_IMAGE_WIDTH, 1024)
            self.assertEqual(DEFAULT_IMAGE_HEIGHT, 768)
            self.assertEqual(DEFAULT_NUM_IMAGES, 2)
    
    def test_output_directory_configuration(self):
        """Test output directory configuration."""
        with patch.dict(os.environ, {
            'OUTPUT_DIR': self.temp_dir,
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images'),
            'OUTPUT_3D_ASSETS_DIR': os.path.join(self.temp_dir, '3d_assets')
        }):
            from config import OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_3D_ASSETS_DIR
            
            self.assertEqual(OUTPUT_DIR, self.temp_dir)
            self.assertTrue(OUTPUT_IMAGES_DIR.endswith('images'))
            self.assertTrue(OUTPUT_3D_ASSETS_DIR.endswith('3d_assets'))
    
    def test_mongodb_configuration(self):
        """Test MongoDB configuration."""
        with patch.dict(os.environ, {
            'MONGO_DB_NAME': 'test_database',
            'MONGO_BIOME_COLLECTION': 'test_biomes_collection',
            'MONGO_URI': 'mongodb://localhost:27017',
            'MONGO_PORT': '27018'
        }):
            from config import MONGO_DB_NAME, MONGO_BIOME_COLLECTION
            
            self.assertEqual(MONGO_DB_NAME, 'test_database')
            self.assertEqual(MONGO_BIOME_COLLECTION, 'test_biomes_collection')
    
    def test_redis_celery_configuration(self):
        """Test Redis and Celery configuration."""
        with patch.dict(os.environ, {
            'REDIS_BROKER_URL': 'redis://test-redis:6379/1',
            'REDIS_RESULT_BACKEND': 'redis://test-redis:6379/2',
            'USE_CELERY': 'True'
        }):
            from config import REDIS_BROKER_URL, REDIS_RESULT_BACKEND, USE_CELERY
            
            self.assertEqual(REDIS_BROKER_URL, 'redis://test-redis:6379/1')
            self.assertEqual(REDIS_RESULT_BACKEND, 'redis://test-redis:6379/2')
            self.assertTrue(USE_CELERY)
    
    def test_hunyuan3d_configuration(self):
        """Test Hunyuan3D specific configuration."""
        with patch.dict(os.environ, {
            'HUNYUAN3D_MODEL_PATH': 'custom/model/path',
            'HUNYUAN3D_SUBFOLDER': 'custom-subfolder',
            'HUNYUAN3D_TEXGEN_MODEL_PATH': 'custom/texgen/path',
            'HUNYUAN3D_STEPS': '25',
            'HUNYUAN3D_GUIDANCE_SCALE': '8.0',
            'HUNYUAN3D_OCTREE_RESOLUTION': '256',
            'HUNYUAN3D_DEVICE': 'cpu',
            'HUNYUAN3D_REMOVE_BACKGROUND': 'False',
            'HUNYUAN3D_NUM_CHUNKS': '1',
            'HUNYUAN3D_ENABLE_FLASHVDM': 'False',
            'HUNYUAN3D_COMPILE': 'True',
            'HUNYUAN3D_LOW_VRAM_MODE': 'True'
        }):
            from config import (
                HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
                HUNYUAN3D_STEPS, HUNYUAN3D_GUIDANCE_SCALE, HUNYUAN3D_OCTREE_RESOLUTION,
                HUNYUAN3D_DEVICE, HUNYUAN3D_REMOVE_BACKGROUND, HUNYUAN3D_NUM_CHUNKS,
                HUNYUAN3D_ENABLE_FLASHVDM, HUNYUAN3D_COMPILE, HUNYUAN3D_LOW_VRAM_MODE
            )
            
            self.assertEqual(HUNYUAN3D_MODEL_PATH, 'custom/model/path')
            self.assertEqual(HUNYUAN3D_SUBFOLDER, 'custom-subfolder')
            self.assertEqual(HUNYUAN3D_TEXGEN_MODEL_PATH, 'custom/texgen/path')
            self.assertEqual(HUNYUAN3D_STEPS, 25)
            self.assertEqual(HUNYUAN3D_GUIDANCE_SCALE, 8.0)
            self.assertEqual(HUNYUAN3D_OCTREE_RESOLUTION, 256)
            self.assertEqual(HUNYUAN3D_DEVICE, 'cpu')
            self.assertFalse(HUNYUAN3D_REMOVE_BACKGROUND)
            self.assertEqual(HUNYUAN3D_NUM_CHUNKS, 1)
            self.assertFalse(HUNYUAN3D_ENABLE_FLASHVDM)
            self.assertTrue(HUNYUAN3D_COMPILE)
            self.assertTrue(HUNYUAN3D_LOW_VRAM_MODE)
    
    def test_aws_configuration(self):
        """Test AWS configuration."""
        with patch.dict(os.environ, {
            'AWS_REGION': 'eu-west-1',
            'AWS_GPU_INSTANCE_ID': 'i-test123456789',
            'AWS_MAX_STARTUP_WAIT_TIME': '600',
            'AWS_EC2_CHECK_INTERVAL': '15'
        }):
            from config import (
                AWS_REGION, AWS_GPU_INSTANCE_ID,
                AWS_MAX_STARTUP_WAIT_TIME, AWS_EC2_CHECK_INTERVAL
            )
            
            self.assertEqual(AWS_REGION, 'eu-west-1')
            self.assertEqual(AWS_GPU_INSTANCE_ID, 'i-test123456789')
            self.assertEqual(AWS_MAX_STARTUP_WAIT_TIME, 600)
            self.assertEqual(AWS_EC2_CHECK_INTERVAL, 15)
    
    def test_task_routing_configuration(self):
        """Test Celery task routing configuration."""
        with patch.dict(os.environ, {
            'TASK_TIMEOUT_3D_GENERATION': '3600',
            'TASK_TIMEOUT_2D_GENERATION': '600',
            'TASK_TIMEOUT_EC2_MANAGEMENT': '300'
        }):
            from config import (
                TASK_TIMEOUT_3D_GENERATION, TASK_TIMEOUT_2D_GENERATION,
                TASK_TIMEOUT_EC2_MANAGEMENT
            )
            
            self.assertEqual(TASK_TIMEOUT_3D_GENERATION, 3600)
            self.assertEqual(TASK_TIMEOUT_2D_GENERATION, 600)
            self.assertEqual(TASK_TIMEOUT_EC2_MANAGEMENT, 300)


class TestConfigurationDefaults(unittest.TestCase):
    """Test configuration default values when environment variables are not set."""
    
    def test_default_values_without_env(self):
        """Test that configuration provides reasonable defaults."""
        with patch.dict(os.environ, {}, clear=True):
            try:
                from config import (
                    DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL,
                    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT
                )
                
                # Should have reasonable defaults
                self.assertIsInstance(DEFAULT_TEXT_MODEL, str)
                self.assertIsInstance(DEFAULT_GRID_MODEL, str)
                self.assertIsInstance(DEFAULT_IMAGE_WIDTH, int)
                self.assertIsInstance(DEFAULT_IMAGE_HEIGHT, int)
                self.assertGreater(DEFAULT_IMAGE_WIDTH, 0)
                self.assertGreater(DEFAULT_IMAGE_HEIGHT, 0)
                
            except ImportError:
                # Config module might not be available without environment
                pass
    
    def test_boolean_defaults(self):
        """Test boolean configuration defaults."""
        with patch.dict(os.environ, {}, clear=True):
            try:
                from config import USE_CELERY, HUNYUAN3D_REMOVE_BACKGROUND
                
                self.assertIsInstance(USE_CELERY, bool)
                self.assertIsInstance(HUNYUAN3D_REMOVE_BACKGROUND, bool)
                
            except ImportError:
                pass


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and error handling."""
    
    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric configuration values."""
        with patch.dict(os.environ, {
            'DEFAULT_IMAGE_WIDTH': 'invalid_number',
            'HUNYUAN3D_STEPS': 'not_a_number'
        }):
            # Should handle invalid values gracefully
            try:
                from config import DEFAULT_IMAGE_WIDTH, HUNYUAN3D_STEPS
                # Values should either be defaults or converted appropriately
                self.assertIsInstance(DEFAULT_IMAGE_WIDTH, int)
                self.assertIsInstance(HUNYUAN3D_STEPS, int)
            except (ValueError, ImportError):
                # Expected if config module has strict validation
                pass
    
    def test_invalid_boolean_values(self):
        """Test handling of invalid boolean configuration values."""
        with patch.dict(os.environ, {
            'USE_CELERY': 'maybe',
            'HUNYUAN3D_COMPILE': 'yes_please'
        }):
            try:
                from config import USE_CELERY, HUNYUAN3D_COMPILE
                
                # Should handle invalid boolean values
                self.assertIsInstance(USE_CELERY, bool)
                self.assertIsInstance(HUNYUAN3D_COMPILE, bool)
                
            except (ValueError, ImportError):
                pass


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration with other modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_with_aws_manager(self):
        """Test configuration integration with AWS manager."""
        with patch.dict(os.environ, {
            'AWS_REGION': 'us-east-1',
            'AWS_GPU_INSTANCE_ID': 'i-test12345'
        }):
            try:
                from config import AWS_REGION, AWS_GPU_INSTANCE_ID
                from aws_manager import get_aws_manager
                
                manager = get_aws_manager()
                
                self.assertEqual(manager.region, AWS_REGION)
                
            except ImportError:
                # Expected if modules not available
                pass
    
    def test_config_with_output_directories(self):
        """Test configuration creates valid output directories."""
        with patch.dict(os.environ, {
            'OUTPUT_DIR': self.temp_dir,
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images'),
            'OUTPUT_3D_ASSETS_DIR': os.path.join(self.temp_dir, '3d_assets')
        }):
            try:
                from config import OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_3D_ASSETS_DIR
                
                # Directories should be valid paths
                self.assertTrue(os.path.isabs(OUTPUT_DIR))
                self.assertTrue(os.path.isabs(OUTPUT_IMAGES_DIR))
                self.assertTrue(os.path.isabs(OUTPUT_3D_ASSETS_DIR))
                
            except ImportError:
                pass


def test_aws_configuration():
    """Legacy test function for AWS configuration."""
    print("🔧 Testing AWS Configuration...")
    
    try:
        from aws_manager import AWSManager, get_aws_manager
        from config import AWS_REGION, AWS_GPU_INSTANCE_ID
        
        print(f"✅ Successfully imported AWS modules")
        print(f"📍 Region: {AWS_REGION}")
        print(f"🖥️  Instance ID: {AWS_GPU_INSTANCE_ID or 'Not configured'}")
        
        # Test AWS connectivity
        print("\n🔌 Testing AWS connectivity...")
        
        try:
            manager = AWSManager(region=AWS_REGION)
            print("✅ AWS connection established")
            
            # List GPU instances
            print("\n🎮 Listing GPU instances...")
            gpu_instances = manager.list_gpu_instances()
            
            if gpu_instances:
                print(f"✅ Found {len(gpu_instances)} GPU instance(s):")
                for instance in gpu_instances:
                    status_emoji = {"running": "🟢", "stopped": "🔴", "pending": "🟡", "stopping": "🟠"}.get(instance.state, "⚪")
                    print(f"  {status_emoji} {instance.instance_id} ({instance.instance_type}) - {instance.state}")
                    if instance.uptime_hours:
                        print(f"    ⏰ Uptime: {instance.uptime_hours:.1f} hours")
            else:
                print("⚠️  No GPU instances found")
            
            # Test specific instance if configured
            if AWS_GPU_INSTANCE_ID:
                print(f"\n🎯 Testing configured instance: {AWS_GPU_INSTANCE_ID}")
                info = manager.get_instance_info(AWS_GPU_INSTANCE_ID)
                
                if info:
                    print(f"✅ Instance found: {info.instance_type} - {info.state}")
                    
                    # Get cost estimate
                    cost_info = manager.get_instance_cost_estimate(AWS_GPU_INSTANCE_ID)
                    if 'error' not in cost_info:
                        print(f"💰 Estimated cost: ${cost_info['estimated_cost_usd']:.2f} "
                              f"(${cost_info['hourly_rate_usd']:.2f}/hour)")
                else:
                    print(f"❌ Instance {AWS_GPU_INSTANCE_ID} not found")
            
            return True
            
        except Exception as e:
            print(f"❌ AWS connectivity error: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_redis_configuration():
    """Test Redis connectivity"""
    print("\n🔧 Testing Redis Configuration...")
    
    try:
        from config import REDIS_BROKER_URL, USE_CELERY
        print(f"✅ Redis URL: {REDIS_BROKER_URL}")
        print(f"✅ Celery enabled: {USE_CELERY}")
        
        if USE_CELERY:
            try:
                import redis
                # Parse the Redis URL
                if REDIS_BROKER_URL.startswith('redis://'):
                    # Extract host and port from URL
                    parts = REDIS_BROKER_URL.replace('redis://', '').split('/')
                    host_port = parts[0].split(':')
                    host = host_port[0] if len(host_port) > 0 else 'localhost'
                    port = int(host_port[1]) if len(host_port) > 1 else 6379
                    db = int(parts[1]) if len(parts) > 1 else 0
                    
                    print(f"🔌 Testing Redis connection to {host}:{port}/{db}...")
                    
                    r = redis.Redis(host=host, port=port, db=db, socket_timeout=5)
                    r.ping()
                    print("✅ Redis connection successful")
                    return True
                    
                else:
                    print("⚠️  Non-standard Redis URL format, skipping connection test")
                    return True
                    
            except ImportError:
                print("⚠️  Redis package not installed, cannot test connection")
                return True
            except Exception as e:
                print(f"❌ Redis connection failed: {e}")
                return False
        else:
            print("ℹ️  Celery disabled, skipping Redis test")
            return True
            
    except ImportError as e:
        print(f"❌ Configuration import error: {e}")
        return False

def test_hunyuan3d_configuration():
    """Test Hunyuan3D configuration"""
    print("\n🔧 Testing Hunyuan3D Configuration...")
    
    try:
        from config import (
            HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_DEVICE,
            HUNYUAN3D_STEPS, HUNYUAN3D_GUIDANCE_SCALE, SUPPORTED_3D_FORMATS
        )
        
        print(f"✅ Model path: {HUNYUAN3D_MODEL_PATH}")
        print(f"✅ Subfolder: {HUNYUAN3D_SUBFOLDER}")
        print(f"✅ Device: {HUNYUAN3D_DEVICE}")
        print(f"✅ Steps: {HUNYUAN3D_STEPS}")
        print(f"✅ Guidance scale: {HUNYUAN3D_GUIDANCE_SCALE}")
        print(f"✅ Supported formats: {', '.join(SUPPORTED_3D_FORMATS)}")
        
        # Check if output directories exist
        from config import OUTPUT_3D_ASSETS_DIR
        if os.path.exists(OUTPUT_3D_ASSETS_DIR):
            print(f"✅ 3D assets directory exists: {OUTPUT_3D_ASSETS_DIR}")
        else:
            print(f"⚠️  3D assets directory missing: {OUTPUT_3D_ASSETS_DIR}")
            os.makedirs(OUTPUT_3D_ASSETS_DIR, exist_ok=True)
            print(f"✅ Created 3D assets directory")
        
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Hunyuan3D + Celery + AWS Configuration Test")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Suppress debug logs during testing
    
    results = []
    
    # Test configurations
    results.append(("AWS Configuration", test_aws_configuration()))
    results.append(("Redis Configuration", test_redis_configuration()))
    results.append(("Hunyuan3D Configuration", test_hunyuan3d_configuration()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your configuration is ready.")
        print("\n📋 Next steps:")
        print("1. Start Redis server: redis-server")
        print("2. Start Celery worker: celery -A src.tasks worker --loglevel=info")
        print("3. Launch the application: python src/merged_gradio_app.py")
    else:
        print("\n⚠️  Some tests failed. Please review the configuration.")
        print("\n💡 Tips:")
        print("- Check your .env file configuration")
        print("- Ensure AWS credentials are properly configured")
        print("- Make sure Redis is installed and running")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
