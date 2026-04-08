#!/usr/bin/env python3
"""
Test suite for Celery Tasks functionality.
Tests all 8 Celery tasks, error handling, and task routing.
"""

import unittest
import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import uuid

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCeleryTasksSetup(unittest.TestCase):
    """Test Celery tasks configuration and setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'REDIS_BROKER_URL': 'redis://localhost:6379/0',
            'REDIS_RESULT_BACKEND': 'redis://localhost:6379/0',
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images'),
            'OUTPUT_3D_ASSETS_DIR': os.path.join(self.temp_dir, '3d_assets'),
            'MONGO_DB_NAME': 'test_db',
            'MONGO_BIOME_COLLECTION': 'test_biomes'
        })
        self.env_patcher.start()
        
        # Create output directories
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, '3d_assets'), exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('tasks.Celery')
    def test_celery_app_configuration(self, mock_celery):
        """Test Celery app configuration."""
        # Import tasks module to trigger app creation
        import tasks
        
        # Verify Celery app was created with correct configuration
        mock_celery.assert_called_once_with(
            'gpu_tasks',
            broker='redis://localhost:6379/0',
            backend='redis://localhost:6379/0'
        )
    
    def test_task_module_loading_flags(self):
        """Test task module loading flags."""
        import tasks
        
        # Should have module loading flags
        self.assertIsInstance(tasks.TASK_2D_MODULES_LOADED, bool)
        self.assertIsInstance(tasks.TASK_3D_MODULES_LOADED, bool)
        self.assertIsInstance(tasks.TASK_MODULES_LOADED, bool)


class TestImageGenerationTasks(unittest.TestCase):
    """Test image generation Celery tasks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images'),
            'DEFAULT_TEXT_MODEL': 'flux',
            'DEFAULT_GRID_MODEL': 'flux'
        })
        self.env_patcher.start()
        
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('tasks._pipeline')
    @patch('tasks.TASK_MODULES_LOADED', True)
    @patch('tasks.create_image_grid')
    @patch('tasks.uuid.uuid4')
    def test_generate_text_image_success(self, mock_uuid, mock_create_grid, mock_pipeline):
        """Test successful text image generation."""
        from tasks import generate_text_image
        
        # Mock pipeline and image generation
        mock_image = Mock()
        mock_image.save = Mock()
        mock_pipeline.process_text.return_value = [mock_image]
        mock_pipeline.model_type = 'flux'
        
        # Mock UUID for consistent filename
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value='12345678-1234-1234-1234-123456789012')
        
        # Mock text processor
        mock_text_processor = Mock()
        mock_text_processor.model_type = 'flux'
        
        with patch('tasks._text_processor', mock_text_processor):
            result = generate_text_image("A beautiful landscape", 512, 512, 1, "flux")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('image_filenames', result)
        self.assertTrue(len(result['image_filenames']) > 0)
        mock_pipeline.process_text.assert_called_once_with("A beautiful landscape")
    
    @patch('tasks._pipeline', None)
    @patch('tasks.TASK_MODULES_LOADED', False)
    def test_generate_text_image_modules_not_loaded(self):
        """Test text image generation when modules not loaded."""
        from tasks import generate_text_image
        
        result = generate_text_image("A landscape", 512, 512, 1, "flux")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('not fully initialized', result['message'])
    
    @patch('tasks._pipeline')
    @patch('tasks.TASK_MODULES_LOADED', True)
    def test_generate_text_image_no_images_generated(self, mock_pipeline):
        """Test text image generation when no images are produced."""
        from tasks import generate_text_image
        
        # Mock pipeline returning empty list
        mock_pipeline.process_text.return_value = []
        
        result = generate_text_image("A landscape", 512, 512, 1, "flux")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('No images generated', result['message'])
    
    @patch('tasks._pipeline')
    @patch('tasks.TASK_MODULES_LOADED', True)
    @patch('tasks.create_image_grid')
    @patch('tasks.uuid.uuid4')
    def test_generate_text_image_multiple_images(self, mock_uuid, mock_create_grid, mock_pipeline):
        """Test text image generation with multiple images."""
        from tasks import generate_text_image
        
        # Mock multiple images
        mock_images = [Mock(), Mock(), Mock()]
        for img in mock_images:
            img.save = Mock()
        
        mock_pipeline.process_text.return_value = mock_images
        
        # Mock grid creation
        mock_grid = Mock()
        mock_grid.save = Mock()
        mock_create_grid.return_value = mock_grid
        
        # Mock UUID
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value='12345678-1234-1234-1234-123456789012')
        
        result = generate_text_image("A landscape", 512, 512, 3, "flux")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('grid', result['message'])
        mock_create_grid.assert_called_once_with(mock_images)
    
    @patch('tasks._pipeline')
    @patch('tasks.TASK_MODULES_LOADED', True)
    @patch('tasks.uuid.uuid4')
    def test_generate_grid_image_success(self, mock_uuid, mock_pipeline):
        """Test successful grid image generation."""
        from tasks import generate_grid_image
        
        # Mock pipeline and grid processing
        mock_image = Mock()
        mock_image.save = Mock()
        mock_viz = Mock()
        mock_viz.save = Mock()
        
        mock_pipeline.process_grid.return_value = ([mock_image], mock_viz)
        
        # Mock UUID
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value='12345678-1234-1234-1234-123456789012')
        
        grid_string = "0 1 0\n1 1 1\n0 1 0"
        result = generate_grid_image(grid_string, 512, 512, 1, "flux")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('image_filenames', result)
        mock_pipeline.process_grid.assert_called_once_with(grid_string)
        mock_viz.save.assert_called_once()
    
    @patch('tasks._pipeline')
    @patch('tasks.TASK_MODULES_LOADED', True)
    def test_generate_grid_image_no_images(self, mock_pipeline):
        """Test grid image generation when no images are produced."""
        from tasks import generate_grid_image
        
        # Mock pipeline returning empty list
        mock_pipeline.process_grid.return_value = ([], None)
        
        result = generate_grid_image("0 1 0", 512, 512, 1, "flux")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('No images generated', result['message'])


class TestBiomeGenerationTask(unittest.TestCase):
    """Test biome generation Celery task."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'MONGO_DB_NAME': 'test_db',
            'MONGO_BIOME_COLLECTION': 'test_biomes'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    @patch('tasks.TASK_MODULES_LOADED', True)
    @patch('tasks.generate_biome')
    def test_run_biome_generation_success(self, mock_generate_biome):
        """Test successful biome generation."""
        from tasks import run_biome_generation
        
        # Mock biome generation
        mock_generate_biome.return_value = "Biome generated successfully"
        
        result = run_biome_generation("A forest theme", "Tree, Rock, Path")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('Biome generated', result['message'])
        mock_generate_biome.assert_called_once_with("A forest theme", ["Tree", "Rock", "Path"])
    
    @patch('tasks.TASK_MODULES_LOADED', False)
    def test_run_biome_generation_modules_not_loaded(self):
        """Test biome generation when modules not loaded."""
        from tasks import run_biome_generation
        
        result = run_biome_generation("A theme", "Tree")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('not fully initialized', result['message'])
    
    @patch('tasks.TASK_MODULES_LOADED', True)
    def test_run_biome_generation_no_structures(self):
        """Test biome generation with no structure types."""
        from tasks import run_biome_generation
        
        result = run_biome_generation("A theme", "")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('No structure types', result['message'])
    
    @patch('tasks.TASK_MODULES_LOADED', True)
    @patch('tasks.generate_biome')
    def test_run_biome_generation_exception(self, mock_generate_biome):
        """Test biome generation with exception."""
        from tasks import run_biome_generation
        
        # Mock exception during generation
        mock_generate_biome.side_effect = Exception("Database error")
        
        result = run_biome_generation("A theme", "Tree, Rock")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Task failed', result['message'])


class Test3DGenerationTasks(unittest.TestCase):
    """Test 3D generation Celery tasks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_3D_ASSETS_DIR': os.path.join(self.temp_dir, '3d_assets'),
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images')
        })
        self.env_patcher.start()
        
        os.makedirs(os.path.join(self.temp_dir, '3d_assets'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('tasks.TASK_3D_MODULES_LOADED', True)
    @patch('tasks.initialize_hunyuan3d_processors')
    @patch('tasks._generate_3d_from_image_core')
    def test_generate_3d_model_from_image_success(self, mock_generate_core, mock_init_processors, mock_init):
        """Test successful 3D model generation from image."""
        from tasks import generate_3d_model_from_image
        
        # Mock initialization
        mock_init_processors.return_value = True
        
        # Mock 3D generation
        mock_generate_core.return_value = {
            'status': 'success',
            'message': '3D model generated',
            'white_mesh_path': '/path/to/model.glb',
            'model_dir': '/path/to/dir'
        }
        
        # Create a mock task instance for bind=True
        mock_task = Mock()
        mock_task.update_state = Mock()
        
        result = generate_3d_model_from_image.apply(
            args=['/path/to/image.jpg', True, 'glb'],
            task=mock_task
        ).result
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('white_mesh_path', result)
    
    @patch('tasks.TASK_3D_MODULES_LOADED', False)
    def test_generate_3d_model_from_image_modules_not_loaded(self):
        """Test 3D generation when modules not loaded."""
        from tasks import generate_3d_model_from_image
        
        # Create a mock task instance
        mock_task = Mock()
        
        result = generate_3d_model_from_image.apply(
            args=['/path/to/image.jpg', True, 'glb'],
            task=mock_task
        ).result
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('modules not loaded', result['message'])
    
    @patch('tasks.TASK_3D_MODULES_LOADED', True)
    @patch('tasks.TASK_2D_MODULES_LOADED', True)
    @patch('tasks.initialize_hunyuan3d_processors')
    @patch('tasks._pipeline')
    @patch('tasks._generate_3d_from_image_core')

class TestGPUManagementTask(unittest.TestCase):
    """Test GPU instance management Celery task."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'AWS_GPU_INSTANCE_ID': 'i-1234567890abcdef0',
            'AWS_REGION': 'us-west-2',
            'AWS_MAX_STARTUP_WAIT_TIME': '300'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    @patch('tasks.get_aws_manager')
    def test_manage_gpu_instance_ensure_running(self, mock_get_manager):
        """Test ensuring GPU instance is running."""
        from tasks import manage_gpu_instance
        
        # Mock AWS manager
        mock_manager = Mock()
        mock_manager.ensure_instance_running.return_value = True
        mock_get_manager.return_value = mock_manager
        
        result = manage_gpu_instance("ensure_running")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('is running', result['message'])
        mock_manager.ensure_instance_running.assert_called_once()
    
    @patch('tasks.get_aws_manager')
    def test_manage_gpu_instance_stop(self, mock_get_manager):
        """Test stopping GPU instance."""
        from tasks import manage_gpu_instance
        
        # Mock AWS manager
        mock_manager = Mock()
        mock_manager.stop_instance.return_value = True
        mock_get_manager.return_value = mock_manager
        
        result = manage_gpu_instance("stop")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('Stop request sent', result['message'])
        mock_manager.stop_instance.assert_called_once()
    
    @patch('tasks.get_aws_manager')
    def test_manage_gpu_instance_status(self, mock_get_manager):
        """Test getting GPU instance status."""
        from tasks import manage_gpu_instance
        
        # Mock AWS manager and instance info
        mock_info = Mock()
        mock_info.state = 'running'
        mock_info.instance_type = 'g4dn.xlarge'
        mock_info.public_ip = '1.2.3.4'
        mock_info.uptime_hours = 2.5
        
        mock_manager = Mock()
        mock_manager.get_instance_info.return_value = mock_info
        mock_manager.get_instance_cost_estimate.return_value = {
            'hourly_rate': 0.526,
            'current_session_cost': 1.315
        }
        mock_get_manager.return_value = mock_manager
        
        result = manage_gpu_instance("status")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('instance_info', result)
        self.assertIn('cost_estimate', result)
        self.assertEqual(result['instance_info']['state'], 'running')
    
    def test_manage_gpu_instance_no_instance_id(self):
        """Test GPU management with no instance ID."""
        from tasks import manage_gpu_instance
        
        with patch.dict(os.environ, {'AWS_GPU_INSTANCE_ID': ''}):
            result = manage_gpu_instance("status")
            
            self.assertEqual(result['status'], 'error')
            self.assertIn('No GPU instance ID', result['message'])
    
    @patch('tasks.get_aws_manager')
    def test_manage_gpu_instance_unknown_action(self, mock_get_manager):
        """Test GPU management with unknown action."""
        from tasks import manage_gpu_instance
        
        result = manage_gpu_instance("unknown_action")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Unknown action', result['message'])


class TestBatchProcessingTask(unittest.TestCase):
    """Test MongoDB batch processing Celery task."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images'),
            'MONGO_DB_NAME': 'test_db'
        })
        self.env_patcher.start()
        
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('tasks.TASK_MODULES_LOADED', True)
    @patch('tasks.MongoDBHelper')
    @patch('tasks._pipeline')
    @patch('tasks.uuid.uuid4')
    def test_batch_process_mongodb_prompts_success(self, mock_uuid, mock_pipeline, mock_mongo_helper_class):
        """Test successful batch processing of MongoDB prompts."""
        from tasks import batch_process_mongodb_prompts_task
        
        # Mock MongoDB helper
        mock_mongo_helper = Mock()
        mock_docs = [
            {'_id': 'doc1', 'theme_prompt': 'A forest scene'},
            {'_id': 'doc2', 'description': 'A mountain landscape'}
        ]
        mock_mongo_helper.find_many.return_value = mock_docs
        mock_mongo_helper.update_by_id.return_value = 1
        mock_mongo_helper_class.return_value = mock_mongo_helper
        
        # Mock pipeline
        mock_image = Mock()
        mock_image.save = Mock()
        mock_pipeline.process_text.return_value = [mock_image]
        
        # Mock UUID
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value='12345678-1234-1234-1234-123456789012')
        
        result = batch_process_mongodb_prompts_task(
            'test_db', 'test_collection', 10, 512, 512, 'flux', True
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('Generated image for', result['message'])
        mock_pipeline.process_text.assert_called()
    
    @patch('tasks.TASK_MODULES_LOADED', True)
    @patch('tasks.MongoDBHelper')
    def test_batch_process_mongodb_prompts_no_documents(self, mock_mongo_helper_class):
        """Test batch processing with no documents found."""
        from tasks import batch_process_mongodb_prompts_task
        
        # Mock MongoDB helper returning empty list
        mock_mongo_helper = Mock()
        mock_mongo_helper.find_many.return_value = []
        mock_mongo_helper_class.return_value = mock_mongo_helper
        
        result = batch_process_mongodb_prompts_task(
            'test_db', 'test_collection', 10, 512, 512, 'flux', True
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('No prompts found', result['message'])


class TestCleanupTask(unittest.TestCase):
    """Test asset cleanup Celery task."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.assets_dir = os.path.join(self.temp_dir, '3d_assets')
        self.images_dir = os.path.join(self.temp_dir, 'images')
        
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_3D_ASSETS_DIR': self.assets_dir,
            'OUTPUT_IMAGES_DIR': self.images_dir
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cleanup_old_assets_success(self):
        """Test successful cleanup of old assets."""
        from tasks import cleanup_old_assets
        import time
        
        # Create some old files
        old_file = os.path.join(self.assets_dir, 'old_model.glb')
        recent_file = os.path.join(self.assets_dir, 'recent_model.glb')
        
        with open(old_file, 'w') as f:
            f.write('old model data')
        with open(recent_file, 'w') as f:
            f.write('recent model data')
        
        # Mock file modification times
        old_time = time.time() - (48 * 3600)  # 48 hours ago
        recent_time = time.time() - (1 * 3600)  # 1 hour ago
        
        with patch('os.path.getmtime') as mock_getmtime:
            mock_getmtime.side_effect = lambda path: old_time if 'old' in path else recent_time
            
            result = cleanup_old_assets(max_age_hours=24)
        
        self.assertEqual(result['status'], 'success')
        self.assertGreater(result['files_cleaned'], 0)
        self.assertGreater(result['size_freed_mb'], 0)
    
    def test_cleanup_old_assets_no_files(self):
        """Test cleanup when no old files exist."""
        from tasks import cleanup_old_assets
        
        result = cleanup_old_assets(max_age_hours=24)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['files_cleaned'], 0)
        self.assertEqual(result['size_freed_mb'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
