#!/usr/bin/env python3
"""
Test suite for Frontend Integration functionality.
Tests Gradio interface, 3D generation UI, and event handling.
"""

import unittest
import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestGradioAppIntegration(unittest.TestCase):
    """Test Gradio app integration and building."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images'),
            'OUTPUT_3D_ASSETS_DIR': os.path.join(self.temp_dir, '3d_assets'),
            'USE_CELERY': 'True',
            'MONGO_DB_NAME': 'test_db',
            'MONGO_BIOME_COLLECTION': 'test_biomes'
        })
        self.env_patcher.start()
        
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, '3d_assets'), exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('merged_gradio_app.gr.Blocks')
    @patch('merged_gradio_app.get_biome_names')
    @patch('merged_gradio_app.display_selected_biome')
    def test_build_app_success(self, mock_display_biome, mock_get_names, mock_blocks):
        """Test successful Gradio app building."""
        from merged_gradio_app import build_app
        
        # Mock biome functions
        mock_get_names.return_value = ['Forest', 'Desert']
        mock_display_biome.return_value = 'Biome details'
        
        # Mock Gradio components
        mock_demo = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_demo
        
        with patch('merged_gradio_app.gr.TabItem'), \
             patch('merged_gradio_app.gr.Textbox'), \
             patch('merged_gradio_app.gr.Button'), \
             patch('merged_gradio_app.gr.Image'), \
             patch('merged_gradio_app.gr.File'), \
             patch('merged_gradio_app.gr.Dropdown'), \
             patch('merged_gradio_app.gr.Slider'), \
             patch('merged_gradio_app.gr.Checkbox'), \
             patch('merged_gradio_app.gr.Accordion'):
            
            result = build_app()
            
            self.assertIsNotNone(result)
            mock_blocks.assert_called_once()
    
    @patch('merged_gradio_app.USE_CELERY', True)
    def test_app_production_mode(self):
        """Test app configuration in production mode (Celery enabled)."""
        from merged_gradio_app import USE_CELERY
        
        self.assertTrue(USE_CELERY)
    
    @patch('merged_gradio_app.USE_CELERY', False)
    def test_app_development_mode(self):
        """Test app configuration in development mode (direct processing)."""
        from merged_gradio_app import USE_CELERY
        
        self.assertFalse(USE_CELERY)


class Test3DGenerationUIFunctions(unittest.TestCase):
    """Test 3D generation UI wrapper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test.jpg')
        
        # Create test image file
        with open(self.test_image_path, 'w') as f:
            f.write('fake image data')
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_3D_ASSETS_DIR': os.path.join(self.temp_dir, '3d_assets'),
            'USE_CELERY': 'True'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('merged_gradio_app.USE_CELERY', True)
    @patch('merged_gradio_app.celery_generate_3d_model_from_image')
    def test_submit_3d_from_image_task_celery_mode(self, mock_celery_task):
        """Test 3D from image submission in Celery mode."""
        from merged_gradio_app import submit_3d_from_image_task
        
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = 'task-123'
        mock_celery_task.delay.return_value = mock_task
        
        result_file, message = submit_3d_from_image_task(
            self.test_image_path, True, 'glb', 'hunyuan3d'
        )
        
        self.assertIsNone(result_file)  # File not returned in Celery mode
        self.assertIn('task submitted', message)
        self.assertIn('task-123', message)
        mock_celery_task.delay.assert_called_once_with(self.test_image_path, True, 'glb')
    
    @patch('merged_gradio_app.USE_CELERY', False)
    @patch('merged_gradio_app.mock_generate_3d_from_image')
    def test_submit_3d_from_image_task_dev_mode(self, mock_generate):
        """Test 3D from image submission in development mode."""
        from merged_gradio_app import submit_3d_from_image_task
        
        # Mock development function
        mock_generate.return_value = 'Mock 3D generation result'
        
        result_file, message = submit_3d_from_image_task(
            self.test_image_path, True, 'glb', 'hunyuan3d'
        )
        
        self.assertIsNone(result_file)  # Mock mode doesn't return files
        self.assertIn('DEV Mode Mock', message)
        mock_generate.assert_called_once_with(self.test_image_path, True, 'glb')
    
    def test_submit_3d_from_image_task_no_image(self):
        """Test 3D from image submission with no image."""
        from merged_gradio_app import submit_3d_from_image_task
        
        result_file, message = submit_3d_from_image_task(None, True, 'glb', 'hunyuan3d')
        
        self.assertIsNone(result_file)
        self.assertIn('No image uploaded', message)

    @patch('merged_gradio_app.USE_CELERY', True)
    @patch('merged_gradio_app.celery_manage_gpu_instance')
    def test_manage_gpu_instance_task_celery_mode(self, mock_celery_task):
        """Test GPU instance management in Celery mode."""
        from merged_gradio_app import manage_gpu_instance_task
        
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = 'task-789'
        mock_celery_task.delay.return_value = mock_task
        
        message = manage_gpu_instance_task('start')
        
        self.assertIn('task submitted', message)
        self.assertIn('task-789', message)
        mock_celery_task.delay.assert_called_once_with('start')
    
    @patch('merged_gradio_app.USE_CELERY', False)
    def test_manage_gpu_instance_task_dev_mode(self):
        """Test GPU instance management in development mode."""
        from merged_gradio_app import manage_gpu_instance_task
        
        message = manage_gpu_instance_task('status')
        
        self.assertIn('DEV Mode Mock', message)
        self.assertIn('status command simulated', message)


class TestImageGenerationUIFunctions(unittest.TestCase):
    """Test existing image generation UI functions still work."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_IMAGES_DIR': os.path.join(self.temp_dir, 'images'),
            'USE_CELERY': 'True'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('merged_gradio_app.USE_CELERY', True)
    @patch('merged_gradio_app.celery_generate_text_image')
    def test_submit_text_prompt_task_celery_mode(self, mock_celery_task):
        """Test text prompt submission in Celery mode."""
        from merged_gradio_app import submit_text_prompt_task
        
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = 'text-task-123'
        mock_celery_task.delay.return_value = mock_task
        
        image, message = submit_text_prompt_task(
            'A beautiful landscape', 512, 512, 1, 'flux'
        )
        
        self.assertIsNone(image)  # No image returned in Celery mode
        self.assertIn('Task submitted', message)
        self.assertIn('text-task-123', message)
        mock_celery_task.delay.assert_called_once_with('A beautiful landscape', 512, 512, 1, 'flux')
    
    def test_submit_text_prompt_task_no_prompt(self):
        """Test text prompt submission with no prompt."""
        from merged_gradio_app import submit_text_prompt_task
        
        image, message = submit_text_prompt_task('', 512, 512, 1, 'flux')
        
        self.assertIsNone(image)
        self.assertIn('No prompt provided', message)
    
    @patch('merged_gradio_app.USE_CELERY', True)
    @patch('merged_gradio_app.celery_generate_grid_image')
    def test_submit_grid_input_task_celery_mode(self, mock_celery_task):
        """Test grid input submission in Celery mode."""
        from merged_gradio_app import submit_grid_input_task
        
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = 'grid-task-456'
        mock_celery_task.delay.return_value = mock_task
        
        image, viz, message = submit_grid_input_task(
            '0 1 0\n1 1 1\n0 1 0', 512, 512, 1, 'flux'
        )
        
        self.assertIsNone(image)
        self.assertIsNone(viz)
        self.assertIn('Task submitted', message)
        mock_celery_task.delay.assert_called_once()
    
    def test_submit_grid_input_task_no_grid(self):
        """Test grid input submission with no grid."""
        from merged_gradio_app import submit_grid_input_task
        
        image, viz, message = submit_grid_input_task('', 512, 512, 1, 'flux')
        
        self.assertIsNone(image)
        self.assertIsNone(viz)
        self.assertIn('No grid provided', message)


class TestBiomeInspectorUI(unittest.TestCase):
    """Test biome inspector UI functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'MONGO_DB_NAME': 'test_db',
            'MONGO_BIOME_COLLECTION': 'test_biomes',
            'USE_CELERY': 'True'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    @patch('merged_gradio_app.fetch_biome')
    def test_display_selected_biome_success(self, mock_fetch_biome):
        """Test successful biome display."""
        from merged_gradio_app import display_selected_biome
        
        # Mock biome data
        mock_biome = {
            'theme': 'A lush forest',
            'structures': ['Tree', 'Rock'],
            'grid_data': [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        }
        mock_fetch_biome.return_value = mock_biome
        
        result = display_selected_biome('test_db', 'test_biomes', 'Forest')
        
        self.assertIn('lush forest', result)
        self.assertIn('Tree', result)
        mock_fetch_biome.assert_called_once_with('test_db', 'test_biomes', 'Forest')
    
    @patch('merged_gradio_app.fetch_biome')
    def test_display_selected_biome_not_found(self, mock_fetch_biome):
        """Test biome display when biome not found."""
        from merged_gradio_app import display_selected_biome
        
        # Mock biome not found
        mock_fetch_biome.return_value = None
        
        result = display_selected_biome('test_db', 'test_biomes', 'Nonexistent')
        
        self.assertIn('not found', result)
        mock_fetch_biome.assert_called_once_with('test_db', 'test_biomes', 'Nonexistent')
    
    def test_display_selected_biome_empty_name(self):
        """Test biome display with empty name."""
        from merged_gradio_app import display_selected_biome
        
        result = display_selected_biome('test_db', 'test_biomes', '')
        
        self.assertEqual(result, '')
    
    @patch('merged_gradio_app.USE_CELERY', True)
    @patch('merged_gradio_app.celery_run_biome_generation')
    @patch('merged_gradio_app.get_biome_names')
    async def test_biome_handler_celery_mode(self, mock_get_names, mock_celery_task):
        """Test biome generation handler in Celery mode."""
        from merged_gradio_app import handler
        
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = 'biome-task-123'
        mock_celery_task.delay.return_value = mock_task
        
        # Mock biome names
        mock_get_names.return_value = ['Forest', 'Desert', 'New Biome']
        
        message, dropdown = await handler(
            'A mystical forest', 'Tree, Rock, Path', 'test_db', 'test_biomes'
        )
        
        self.assertIn('task submitted', message)
        self.assertIn('biome-task-123', message)
        mock_celery_task.delay.assert_called_once_with('A mystical forest', 'Tree, Rock, Path')


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in the Gradio app."""
    
    def test_create_sample_grid(self):
        """Test sample grid creation."""
        from merged_gradio_app import create_sample_grid
        
        sample = create_sample_grid()
        
        self.assertIsInstance(sample, str)
        self.assertIn('0', sample)
        self.assertIn('1', sample)
        self.assertIn('2', sample)
        self.assertIn('3', sample)
    
    @patch('merged_gradio_app.get_prompts_from_mongodb')
    def test_mongodb_integration_functions_exist(self, mock_get_prompts):
        """Test that MongoDB integration functions exist and are callable."""
        from merged_gradio_app import get_prompts_from_mongodb
        
        # Mock MongoDB function
        mock_get_prompts.return_value = (['prompt1', 'prompt2'], 'Success')
        
        prompts, status = get_prompts_from_mongodb()
        
        self.assertIsInstance(prompts, list)
        self.assertIsInstance(status, str)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the frontend integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OUTPUT_3D_ASSETS_DIR': os.path.join(self.temp_dir, '3d_assets'),
            'USE_CELERY': 'True'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('merged_gradio_app.celery_generate_3d_model_from_image')
    def test_3d_generation_task_exception(self, mock_celery_task):
        """Test 3D generation task exception handling."""
        from merged_gradio_app import submit_3d_from_image_task
        
        # Mock Celery task raising exception
        mock_celery_task.delay.side_effect = Exception("Celery connection error")
        
        result_file, message = submit_3d_from_image_task(
            '/fake/path.jpg', True, 'glb', 'hunyuan3d'
        )
        
        self.assertIsNone(result_file)
        self.assertIn('Error:', message)
        self.assertIn('Celery connection error', message)


if __name__ == '__main__':
    unittest.main(verbosity=2)
