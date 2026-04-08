#!/usr/bin/env python3
"""
Test suite for Hunyuan3D Worker functionality.
Tests 3D model generation, initialization, and GPU operations.
"""

import unittest
import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from PIL import Image as PILImage
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHunyuan3DWorker(unittest.TestCase):
    """Test cases for Hunyuan3D Worker."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.png')
        
        # Create a test image
        test_image = PILImage.new('RGB', (256, 256), color='red')
        test_image.save(self.test_image_path)
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'HUNYUAN3D_MODEL_PATH': 'tencent/Hunyuan3D-2',
            'HUNYUAN3D_SUBFOLDER': 'hy3d-base',
            'HUNYUAN3D_TEXGEN_MODEL_PATH': 'tencent/Hunyuan3D-2',
            'HUNYUAN3D_STEPS': '30',
            'HUNYUAN3D_GUIDANCE_SCALE': '7.5',
            'HUNYUAN3D_OCTREE_RESOLUTION': '512',
            'HUNYUAN3D_DEVICE': 'cuda',
            'HUNYUAN3D_REMOVE_BACKGROUND': 'True',
            'HUNYUAN3D_ENABLE_FLASHVDM': 'True',
            'HUNYUAN3D_COMPILE': 'False',
            'HUNYUAN3D_LOW_VRAM_MODE': 'False',
            'OUTPUT_3D_ASSETS_DIR': self.temp_dir
        })
        self.env_patcher.start()
        
        # Reset global state
        self._reset_worker_state()
    
    def tearDown(self):
        """Clean up after each test method."""
        self.env_patcher.stop()
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self._reset_worker_state()
    
    def _reset_worker_state(self):
        """Reset global worker state for clean tests."""
        import hunyuan3d_worker
        hunyuan3d_worker._hunyuan_i23d_worker = None
        hunyuan3d_worker._hunyuan_rembg_worker = None
        hunyuan3d_worker._hunyuan_texgen_worker = None
        hunyuan3d_worker._models_initialized = False
        hunyuan3d_worker._initialization_error = None
    
    @patch('hunyuan3d_worker.torch')
    @patch('hunyuan3d_worker.BackgroundRemover')
    @patch('hunyuan3d_worker.Hunyuan3DDiTFlowMatchingPipeline')
    @patch('hunyuan3d_worker.Hunyuan3DPaintPipeline')
    def test_initialize_hunyuan3d_processors_success(self, mock_paint, mock_pipeline, mock_bg_remover, mock_torch):
        """Test successful initialization of Hunyuan3D processors."""
        from hunyuan3d_worker import initialize_hunyuan3d_processors
        
        # Mock successful initialization
        mock_pipeline.from_pretrained.return_value = Mock()
        mock_bg_remover.return_value = Mock()
        mock_paint.from_pretrained.return_value = Mock()
        
        result = initialize_hunyuan3d_processors()
        
        self.assertTrue(result)
        mock_pipeline.from_pretrained.assert_called_once()
        mock_bg_remover.assert_called_once()
        mock_paint.from_pretrained.assert_called_once()
    
    @patch('hunyuan3d_worker.torch')
    def test_initialize_hunyuan3d_processors_import_error(self, mock_torch):
        """Test initialization failure due to import error."""
        from hunyuan3d_worker import initialize_hunyuan3d_processors
        
        # Mock import error
        with patch('hunyuan3d_worker.Hunyuan3DDiTFlowMatchingPipeline', side_effect=ImportError("Module not found")):
            result = initialize_hunyuan3d_processors()
            
            self.assertFalse(result)
    
    @patch('hunyuan3d_worker.initialize_hunyuan3d_processors')
    @patch('hunyuan3d_worker.PILImage')
    @patch('hunyuan3d_worker.torch')
    @patch('hunyuan3d_worker.os.makedirs')
    def test_generate_3d_from_image_core_success(self, mock_makedirs, mock_torch, mock_pil, mock_init):
        """Test successful 3D generation from image."""
        from hunyuan3d_worker import generate_3d_from_image_core
        
        # Mock successful initialization
        mock_init.return_value = True
        
        # Mock image loading
        mock_image = Mock()
        mock_pil.open.return_value.convert.return_value = mock_image
        
        # Mock processors
        mock_rembg = Mock()
        mock_rembg.return_value = mock_image
        
        mock_i23d = Mock()
        mock_mesh = Mock()
        mock_mesh.export = Mock()
        mock_i23d.return_value = mock_mesh
        
        # Mock global workers
        with patch('hunyuan3d_worker._hunyuan_rembg_worker', mock_rembg), \
             patch('hunyuan3d_worker._hunyuan_i23d_worker', mock_i23d):
            
            result = generate_3d_from_image_core(self.test_image_path)
            
            self.assertEqual(result['status'], 'success')
            self.assertIn('white_mesh_path', result)
            self.assertIn('model_dir', result)
            mock_mesh.export.assert_called()
    
    @patch('hunyuan3d_worker.initialize_hunyuan3d_processors')
    def test_generate_3d_from_image_core_init_failure(self, mock_init):
        """Test 3D generation failure due to initialization error."""
        from hunyuan3d_worker import generate_3d_from_image_core
        
        # Mock initialization failure
        mock_init.return_value = False
        
        result = generate_3d_from_image_core(self.test_image_path)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Failed to initialize', result['message'])
    
    @patch('hunyuan3d_worker.initialize_hunyuan3d_processors')
    @patch('hunyuan3d_worker.PILImage')
    def test_generate_3d_from_image_core_invalid_image(self, mock_pil, mock_init):
        """Test 3D generation failure due to invalid image."""
        from hunyuan3d_worker import generate_3d_from_image_core
        
        # Mock successful initialization
        mock_init.return_value = True
        
        # Mock image loading failure
        mock_pil.open.side_effect = Exception("Invalid image file")
        
        result = generate_3d_from_image_core('/invalid/path.jpg')
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Failed to load image', result['message'])
    
    @patch('hunyuan3d_worker.initialize_hunyuan3d_processors')
    @patch('hunyuan3d_worker.PILImage')
    @patch('hunyuan3d_worker.torch')
    @patch('hunyuan3d_worker.os.makedirs')
    def test_generate_3d_with_texture(self, mock_makedirs, mock_torch, mock_pil, mock_init):
        """Test 3D generation with texture."""
        from hunyuan3d_worker import generate_3d_from_image_core
        
        # Mock successful initialization
        mock_init.return_value = True
        
        # Mock image loading
        mock_image = Mock()
        mock_pil.open.return_value.convert.return_value = mock_image
        
        # Mock processors
        mock_i23d = Mock()
        mock_mesh = Mock()
        mock_mesh.export = Mock()
        mock_i23d.return_value = mock_mesh
        
        mock_texgen = Mock()
        mock_textured_mesh = Mock()
        mock_textured_mesh.export = Mock()
        mock_texgen.return_value = mock_textured_mesh
        
        # Mock global workers
        with patch('hunyuan3d_worker._hunyuan_i23d_worker', mock_i23d), \
             patch('hunyuan3d_worker._hunyuan_texgen_worker', mock_texgen):
            
            result = generate_3d_from_image_core(self.test_image_path, with_texture=True)
            
            self.assertEqual(result['status'], 'success')
            self.assertIn('textured_mesh_path', result)
            mock_textured_mesh.export.assert_called()
    
    @patch('hunyuan3d_worker.initialize_hunyuan3d_processors')
    @patch('hunyuan3d_worker.PILImage')
    @patch('hunyuan3d_worker.torch')
    @patch('hunyuan3d_worker.os.makedirs')
    def test_generate_3d_progress_callback(self, mock_makedirs, mock_torch, mock_pil, mock_init):
        """Test 3D generation with progress callback."""
        from hunyuan3d_worker import generate_3d_from_image_core
        
        # Mock successful initialization
        mock_init.return_value = True
        
        # Mock image loading
        mock_image = Mock()
        mock_pil.open.return_value.convert.return_value = mock_image
        
        # Mock processors
        mock_i23d = Mock()
        mock_mesh = Mock()
        mock_mesh.export = Mock()
        mock_i23d.return_value = mock_mesh
        
        # Progress callback tracker
        progress_calls = []
        def progress_callback(progress, status):
            progress_calls.append((progress, status))
        
        # Mock global workers
        with patch('hunyuan3d_worker._hunyuan_i23d_worker', mock_i23d):
            
            result = generate_3d_from_image_core(
                self.test_image_path, 
                progress_callback=progress_callback
            )
            
            self.assertEqual(result['status'], 'success')
            self.assertTrue(len(progress_calls) > 0)
            # Check that progress values are reasonable
            for progress, status in progress_calls:
                self.assertGreaterEqual(progress, 0)
                self.assertLessEqual(progress, 100)
                self.assertIsInstance(status, str)
    
    def test_get_model_info(self):
        """Test getting model information."""
        from hunyuan3d_worker import get_model_info
        
        info = get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('hunyuan3d_available', info)
        self.assertIn('device', info)
        self.assertIn('models_initialized', info)
    
    @patch('hunyuan3d_worker._hunyuan_i23d_worker')
    @patch('hunyuan3d_worker._hunyuan_rembg_worker')
    @patch('hunyuan3d_worker._hunyuan_texgen_worker')
    def test_cleanup_models(self, mock_texgen, mock_rembg, mock_i23d):
        """Test model cleanup."""
        from hunyuan3d_worker import cleanup_models
        
        # Mock models with cleanup methods
        mock_i23d.cleanup = Mock()
        mock_texgen.cleanup = Mock()
        
        cleanup_models()
        
        # Verify cleanup was attempted (even if methods don't exist)
        # The function should handle missing cleanup methods gracefully
        # This is more about testing the function doesn't crash
    
    def test_multiple_initialization_calls(self):
        """Test that multiple initialization calls don't reinitialize."""
        from hunyuan3d_worker import initialize_hunyuan3d_processors
        import hunyuan3d_worker
        
        # Mock already initialized state
        hunyuan3d_worker._models_initialized = True
        
        result = initialize_hunyuan3d_processors()
        
        self.assertTrue(result)
    
    def test_initialization_with_previous_error(self):
        """Test initialization when there was a previous error."""
        from hunyuan3d_worker import initialize_hunyuan3d_processors
        import hunyuan3d_worker
        
        # Mock previous initialization error
        hunyuan3d_worker._initialization_error = "Previous error"
        
        result = initialize_hunyuan3d_processors()
        
        self.assertFalse(result)


class TestHunyuan3DWorkerConfiguration(unittest.TestCase):
    """Test Hunyuan3D Worker configuration handling."""
    
    def test_configuration_loading(self):
        """Test loading configuration from config module."""
        with patch.dict(os.environ, {
            'HUNYUAN3D_STEPS': '50',
            'HUNYUAN3D_DEVICE': 'cpu',
            'HUNYUAN3D_OCTREE_RESOLUTION': '256'
        }):
            # Import config values
            from config import HUNYUAN3D_STEPS, HUNYUAN3D_DEVICE, HUNYUAN3D_OCTREE_RESOLUTION
            
            self.assertEqual(HUNYUAN3D_STEPS, 50)
            self.assertEqual(HUNYUAN3D_DEVICE, 'cpu')
            self.assertEqual(HUNYUAN3D_OCTREE_RESOLUTION, 256)
    
    def test_default_configuration(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            # Import should still work with defaults
            try:
                from config import HUNYUAN3D_DEVICE
                # Should have a reasonable default
                self.assertIn(HUNYUAN3D_DEVICE, ['cuda', 'cpu'])
            except ImportError:
                # Config might not be available in test environment
                pass


class TestHunyuan3DWorkerIntegration(unittest.TestCase):
    """Integration tests for Hunyuan3D Worker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_mock_generation(self):
        """Test end-to-end 3D generation with all mocks."""
        from hunyuan3d_worker import generate_3d_from_image_core
        
        # Create test image
        test_image_path = os.path.join(self.temp_dir, 'test.png')
        test_image = PILImage.new('RGB', (256, 256), color='blue')
        test_image.save(test_image_path)
        
        # Mock all dependencies
        with patch('hunyuan3d_worker.initialize_hunyuan3d_processors', return_value=True), \
             patch('hunyuan3d_worker.PILImage.open') as mock_open, \
             patch('hunyuan3d_worker.torch'), \
             patch('hunyuan3d_worker.os.makedirs'), \
             patch('hunyuan3d_worker._hunyuan_i23d_worker') as mock_i23d:
            
            # Setup mocks
            mock_open.return_value.convert.return_value = test_image
            mock_mesh = Mock()
            mock_mesh.export = Mock()
            mock_i23d.return_value = mock_mesh
            
            # Run generation
            result = generate_3d_from_image_core(test_image_path)
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)
            self.assertIn('message', result)
            
            if result['status'] == 'success':
                self.assertIn('white_mesh_path', result)
                self.assertIn('model_dir', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
