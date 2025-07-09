"""
Hunyuan3D Worker Module

This module handles the initialization and core processing logic for Hunyuan3D models.
It's designed to be imported by the Celery worker tasks for 3D model generation.
"""

import os
import sys
import logging
import time
import uuid
from typing import Optional, Dict, Any, Callable
from PIL import Image as PILImage
import torch

# Import configurations for Hunyuan3D-2.1
from config import (
    HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
    HUNYUAN3D_COMPILE, HUNYUAN3D_ENABLE_FLASHVDM, HUNYUAN3D_DEVICE,
    HUNYUAN3D_PAINT_CONFIG_MAX_VIEWS, HUNYUAN3D_PAINT_CONFIG_RESOLUTION
)

# Fix Python path for Hunyuan3D imports
def _fix_python_path():
    """Add Hunyuan3D directory to Python path if needed."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    hunyuan_dirs = [
        os.path.join(project_root, "Hunyuan3D-2.1"),
        os.path.join(project_root, "Hunyuan3D-2"),
        os.path.join(project_root, "Hunyuan3D-1"),
        os.path.join(project_root, "models", "Hunyuan3D-2.1"),
        os.path.join(project_root, "models", "Hunyuan3D-2"),
        os.path.join(project_root, "models", "Hunyuan3D-1")
    ]
    
    for hunyuan_dir in hunyuan_dirs:
        if os.path.exists(hunyuan_dir):
            if hunyuan_dir not in sys.path:
                sys.path.insert(0, hunyuan_dir)
                print(f"Added {hunyuan_dir} to Python path")
            
            # Also add the specific subdirectories for Hunyuan3D-2.1
            hy3dshape_path = os.path.join(hunyuan_dir, "hy3dshape")
            hy3dpaint_path = os.path.join(hunyuan_dir, "hy3dpaint")
            
            if os.path.exists(hy3dshape_path) and hy3dshape_path not in sys.path:
                sys.path.insert(0, hy3dshape_path)
                print(f"Added {hy3dshape_path} to Python path")
                
            if os.path.exists(hy3dpaint_path) and hy3dpaint_path not in sys.path:
                sys.path.insert(0, hy3dpaint_path)
                print(f"Added {hy3dpaint_path} to Python path")
                
            return hunyuan_dir
    
    return None

# Fix path on import
_hunyuan_path = _fix_python_path()

# Setup logging
logger = logging.getLogger(__name__)

# Global worker instances (lazy loaded)
_hunyuan_i23d_worker = None
_hunyuan_rembg_worker = None
_hunyuan_texgen_worker = None

# Model loading flags
_models_initialized = False
_initialization_error = None
_dependencies_available = False

# Check dependencies at module import time
def _check_dependencies():
    """Check if all required dependencies are available."""
    global _dependencies_available
    
    try:
        # Check PyTorch and CUDA
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} available")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"✓ GPU: {torch.cuda.get_device_name()}")
        
        # Check basic ML libraries
        import transformers
        import diffusers
        logger.info(f"✓ Transformers {transformers.__version__} available")
        logger.info(f"✓ Diffusers {diffusers.__version__} available")
        
        # Try to import 3D processing libraries
        import trimesh
        logger.info("✓ Trimesh available")
        
        _dependencies_available = True
        logger.info("✅ All basic dependencies available")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        _dependencies_available = False
        return False

# Check dependencies when module is imported
_check_dependencies()


def get_device():
    """Get the appropriate device for Hunyuan3D processing."""
    device = HUNYUAN3D_DEVICE if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device


def initialize_hunyuan3d_processors():
    """
    Initialize Hunyuan3D processors for 3D generation.
    This function should be called once per worker process.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _hunyuan_i23d_worker, _hunyuan_texgen_worker, _hunyuan_rembg_worker
    global _models_initialized, _initialization_error
    
    if _models_initialized:
        logger.info("✅ Hunyuan3D processors already initialized")
        return True
    
    if _initialization_error:
        logger.error(f"❌ Previous initialization failed: {_initialization_error}")
        return False
    
    try:
        logger.info("🚀 Initializing Hunyuan3D-2.1 processors...")
        device = get_device()
        
        # Apply torchvision compatibility fix for 2.1
        try:
            # Import torchvision fix from Hunyuan3D-2.1
            hunyuan_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Hunyuan3D-2.1")
            torchvision_fix_path = os.path.join(hunyuan_path, "torchvision_fix.py")
            if os.path.exists(torchvision_fix_path):
                sys.path.insert(0, hunyuan_path)
                from torchvision_fix import apply_fix
                apply_fix()
                logger.info("✓ Applied torchvision compatibility fix")
        except Exception as e:
            logger.warning(f"⚠️  Could not apply torchvision fix: {e}")
        
        # Initialize shape generation pipeline for 2.1
        logger.info(f"🎯 Loading shape generation model from: {HUNYUAN3D_MODEL_PATH}")
        _log_memory_usage("before shape model loading")
        try:
            # Updated import path for 2.1 - use absolute path
            # The path should already be set by _fix_python_path(), but let's ensure it
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            hy3dshape_path = os.path.join(project_root, 'Hunyuan3D-2.1', 'hy3dshape')
            if hy3dshape_path not in sys.path:
                sys.path.insert(0, hy3dshape_path)
            
            # Import and load model directly to GPU to avoid CPU memory usage
            logger.info("🚀 Loading model directly to GPU to avoid system RAM usage...")
            _hunyuan_i23d_worker = _load_hunyuan3d_directly_to_gpu(
                HUNYUAN3D_MODEL_PATH,
                HUNYUAN3D_SUBFOLDER,
                device=device,
                dtype=torch.float16
            )
            
            _log_memory_usage("after direct GPU loading")
            
            # Enable FlashVDM for 2.1 with GPU configuration
            _hunyuan_i23d_worker.enable_flashvdm(mc_algo='mc')
            
            logger.info(f"✓ Shape generation model loaded successfully to {device}")
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "CPU mode")
            
            if HUNYUAN3D_COMPILE and hasattr(_hunyuan_i23d_worker, 'compile'):
                logger.info("⚡ Compiling Hunyuan3D model for optimization...")
                _hunyuan_i23d_worker.compile()
                
        except Exception as e:
            logger.error(f"❌ Failed to load shape generation model: {e}")
            _initialization_error = f"Shape model loading failed: {e}"
            return False
        
        # Initialize background remover
        logger.info("🖼️  Loading background removal model...")
        try:
            from hy3dshape.rembg import BackgroundRemover
            _hunyuan_rembg_worker = BackgroundRemover()
            logger.info("✓ Background removal model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load background removal model: {e}")
            _hunyuan_rembg_worker = None
        
        # Initialize texture generation pipeline for 2.1 (with PBR)
        if HUNYUAN3D_TEXGEN_MODEL_PATH:
            logger.info(f"🎨 Loading PBR texture generation model from: {HUNYUAN3D_TEXGEN_MODEL_PATH}")
            try:
                # Updated import path for 2.1 - use absolute path
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                hy3dpaint_path = os.path.join(project_root, 'Hunyuan3D-2.1', 'hy3dpaint')
                if hy3dpaint_path not in sys.path:
                    sys.path.insert(0, hy3dpaint_path)
                
                # Check and compile mesh_inpaint_processor if needed
                diff_renderer_path = os.path.join(hy3dpaint_path, 'DifferentiableRenderer')
                if diff_renderer_path not in sys.path:
                    sys.path.insert(0, diff_renderer_path)
                
                mesh_inpaint_cpp = os.path.join(diff_renderer_path, 'mesh_inpaint_processor.cpp')
                if os.path.exists(mesh_inpaint_cpp):
                    # Check if mesh_inpaint_processor is already compiled
                    import glob
                    inpaint_so_files = glob.glob(os.path.join(diff_renderer_path, 'mesh_inpaint_processor*.so'))
                    
                    if not inpaint_so_files:
                        logger.info("🔧 Compiling mesh_inpaint_processor for texture inpainting...")
                        try:
                            import subprocess
                            # Install pybind11 if needed
                            subprocess.run([sys.executable, "-m", "pip", "install", "pybind11"], check=True)
                            
                            # Change to DifferentiableRenderer directory
                            cwd = os.getcwd()
                            os.chdir(diff_renderer_path)
                            
                            # Compile the mesh_inpaint_processor
                            compile_cmd = f"c++ -O3 -Wall -shared -std=c++11 -fPIC `{sys.executable} -m pybind11 --includes` mesh_inpaint_processor.cpp -o mesh_inpaint_processor`{sys.executable}-config --extension-suffix`"
                            subprocess.run(compile_cmd, shell=True, check=True)
                            
                            # Return to original directory
                            os.chdir(cwd)
                            logger.info("✅ mesh_inpaint_processor compiled successfully")
                        except Exception as e:
                            logger.error(f"❌ Failed to compile mesh_inpaint_processor: {e}")
                    else:
                        logger.info(f"✅ mesh_inpaint_processor already compiled: {inpaint_so_files[0]}")
                
                # Try importing mesh_inpaint_processor to verify it's available
                try:
                    from mesh_inpaint_processor import meshVerticeInpaint
                    logger.info("✅ Mesh inpaint processor successfully loaded")
                except ImportError as e:
                    logger.error(f"❌ Mesh inpaint processor failed to import: {e}")
                    logger.error("This will cause texture generation to fail")
                except Exception as e:
                    logger.error(f"❌ Mesh inpaint processor error: {e}")
                
                from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
                
                try:
                    # First try with torch_dtype parameter
                    logger.info("Creating PBR-enabled configuration for texture generation...")
                    paint_config = Hunyuan3DPaintConfig(
                        max_num_view=HUNYUAN3D_PAINT_CONFIG_MAX_VIEWS,
                        resolution=HUNYUAN3D_PAINT_CONFIG_RESOLUTION,
                        torch_dtype=torch.float16,  # Use float16 for memory efficiency
                    )
                except TypeError as e:
                    # If torch_dtype parameter is not supported, retry without it
                    if "unexpected keyword argument 'torch_dtype'" in str(e) or "got an unexpected" in str(e):
                        logger.info("torch_dtype parameter not supported, using default configuration")
                        paint_config = Hunyuan3DPaintConfig(
                            max_num_view=HUNYUAN3D_PAINT_CONFIG_MAX_VIEWS,
                            resolution=HUNYUAN3D_PAINT_CONFIG_RESOLUTION,
                        )
                    else:
                        raise
                    
                # Configure paths
                #paint_config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
                import os
                paint_config.realesrgan_ckpt_path = os.getenv(
                 "HUNYUAN3D_REALESRGAN_CKPT_PATH",
                 "/mnt/persistant_data/spark/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
                )
                paint_config.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
                paint_config.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
                
                # Initialize pipeline
                logger.info("Initializing Hunyuan3D texture generation pipeline...")
                _hunyuan_texgen_worker = Hunyuan3DPaintPipeline(paint_config)
                
                # Explicitly move to GPU and ensure it stays there
                if hasattr(_hunyuan_texgen_worker, 'to'):
                    _hunyuan_texgen_worker = _hunyuan_texgen_worker.to(device)
                logger.info(f"✓ PBR texture generation model loaded successfully to {device}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load texture generation model: {e}")
                _hunyuan_texgen_worker = None
        else:
            logger.info("⏭️  Texture generation model not configured (optional)")
        
        _models_initialized = True
        logger.info("🎉 Hunyuan3D-2.1 processors initialized successfully!")
        return True
        
    except Exception as e:
        _initialization_error = str(e)
        logger.error(f"❌ Failed to initialize Hunyuan3D-2.1 processors: {e}")
        return False


# Memory monitoring function
def _log_memory_usage(stage=""):
    """Log current RAM and GPU memory usage."""
    try:
        import psutil
        # RAM usage
        ram_usage = psutil.virtual_memory()
        ram_used_gb = ram_usage.used / (1024**3)
        ram_total_gb = ram_usage.total / (1024**3)
        
        # GPU usage
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"📊 Memory usage {stage}:")
            logger.info(f"   RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB ({ram_usage.percent:.1f}%)")
            logger.info(f"   GPU: {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")
        else:
            logger.info(f"📊 RAM usage {stage}: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB ({ram_usage.percent:.1f}%)")
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")

def get_model_info():
    """Get information about loaded models."""
    info = {
        "initialized": _models_initialized,
        "shape_model": _hunyuan_i23d_worker is not None,
        "background_remover": _hunyuan_rembg_worker is not None,
        "texture_model": _hunyuan_texgen_worker is not None,
        "dependencies_available": _dependencies_available,
        "initialization_error": _initialization_error
    }
    
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["gpu_name"] = torch.cuda.get_device_name()
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        info["cuda_available"] = False
    
    return info


def cleanup_models():
    """Cleanup loaded models to free memory."""
    global _hunyuan_i23d_worker, _hunyuan_rembg_worker, _hunyuan_texgen_worker
    global _models_initialized
    
    logger.info("🧹 Cleaning up Hunyuan3D models...")
    
    if _hunyuan_i23d_worker is not None:
        del _hunyuan_i23d_worker
        _hunyuan_i23d_worker = None
    
    if _hunyuan_rembg_worker is not None:
        del _hunyuan_rembg_worker
        _hunyuan_rembg_worker = None
    
    if _hunyuan_texgen_worker is not None:
        del _hunyuan_texgen_worker
        _hunyuan_texgen_worker = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    _models_initialized = False
    logger.info("✅ Models cleaned up successfully")


def generate_3d_from_image_core(
    image_path: str,
    with_texture: bool = False,
    output_format: str = 'glb',
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Core 3D generation logic from an image using Hunyuan3D.
    
    Args:
        image_path: Path to the input image
        with_texture: Whether to generate texture for the model
        output_format: Format of the output model file ('glb', 'obj', 'ply', 'stl')
        progress_callback: Optional callback function for progress updates (progress, status)
        
    Returns:
        Dictionary with status, message, and file paths
    """
    logger.info(f"🎯 Starting 3D model generation from image: {image_path}")
    
    # Progress callback helper
    def update_progress(progress: float, status: str):
        if progress_callback:
            progress_callback(progress, status)
        logger.info(f"📊 Progress: {progress:.1f}% - {status}")
    
    update_progress(0, "Initializing...")
    
    # Check if dependencies are available
    if not _dependencies_available:
        error_msg = "Required dependencies not available. Please install PyTorch, transformers, diffusers."
        logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # Initialize processors if not already done
    update_progress(5, "Checking model initialization...")
    if not _models_initialized:
        logger.info("🔧 Initializing Hunyuan3D processors...")
        if not initialize_hunyuan3d_processors():
            error_msg = f"Failed to initialize Hunyuan3D processors: {_initialization_error}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
    
    # Additional safety check for the worker instance
    if _hunyuan_i23d_worker is None:
        error_msg = "Hunyuan3D shape generation model not properly initialized"
        logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    try:
        # Import config with fallbacks
        try:
            from config import (
                OUTPUT_3D_ASSETS_DIR, HUNYUAN3D_REMOVE_BACKGROUND, HUNYUAN3D_STEPS,
                HUNYUAN3D_GUIDANCE_SCALE, HUNYUAN3D_OCTREE_RESOLUTION, HUNYUAN3D_NUM_CHUNKS,
                HUNYUAN3D_ENABLE_FLASHVDM
            )
        except ImportError:
            logger.warning("⚠️  Config not available, using default values")
            OUTPUT_3D_ASSETS_DIR = "./generated_assets/3d_assets"
            HUNYUAN3D_REMOVE_BACKGROUND = True
            HUNYUAN3D_STEPS = 30
            HUNYUAN3D_GUIDANCE_SCALE = 7.5
            HUNYUAN3D_OCTREE_RESOLUTION = 256
            HUNYUAN3D_NUM_CHUNKS = 200000
            HUNYUAN3D_ENABLE_FLASHVDM = False
        
        # Create output directory
        unique_id = str(uuid.uuid4())[:8]
        model_dir = os.path.join(OUTPUT_3D_ASSETS_DIR, f"model_{unique_id}")
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"📁 Created output directory: {model_dir}")
        
        # Load and preprocess image
        update_progress(10, "Loading image...")
        try:
            image = PILImage.open(image_path).convert('RGB')
            logger.info(f"✓ Loaded image: {image.size}")
            
            # Validate image dimensions
            if image.size[0] < 50 or image.size[1] < 50:
                logger.warning(f"⚠️  Image is very small: {image.size}")
            
            if image.size[0] > 2048 or image.size[1] > 2048:
                logger.info(f"ℹ️  Resizing large image from {image.size} to max 1024x1024")
                image.thumbnail((1024, 1024), PILImage.Resampling.LANCZOS)
                logger.info(f"✓ Resized image to: {image.size}")
                
        except Exception as e:
            error_msg = f"Failed to load image: {e}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Remove background if enabled
        if HUNYUAN3D_REMOVE_BACKGROUND and _hunyuan_rembg_worker is not None:
            update_progress(20, "Removing background...")
            start_time = time.time()
            try:
                image = _hunyuan_rembg_worker(image)
                logger.info(f"✓ Background removal took {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.warning(f"⚠️  Background removal failed: {e}")
        
        update_progress(30, "Generating 3D shape...")
        
        # Generate 3D mesh from image
        start_time = time.time()
        
        # Prepare generation parameters
        generation_kwargs = {
            "num_inference_steps": HUNYUAN3D_STEPS,
            "guidance_scale": HUNYUAN3D_GUIDANCE_SCALE,
        }
        
        # Add optional parameters if supported
        try:
            generation_kwargs.update({
                "octree_resolution": HUNYUAN3D_OCTREE_RESOLUTION,
                "num_chunks": HUNYUAN3D_NUM_CHUNKS,
            })
            
            if HUNYUAN3D_ENABLE_FLASHVDM:
                generation_kwargs["enable_flashvdm"] = True
        except:
            logger.info("ℹ️  Using basic generation parameters")
        
        logger.info(f"🔥 Generating 3D model with parameters: {generation_kwargs}")
        
        # Generate the 3D model
        try:
            mesh_result = _hunyuan_i23d_worker(image, **generation_kwargs)
            shape_gen_time = time.time() - start_time
            logger.info(f"✓ Shape generation took {shape_gen_time:.2f}s")
        except Exception as e:
            error_msg = f"3D shape generation failed: {e}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        update_progress(60, "Processing mesh...")
        
        # Get the generated mesh
        try:
            logger.info(f"🔍 Analyzing mesh_result type: {type(mesh_result)}")
            logger.info(f"🔍 Mesh_result attributes: {dir(mesh_result)}")
            
            # Hunyuan3D pipeline returns a list of meshes, get the first one
            if isinstance(mesh_result, list) and len(mesh_result) > 0:
                mesh = mesh_result[0]
                logger.info(f"✓ Extracted mesh from list[0], type: {type(mesh)}")
            elif hasattr(mesh_result, 'meshes') and len(mesh_result.meshes) > 0:
                mesh = mesh_result.meshes[0]
                logger.info(f"✓ Extracted mesh from meshes[0], type: {type(mesh)}")
            else:
                mesh = mesh_result
                logger.info(f"✓ Using mesh_result directly, type: {type(mesh)}")
            
            # Check if mesh has vertices and faces
            if hasattr(mesh, 'vertices'):
                vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
                logger.info(f"🔍 Mesh has {len(vertices)} vertices")
            else:
                logger.warning("⚠️  Mesh has no vertices attribute")
                
            if hasattr(mesh, 'faces'):
                faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
                logger.info(f"🔍 Mesh has {len(faces)} faces")
            else:
                logger.warning("⚠️  Mesh has no faces attribute")
            
            # Additional debugging for mesh object
            logger.info(f"🔍 Mesh object type: {type(mesh)}")
            logger.info(f"🔍 Mesh object attributes: {[attr for attr in dir(mesh) if not attr.startswith('_')]}")
            
            logger.info("✓ Extracted mesh from generation result")
        except Exception as e:
            error_msg = f"Failed to extract mesh: {e}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Post-process the mesh
        update_progress(70, "Processing mesh...")
        start_time = time.time()
        
        # Apply mesh cleaning operations if available
        try:
            # Try to import mesh processing tools from 2.1
            from hy3dshape.postprocessors import MeshSimplifier, mesh_normalize
            
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                # Use built-in mesh cleaners from 2.1
                mesh_simplifier = MeshSimplifier()
                mesh = mesh_simplifier(mesh)
                mesh = mesh_normalize(mesh)
                
                logger.info("✓ Applied mesh cleaning operations")
        except ImportError:
            logger.info("ℹ️  Mesh cleaning tools not available, skipping")
        except Exception as e:
            logger.warning(f"⚠️  Mesh cleaning failed: {e}")
        
        process_time = time.time() - start_time
        logger.info(f"✓ Mesh processing took {process_time:.2f}s")
        
        # Convert to trimesh for export
        update_progress(80, "Converting mesh format...")
        try:
            import trimesh
            
            if not isinstance(mesh, trimesh.Trimesh):
                # Convert from Hunyuan3D mesh format to trimesh
                if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                    vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
                    faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
                    
                    # Validate mesh data
                    if len(vertices) == 0:
                        error_msg = "Invalid mesh: no vertices found"
                        logger.error(f"❌ {error_msg}")
                        return {"status": "error", "message": error_msg}
                    
                    if len(faces) == 0:
                        error_msg = "Invalid mesh: no faces found"
                        logger.error(f"❌ {error_msg}")
                        return {"status": "error", "message": error_msg}
                    
                    logger.info(f"✓ Creating trimesh with {len(vertices)} vertices and {len(faces)} faces")
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    logger.info("✓ Converted to trimesh format")
                else:
                    error_msg = f"Invalid mesh format generated - missing vertices/faces attributes. Mesh type: {type(mesh)}, attributes: {dir(mesh)}"
                    logger.error(f"❌ {error_msg}")
                    return {"status": "error", "message": error_msg}
            else:
                logger.info("✓ Mesh is already in trimesh format")
            
        except ImportError:
            error_msg = "Trimesh library not available for mesh export"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Mesh conversion failed: {e}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Generate texture if requested
        texture_path = None
        texture_gen_time = None
        if with_texture and _hunyuan_texgen_worker is not None:
            update_progress(85, "Generating texture...")
            start_time = time.time()
            try:
                # Generate texture using the texture generation pipeline
                textured_mesh = _hunyuan_texgen_worker(image, mesh)
                mesh = textured_mesh
                texture_gen_time = time.time() - start_time
                logger.info(f"✓ Texture generation took {texture_gen_time:.2f}s")
            except Exception as e:
                logger.warning(f"⚠️  Texture generation failed: {e}")
                with_texture = False  # Fall back to non-textured
        elif with_texture:
            logger.warning("⚠️  Texture generation requested but texture model not available")
        
        update_progress(90, "Exporting model...")
        
        # Export the mesh
        output_filename = f"model_{unique_id}.{output_format}"
        output_path = os.path.join(model_dir, output_filename)
        
        try:
            mesh.export(output_path)
            logger.info(f"✓ Model exported to: {output_path}")
        except Exception as e:
            error_msg = f"Failed to export mesh: {e}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        update_progress(95, "Finalizing...")
        
        # Generate metadata
        metadata = {
            "model_id": unique_id,
            "input_image": image_path,
            "output_format": output_format,
            "with_texture": with_texture,
            "generation_time": shape_gen_time,
            "processing_time": process_time,
            "texture_time": texture_gen_time,
            "total_vertices": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
            "total_faces": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
        }
        
        # Save metadata
        metadata_path = os.path.join(model_dir, f"metadata_{unique_id}.json")
        try:
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save metadata: {e}")
        
        update_progress(100, "Completed successfully!")
        
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        logger.info(f"🎉 3D model generation completed successfully in {total_time:.2f}s")
        
        result = {
            "status": "success",
            "message": "3D model generated successfully",
            "model_path": output_path,
            "metadata_path": metadata_path,
            "model_id": unique_id,
            "generation_stats": metadata
        }
        
        return result
        
        # Also save as GLB for web viewing
        if output_format != 'glb':
            glb_path = os.path.join(model_dir, f"model_{unique_id}.glb")
            try:
                mesh.export(glb_path)
            except Exception as e:
                logger.warning(f"Failed to export GLB version: {e}")
                glb_path = None
        else:
            glb_path = output_path
        
        update_progress(100, "Complete!")
        
        # Prepare result
        result = {
            "status": "success",
            "message": f"Successfully generated 3D model from image",
            "model_path": output_path,
            "model_dir": model_dir,
            "output_format": output_format,
            "with_texture": with_texture,
            "processing_time": {
                "shape_generation": shape_gen_time,
                "mesh_processing": process_time,
            }
        }
        
        if glb_path and glb_path != output_path:
            result["glb_path"] = glb_path
        if with_texture and texture_gen_time is not None:
            result["processing_time"]["texture_generation"] = texture_gen_time
        
        logger.info(f"3D generation completed successfully: {output_path}")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error during 3D generation: {str(e)}"
        logger.error(f"💥 {error_msg}", exc_info=True)
        return {"status": "error", "message": error_msg}


def _load_hunyuan3d_direct_to_gpu(device):
    """
    Load Hunyuan3D model directly to GPU to avoid using system RAM.
    This bypasses the default CPU loading in the original implementation.
    """
    import yaml
    import torch
    import os
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.utils.utils import instantiate_from_config
    
    # Get model paths
    model_path = os.path.expanduser(HUNYUAN3D_MODEL_PATH)
    config_path = os.path.join(model_path, HUNYUAN3D_SUBFOLDER, 'config.yaml')
    ckpt_path = os.path.join(model_path, HUNYUAN3D_SUBFOLDER, 'model.fp16.ckpt')
    
    logger.info(f"Loading config from: {config_path}")
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint directly to GPU
    logger.info(f"📥 Loading checkpoint directly to {device}...")
    _log_memory_usage("before checkpoint loading")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    _log_memory_usage("after checkpoint loaded to GPU")
    
    # Create model components and load directly to GPU
    logger.info("🏗️ Creating model components...")
    
    # Create model
    model = instantiate_from_config(config['model'])
    model.load_state_dict(ckpt['model'])
    model.to(device, dtype=torch.float16)
    
    # Create VAE
    vae = instantiate_from_config(config['vae'])
    vae.load_state_dict(ckpt['vae'], strict=False)
    vae.to(device, dtype=torch.float16)
    
    # Create conditioner
    conditioner = instantiate_from_config(config['conditioner'])
    if 'conditioner' in ckpt:
        conditioner.load_state_dict(ckpt['conditioner'])
    conditioner.to(device, dtype=torch.float16)
    
    # Create other components
    image_processor = instantiate_from_config(config['image_processor'])
    scheduler = instantiate_from_config(config['scheduler'])
    
    # Free checkpoint memory
    del ckpt
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    _log_memory_usage("after checkpoint freed")
    
    # Create pipeline
    pipeline = Hunyuan3DDiTFlowMatchingPipeline(
        vae=vae,
        model=model,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
        device=device,
        dtype=torch.float16,
    )
    
    # Verify all components are on GPU
    vae_device = next(pipeline.vae.parameters()).device
    model_device = next(pipeline.model.parameters()).device
    conditioner_device = next(pipeline.conditioner.parameters()).device
    
    logger.info(f"✅ Model loaded directly to GPU!")
    logger.info(f"📍 Component locations - VAE: {vae_device}, Model: {model_device}, Conditioner: {conditioner_device}")
    
    return pipeline


def _load_hunyuan3d_directly_to_gpu(model_path, subfolder, device='cuda', dtype=torch.float16):
    """
    Custom loading function that loads the Hunyuan3D model directly to GPU
    to avoid using system RAM during the loading process.
    """
    import yaml
    import os
    from hy3dshape.utils.utils import smart_load_model
    from hy3dshape.utils.misc import instantiate_from_config
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    
    logger.info(f"🚀 Loading Hunyuan3D model directly to GPU: {device}")
    
    # Get config and checkpoint paths
    config_path, ckpt_path = smart_load_model(
        model_path, subfolder=subfolder, use_safetensors=False, variant='fp16'
    )
    
    # Expand the model path properly
    base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
    full_model_path = os.path.expanduser(os.path.join(base_dir, model_path))
    config_path = os.path.join(full_model_path, subfolder, 'config.yaml')
    ckpt_path = os.path.join(full_model_path, subfolder, 'model.fp16.ckpt')
    
    logger.info(f"📁 Config path: {config_path}")
    logger.info(f"📁 Checkpoint path: {ckpt_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint directly to GPU
    logger.info(f"⬇️  Loading checkpoint directly to {device}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Initialize models directly on GPU
    logger.info("🏗️  Creating model components...")
    model = instantiate_from_config(config['model']).to(device, dtype=dtype)
    model.load_state_dict(ckpt['model'])
    
    vae = instantiate_from_config(config['vae']).to(device, dtype=dtype)
    vae.load_state_dict(ckpt['vae'], strict=False)
    
    conditioner = instantiate_from_config(config['conditioner']).to(device, dtype=dtype)
    if 'conditioner' in ckpt:
        conditioner.load_state_dict(ckpt['conditioner'])
    
    image_processor = instantiate_from_config(config['image_processor'])
    scheduler = instantiate_from_config(config['scheduler'])
    
    # Create pipeline with required kwargs
    pipeline = Hunyuan3DDiTFlowMatchingPipeline(
        vae=vae,
        model=model,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
        device=device,
        dtype=dtype,
        from_pretrained_kwargs=dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=False,
            variant='fp16',
            dtype=dtype,
            device=device,
        )
    )
    
    logger.info("✅ Model loaded directly to GPU successfully")
    return pipeline


def get_model_info() -> Dict[str, Any]:
    """
    Get information about loaded models.
    
    Returns:
        Dictionary with model status and information
    """
    return {
        "models_initialized": _models_initialized,
        "initialization_error": _initialization_error,
        "shape_gen_loaded": _hunyuan_i23d_worker is not None,
        "rembg_loaded": _hunyuan_rembg_worker is not None,
        "texgen_loaded": _hunyuan_texgen_worker is not None,
        "device": getattr(_hunyuan_i23d_worker, 'device', None) if _hunyuan_i23d_worker else None
    }


def cleanup_models():
    """
    Clean up loaded models to free memory.
    """
    global _hunyuan_i23d_worker, _hunyuan_rembg_worker, _hunyuan_texgen_worker
    global _models_initialized
    
    try:
        if _hunyuan_i23d_worker is not None:
            del _hunyuan_i23d_worker
            _hunyuan_i23d_worker = None
        
        if _hunyuan_rembg_worker is not None:
            del _hunyuan_rembg_worker
            _hunyuan_rembg_worker = None
        
        if _hunyuan_texgen_worker is not None:
            del _hunyuan_texgen_worker
            _hunyuan_texgen_worker = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        _models_initialized = False
        logger.info("Hunyuan3D models cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")
