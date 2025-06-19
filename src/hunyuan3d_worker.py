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

# Fix Python path for Hunyuan3D imports
def _fix_python_path():
    """Add Hunyuan3D directory to Python path if needed."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    hunyuan_dirs = [
        os.path.join(project_root, "Hunyuan3D-2"),
        os.path.join(project_root, "Hunyuan3D-1"),
        os.path.join(project_root, "models", "Hunyuan3D-1"),
        os.path.join(project_root, "models", "Hunyuan3D-2")
    ]
    
    for hunyuan_dir in hunyuan_dirs:
        if os.path.exists(hunyuan_dir):
            if hunyuan_dir not in sys.path:
                sys.path.insert(0, hunyuan_dir)
                print(f"Added {hunyuan_dir} to Python path")
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


def initialize_hunyuan3d_processors():
    """
    Initialize Hunyuan3D processors for 3D generation.
    This function should be called once per worker process.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _hunyuan_i23d_worker, _hunyuan_rembg_worker, _hunyuan_texgen_worker
    global _models_initialized, _initialization_error
    
    if _models_initialized:
        logger.info("✅ Hunyuan3D processors already initialized")
        return True
    
    if _initialization_error:
        logger.error(f"❌ Previous initialization failed: {_initialization_error}")
        return False
    
    if not _dependencies_available:
        _initialization_error = "Required dependencies not available"
        logger.error(f"❌ {_initialization_error}")
        return False
    
    try:
        logger.info("🚀 Starting Hunyuan3D processor initialization...")
        
        # Try to import Hunyuan3D modules
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.rembg import BackgroundRemover
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            logger.info("✓ Hunyuan3D modules imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import Hunyuan3D modules: {e}")
            logger.error("   Please ensure Hunyuan3D is properly installed")
            logger.error("   Check if you have the Hunyuan3D-2 directory with hy3dgen module")
            _initialization_error = f"Hunyuan3D modules not found: {e}"
            return False
        
        # Try to import config
        try:
            from config import (
                HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
                HUNYUAN3D_COMPILE, HUNYUAN3D_LOW_VRAM_MODE, HUNYUAN3D_DEVICE
            )
            logger.info("✓ Configuration loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️  Config import failed, using defaults: {e}")
            # Set default values
            HUNYUAN3D_MODEL_PATH = "Tencent/Hunyuan3D-1"
            HUNYUAN3D_SUBFOLDER = "lite"
            HUNYUAN3D_TEXGEN_MODEL_PATH = None
            HUNYUAN3D_COMPILE = False
            HUNYUAN3D_LOW_VRAM_MODE = True
            HUNYUAN3D_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("🔧 Initializing Hunyuan3D processors...")
        
        # Determine device
        device = HUNYUAN3D_DEVICE if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 Using device: {device}")
        
        # Initialize 3D shape generation pipeline
        logger.info(f"📦 Loading Hunyuan3D shape generation model from: {HUNYUAN3D_MODEL_PATH}")
        try:
            _hunyuan_i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN3D_MODEL_PATH,
                subfolder=HUNYUAN3D_SUBFOLDER,
                torch_dtype=torch.float16 if HUNYUAN3D_LOW_VRAM_MODE else torch.float32,
            )
            
            # Move to GPU if available
            _hunyuan_i23d_worker = _hunyuan_i23d_worker.to(device)
            logger.info("✓ Shape generation model loaded successfully")
            
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
            _hunyuan_rembg_worker = BackgroundRemover()
            logger.info("✓ Background removal model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load background removal model: {e}")
            _hunyuan_rembg_worker = None
        
        # Initialize texture generation pipeline (optional)
        if HUNYUAN3D_TEXGEN_MODEL_PATH:
            logger.info(f"🎨 Loading texture generation model from: {HUNYUAN3D_TEXGEN_MODEL_PATH}")
            try:
                _hunyuan_texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
                    HUNYUAN3D_TEXGEN_MODEL_PATH,
                )
                if hasattr(_hunyuan_texgen_worker, 'to'):
                    _hunyuan_texgen_worker = _hunyuan_texgen_worker.to(device)
                logger.info("✓ Texture generation model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load texture generation model: {e}")
                _hunyuan_texgen_worker = None
        else:
            logger.info("⏭️  Texture generation model not configured (optional)")
        
        _models_initialized = True
        logger.info("🎉 Hunyuan3D processors initialized successfully!")
        return True
        
    except Exception as e:
        _initialization_error = str(e)
        logger.error(f"💥 Failed to initialize Hunyuan3D processors: {e}", exc_info=True)
        return False


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
            if hasattr(mesh_result, 'meshes') and len(mesh_result.meshes) > 0:
                mesh = mesh_result.meshes[0]
            else:
                mesh = mesh_result
            
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
            # Try to import mesh processing tools
            from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
            
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                # Use built-in mesh cleaners
                face_reducer = FaceReducer()
                floater_remover = FloaterRemover()
                degenerate_remover = DegenerateFaceRemover()
                
                mesh = face_reducer(mesh)
                mesh = floater_remover(mesh)
                mesh = degenerate_remover(mesh)
                
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
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    logger.info("✓ Converted to trimesh format")
                else:
                    error_msg = "Invalid mesh format generated - no vertices/faces found"
                    logger.error(f"❌ {error_msg}")
                    return {"status": "error", "message": error_msg}
            
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
        
    except Exception as e:
        error_msg = f"Unexpected error during 3D generation: {str(e)}"
        logger.error(f"💥 {error_msg}", exc_info=True)
        return {"status": "error", "message": error_msg}
        except Exception as e:
            return {"status": "error", "message": f"Failed to export mesh: {e}"}
        
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
            result["processing_time"]["texture_generation"] = texture_gen_time
        
        logger.info(f"3D generation completed successfully: {output_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error in 3D generation core: {e}", exc_info=True)
        return {"status": "error", "message": f"3D generation failed: {e}"}


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
