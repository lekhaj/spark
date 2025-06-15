"""
Hunyuan3D Worker Module

This module handles the initialization and core processing logic for Hunyuan3D models.
It's designed to be imported by the Celery worker tasks for 3D model generation.
"""

import os
import logging
import time
import uuid
from typing import Optional, Dict, Any, Callable
from PIL import Image as PILImage
import torch

# Setup logging
logger = logging.getLogger(__name__)

# Global worker instances (lazy loaded)
_hunyuan_i23d_worker = None
_hunyuan_rembg_worker = None
_hunyuan_texgen_worker = None

# Model loading flags
_models_initialized = False
_initialization_error = None


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
        return True
    
    if _initialization_error:
        logger.error(f"Previous initialization failed: {_initialization_error}")
        return False
    
    try:
        # Import Hunyuan3D modules
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        
        # Import config
        from config import (
            HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
            HUNYUAN3D_COMPILE, HUNYUAN3D_LOW_VRAM_MODE, HUNYUAN3D_DEVICE
        )
        
        logger.info("Initializing Hunyuan3D processors...")
        
        # Initialize 3D shape generation pipeline
        logger.info(f"Loading Hunyuan3D shape generation model from: {HUNYUAN3D_MODEL_PATH}")
        _hunyuan_i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN3D_MODEL_PATH,
            subfolder=HUNYUAN3D_SUBFOLDER,
            torch_dtype=torch.float16 if HUNYUAN3D_LOW_VRAM_MODE else torch.float32,
        )
        
        # Move to GPU if available
        device = HUNYUAN3D_DEVICE if torch.cuda.is_available() else "cpu"
        _hunyuan_i23d_worker = _hunyuan_i23d_worker.to(device)
        
        if HUNYUAN3D_COMPILE and hasattr(_hunyuan_i23d_worker, 'compile'):
            logger.info("Compiling Hunyuan3D model for optimization...")
            _hunyuan_i23d_worker.compile()
        
        # Initialize background remover
        logger.info("Loading background removal model...")
        _hunyuan_rembg_worker = BackgroundRemover()
        
        # Initialize texture generation pipeline (optional)
        if HUNYUAN3D_TEXGEN_MODEL_PATH:
            logger.info(f"Loading texture generation model from: {HUNYUAN3D_TEXGEN_MODEL_PATH}")
            _hunyuan_texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
                HUNYUAN3D_TEXGEN_MODEL_PATH,
            )
            if hasattr(_hunyuan_texgen_worker, 'to'):
                _hunyuan_texgen_worker = _hunyuan_texgen_worker.to(device)
        
        _models_initialized = True
        logger.info("Hunyuan3D processors initialized successfully!")
        return True
        
    except Exception as e:
        _initialization_error = str(e)
        logger.error(f"Failed to initialize Hunyuan3D processors: {e}", exc_info=True)
        return False


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
    if not _models_initialized:
        if not initialize_hunyuan3d_processors():
            return {"status": "error", "message": "Failed to initialize Hunyuan3D processors"}
    
    # Additional safety check for the worker instance
    if _hunyuan_i23d_worker is None:
        return {"status": "error", "message": "Hunyuan3D shape generation model not properly initialized"}
    
    try:
        # Import required modules
        from config import (
            OUTPUT_3D_ASSETS_DIR, HUNYUAN3D_REMOVE_BACKGROUND, HUNYUAN3D_STEPS,
            HUNYUAN3D_GUIDANCE_SCALE, HUNYUAN3D_OCTREE_RESOLUTION, HUNYUAN3D_NUM_CHUNKS,
            HUNYUAN3D_ENABLE_FLASHVDM
        )
        from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        import trimesh
        
        # Progress callback helper
        def update_progress(progress: float, status: str):
            if progress_callback:
                progress_callback(progress, status)
            logger.info(f"Progress: {progress:.1f}% - {status}")
        
        update_progress(0, "Starting 3D generation...")
        
        # Create output directory
        unique_id = str(uuid.uuid4())[:8]
        model_dir = os.path.join(OUTPUT_3D_ASSETS_DIR, f"model_{unique_id}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Load and preprocess image
        update_progress(10, "Loading image...")
        try:
            image = PILImage.open(image_path).convert('RGB')
        except Exception as e:
            return {"status": "error", "message": f"Failed to load image: {e}"}
        
        # Remove background if enabled
        if HUNYUAN3D_REMOVE_BACKGROUND and _hunyuan_rembg_worker is not None:
            update_progress(20, "Removing background...")
            start_time = time.time()
            image = _hunyuan_rembg_worker(image)
            logger.info(f"Background removal took {time.time() - start_time:.2f}s")
        
        update_progress(30, "Generating 3D shape...")
        
        # Generate 3D mesh from image
        start_time = time.time()
        
        # Prepare generation parameters
        generation_kwargs = {
            "num_inference_steps": HUNYUAN3D_STEPS,
            "guidance_scale": HUNYUAN3D_GUIDANCE_SCALE,
            "octree_resolution": HUNYUAN3D_OCTREE_RESOLUTION,
            "num_chunks": HUNYUAN3D_NUM_CHUNKS,
        }
        
        if HUNYUAN3D_ENABLE_FLASHVDM:
            generation_kwargs["enable_flashvdm"] = True
        
        # Generate the 3D model
        mesh_result = _hunyuan_i23d_worker(
            image,
            **generation_kwargs
        )
        
        shape_gen_time = time.time() - start_time
        logger.info(f"Shape generation took {shape_gen_time:.2f}s")
        
        update_progress(60, "Processing mesh...")
        
        # Get the generated mesh
        if hasattr(mesh_result, 'meshes') and len(mesh_result.meshes) > 0:
            mesh = mesh_result.meshes[0]
        else:
            mesh = mesh_result
        
        # Post-process the mesh
        start_time = time.time()
        
        # Apply mesh cleaning operations
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            # Use built-in mesh cleaners
            face_reducer = FaceReducer()
            floater_remover = FloaterRemover()
            degenerate_remover = DegenerateFaceRemover()
            
            mesh = face_reducer(mesh)
            mesh = floater_remover(mesh)
            mesh = degenerate_remover(mesh)
        
        process_time = time.time() - start_time
        logger.info(f"Mesh processing took {process_time:.2f}s")
        
        # Convert to trimesh for export
        if not isinstance(mesh, trimesh.Trimesh):
            # Convert from Hunyuan3D mesh format to trimesh
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
                faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            else:
                return {"status": "error", "message": "Invalid mesh format generated"}
        
        update_progress(80, "Generating texture..." if with_texture else "Saving model...")
        
        # Generate texture if requested
        texture_path = None
        texture_gen_time = None
        if with_texture and _hunyuan_texgen_worker is not None:
            start_time = time.time()
            try:
                # Generate texture using the texture generation pipeline
                textured_mesh = _hunyuan_texgen_worker(
                    image,
                    mesh
                )
                mesh = textured_mesh
                texture_gen_time = time.time() - start_time
                logger.info(f"Texture generation took {texture_gen_time:.2f}s")
            except Exception as e:
                logger.warning(f"Texture generation failed: {e}")
                with_texture = False  # Fall back to non-textured
        
        update_progress(90, "Exporting model...")
        
        # Export the mesh
        output_filename = f"model_{unique_id}.{output_format}"
        output_path = os.path.join(model_dir, output_filename)
        
        try:
            mesh.export(output_path)
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
