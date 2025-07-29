# src/tasks.py
# Updated for Hunyuan3D-2.1 compatibility with S3 integration
import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import multiprocessing as mp
mp.set_start_method('spawn', force=True)


import os
import sys
import logging
from celery import Celery
import numpy as np
from PIL import Image
from datetime import datetime
import pymongo
from celery.signals import worker_process_init
from celery.utils.log import get_task_logger
import uuid
import tempfile
import requests
import shutil
import time





# Ensure current directory is in Python path for local module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Initialize task logger
task_logger = get_task_logger(__name__)



# Default implementations for missing modules
class DummyProcessor:
    def __init__(self, model_type=None):
        self.model_type = model_type or "dummy"
    
class DummyPipeline:
    def __init__(self, text_processor, grid_processor):
        self.text_processor = text_processor
        self.grid_processor = grid_processor
    
    def process_text(self, prompt):
        return []
    
    def process_grid(self, grid_string):
        return [], None

class DummyMongoHelper:
    def find_many(self, db_name, collection_name, query=None, limit=0):
        return []
    
    def update_by_id(self, db_name, collection_name, doc_id, update):
        return 0

class DummyImage:
    def save(self, path):
        pass

def dummy_create_image_grid(images):
    return DummyImage()

def dummy_save_image(image, path):
    pass

def dummy_generate_biome(theme, structure_list):
    return "Biome generation not available"

def dummy_get_aws_manager():
    return None

def dummy_generate_3d_from_image_core(image_path, with_texture=False, output_format='glb', progress_callback=None):
    return {"status": "error", "message": "Hunyuan3D modules not loaded"}

def dummy_initialize_hunyuan3d_processors():
    return False

# Initialize defaults
TextProcessor = DummyProcessor
GridProcessor = DummyProcessor
Pipeline = DummyPipeline
save_image = dummy_save_image
create_image_grid = dummy_create_image_grid
MongoDBHelper = DummyMongoHelper
get_biome_names = lambda: []
fetch_biome = lambda db, col, name: None
generate_biome = dummy_generate_biome
get_aws_manager = dummy_get_aws_manager
initialize_hunyuan3d_processors = dummy_initialize_hunyuan3d_processors
generate_3d_from_image_core = dummy_generate_3d_from_image_core

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
hunyuan_shape_path = os.path.join(project_root, "Hunyuan3D-2.1", "hy3dshape")
hunyuan_paint_path = os.path.join(project_root, "Hunyuan3D-2.1", "hy3dpaint")

if os.path.exists(hunyuan_shape_path):
    sys.path.insert(0, hunyuan_shape_path)
if os.path.exists(hunyuan_paint_path):
    sys.path.insert(0, hunyuan_paint_path)

# Try importing the specific Hunyuan3D 2.1 modules
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.postprocessors import MeshSimplifier, mesh_normalize
from hy3dshape.rembg import BackgroundRemover
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
import trimesh

task_logger.info("✓ Hunyuan3D-2.1 core modules imported successfully")

# Try importing the worker functions
from hunyuan3d_worker import (
    initialize_hunyuan3d_processors, 
    generate_3d_from_image_core,
    get_model_info,
    cleanup_models
)

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
task_logger = logging.getLogger('celery_tasks')

# Initialize default values
DEFAULT_TEXT_MODEL = "stability"
DEFAULT_GRID_MODEL = "stability"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_3D_ASSETS_DIR = os.path.join(OUTPUT_DIR, "3d_assets")
MONGO_DB_NAME = "World_builder"
MONGO_BIOME_COLLECTION = "biomes"


# Try to import real modules
try:
    from config import (
        DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL,
        OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_3D_ASSETS_DIR,
        MONGO_DB_NAME, MONGO_BIOME_COLLECTION,
        HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
        HUNYUAN3D_PAINT_CONFIG_MAX_VIEWS, HUNYUAN3D_PAINT_CONFIG_RESOLUTION,
        HUNYUAN3D_REALESRGAN_CKPT_PATH, HUNYUAN3D_MULTIVIEW_CFG_PATH, HUNYUAN3D_CUSTOM_PIPELINE_PATH,
        HUNYUAN3D_APPLY_TORCHVISION_FIX,
        HUNYUAN3D_STEPS, HUNYUAN3D_GUIDANCE_SCALE, HUNYUAN3D_OCTREE_RESOLUTION,
        HUNYUAN3D_REMOVE_BACKGROUND, HUNYUAN3D_NUM_CHUNKS, HUNYUAN3D_ENABLE_FLASHVDM,
        HUNYUAN3D_COMPILE, HUNYUAN3D_LOW_VRAM_MODE, HUNYUAN3D_DEVICE,
        AWS_REGION, AWS_GPU_INSTANCE_ID, AWS_MAX_STARTUP_WAIT_TIME, AWS_EC2_CHECK_INTERVAL,
        CELERY_TASK_ROUTES, TASK_TIMEOUT_3D_GENERATION, TASK_TIMEOUT_2D_GENERATION, TASK_TIMEOUT_EC2_MANAGEMENT
    )
    from pipeline.text_processor import TextProcessor
    from pipeline.grid_processor import GridProcessor
    from pipeline.pipeline import Pipeline
    from utils.image_utils import save_image, create_image_grid
    from db_helper import MongoDBHelper
    from text_grid.structure_registry import get_biome_names, fetch_biome
    from text_grid.grid_generator import generate_biome
    from aws_manager import get_aws_manager
    
    TASK_2D_MODULES_LOADED = True
    task_logger.info("All necessary 2D pipeline task modules loaded for Celery worker.")
except ImportError as e:
    task_logger.error(f"Could not load all 2D task modules for Celery worker: {e}")
    TASK_2D_MODULES_LOADED = False

# Try to import Hunyuan3D modules
_hunyuan_i23d_worker = None
_hunyuan_rembg_worker = None
_hunyuan_texgen_worker = None

# Global flag for Hunyuan3D availability
TASK_3D_MODULES_LOADED = False

# Step 1: Test basic PyTorch and CUDA
try:
    import torch
    task_logger.info(f"✓ PyTorch loaded: {torch.__version__}")
    task_logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        task_logger.info(f"✓ GPU device: {torch.cuda.get_device_name()}")
        task_logger.info(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        task_logger.warning("⚠ CUDA not available, will use CPU")
        
except ImportError as e:
    task_logger.error(f"✗ PyTorch import failed: {e}")
    torch = None

# Step 2: Test Transformers and Diffusers
try:
    if torch is not None:
        from transformers import AutoTokenizer, AutoModel
        import diffusers
        task_logger.info(f"✓ Transformers loaded: {diffusers.__version__}")
        task_logger.info("✓ Diffusers loaded successfully")
except ImportError as e:
    task_logger.error(f"✗ Transformers/Diffusers import failed: {e}")

# Step 3: Try to import Hunyuan3D modules
try:
    if torch is not None:
        # Add Hunyuan3D-2.1 paths to sys.path for proper imports
        # import sys
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # project_root = os.path.dirname(current_dir)
        # hunyuan_shape_path = os.path.join(project_root, "Hunyuan3D-2.1", "hy3dshape")
        # hunyuan_paint_path = os.path.join(project_root, "Hunyuan3D-2.1", "hy3dpaint")
        
        # if os.path.exists(hunyuan_shape_path):
        #     sys.path.insert(0, hunyuan_shape_path)
        # if os.path.exists(hunyuan_paint_path):
        #     sys.path.insert(0, hunyuan_paint_path)
        
        # # Try importing the specific Hunyuan3D 2.1 modules
        # from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        # from hy3dshape.postprocessors import MeshSimplifier, mesh_normalize
        # from hy3dshape.rembg import BackgroundRemover
        # from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        # import trimesh
        
        # task_logger.info("✓ Hunyuan3D-2.1 core modules imported successfully")
        
        # # Try importing the worker functions
        # from hunyuan3d_worker import (
        #     initialize_hunyuan3d_processors, 
        #     generate_3d_from_image_core,
        #     get_model_info,
        #     cleanup_models
        # )
        
        task_logger.info("✓ Hunyuan3D-2.1 worker functions imported successfully")
        
        TASK_3D_MODULES_LOADED = True
        task_logger.info("✅ All necessary Hunyuan3D-2.1 modules loaded for Celery worker.")
        
except ImportError as e:
    task_logger.error(f"✗ Could not load Hunyuan3D-2.1 modules for Celery worker: {e}")
    task_logger.error("   Make sure you have installed: transformers, diffusers, torch with CUDA support, and Hunyuan3D-2.1")
    TASK_3D_MODULES_LOADED = False
except Exception as e:
    task_logger.error(f"✗ Unexpected error loading Hunyuan3D-2.1 modules: {e}")
    TASK_3D_MODULES_LOADED = False

# Try to import SDXL Turbo worker
TASK_SDXL_MODULES_LOADED = False
_sdxl_worker = None

try:
    from sdxl_turbo_worker import SDXLTurboWorker, get_sdxl_worker
    TASK_SDXL_MODULES_LOADED = True
    task_logger.info("✅ SDXL Turbo worker modules loaded successfully")
except ImportError as e:
    task_logger.error(f"✗ Failed to load SDXL Turbo worker modules: {e}")
    TASK_SDXL_MODULES_LOADED = False

# Create function aliases
_generate_3d_from_image_core = generate_3d_from_image_core

# Overall flag for task modules
TASK_MODULES_LOADED = TASK_2D_MODULES_LOADED

# Celery App Setup
try:
    from config import REDIS_BROKER_URL, REDIS_RESULT_BACKEND, CELERY_TASK_ROUTES, REDIS_CONFIG
    
    # Test Redis connections before using them
    redis_test = REDIS_CONFIG.test_connection()
    task_logger.info(f"Redis connection test results: {redis_test}")
    
    # Use the Redis configuration
    broker_url = REDIS_BROKER_URL
    result_backend = REDIS_RESULT_BACKEND
    task_routes = CELERY_TASK_ROUTES
    
    # Log configuration
    task_logger.info(f"Using Redis broker: {broker_url}")
    task_logger.info(f"Using Redis result backend: {result_backend}")
    task_logger.info(f"Worker type: {getattr(REDIS_CONFIG, 'worker_type', 'unknown')}")
    
except ImportError as e:
    task_logger.warning(f"Could not import Redis config: {e}")
    broker_url = os.getenv('REDIS_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.getenv('REDIS_RESULT_BACKEND', 'redis://localhost:6379/0')
    task_routes = {}

app = Celery('gpu_tasks', broker=broker_url, backend=result_backend)

# Configure task routing if available
if task_routes:
    app.conf.task_routes = task_routes

# Configure task timeouts
app.conf.update(
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
)

# Global variables for processors
_text_processor = None
_grid_processor = None
_pipeline = None

# Global variables for Hunyuan3D processors
_hunyuan_i23d_worker = None
_hunyuan_rembg_worker = None
_hunyuan_texgen_worker = None
_hunyuan_mesh_simplifier = None

# Global variable for local image generation model
_local_image_model = None
_local_model_loaded = False

def initialize_local_image_model():
    """Initialize local image generation model for the worker"""
    global _local_image_model, _local_model_loaded
    
    if _local_model_loaded and _local_image_model is not None:
        return True
        
    try:
        task_logger.info("🔧 Initializing local image generation model...")
        from models.local_model import LocalModel
        _local_image_model = LocalModel()
        
        # Load the model 
        success = _local_image_model.load_model()
        if success:
            _local_model_loaded = True
            task_logger.info("✅ Local image generation model initialized successfully")
            return True
        else:
            task_logger.error("❌ Failed to load local image generation model")
            _local_model_loaded = False
            return False
            
    except Exception as e:
        task_logger.error(f"❌ Error initializing local image generation model: {e}")
        _local_model_loaded = False
        return False

def ensure_local_model_initialized():
    """Ensure local image model is initialized before use"""
    global _local_image_model, _local_model_loaded
    
    if _local_model_loaded and _local_image_model is not None:
        return True
        
    return initialize_local_image_model()

@worker_process_init.connect
def initialize_processors_for_worker(**kwargs):
    """Initialize processors for the worker process"""
    global _text_processor, _grid_processor, _pipeline
    
    if not TASK_2D_MODULES_LOADED:
        task_logger.error("Skipping processor initialization: Core task modules failed to load.")
        return

    try:
        task_logger.info("Initializing TextProcessor and GridProcessor for Celery worker process...")
        _text_processor = TextProcessor(model_type=DEFAULT_TEXT_MODEL)
        _grid_processor = GridProcessor(model_type=DEFAULT_GRID_MODEL)
        _pipeline = Pipeline(_text_processor, _grid_processor)
        task_logger.info("Processors initialized/ready for task execution.")
    except Exception as e:
        task_logger.critical(f"FATAL: Failed to initialize core processors in Celery worker: {e}", exc_info=True)

# Helper function to ensure processors are initialized
def ensure_processors_initialized():
    """Ensure processors are initialized before use"""
    global _text_processor, _grid_processor, _pipeline
    
    if not TASK_2D_MODULES_LOADED:
        return False
        
    if _pipeline is None:
        try:
            _text_processor = TextProcessor(model_type=DEFAULT_TEXT_MODEL)
            _grid_processor = GridProcessor(model_type=DEFAULT_GRID_MODEL)
            _pipeline = Pipeline(_text_processor, _grid_processor)
            return True
        except Exception as e:
            task_logger.error(f"Failed to initialize processors: {e}")
            return False
    return True

@app.task(name='generate_text_image')
def generate_text_image(prompt: str, width: int, height: int, num_images: int, model_type: str, 
                       doc_id=None, update_collection=None, category_key=None, item_key=None, 
                       output_s3_prefix="text_generation"):
    """Celery task to process text prompt and generate images with S3 upload and MongoDB update."""
    global _text_processor, _pipeline
    
    if not ensure_processors_initialized():
        return {"status": "error", "message": "Worker not fully initialized or modules missing."}

    # Initialize managers
    s3_mgr = get_s3_manager()
    mongo_mgr = get_mongo_helper()

    try:
        task_logger.info(f"Task: Processing text prompt: '{prompt[:50]}' with model {model_type}")
        
        # Ensure correct model is used
        if hasattr(_text_processor, 'model_type') and _text_processor.model_type != model_type:
            task_logger.info(f"Re-initializing TextProcessor to {model_type} for task.")
            _text_processor = TextProcessor(model_type=model_type)
            _pipeline = Pipeline(_text_processor, _grid_processor)

        images = _pipeline.process_text(prompt)
        
        if not images:
            task_logger.error("No images were generated by the pipeline.")
            return {"status": "error", "message": "No images generated."}
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
        
        # Save images
        image_relative_paths = []
        s3_urls = []
        mongodb_updated = False
        mongodb_update_error = None
        
        if len(images) > 1:
            grid_image = create_image_grid(images)
            unique_id = str(uuid.uuid4())[:8]
            grid_filename = f"text_grid_{unique_id}.png"
            grid_path = os.path.join(OUTPUT_IMAGES_DIR, grid_filename)
            grid_image.save(grid_path)
            image_relative_paths.append(grid_filename)

            # Upload grid image to S3
            if s3_mgr:
                upload_result = s3_mgr.upload_image(grid_path, f"{output_s3_prefix}/grid")
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Grid image uploaded to S3: {upload_result.get('s3_url')}")

            for i, img in enumerate(images):
                single_filename = f"text_{unique_id}_{i}.png"
                single_img_path = os.path.join(OUTPUT_IMAGES_DIR, single_filename)
                img.save(single_img_path)
                image_relative_paths.append(single_filename)
                
                # Upload individual image to S3
                if s3_mgr:
                    upload_result = s3_mgr.upload_image(single_img_path, f"{output_s3_prefix}/individual")
                    if upload_result.get("status") == "success":
                        s3_urls.append(upload_result.get("s3_url"))
                        task_logger.info(f"✅ Individual image {i} uploaded to S3: {upload_result.get('s3_url')}")
                        
            message = f"Generated {len(images)} images (grid). Saved to {OUTPUT_IMAGES_DIR}"
        else:
            unique_id = str(uuid.uuid4())[:8]
            img_filename = f"text_image_{unique_id}.png"
            img_path = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
            images[0].save(img_path)
            image_relative_paths.append(img_filename)
            
            # Upload single image to S3
            if s3_mgr:
                upload_result = s3_mgr.upload_image(img_path, output_s3_prefix)
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Single image uploaded to S3: {upload_result.get('s3_url')}")
                    
            message = f"Generated 1 image. Saved to {OUTPUT_IMAGES_DIR}"

        # Update MongoDB if requested and we have at least one S3 URL
        if mongo_mgr and doc_id and update_collection and s3_urls:
            # Use the first S3 URL (main image) for MongoDB update
            main_s3_url = s3_urls[0]
            mongodb_updated, mongodb_update_error = update_image_url_in_mongodb(
                mongo_mgr,
                doc_id,
                update_collection,
                main_s3_url,
                local_image_path=img_path if len(images) == 1 else grid_path,
                category_key=category_key,
                item_key=item_key,
                metadata={
                    "prompt": prompt,
                    "model_type": model_type,
                    "dimensions": f"{width}x{height}",
                    "num_images": len(images),
                    "generation_type": "text_to_image"
                }
            )

        result = {
            "status": "success", 
            "message": message, 
            "image_filenames": image_relative_paths,
            "mongodb_updated": mongodb_updated
        }
        
        if s3_urls:
            result["s3_image_urls"] = s3_urls
            result["main_s3_url"] = s3_urls[0]
            
        if mongodb_update_error:
            result["mongodb_update_error"] = mongodb_update_error

        return result

    except Exception as e:
        task_logger.error(f"Error in generate_text_image task for prompt '{prompt[:50]}': {e}", exc_info=True)
        return {"status": "error", "message": f"Task failed: {e}"}

@app.task(name='generate_local_image', bind=True)
def generate_local_image(self, prompt: str, width: int, height: int, num_images: int,
                        doc_id=None, update_collection=None, category_key=None, item_key=None,
                        output_s3_prefix="local_generation"):
    """
    Celery task specifically for local image generation using the local diffusion model.
    This bypasses the API clients and directly uses the local model for processing.
    Now includes S3 upload and MongoDB update functionality.
    """
    global _local_image_model
    
    task_logger.info("🎨 Local image generation task started")
    task_logger.info(f"   Prompt: '{prompt[:50]}...'")
    task_logger.info(f"   Dimensions: {width}x{height}")
    task_logger.info(f"   Number of images: {num_images}")
    
    # Initialize managers
    s3_mgr = get_s3_manager()
    mongo_mgr = get_mongo_helper()
    
    # Update progress
    self.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Initializing local model...'})
    
    # Ensure local model is initialized
    if not ensure_local_model_initialized():
        error_msg = "Local image generation model not available. Please check GPU setup and model installation."
        task_logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    try:
        self.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Generating image(s)...'})
        
        # Generate images using local model
        task_logger.info(f"🎯 Generating {num_images} image(s) locally...")
        images = _local_image_model.generate_image(
            input_data=prompt,
            width=width,
            height=height,
            num_images=num_images
        )
        
        if not images or len(images) == 0:
            error_msg = "No images were generated by the local model"
            task_logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        self.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Saving generated images...'})
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
        
        # Save images
        image_relative_paths = []
        s3_urls = []
        unique_id = str(uuid.uuid4())[:8]
        mongodb_updated = False
        mongodb_update_error = None
        
        if len(images) > 1:
            # Create grid image for multiple images
            grid_image = create_image_grid(images)
            grid_filename = f"local_grid_{unique_id}.png"
            grid_path = os.path.join(OUTPUT_IMAGES_DIR, grid_filename)
            grid_image.save(grid_path)
            image_relative_paths.append(grid_filename)
            
            # Upload grid image to S3
            if s3_mgr:
                self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Uploading grid image to S3...'})
                upload_result = s3_mgr.upload_image(grid_path, f"{output_s3_prefix}/grid")
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Grid image uploaded to S3: {upload_result.get('s3_url')}")
            
            # Save individual images as well
            for i, img in enumerate(images):
                single_filename = f"local_{unique_id}_{i}.png"
                single_img_path = os.path.join(OUTPUT_IMAGES_DIR, single_filename)
                img.save(single_img_path)
                image_relative_paths.append(single_filename)
                
                # Upload individual image to S3
                if s3_mgr:
                    upload_result = s3_mgr.upload_image(single_img_path, f"{output_s3_prefix}/individual")
                    if upload_result.get("status") == "success":
                        s3_urls.append(upload_result.get("s3_url"))
                        task_logger.info(f"✅ Individual image {i} uploaded to S3: {upload_result.get('s3_url')}")
                
            task_logger.info(f"✅ Generated {len(images)} images locally (grid + individual)")
            message = f"Generated {len(images)} images locally (with grid). Saved to {OUTPUT_IMAGES_DIR}"
        else:
            # Single image
            img_filename = f"local_image_{unique_id}.png"
            img_path = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
            images[0].save(img_path)
            image_relative_paths.append(img_filename)
            
            # Upload single image to S3
            if s3_mgr:
                self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Uploading image to S3...'})
                upload_result = s3_mgr.upload_image(img_path, output_s3_prefix)
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Single image uploaded to S3: {upload_result.get('s3_url')}")
            
            task_logger.info("✅ Generated 1 image locally")
            message = f"Generated 1 image locally. Saved to {OUTPUT_IMAGES_DIR}"
        
        # Update MongoDB if requested and we have at least one S3 URL
        if mongo_mgr and doc_id and update_collection and s3_urls:
            self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Updating database...'})
            # Use the first S3 URL (main image) for MongoDB update
            main_s3_url = s3_urls[0]
            mongodb_updated, mongodb_update_error = update_image_url_in_mongodb(
                mongo_mgr,
                doc_id,
                update_collection,
                main_s3_url,
                local_image_path=img_path if len(images) == 1 else grid_path,
                category_key=category_key,
                item_key=item_key,
                metadata={
                    "prompt": prompt,
                    "model_type": "local",
                    "dimensions": f"{width}x{height}",
                    "num_images": len(images),
                    "generation_type": "local_image_generation"
                }
            )
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Local image generation completed!'})
        
        result = {
            "status": "success", 
            "message": message, 
            "image_filenames": image_relative_paths,
            "model_type": "local",
            "mongodb_updated": mongodb_updated,
            "generation_stats": {
                "prompt": prompt,
                "dimensions": f"{width}x{height}",
                "num_images": len(images)
            }
        }
        
        if s3_urls:
            result["s3_image_urls"] = s3_urls
            result["main_s3_url"] = s3_urls[0]
            
        if mongodb_update_error:
            result["mongodb_update_error"] = mongodb_update_error
        
        return result
        
    except Exception as e:
        error_msg = f"Local image generation failed: {str(e)}"
        task_logger.error(f"❌ {error_msg}", exc_info=True)
        return {"status": "error", "message": error_msg}

@app.task(name='generate_grid_image')
def generate_grid_image(grid_string: str, width: int, height: int, num_images: int, model_type: str,
                       doc_id=None, update_collection=None, category_key=None, item_key=None,
                       output_s3_prefix="grid_generation"):
    """Celery task to process grid data and generate terrain images with S3 upload and MongoDB update."""
    global _grid_processor, _pipeline
    
    if not ensure_processors_initialized():
        return {"status": "error", "message": "Worker not fully initialized or modules missing."}

    # Initialize managers
    s3_mgr = get_s3_manager()
    mongo_mgr = get_mongo_helper()

    try:
        task_logger.info(f"Task: Processing grid data with model {model_type}")

        if hasattr(_grid_processor, 'model_type') and _grid_processor.model_type != model_type:
            task_logger.info(f"Re-initializing GridProcessor to {model_type} for task.")
            _grid_processor = GridProcessor(model_type=model_type)
            _pipeline = Pipeline(_text_processor, _grid_processor)

        images, grid_viz = _pipeline.process_grid(grid_string)
        
        if not images:
            task_logger.error("No images were generated by the pipeline.")
            return {"status": "error", "message": "No images generated."}

        # Ensure output directory exists
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
        
        unique_id = str(uuid.uuid4())[:8]
        image_filenames = []
        s3_urls = []
        mongodb_updated = False
        mongodb_update_error = None
        
        # Save visualization if available
        if grid_viz is not None:
            viz_filename = f"grid_visualization_{unique_id}.png"
            viz_path = os.path.join(OUTPUT_IMAGES_DIR, viz_filename)
            grid_viz.save(viz_path)
            image_filenames.append(viz_filename)
            
            # Upload visualization to S3
            if s3_mgr:
                upload_result = s3_mgr.upload_image(viz_path, f"{output_s3_prefix}/visualization")
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Grid visualization uploaded to S3: {upload_result.get('s3_url')}")

        if len(images) > 1:
            grid_image = create_image_grid(images)
            grid_filename = f"terrain_grid_{unique_id}.png"
            grid_path = os.path.join(OUTPUT_IMAGES_DIR, grid_filename)
            grid_image.save(grid_path)
            image_filenames.append(grid_filename)
            
            # Upload grid image to S3
            if s3_mgr:
                upload_result = s3_mgr.upload_image(grid_path, f"{output_s3_prefix}/grid")
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Terrain grid uploaded to S3: {upload_result.get('s3_url')}")
            
            for i, img in enumerate(images):
                single_filename = f"terrain_{unique_id}_{i}.png"
                single_img_path = os.path.join(OUTPUT_IMAGES_DIR, single_filename)
                img.save(single_img_path)
                image_filenames.append(single_filename)
                
                # Upload individual terrain image to S3
                if s3_mgr:
                    upload_result = s3_mgr.upload_image(single_img_path, f"{output_s3_prefix}/individual")
                    if upload_result.get("status") == "success":
                        s3_urls.append(upload_result.get("s3_url"))
                        task_logger.info(f"✅ Individual terrain image {i} uploaded to S3: {upload_result.get('s3_url')}")
                        
            message = f"Generated {len(images)} images (grid). Saved to {OUTPUT_IMAGES_DIR}"
        else:
            img_filename = f"terrain_image_{unique_id}.png"
            img_path = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
            images[0].save(img_path)
            image_filenames.append(img_filename)
            
            # Upload single terrain image to S3
            if s3_mgr:
                upload_result = s3_mgr.upload_image(img_path, output_s3_prefix)
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Single terrain image uploaded to S3: {upload_result.get('s3_url')}")
                    
            message = f"Generated 1 image. Saved to {OUTPUT_IMAGES_DIR}"

        # Update MongoDB if requested and we have at least one S3 URL
        if mongo_mgr and doc_id and update_collection and s3_urls:
            # Use the first S3 URL (main image) for MongoDB update
            main_s3_url = s3_urls[0]
            mongodb_updated, mongodb_update_error = update_image_url_in_mongodb(
                mongo_mgr,
                doc_id,
                update_collection,
                main_s3_url,
                local_image_path=img_path if len(images) == 1 else grid_path,
                category_key=category_key,
                item_key=item_key,
                metadata={
                    "grid_string": grid_string[:100] + "..." if len(grid_string) > 100 else grid_string,
                    "model_type": model_type,
                    "dimensions": f"{width}x{height}",
                    "num_images": len(images),
                    "generation_type": "grid_to_terrain",
                    "has_visualization": grid_viz is not None
                }
            )

        result = {
            "status": "success", 
            "message": message, 
            "image_filenames": image_filenames,
            "mongodb_updated": mongodb_updated
        }
        
        if s3_urls:
            result["s3_image_urls"] = s3_urls
            result["main_s3_url"] = s3_urls[0]
            
        if mongodb_update_error:
            result["mongodb_update_error"] = mongodb_update_error

        return result

    except Exception as e:
        task_logger.error(f"Error in generate_grid_image task: {e}", exc_info=True)
        return {"status": "error", "message": f"Task failed: {e}"}

@app.task(name='process_local_generation', bind=True)
def process_local_generation(self, input_data: str, input_type: str, width: int, height: int, num_images: int, 
                           enhance_prompt: bool = True, doc_id=None, update_collection=None, 
                           category_key=None, item_key=None, output_s3_prefix="local_processing"):
    """
    Comprehensive local processing task that can handle various input types with local models.
    Now includes S3 upload and MongoDB update functionality.
    
    Args:
        input_data: The input content (prompt, grid string, etc.)
        input_type: Type of input ('text', 'grid', 'biome')
        width: Image width
        height: Image height 
        num_images: Number of images to generate
        enhance_prompt: Whether to enhance text prompts for better results
        doc_id: MongoDB document ID to update
        update_collection: Collection name to update
        category_key: Category key for nested document updates
        item_key: Item key for nested document updates
        output_s3_prefix: S3 prefix for uploaded images
    """
    task_logger.info("🚀 Local processing task started")
    task_logger.info(f"   Input type: {input_type}")
    task_logger.info(f"   Input data: '{input_data[:50]}...'")
    task_logger.info(f"   Dimensions: {width}x{height}")
    task_logger.info(f"   Number of images: {num_images}")
    
    # Initialize managers
    s3_mgr = get_s3_manager()
    mongo_mgr = get_mongo_helper()
    
    # Update progress
    self.update_state(state='PROGRESS', meta={'progress': 5, 'status': f'Initializing local {input_type} processing...'})
    
    try:
        if input_type == 'text':
            # Ensure local model is initialized
            if not ensure_local_model_initialized():
                error_msg = "Local image generation model not available"
                return {"status": "error", "message": error_msg}
            
            self.update_state(state='PROGRESS', meta={'progress': 15, 'status': 'Processing text prompt...'})
            
            # Enhance prompt if requested
            if enhance_prompt:
                from pipeline.text_processor import TextProcessor
                temp_processor = TextProcessor(model_type='local')
                enhanced_prompt = temp_processor.enhance_prompt(input_data)
                task_logger.info(f"Enhanced prompt: {enhanced_prompt}")
                input_data = enhanced_prompt
            
            self.update_state(state='PROGRESS', meta={'progress': 25, 'status': 'Generating images with local model...'})
            
            # Generate images
            images = _local_image_model.generate_image(
                input_data=input_data,
                width=width,
                height=height,
                num_images=num_images
            )
            
        elif input_type == 'grid':
            # Initialize processors if needed
            if not ensure_processors_initialized():
                return {"status": "error", "message": "Grid processors not available"}
            
            self.update_state(state='PROGRESS', meta={'progress': 15, 'status': 'Processing grid data...'})
            
            # Use local model for grid processing
            if hasattr(_grid_processor, 'model_type') and _grid_processor.model_type != 'local':
                from pipeline.grid_processor import GridProcessor
                from pipeline.pipeline import Pipeline
                local_grid_processor = GridProcessor(model_type='local')
                local_pipeline = Pipeline(_text_processor, local_grid_processor)
            else:
                local_pipeline = _pipeline
            
            self.update_state(state='PROGRESS', meta={'progress': 25, 'status': 'Generating terrain images locally...'})
            
            images, grid_viz = local_pipeline.process_grid(input_data)
            
        else:
            return {"status": "error", "message": f"Unsupported input type: {input_type}"}
        
        if not images or len(images) == 0:
            error_msg = f"No images were generated for {input_type} input"
            task_logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        self.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Saving generated content...'})
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
        
        # Save images
        image_relative_paths = []
        s3_urls = []
        unique_id = str(uuid.uuid4())[:8]
        mongodb_updated = False
        mongodb_update_error = None
        
        if len(images) > 1:
            # Create grid image for multiple images
            grid_image = create_image_grid(images)
            grid_filename = f"local_{input_type}_grid_{unique_id}.png"
            grid_path = os.path.join(OUTPUT_IMAGES_DIR, grid_filename)
            grid_image.save(grid_path)
            image_relative_paths.append(grid_filename)
            
            # Upload grid image to S3
            if s3_mgr:
                self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Uploading grid image to S3...'})
                upload_result = s3_mgr.upload_image(grid_path, f"{output_s3_prefix}/grid")
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Grid image uploaded to S3: {upload_result.get('s3_url')}")
            
            # Save individual images
            for i, img in enumerate(images):
                single_filename = f"local_{input_type}_{unique_id}_{i}.png"
                single_img_path = os.path.join(OUTPUT_IMAGES_DIR, single_filename)
                img.save(single_img_path)
                image_relative_paths.append(single_filename)
                
                # Upload individual image to S3
                if s3_mgr:
                    upload_result = s3_mgr.upload_image(single_img_path, f"{output_s3_prefix}/individual")
                    if upload_result.get("status") == "success":
                        s3_urls.append(upload_result.get("s3_url"))
                        task_logger.info(f"✅ Individual image {i} uploaded to S3: {upload_result.get('s3_url')}")
                
            message = f"Generated {len(images)} {input_type} images locally (with grid)"
        else:
            # Single image
            img_filename = f"local_{input_type}_{unique_id}.png"
            img_path = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
            images[0].save(img_path)
            image_relative_paths.append(img_filename)
            
            # Upload single image to S3
            if s3_mgr:
                self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Uploading image to S3...'})
                upload_result = s3_mgr.upload_image(img_path, output_s3_prefix)
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Single image uploaded to S3: {upload_result.get('s3_url')}")
            
            message = f"Generated 1 {input_type} image locally"
        
        # Save grid visualization if available (for grid inputs)
        if input_type == 'grid' and 'grid_viz' in locals() and grid_viz is not None:
            viz_filename = f"local_grid_viz_{unique_id}.png"
            viz_path = os.path.join(OUTPUT_IMAGES_DIR, viz_filename)
            grid_viz.save(viz_path)
            image_relative_paths.append(viz_filename)
            
            # Upload visualization to S3
            if s3_mgr:
                upload_result = s3_mgr.upload_image(viz_path, f"{output_s3_prefix}/visualization")
                if upload_result.get("status") == "success":
                    s3_urls.append(upload_result.get("s3_url"))
                    task_logger.info(f"✅ Grid visualization uploaded to S3: {upload_result.get('s3_url')}")
        
        # Update MongoDB if requested and we have at least one S3 URL
        if mongo_mgr and doc_id and update_collection and s3_urls:
            self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Updating database...'})
            # Use the first S3 URL (main image) for MongoDB update
            main_s3_url = s3_urls[0]
            mongodb_updated, mongodb_update_error = update_image_url_in_mongodb(
                mongo_mgr,
                doc_id,
                update_collection,
                main_s3_url,
                local_image_path=img_path if len(images) == 1 else grid_path,
                category_key=category_key,
                item_key=item_key,
                metadata={
                    "input_data": input_data[:100] + "..." if len(input_data) > 100 else input_data,
                    "input_type": input_type,
                    "dimensions": f"{width}x{height}",
                    "num_images": len(images),
                    "enhanced_prompt": enhance_prompt if input_type == 'text' else False,
                    "generation_type": f"local_{input_type}_processing"
                }
            )
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': f'Local {input_type} processing completed!'})
        
        result = {
            "status": "success", 
            "message": f"{message}. Saved to {OUTPUT_IMAGES_DIR}", 
            "image_filenames": image_relative_paths,
            "model_type": "local",
            "input_type": input_type,
            "mongodb_updated": mongodb_updated,
            "generation_stats": {
                "input_data": input_data[:100] + "..." if len(input_data) > 100 else input_data,
                "dimensions": f"{width}x{height}",
                "num_images": len(images),
                "enhanced_prompt": enhance_prompt if input_type == 'text' else False
            }
        }
        
        if s3_urls:
            result["s3_image_urls"] = s3_urls
            result["main_s3_url"] = s3_urls[0]
            
        if mongodb_update_error:
            result["mongodb_update_error"] = mongodb_update_error
        
        task_logger.info(f"✅ Local {input_type} processing completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Local {input_type} processing failed: {str(e)}"
        task_logger.error(f"❌ {error_msg}", exc_info=True)
        return {"status": "error", "message": error_msg}

@app.task(name='run_biome_generation')
def run_biome_generation(theme: str, structure_types_str: str):
    """Celery task to generate biome data and save it to MongoDB."""
    if not TASK_MODULES_LOADED:
        return {"status": "error", "message": "Worker not fully initialized or modules missing."}

    task_logger.info(f"Task: Running biome generation for theme: '{theme[:50]}', structures: '{structure_types_str}'")
    structure_type_list = [s.strip() for s in structure_types_str.split(',') if s.strip()]

    if not structure_type_list:
        task_logger.warning("No structure types provided for biome generation task.")
        return {"status": "error", "message": "No structure types provided."}
    
    try:
        msg = generate_biome(theme, structure_type_list)
        task_logger.info(f"Biome generation finished with message: {msg}")
        return {"status": "success", "message": msg}

    except Exception as e:
        task_logger.error(f"Error in run_biome_generation task for theme '{theme[:50]}': {e}", exc_info=True)
        return {"status": "error", "message": f"Task failed: {e}"}

@app.task(name='batch_process_mongodb_prompts_task')
def batch_process_mongodb_prompts_task(db_name: str, collection_name: str, limit: int, width: int, height: int, model_type: str, update_db: bool):
    """Celery task to batch process multiple prompts from MongoDB and generate images."""
    global _text_processor, _pipeline 
    
    if not ensure_processors_initialized():
        return {"status": "error", "message": "Worker not fully initialized or modules missing."}

    mongo_helper = MongoDBHelper()
    s3_mgr = get_s3_manager()
    
    try:
        query = {"$or": [{"theme_prompt": {"$exists": True}}, {"description": {"$exists": True}}]}
        prompt_documents = mongo_helper.find_many(MONGO_DB_NAME, collection_name, query=query, limit=limit)
        
        if not prompt_documents:
            return {"status": "success", "message": "No prompts found to process in batch."}
    except Exception as e:
        task_logger.error(f"Error fetching prompts for batch processing from MongoDB: {e}")
        return {"status": "error", "message": f"Failed to fetch prompts for batch: {e}"}

    results = []
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)  # Ensure output dir exists
    for doc in prompt_documents:
        doc_id = str(doc.get("_id"))
        # If possible_structures exists, process each structure
        if "possible_structures" in doc:
            for category_key, category in doc["possible_structures"].items():
                for item_key, struct in category.items():
                    prompt = struct.get("description")
                    structure_id = struct.get("structureId") or item_key  # fallback to item_key if structureId missing
                    if not prompt or not structure_id:
                        results.append(f"Skipping structure in {doc_id}: No prompt or structureId.")
                        continue
                    try:
                        if hasattr(_text_processor, 'model_type') and _text_processor.model_type != model_type:
                            _text_processor = TextProcessor(model_type=model_type)
                            _pipeline = Pipeline(_text_processor, _grid_processor)
                        # --- Use SDXL Turbo for image generation and MongoDB update ---
                        sdxl_result = generate_image_sdxl_turbo.apply_async(
                            args=[prompt],
                            kwargs={
                                "doc_id": doc_id,
                                "update_collection": collection_name,
                                "output_s3_prefix": f"images/{structure_id}",
                                "width": width,
                                "height": height,
                                "enhance_prompt": True,
                                "category_key": category_key,
                                "item_key": item_key
                            }
                        )
                        result = sdxl_result.get(timeout=180)
                        # Fetch the updated document from MongoDB to get the image URL
                        updated_doc = mongo_helper.find_many(MONGO_DB_NAME, collection_name, query={"_id": doc.get("_id")}, limit=1)
                        image_url = None
                        if updated_doc and category_key in updated_doc[0].get("possible_structures", {}):
                            struct_data = updated_doc[0]["possible_structures"][category_key].get(item_key, {})
                            image_url = struct_data.get("imageUrl")
                        if result.get("status") == "success":
                            results.append(f"Generated image for structure {structure_id} -> {image_url if image_url else 'No imageUrl found in DB'}")
                        else:
                            results.append(f"Failed to generate image for structure {structure_id} - {result.get('message')}")
                    except Exception as e:
                        task_logger.error(f"Error in batch processing for structure {structure_id}: {e}", exc_info=True)
                        results.append(f"Failed to process structure {structure_id} - Error: {e}")
        else:
            # Fallback: process the document as a whole (legacy)
            prompt = doc.get("theme_prompt") or doc.get("description")
            if not prompt:
                results.append(f"Skipping {doc_id}: No valid prompt found.")
                continue
            try:
                if hasattr(_text_processor, 'model_type') and _text_processor.model_type != model_type:
                    _text_processor = TextProcessor(model_type=model_type)
                    _pipeline = Pipeline(_text_processor, _grid_processor)
                sdxl_result = generate_image_sdxl_turbo.apply_async(
                    args=[prompt],
                    kwargs={
                        "doc_id": doc_id,
                        "update_collection": collection_name,
                        "output_s3_prefix": f"images/{doc_id}",
                        "width": width,
                        "height": height,
                        "enhance_prompt": True
                    }
                )
                result = sdxl_result.get(timeout=180)
                # Fetch the updated document from MongoDB to get the image URL
                updated_doc = mongo_helper.find_many(MONGO_DB_NAME, collection_name, query={"_id": doc.get("_id")}, limit=1)
                image_url = None
                if updated_doc:
                    image_url = updated_doc[0].get("image_path")
                if result.get("status") == "success":
                    results.append(f"Generated image for: '{prompt[:30]}...' -> {image_url if image_url else 'No image_path found in DB'}")
                else:
                    results.append(f"Failed to generate image for: '{prompt[:30]}' - {result.get('message')}")
            except Exception as e:
                task_logger.error(f"Error in batch processing for {doc_id}: {e}", exc_info=True)
                results.append(f"Failed to process '{prompt[:30]}...' - Error: {e}")
    return {"status": "success", "message": "\n".join(results)}

@app.task(name='generate_3d_model_from_image', bind=True)
def generate_3d_model_from_image(self, image_s3_key_or_path, with_texture=False, output_format='glb', doc_id=None, update_collection=None, category_key=None, item_key=None):
    """
    Celery task to generate a 3D model from an image using Hunyuan3D.
    Now supports S3 integration for both input and output.
    
    Args:
        image_s3_key_or_path: S3 key or local path to image
        with_texture: Whether to generate textures
        output_format: Output format (glb, obj, etc.)
        doc_id: MongoDB document ID to update with results
        update_collection: Collection name to update
        category_key: Category key for nested structure updates (e.g., 'settlements')
        item_key: Item key for nested structure updates (e.g., specific structure ID)
    """
    task_logger.info("🚀 Starting 3D model generation task")
    task_logger.info(f"   Image path/key: {image_s3_key_or_path}")
    task_logger.info(f"   With texture: {with_texture}")
    task_logger.info(f"   Output format: {output_format}")
    task_logger.info(f"   MongoDB doc_id: {doc_id}")
    task_logger.info(f"   Hunyuan3D modules loaded: {TASK_3D_MODULES_LOADED}")
    
    if not TASK_3D_MODULES_LOADED:
        error_msg = "Hunyuan3D modules not loaded on this worker. Please check module installation."
        task_logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # Initialize S3 and MongoDB helpers
    s3_mgr = get_s3_manager()
    mongo_mgr = get_mongo_helper()
    
    local_image_path = None
    temp_dir = None
    
    try:
        # Step 1: Download image from S3, HTTP URL, or use local file
        self.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Preparing image...'})
        
        if image_s3_key_or_path.startswith(('http://', 'https://')):
            # Handle HTTP/HTTPS URLs (MongoDB image URLs)
            task_logger.info(f"🌐 Downloading image from URL: {image_s3_key_or_path}")
            self.update_state(state='PROGRESS', meta={'progress': 8, 'status': 'Downloading image from URL...'})
            
            import tempfile
            import requests
            from urllib.parse import urlparse
            
            temp_dir = tempfile.mkdtemp()
            
            # Parse URL to get file extension
            parsed_url = urlparse(image_s3_key_or_path)
            file_ext = os.path.splitext(parsed_url.path)[1] or '.png'
            local_image_path = os.path.join(temp_dir, f"downloaded_image{file_ext}")
            
            # Download the image
            response = requests.get(image_s3_key_or_path, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(local_image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            task_logger.info(f"✅ Downloaded image from URL to: {local_image_path}")
            
        elif image_s3_key_or_path.startswith('s3://') or image_s3_key_or_path.startswith('https://') or not os.path.exists(image_s3_key_or_path):
            # Treat as S3 key or URL, download to temp location
            if s3_mgr is None:
                return {"status": "error", "message": "S3 manager not available for image download"}
            
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
            # Parse S3 key from different URL formats
            # First, remove any s3://bucket/ prefix that might have been incorrectly added
            clean_path = image_s3_key_or_path
            if clean_path.startswith('s3://sparkassets/'):
                clean_path = clean_path.replace('s3://sparkassets/', '')
            
            if clean_path.startswith('s3://'):
                # s3://bucket/key format
                s3_key = clean_path.replace('s3://', '').split('/', 1)[1]
            elif clean_path.startswith('https://'):
                # https://bucket.s3.region.amazonaws.com/key format
                from urllib.parse import urlparse
                parsed_url = urlparse(clean_path)
                s3_key = parsed_url.path.lstrip('/')  # Remove leading slash
            else:
                # Assume it's already an S3 key
                s3_key = clean_path
            
            download_result = s3_mgr.download_image(s3_key, temp_dir)
            if download_result.get("status") != "success":
                return {"status": "error", "message": f"Failed to download image from S3: {download_result.get('message')}"}
            
            local_image_path = download_result["local_path"]
            task_logger.info(f"✅ Downloaded image from S3: {s3_key} -> {local_image_path}")
        else:
            # Use local path directly
            local_image_path = image_s3_key_or_path
            task_logger.info(f"✅ Using local image path: {local_image_path}")
        
        # Step 2: Initialize processors if needed
        task_logger.info("🔧 Checking Hunyuan3D processor initialization...")
        if _hunyuan_i23d_worker is None:
            task_logger.info("⚙️ Initializing Hunyuan3D processors...")
            self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Initializing Hunyuan3D processors...'})
            
            if not initialize_hunyuan3d_processors():
                error_msg = "Failed to initialize Hunyuan3D processors."
                task_logger.error(f"❌ {error_msg}")
                return {"status": "error", "message": error_msg}
            
            task_logger.info("✅ Hunyuan3D processors initialized successfully")
        else:
            task_logger.info("✅ Hunyuan3D processors already initialized")
        
        # Step 3: Generate 3D model
        task_logger.info("🎯 Starting 3D model generation...")
        self.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Generating 3D model...'})
        
        result = generate_3d_from_image_core(
            local_image_path, 
            with_texture, 
            output_format, 
            progress_callback=lambda p, s: self.update_state(
                state='PROGRESS', 
                meta={'progress': 20 + int(p * 0.6), 'status': s}  # 20-80% range
            )
        )
        
        if result.get('status') != 'success':
            task_logger.error(f"❌ 3D model generation failed: {result.get('message', 'Unknown error')}")
            return result
        
        # Step 4: Upload 3D assets to S3
        self.update_state(state='PROGRESS', meta={'progress': 85, 'status': 'Uploading 3D assets to S3...'})
        
        model_path = result.get('model_path')
        s3_urls = {}
        
        if s3_mgr and model_path:
            # Extract source image name for consistent naming
            source_image_name = None
            if image_s3_key_or_path:
                if image_s3_key_or_path.startswith('s3://') or not os.path.exists(image_s3_key_or_path):
                    # S3 key - extract filename
                    source_image_name = s3_mgr.get_filename_from_s3_key(image_s3_key_or_path)
                else:
                    # Local path - extract filename
                    source_image_name = os.path.basename(image_s3_key_or_path)
            
            upload_result = s3_mgr.upload_3d_asset(model_path, "generated", source_image_name)
            
            if upload_result.get("status") == "success":
                s3_urls['model'] = upload_result.get("s3_url")
                result['s3_model_url'] = s3_urls['model']
                task_logger.info(f"✅ Uploaded 3D model to S3: {s3_urls['model']}")
            else:
                task_logger.warning(f"Failed to upload 3D model to S3: {upload_result.get('message')}")
        
        # Step 5: Update MongoDB with 3D asset link for structure images
        if mongo_mgr and doc_id and update_collection and s3_urls.get('model'):
            self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Updating database...'})
            try:
                # Import MONGO_DB_NAME here to avoid scope issues
                try:
                    from config import MONGO_DB_NAME
                except ImportError:
                    MONGO_DB_NAME = "World_builder"
                
                model_s3_url = s3_urls['model']  # Get the 3D model URL
                
                if category_key and item_key:
                    # Update nested field for structure images - add 3D asset link
                    update_path = f"possible_structures.{category_key}.{item_key}.asset_3d_url"
                    update_data = {
                        update_path: model_s3_url,
                        f"possible_structures.{category_key}.{item_key}.status": "3d_asset_generated",
                        f"possible_structures.{category_key}.{item_key}.asset_3d_generated_at": datetime.now(),
                        f"possible_structures.{category_key}.{item_key}.asset_3d_format": output_format
                    }
                    update_op = {"$set": update_data}
                    result_count = mongo_mgr.update_by_id(
                        db_name=MONGO_DB_NAME,
                        collection_name=update_collection,
                        document_id=doc_id,
                        update=update_op
                    )
                    task_logger.info(f"✅ Updated MongoDB document {doc_id} at {update_path} with 3D asset URL: {model_s3_url}")
                else:
                    # Update root-level for theme images
                    update_data = {
                        "asset_3d_url": model_s3_url,
                        "status": "3d_asset_generated",
                        "asset_3d_generated_at": datetime.now(),
                        "asset_3d_format": output_format
                    }
                    update_op = {"$set": update_data}
                    result_count = mongo_mgr.update_by_id(
                        db_name=MONGO_DB_NAME,
                        collection_name=update_collection,
                        document_id=doc_id,
                        update=update_op
                    )
                    task_logger.info(f"✅ Updated MongoDB document {doc_id} at root level with 3D asset URL: {model_s3_url}")
                
                result['mongodb_updated'] = result_count > 0
                
            except Exception as e:
                task_logger.error(f"Failed to update MongoDB with 3D asset URL: {e}")
                result['mongodb_updated'] = False
        
        # Step 6: Update MongoDB document by image_path for 3D asset link
        if mongo_mgr and result.get('status') == 'success' and s3_urls.get('model'):
            task_logger.info(f"🔍 Checking MongoDB update conditions:")
            task_logger.info(f"   mongo_mgr: {mongo_mgr is not None}")
            task_logger.info(f"   update_collection: {update_collection}")
            task_logger.info(f"   result status: {result.get('status')}")
            task_logger.info(f"   s3_model_url: {s3_urls.get('model')}")
            
            # Use a default collection if update_collection is not provided
            collection_to_update = update_collection or "biomes"  # Default to biomes collection
            
            try:
                from config import MONGO_DB_NAME
                
                task_logger.info(f"🔍 Searching for document with image_path: {image_s3_key_or_path}")
                
                # Debug: Show sample image paths in the collection
                try:
                    sample_docs = mongo_mgr.debug_image_paths(MONGO_DB_NAME, collection_to_update, limit=5)
                    task_logger.info(f"🔍 Sample image_path values in collection:")
                    for doc in sample_docs:
                        task_logger.info(f"   ID: {doc.get('_id')}, image_path: {doc.get('image_path')}")
                except Exception as e:
                    task_logger.warning(f"Could not fetch sample image paths: {e}")
                
                # Find document by image_path using the helper method
                doc = mongo_mgr.find_by_image_path(MONGO_DB_NAME, collection_to_update, image_s3_key_or_path)
                
                if doc:
                    task_logger.info(f"✅ Found document: {doc.get('_id')} in collection: {collection_to_update}")
                    
                    # Update document with 3D asset link and set processed to false
                    asset_update_data = {
                        "$set": {
                            "3d_asset_url": s3_urls['model'],
                            "3d_asset_generated_at": datetime.now(),
                            "3d_asset_format": output_format,
                            "status": "3d_asset_generated",
                            "processed": False  # Set processed to false as requested
                        }
                    }
                    
                    task_logger.info(f"🔄 Updating document with data: {asset_update_data}")
                    update_result = mongo_mgr.update_by_id(MONGO_DB_NAME, collection_to_update, str(doc['_id']), asset_update_data)
                    task_logger.info(f"✅ Updated document with 3D asset URL and set processed=false: {update_result} records modified")
                    result['asset_mongodb_updated'] = True
                    result['updated_doc_id'] = str(doc['_id'])
                    result['updated_collection'] = collection_to_update
                else:
                    task_logger.warning(f"⚠️ No document found with image_path matching: {image_s3_key_or_path}")
                    task_logger.info(f"🔍 Tried searching in collection: {collection_to_update}")
                    result['asset_mongodb_updated'] = False
                    result['asset_mongodb_warning'] = f"No document found with image_path: {image_s3_key_or_path}"
                
            except Exception as e:
                task_logger.error(f"Failed to update MongoDB with 3D asset link: {e}")
                result['asset_mongodb_error'] = str(e)
        else:
            task_logger.info("⏭️ Skipping MongoDB update - conditions not met")
            if not mongo_mgr:
                task_logger.info("   Reason: mongo_mgr is None")
            if result.get('status') != 'success':
                task_logger.info(f"   Reason: result status is {result.get('status')}")
            if not s3_urls.get('model'):
                task_logger.info("   Reason: no S3 model URL available")
        
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            task_logger.info("🧹 Cleaned up temporary files")
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Completed successfully'})
        task_logger.info(f"✅ 3D model generated successfully with S3 integration")
        
        return result
        
    except Exception as e:
        error_msg = f"Error generating 3D model: {str(e)}"
        task_logger.error(f"❌ {error_msg}", exc_info=True)
        
        # Cleanup on error
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        
        return {"status": "error", "message": error_msg}

@app.task(name='manage_gpu_instance', bind=True)
def manage_gpu_instance(self, action="ensure_running", instance_id=None, region=None):
    """
    Enhanced Celery task to manage GPU spot instances.
    Includes spot instance specific handling.
    """
    try:
        aws_manager = get_aws_manager()
        
        if aws_manager is None:
            return {"status": "error", "message": "AWS manager not available."}
        
        instance_id = instance_id or getattr(aws_manager, 'instance_id', None)
        region = region or getattr(aws_manager, 'region', None)
        
        if not instance_id:
            return {"status": "error", "message": "No GPU instance ID provided or configured."}
        
        if action == "ensure_running":
            task_logger.info(f"Ensuring GPU spot instance {instance_id} is running...")
            
            # Check current status first
            info = aws_manager.get_instance_info()
            if info and info.state == 'running':
                return {"status": "success", "message": f"GPU spot instance {instance_id} is already running."}
            
            # For spot instances, we may need to request a new one if terminated
            if info and info.state == 'terminated':
                task_logger.warning(f"Spot instance {instance_id} was terminated. May need to launch new instance.")
                return {"status": "warning", "message": f"Spot instance {instance_id} was terminated. Please launch a new spot instance."}
            
            success = aws_manager.ensure_instance_running(max_wait_time=600)  # Longer wait for spot instances
            
            if success:
                return {"status": "success", "message": f"GPU spot instance {instance_id} is now running."}
            else:
                return {"status": "error", "message": f"Failed to ensure GPU spot instance {instance_id} is running."}
                
        elif action == "stop":
            task_logger.info(f"Stopping GPU spot instance {instance_id}...")
            success = aws_manager.stop_instance()
            
            if success:
                return {"status": "success", "message": f"Stop request sent for GPU spot instance {instance_id}."}
            else:
                return {"status": "error", "message": f"Failed to stop GPU spot instance {instance_id}."}
        
        elif action == "status":
            task_logger.info(f"Getting status of GPU spot instance {instance_id}...")
            info = aws_manager.get_instance_info()
            
            if info:
                cost_info = aws_manager.get_instance_cost_estimate()
                return {
                    "status": "success", 
                    "message": f"Spot instance {instance_id} status retrieved.",
                    "instance_info": {
                        "state": info.state,
                        "instance_type": info.instance_type,
                        "public_ip": info.public_ip,
                        "uptime_hours": info.uptime_hours,
                        "is_spot": True  # Flag for spot instance
                    },
                    "cost_estimate": cost_info
                }
            else:
                return {"status": "error", "message": f"Failed to get status for spot instance {instance_id}."}
        
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    except Exception as e:
        task_logger.error(f"Error managing GPU spot instance: {e}", exc_info=True)
        # Retry for spot instance specific errors
        if 'spot' in str(e).lower() or 'insufficient' in str(e).lower():
            task_logger.info("Retrying spot instance operation...")
            self.retry(countdown=60, max_retries=3)
        return {"status": "error", "message": f"GPU spot instance management failed: {e}"}

@app.task(name='cleanup_old_assets')
def cleanup_old_assets(max_age_hours=24):
    """
    Celery task to clean up old generated assets to save disk space.
    """
    try:
        import time
        
        task_logger.info(f"Starting cleanup of assets older than {max_age_hours} hours...")
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        cleaned_files = []
        total_size_freed = 0
        
        # Clean up 3D assets
        if os.path.exists(OUTPUT_3D_ASSETS_DIR):
            for root, dirs, files in os.walk(OUTPUT_3D_ASSETS_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        if file_mtime < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_files.append(file_path)
                            total_size_freed += file_size
                    except Exception as e:
                        task_logger.warning(f"Failed to clean up file {file_path}: {e}")
                
                # Remove empty directories
                try:
                    if not os.listdir(root) and root != OUTPUT_3D_ASSETS_DIR:
                        os.rmdir(root)
                        task_logger.info(f"Removed empty directory: {root}")
                except Exception as e:
                    task_logger.warning(f"Failed to remove directory {root}: {e}")
        
        # Clean up old temporary images
        temp_image_patterns = ['text_to_3d_', 'upload_for_3d_']
        if os.path.exists(OUTPUT_IMAGES_DIR):
            for file in os.listdir(OUTPUT_IMAGES_DIR):
                if any(pattern in file for pattern in temp_image_patterns):
                    file_path = os.path.join(OUTPUT_IMAGES_DIR, file)
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        if file_mtime < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_files.append(file_path)
                            total_size_freed += file_size
                    except Exception as e:
                        task_logger.warning(f"Failed to clean up file {file_path}: {e}")
        
        size_mb = total_size_freed / (1024 * 1024)
        
        task_logger.info(f"Cleanup completed: {len(cleaned_files)} files removed, {size_mb:.2f} MB freed")
        
        return {
            "status": "success",
            "message": f"Cleanup completed successfully",
            "files_cleaned": len(cleaned_files),
            "size_freed_mb": round(size_mb, 2),
            "files_list": cleaned_files[:10]
        }
        
    except Exception as e:
        task_logger.error(f"Error during asset cleanup: {e}", exc_info=True)
        return {"status": "error", "message": f"Cleanup failed: {e}"}

# S3 and MongoDB integration imports
s3_manager = None
mongo_helper = None

def get_s3_manager():
    """Get S3 manager instance (lazy loading)"""
    global s3_manager
    if s3_manager is None:
        try:
            from s3_manager import S3Manager
            from config import USE_S3_STORAGE
            if USE_S3_STORAGE:
                s3_manager = S3Manager()
                task_logger.info("✅ S3 Manager initialized")
            else:
                task_logger.info("S3 storage disabled in config")
        except Exception as e:
            task_logger.error(f"Failed to initialize S3 manager: {e}")
            s3_manager = None
    return s3_manager

def get_mongo_helper():
    """Get MongoDB helper instance (lazy loading)"""
    global mongo_helper
    if mongo_helper is None:
        try:
            from db_helper import MongoDBHelper
            mongo_helper = MongoDBHelper()
            task_logger.info("✅ MongoDB Helper initialized")
        except Exception as e:
            task_logger.error(f"Failed to initialize MongoDB helper: {e}")
            mongo_helper = None
    return mongo_helper

@app.task(name='process_image_for_3d_generation', bind=True)
def process_image_for_3d_generation(self, image_s3_key, doc_id, collection_name, processing_options=None):
    """
    GPU-side task to process an image from S3 and generate 3D assets.
    
    Args:
        image_s3_key: S3 key of the image to process
        doc_id: MongoDB document ID to update
        collection_name: MongoDB collection name
        processing_options: Dict with processing options (texture, format, etc.)
    """
    if processing_options is None:
        processing_options = {}
    
    with_texture = processing_options.get('with_texture', False)
    output_format = processing_options.get('output_format', 'glb')
    
    task_logger.info(f"🔄 Processing image from S3 for 3D generation")
    task_logger.info(f"   S3 Key: {image_s3_key}")
    task_logger.info(f"   Document ID: {doc_id}")
    task_logger.info(f"   Collection: {collection_name}")
    task_logger.info(f"   Options: {processing_options}")
    
    # Call the main 3D generation task with S3 integration
    return generate_3d_model_from_image(
        self,
        image_s3_key_or_path=image_s3_key,
        with_texture=with_texture,
        output_format=output_format,
        doc_id=doc_id,
        update_collection=collection_name
    )

@app.task(name='batch_process_s3_images_for_3d', bind=True)
def batch_process_s3_images_for_3d(self, image_s3_keys, processing_options=None):
    """
    Batch process multiple images from S3 for 3D generation.
    
    Args:
        image_s3_keys: List of S3 keys for images to process
        processing_options: Dict with processing options
    """
    if processing_options is None:
        processing_options = {}
    
    task_logger.info(f"🔄 Batch processing {len(image_s3_keys)} images from S3")
    
    results = []
    
    for i, s3_key in enumerate(image_s3_keys):
        self.update_state(
            state='PROGRESS', 
            meta={
                'progress': int((i / len(image_s3_keys)) * 100), 
                'status': f'Processing image {i+1}/{len(image_s3_keys)}: {s3_key}'
            }
        )
        
        try:
            result = generate_3d_model_from_image(
                self,
                image_s3_key_or_path=s3_key,
                with_texture=processing_options.get('with_texture', False),
                output_format=processing_options.get('output_format', 'glb'),
                doc_id=None,
                update_collection=None
            )
            
            results.append({
                "s3_key": s3_key,
                "status": result.get("status"),
                "message": result.get("message"),
                "model_s3_url": result.get("s3_model_url"),
                "local_model_path": result.get("model_path")
            })
            
        except Exception as e:
            task_logger.error(f"Error processing {s3_key}: {e}")
            results.append({
                "s3_key": s3_key,
                "status": "error",
                "message": str(e)
            })
    
    self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Batch processing complete'})
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    
    return {
        "status": "completed",
        "total_processed": len(image_s3_keys),
        "successful": success_count,
        "failed": len(image_s3_keys) - success_count,
        "results": results
    }

@app.task(name='generate_image_sdxl_turbo', bind=True,queue='gpu')
def generate_image_sdxl_turbo(
    self, 
    prompt, 
    negative_prompt=None,
    width=1024, 
    height=1024,
    num_inference_steps=2,
    guidance_scale=0.0,
    seed=None,
    enhance_prompt=True,
    output_s3_prefix="sdxl_turbo",
    doc_id=None,
    update_collection=None,
    category_key=None,
    item_key=None
):
    """
    Celery task to generate high-quality images using SDXL Turbo model.
    Optimized for 15-20GB VRAM with memory management.
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt (optional)
        width: Image width (default 1024)
        height: Image height (default 1024)  
        num_inference_steps: Number of denoising steps (1-4 for Turbo)
        guidance_scale: Guidance scale (0.0 for Turbo)
        seed: Random seed for reproducibility
        enhance_prompt: Whether to enhance prompt for 3D generation
        output_s3_prefix: S3 prefix for uploaded image
        doc_id: MongoDB document ID to update
        update_collection: Collection name to update
    """
    task_logger.info("🚀 Starting SDXL Turbo image generation task")
    task_logger.info(f"   Prompt: {prompt}")
    task_logger.info(f"   Dimensions: {width}x{height}")
    task_logger.info(f"   Steps: {num_inference_steps}")
    task_logger.info(f"   SDXL modules loaded: {TASK_SDXL_MODULES_LOADED}")
    
    # Debug logging for MongoDB parameters
    task_logger.info(f"🔍 MongoDB Parameters Debug:")
    task_logger.info(f"   doc_id: {doc_id}")
    task_logger.info(f"   update_collection: {update_collection}")
    task_logger.info(f"   category_key: {category_key}")
    task_logger.info(f"   item_key: {item_key}")
    task_logger.info(f"   output_s3_prefix: {output_s3_prefix}")
    
    if not TASK_SDXL_MODULES_LOADED:
        error_msg = "SDXL Turbo modules not loaded on this worker. Please check module installation."
        task_logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # Initialize managers
    s3_mgr = get_s3_manager()
    mongo_mgr = get_mongo_helper()
    
    try:
        # Step 1: Initialize SDXL worker
        self.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Initializing SDXL Turbo...'})
        
        task_logger.info("🔧 Initializing SDXL Turbo worker...")
        from sdxl_turbo_worker import get_sdxl_worker
        
        sdxl_worker = get_sdxl_worker()
        
        # Check worker health
        health = sdxl_worker.health_check()
        task_logger.info(f"SDXL worker health: {health}")
        
        # Step 2: Load model if needed
        if not health.get('model_loaded', False):
            self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Loading SDXL Turbo model...'})
            task_logger.info("⚙️ Loading SDXL Turbo model...")
            
            if not sdxl_worker.load_model():
                error_msg = "Failed to load SDXL Turbo model"
                task_logger.error(f"❌ {error_msg}")
                return {"status": "error", "message": error_msg}
            
            task_logger.info("✅ SDXL Turbo model loaded successfully")
        
        # Step 3: Generate image
        self.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Generating image...'})
        task_logger.info("🎨 Starting image generation...")

        # Use outputs/images directory instead of temp
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_IMAGES_DIR, f"sdxl_turbo_{uuid.uuid4().hex[:8]}.png")

        # Generate and save image
        output_path, metadata = sdxl_worker.generate_and_save(
            prompt=prompt,
            output_dir=OUTPUT_IMAGES_DIR,
            filename_prefix="sdxl_turbo",
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            enhance_prompt=enhance_prompt
        )
        
        if output_path is None:
            error_msg = f"Image generation failed: {metadata.get('error', 'Unknown error')}"
            task_logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
        
        task_logger.info(f"✅ Image generated successfully: {output_path}")
        
        # Step 4: Upload to S3
        self.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Uploading to S3...'})
        
        s3_url = None
        if s3_mgr:
            upload_result = s3_mgr.upload_image(
                output_path, 
                output_s3_prefix
            )
            
            if upload_result.get("status") == "success":
                s3_url = upload_result.get("s3_url")
                task_logger.info(f"✅ Image uploaded to S3: {s3_url}")
            else:
                task_logger.warning(f"Failed to upload image to S3: {upload_result.get('message')}")
        
        # Step 5: Update MongoDB using the new helper
        task_logger.info("🔍 Step 5: MongoDB Update Check")
        task_logger.info(f"   mongo_mgr available: {mongo_mgr is not None}")
        task_logger.info(f"   doc_id provided: {doc_id}")
        task_logger.info(f"   update_collection provided: {update_collection}")
        task_logger.info(f"   s3_url available: {s3_url}")
        
        mongodb_updated = False
        mongodb_update_error = None
        
        if mongo_mgr and doc_id and update_collection and s3_url:
            task_logger.info("✅ All conditions met, proceeding with MongoDB update...")
            self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Updating database...'})
            
            mongodb_updated, mongodb_update_error = update_image_url_in_mongodb(
                mongo_mgr,
                doc_id,
                update_collection,
                s3_url,
                local_image_path=output_path,
                category_key=category_key,
                item_key=item_key,
                metadata=metadata
            )
            
            task_logger.info(f"📊 MongoDB Update Result:")
            task_logger.info(f"   Updated: {mongodb_updated}")
            task_logger.info(f"   Error: {mongodb_update_error}")
        else:
            task_logger.warning("⚠️ MongoDB update skipped due to missing conditions:")
            if not mongo_mgr:
                task_logger.warning("   - mongo_mgr is None")
            if not doc_id:
                task_logger.warning("   - doc_id is None or empty")
            if not update_collection:
                task_logger.warning("   - update_collection is None or empty")
            if not s3_url:
                task_logger.warning("   - s3_url is None or empty")
        # Step 6: Final result
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Completed successfully'})
        result = {
            "status": "success",
            "message": "Image generated successfully with SDXL Turbo",
            "image_path": output_path,
            "metadata": metadata,
            "mongodb_updated": mongodb_updated
        }
        if s3_url:
            result["s3_image_url"] = s3_url
        if mongodb_update_error:
            result["mongodb_update_error"] = mongodb_update_error
        task_logger.info("✅ SDXL Turbo image generation completed successfully")
        return result

    except Exception as e:
        error_msg = f"SDXL Turbo image generation failed: {str(e)}"
        task_logger.error(f"❌ {error_msg}", exc_info=True)
        return {"status": "error", "message": error_msg}


@app.task(name='batch_generate_images_sdxl_turbo', bind=True)
def batch_generate_images_sdxl_turbo(
    self,
    prompts_list,
    batch_settings=None,
    output_s3_prefix="sdxl_turbo_batch"
):
    """
    Batch generate multiple images using SDXL Turbo
    
    Args:
        prompts_list: List of prompts to generate images for
        batch_settings: Common settings for all images
        output_s3_prefix: S3 prefix for batch upload
    """
    task_logger.info(f"🚀 Starting SDXL Turbo batch generation for {len(prompts_list)} prompts")
    
    if not TASK_SDXL_MODULES_LOADED:
        error_msg = "SDXL Turbo modules not loaded on this worker"
        task_logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # Default batch settings
    default_settings = {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "enhance_prompt": True
    }
    
    if batch_settings:
        default_settings.update(batch_settings)
    
    results = []
    total_prompts = len(prompts_list)
    
    try:
        # Initialize worker once for batch
        from sdxl_turbo_worker import get_sdxl_worker
        sdxl_worker = get_sdxl_worker()
        
        if not sdxl_worker.load_model():
            return {"status": "error", "message": "Failed to load SDXL Turbo model"}
        
        # Change prompts_list to a list of dicts: [{prompt, doc_id, update_collection, category_key, item_key}]
        for i, item in enumerate(prompts_list):
            prompt = item["prompt"]
            doc_id = item.get("doc_id")
            update_collection = item.get("update_collection")
            category_key = item.get("category_key")
            item_key = item.get("item_key")
            progress = int((i / total_prompts) * 100)
            self.update_state(
                state='PROGRESS', 
                meta={'progress': progress, 'status': f'Generating image {i+1}/{total_prompts}'}
            )
            task_logger.info(f"Generating image {i+1}/{total_prompts}: {prompt}")
            temp_output_dir = tempfile.mkdtemp()
            try:
                sdxl_result = generate_image_sdxl_turbo.apply_async(
                    args=[prompt],
                    kwargs={
                        "doc_id": doc_id,
                        "update_collection": update_collection,
                        "output_s3_prefix": output_s3_prefix,
                        "width": default_settings["width"],
                        "height": default_settings["height"],
                        "num_inference_steps": default_settings["num_inference_steps"],
                        "guidance_scale": default_settings["guidance_scale"],
                        "enhance_prompt": default_settings["enhance_prompt"],
                        "category_key": category_key,
                        "item_key": item_key
                    }
                )
                result = sdxl_result.get(timeout=180)
                results.append({
                    "prompt": prompt,
                    "status": result.get("status"),
                    "image_path": result.get("image_path"),
                    "metadata": result.get("metadata"),
                    "s3_image_url": result.get("s3_image_url"),
                    "mongodb_updated": result.get("mongodb_updated"),
                    "mongodb_update_error": result.get("mongodb_update_error")
                })
                if result.get("status") == "success":
                    task_logger.info(f"✅ Generated image {i+1}/{total_prompts}")
                else:
                    task_logger.error(f"❌ Failed to generate image {i+1}/{total_prompts}")
            finally:
                try:
                    import shutil
                    if os.path.exists(temp_output_dir):
                        shutil.rmtree(temp_output_dir)
                except Exception as e:
                    task_logger.warning(f"Failed to cleanup temp directory: {e}")
    
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Batch generation completed'})
        
        successful_count = len([r for r in results if r.get('status') == 'success'])
        task_logger.info(f"✅ Batch generation completed: {successful_count}/{total_prompts} successful")
        
        return {
            "status": "success",
            "message": f"Batch generation completed: {successful_count}/{total_prompts} successful",
            "results": results,
            "total_prompts": total_prompts,
            "successful_count": successful_count
        }
        
    except Exception as e:
        error_msg = f"Batch generation failed: {str(e)}"
        task_logger.error(f"❌ {error_msg}", exc_info=True)
        return {"image_generated": "error", "message": error_msg}

def update_image_url_in_mongodb(mongo_mgr, doc_id, update_collection, s3_url, local_image_path=None, category_key=None, item_key=None, metadata=None):
    """
    Helper to update MongoDB with the S3 image URL and metadata, similar to 3D asset update logic.
    """
    task_logger.info("🔧 Starting MongoDB image URL update...")
    task_logger.info(f"   doc_id: {doc_id}")
    task_logger.info(f"   update_collection: {update_collection}")
    task_logger.info(f"   s3_url: {s3_url}")
    task_logger.info(f"   category_key: {category_key}")
    task_logger.info(f"   item_key: {item_key}")
    
    try:
        # Import MONGO_DB_NAME in the function to avoid import issues
        try:
            from config import MONGO_DB_NAME
        except ImportError:
            MONGO_DB_NAME = "World_builder"  # fallback default
            
        if not (mongo_mgr and doc_id and update_collection and s3_url):
            error_msg = "Missing required parameters for MongoDB update"
            task_logger.error(f"❌ {error_msg}")
            task_logger.error(f"   mongo_mgr: {mongo_mgr is not None}")
            task_logger.error(f"   doc_id: {bool(doc_id)}")
            task_logger.error(f"   update_collection: {bool(update_collection)}")
            task_logger.error(f"   s3_url: {bool(s3_url)}")
            return False, error_msg
            
        update_data = {}
        if category_key and item_key:
            update_path = f"possible_structures.{category_key}.{item_key}.imageUrl"
            update_data[update_path] = s3_url
            update_data[f"possible_structures.{category_key}.{item_key}.status"] = "image_generated"
            if local_image_path:
                update_data[f"possible_structures.{category_key}.{item_key}.local_image_path"] = local_image_path
            if metadata:
                update_data[f"possible_structures.{category_key}.{item_key}.image_metadata"] = metadata
            task_logger.info(f"📝 Nested update path: {update_path}")
        else:
            update_data["image_path"] = s3_url
            update_data["status"] = "image_generated"
            if local_image_path:
                update_data["local_image_path"] = local_image_path
            if metadata:
                update_data["image_metadata"] = metadata
            task_logger.info("📝 Root-level update: image_path")
        
        task_logger.info(f"📋 Update data: {update_data}")
        update_op = {"$set": update_data}
        
        task_logger.info("🔄 Executing MongoDB update...")
        result_count = mongo_mgr.update_by_id(
            db_name=MONGO_DB_NAME,
            collection_name=update_collection,
            document_id=doc_id,
            update=update_op
        )
        
        task_logger.info(f"📊 Update result count: {result_count}")
        
        if result_count > 0:
            task_logger.info(f"✅ Updated MongoDB document {doc_id} in {update_collection} with S3 URL: {s3_url}")
            return True, None
        else:
            warning_msg = f"No MongoDB document updated for {doc_id} in {update_collection}"
            task_logger.warning(f"⚠️ {warning_msg}")
            return False, "No document updated"
    except Exception as e:
        error_msg = f"Failed to update MongoDB with S3 URL: {e}"
        task_logger.error(f"❌ {error_msg}", exc_info=True)
        return False, str(e)
