# src/tasks.py

import os
import logging
from celery import Celery
import numpy as np
from PIL import Image
from datetime import datetime
import pymongo
from celery.signals import worker_process_init
import uuid

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
task_logger = logging.getLogger('celery_tasks')

# Initialize default values
DEFAULT_TEXT_MODEL = "flux"
DEFAULT_GRID_MODEL = "flux"
OUTPUT_DIR = "/tmp/output"
OUTPUT_IMAGES_DIR = "/tmp/output/images"
OUTPUT_3D_ASSETS_DIR = "/tmp/output/3d_assets"
MONGO_DB_NAME = "biome_db"
MONGO_BIOME_COLLECTION = "biomes"

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

# Try to import real modules
try:
    from config import (
        DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL,
        OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_3D_ASSETS_DIR,
        MONGO_DB_NAME, MONGO_BIOME_COLLECTION,
        HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
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
        # Try importing the specific Hunyuan3D modules
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        import trimesh
        
        task_logger.info("✓ Hunyuan3D core modules imported successfully")
        
        # Try importing the worker functions
        from hunyuan3d_worker import (
            initialize_hunyuan3d_processors, 
            generate_3d_from_image_core,
            get_model_info,
            cleanup_models
        )
        
        task_logger.info("✓ Hunyuan3D worker functions imported successfully")
        
        TASK_3D_MODULES_LOADED = True
        task_logger.info("✅ All necessary Hunyuan3D modules loaded for Celery worker.")
        
except ImportError as e:
    task_logger.error(f"✗ Could not load Hunyuan3D modules for Celery worker: {e}")
    task_logger.error("   Make sure you have installed: transformers, diffusers, torch with CUDA support")
    TASK_3D_MODULES_LOADED = False
except Exception as e:
    task_logger.error(f"✗ Unexpected error loading Hunyuan3D modules: {e}")
    TASK_3D_MODULES_LOADED = False

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
_hunyuan_floater_remove_worker = None
_hunyuan_degenerate_face_remove_worker = None
_hunyuan_face_reduce_worker = None

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
def generate_text_image(prompt: str, width: int, height: int, num_images: int, model_type: str):
    """Celery task to process text prompt and generate images."""
    global _text_processor, _pipeline
    
    if not ensure_processors_initialized():
        return {"status": "error", "message": "Worker not fully initialized or modules missing."}

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
        if len(images) > 1:
            grid_image = create_image_grid(images)
            unique_id = str(uuid.uuid4())[:8]
            grid_filename = f"text_grid_{unique_id}.png"
            grid_path = os.path.join(OUTPUT_IMAGES_DIR, grid_filename)
            grid_image.save(grid_path)
            image_relative_paths.append(grid_filename)

            for i, img in enumerate(images):
                single_filename = f"text_{unique_id}_{i}.png"
                single_img_path = os.path.join(OUTPUT_IMAGES_DIR, single_filename)
                img.save(single_img_path)
                image_relative_paths.append(single_filename)
            message = f"Generated {len(images)} images (grid). Saved to {OUTPUT_IMAGES_DIR}"
        else:
            unique_id = str(uuid.uuid4())[:8]
            img_filename = f"text_image_{unique_id}.png"
            img_path = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
            images[0].save(img_path)
            image_relative_paths.append(img_filename)
            message = f"Generated 1 image. Saved to {OUTPUT_IMAGES_DIR}"

        return {"status": "success", "message": message, "image_filenames": image_relative_paths}

    except Exception as e:
        task_logger.error(f"Error in generate_text_image task for prompt '{prompt[:50]}': {e}", exc_info=True)
        return {"status": "error", "message": f"Task failed: {e}"}

@app.task(name='generate_grid_image')
def generate_grid_image(grid_string: str, width: int, height: int, num_images: int, model_type: str):
    """Celery task to process grid data and generate terrain images."""
    global _grid_processor, _pipeline
    
    if not ensure_processors_initialized():
        return {"status": "error", "message": "Worker not fully initialized or modules missing."}

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
        
        # Save visualization if available
        if grid_viz is not None:
            viz_filename = f"grid_visualization_{unique_id}.png"
            viz_path = os.path.join(OUTPUT_IMAGES_DIR, viz_filename)
            grid_viz.save(viz_path)
            image_filenames.append(viz_filename)

        if len(images) > 1:
            grid_image = create_image_grid(images)
            grid_filename = f"terrain_grid_{unique_id}.png"
            grid_path = os.path.join(OUTPUT_IMAGES_DIR, grid_filename)
            grid_image.save(grid_path)
            image_filenames.append(grid_filename)
            
            for i, img in enumerate(images):
                single_filename = f"terrain_{unique_id}_{i}.png"
                single_img_path = os.path.join(OUTPUT_IMAGES_DIR, single_filename)
                img.save(single_img_path)
                image_filenames.append(single_filename)
            message = f"Generated {len(images)} images (grid). Saved to {OUTPUT_IMAGES_DIR}"
        else:
            img_filename = f"terrain_image_{unique_id}.png"
            img_path = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
            images[0].save(img_path)
            image_filenames.append(img_filename)
            message = f"Generated 1 image. Saved to {OUTPUT_IMAGES_DIR}"

        return {"status": "success", "message": message, "image_filenames": image_filenames}

    except Exception as e:
        task_logger.error(f"Error in generate_grid_image task: {e}", exc_info=True)
        return {"status": "error", "message": f"Task failed: {e}"}

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
    
    try:
        query = {"$or": [{"theme_prompt": {"$exists": True}}, {"description": {"$exists": True}}]}
        prompt_documents = mongo_helper.find_many(MONGO_DB_NAME, collection_name, query=query, limit=limit)
        
        if not prompt_documents:
            return {"status": "success", "message": "No prompts found to process in batch."}
    except Exception as e:
        task_logger.error(f"Error fetching prompts for batch processing from MongoDB: {e}")
        return {"status": "error", "message": f"Failed to fetch prompts for batch: {e}"}

    results = []
    
    for doc in prompt_documents:
        doc_id = str(doc.get("_id"))
        prompt = doc.get("theme_prompt") or doc.get("description")
        
        if not prompt and "possible_structures" in doc: 
            for category_key in doc["possible_structures"]:
                for item_key in doc["possible_structures"][category_key]:
                    if "description" in doc["possible_structures"][category_key][item_key]:
                        prompt = doc["possible_structures"][category_key][item_key]["description"]
                        break
                if prompt: 
                    break
        
        if not prompt:
            results.append(f"Skipping {doc_id}: No valid prompt found.")
            continue
        
        task_logger.info(f"Batch processing prompt: '{prompt[:50]}'")
        try:
            if hasattr(_text_processor, 'model_type') and _text_processor.model_type != model_type:
                _text_processor = TextProcessor(model_type=model_type)
                _pipeline = Pipeline(_text_processor, _grid_processor)

            images = _pipeline.process_text(prompt)
            
            if images and len(images) > 0: 
                unique_id = str(uuid.uuid4())[:8]
                image_filename = f"mongo_batch_{unique_id}_{doc_id[:8]}.png" 
                image_path = os.path.join(OUTPUT_IMAGES_DIR, image_filename)
                images[0].save(image_path)
                results.append(f"Generated image for: '{prompt[:30]}...' -> {image_filename}")
                
                if update_db:
                    update_data = {
                        "processed": True,
                        "processed_at": datetime.now(),
                        "model_used": model_type,
                        "image_path": image_filename 
                    }
                    mongo_helper.update_by_id(MONGO_DB_NAME, collection_name, doc_id, update_data)
            else:
                results.append(f"Failed to generate image for: '{prompt[:30]}' - No image output.")
        except Exception as e:
            task_logger.error(f"Error in batch processing for {doc_id}: {e}", exc_info=True)
            results.append(f"Failed to process '{prompt[:30]}...' - Error: {e}")
            
    return {"status": "success", "message": "\n".join(results)}

@app.task(name='generate_3d_model_from_image', bind=True)
def generate_3d_model_from_image(self, image_path, with_texture=False, output_format='glb'):
    """
    Celery task to generate a 3D model from an image using Hunyuan3D.
    """
    task_logger.info(f"🚀 Starting 3D model generation task")
    task_logger.info(f"   Image path: {image_path}")
    task_logger.info(f"   With texture: {with_texture}")
    task_logger.info(f"   Output format: {output_format}")
    task_logger.info(f"   Hunyuan3D modules loaded: {TASK_3D_MODULES_LOADED}")
    
    if not TASK_3D_MODULES_LOADED:
        error_msg = "Hunyuan3D modules not loaded on this worker. Please check module installation."
        task_logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # Initialize processors if needed
    try:
        task_logger.info("🔧 Checking Hunyuan3D processor initialization...")
        if _hunyuan_i23d_worker is None:
            task_logger.info("⚙️ Initializing Hunyuan3D processors...")
            self.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Initializing Hunyuan3D processors...'})
            
            if not initialize_hunyuan3d_processors():
                error_msg = "Failed to initialize Hunyuan3D processors."
                task_logger.error(f"❌ {error_msg}")
                return {"status": "error", "message": error_msg}
            
            task_logger.info("✅ Hunyuan3D processors initialized successfully")
        else:
            task_logger.info("✅ Hunyuan3D processors already initialized")
    except Exception as e:
        error_msg = f"Error during processor initialization: {str(e)}"
        task_logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}
    
    # Generate 3D model
    try:
        task_logger.info("🎯 Starting 3D model generation...")
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Generating 3D model...'})
        
        result = generate_3d_from_image_core(
            image_path, 
            with_texture, 
            output_format, 
            progress_callback=lambda p, s: self.update_state(
                state='PROGRESS', 
                meta={'progress': min(p, 95), 'status': s}
            )
        )
        
        if result.get('status') == 'success':
            task_logger.info(f"✅ 3D model generated successfully: {result.get('model_path', 'No path returned')}")
            self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Completed successfully'})
        else:
            task_logger.error(f"❌ 3D model generation failed: {result.get('message', 'Unknown error')}")
            
        return result
        
    except Exception as e:
        error_msg = f"Error generating 3D model: {str(e)}"
        task_logger.error(f"❌ {error_msg}", exc_info=True)
        return {"status": "error", "message": error_msg}

@app.task(name='generate_3d_model_from_prompt', bind=True)
def generate_3d_model_from_prompt(self, text_prompt, with_texture=False, output_format='glb'):
    """
    Celery task to generate a 3D model from a text prompt using Hunyuan3D.
    """
    if not TASK_3D_MODULES_LOADED or not TASK_2D_MODULES_LOADED:
        return {"status": "error", "message": "Required modules not loaded on this worker."}
    
    try:
        task_logger.info(f"Starting text-to-3D pipeline for prompt: '{text_prompt[:50]}...'")
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Initializing processors...'})
        
        # Initialize processors if needed
        if _hunyuan_i23d_worker is None:
            if not initialize_hunyuan3d_processors():
                return {"status": "error", "message": "Failed to initialize Hunyuan3D processors."}
        
        if not ensure_processors_initialized():
            return {"status": "error", "message": "Failed to initialize 2D pipeline processors."}
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Generating image from text...'})
        
        # Generate image from text
        task_logger.info(f"Generating image from text prompt: '{text_prompt[:50]}'")
        images = _pipeline.process_text(text_prompt)
        
        if not images or len(images) == 0:
            return {"status": "error", "message": "Failed to generate image from text prompt."}
        
        # Save the generated image
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        image_path = os.path.join(OUTPUT_IMAGES_DIR, f"text_to_3d_{unique_id}.png")
        images[0].save(image_path)
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Converting image to 3D...'})
        
        # Now generate 3D model from the image
        task_logger.info(f"Generating 3D model from generated image: {image_path}")
        
        # Call the core 3D generation logic
        result = _generate_3d_from_image_core(
            image_path, 
            with_texture, 
            output_format, 
            progress_callback=lambda p, s: self.update_state(
                state='PROGRESS', 
                meta={'progress': 50 + (p * 0.5), 'status': s}
            )
        )
        
        # Combine results
        if result["status"] == "success":
            result["image_path"] = image_path
            result["message"] = f"Successfully generated 3D model from text prompt: '{text_prompt[:50]}...'"
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete'})
        
        return result
    
    except Exception as e:
        task_logger.error(f"Error in text-to-3D pipeline: {e}", exc_info=True)
        return {"status": "error", "message": f"Text-to-3D pipeline failed: {e}"}

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
