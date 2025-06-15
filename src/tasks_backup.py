# src/tasks.py

import os
import logging
from celery import Celery
import numpy as np
from PIL import Image
from datetime import datetime
import pymongo # Used for pymongo.results.ObjectId if you have it in your db_helper
from celery.signals import worker_process_init # For lazy loading models
import uuid # Needed for unique filenames in tasks

# --- Imports for actual processing logic ---
# These paths are crucial. They must be correct relative to where tasks.py will run
# On the GPU EC2, your 'src' directory (or just the relevant parts) should be present.

# Initialize default values for fallbacks
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
        # Return dummy image data
        return []
    
    def process_grid(self, grid_string):
        # Return dummy image data and viz
        return [], None

class DummyMongoHelper:
    def find_many(self, db_name, collection_name, query=None, limit=0):
        return []
    
    def update_by_id(self, db_name, collection_name, doc_id, update):
        return 0

def dummy_create_image_grid(images):
    return None

def dummy_save_image(image, path):
    pass

def dummy_generate_biome(theme, structure_list):
    return "Biome generation not available"

def dummy_get_aws_manager():
    return None

def dummy_generate_3d_from_image_core(image_path, with_texture=False, output_format='glb', progress_callback=None):
    return {"status": "error", "message": "Hunyuan3D modules not loaded"}

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

try:
    from config import (
        DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL,
        OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_3D_ASSETS_DIR, # Ensure these are paths on the worker machine
        MONGO_DB_NAME, MONGO_BIOME_COLLECTION,
        # Hunyuan3D configuration
        HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
        HUNYUAN3D_STEPS, HUNYUAN3D_GUIDANCE_SCALE, HUNYUAN3D_OCTREE_RESOLUTION,
        HUNYUAN3D_REMOVE_BACKGROUND, HUNYUAN3D_NUM_CHUNKS, HUNYUAN3D_ENABLE_FLASHVDM,
        HUNYUAN3D_COMPILE, HUNYUAN3D_LOW_VRAM_MODE, HUNYUAN3D_DEVICE,
        # AWS configuration
        AWS_REGION, AWS_GPU_INSTANCE_ID, AWS_MAX_STARTUP_WAIT_TIME, AWS_EC2_CHECK_INTERVAL,
        # Task routing and timeouts
        CELERY_TASK_ROUTES, TASK_TIMEOUT_3D_GENERATION, TASK_TIMEOUT_2D_GENERATION, TASK_TIMEOUT_EC2_MANAGEMENT
    )
    from pipeline.text_processor import TextProcessor
    from pipeline.grid_processor import GridProcessor
    from pipeline.pipeline import Pipeline
    # from terrain.grid_parser import GridParser # Only needed if GridProcessor doesn't handle visualization internally
    from utils.image_utils import save_image, create_image_grid
    from db_helper import MongoDBHelper # For saving results to DB

    # Biome Generation modules (these need to be present on the worker)
    from text_grid.structure_registry import get_biome_names, fetch_biome
    from text_grid.grid_generator import generate_biome

    # AWS Manager for EC2 operations
    from aws_manager import get_aws_manager

    # Flag for 2D Pipeline modules
    TASK_2D_MODULES_LOADED = True
    print("INFO: All necessary 2D pipeline task modules loaded for Celery worker.")
except ImportError as e:
    print(f"ERROR: Could not load all 2D task modules for Celery worker: {e}")
    print("Please ensure your 'src' directory and its subfolders (config, pipeline, db_helper, text_grid, utils, aws_manager) are accessible to the Celery worker.")
    TASK_2D_MODULES_LOADED = False

# --- Try to import Hunyuan3D modules ---
# Initialize Hunyuan3D-related variables with defaults
_hunyuan_i23d_worker = None
_hunyuan_rembg_worker = None
_hunyuan_texgen_worker = None
initialize_hunyuan3d_processors = None
generate_3d_from_image_core = None

try:
    # Hunyuan3D modules (these need to be present on the GPU worker)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    import torch
    import trimesh
      # Import Hunyuan3D worker functions
    from hunyuan3d_worker import (
        initialize_hunyuan3d_processors, 
        generate_3d_from_image_core,
        get_model_info,
        cleanup_models
    )
    
    # Create alias for the core function
    _generate_3d_from_image_core = generate_3d_from_image_core

    # Flag for 3D modules
    TASK_3D_MODULES_LOADED = True
    print("INFO: All necessary Hunyuan3D modules loaded for Celery worker.")
except ImportError as e:
    print(f"ERROR: Could not load Hunyuan3D modules for Celery worker: {e}")
    print("Please ensure Hunyuan3D-2 is installed and accessible to the Celery worker.")
    TASK_3D_MODULES_LOADED = False
    # Use dummy functions
    _generate_3d_from_image_core = dummy_generate_3d_from_image_core

# Overall flag for task modules
TASK_MODULES_LOADED = TASK_2D_MODULES_LOADED

# --- Celery App Setup ---
# The broker URL should point to your Redis instance.
# It will be read from the REDIS_BROKER_URL environment variable or config.py
# If not set, it defaults to a local Redis instance.
try:
    from config import REDIS_BROKER_URL, REDIS_RESULT_BACKEND, CELERY_TASK_ROUTES
    broker_url = REDIS_BROKER_URL
    result_backend = REDIS_RESULT_BACKEND
    task_routes = CELERY_TASK_ROUTES
except ImportError:
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

# Set up logging for Celery tasks
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
task_logger = logging.getLogger('celery_tasks')

# Global variables for processors (will be initialized lazily within tasks if needed)
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

# --- Lazy Initialization of Processors ---
# Use Celery's worker_process_init signal to load models once per worker process.
# This avoids loading models multiple times if the same process handles many tasks.
@worker_process_init.connect
def initialize_processors_for_worker(**kwargs):
    global _text_processor, _grid_processor, _pipeline
    if not TASK_2D_MODULES_LOADED:
        task_logger.error("Skipping processor initialization: Core task modules failed to load.")
        return

    try:
        task_logger.info("Initializing TextProcessor and GridProcessor for Celery worker process...")
        if TASK_2D_MODULES_LOADED:
            from config import DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL
            from pipeline.text_processor import TextProcessor
            from pipeline.grid_processor import GridProcessor
            from pipeline.pipeline import Pipeline
            
            _text_processor = TextProcessor(model_type=DEFAULT_TEXT_MODEL)
            _grid_processor = GridProcessor(model_type=DEFAULT_GRID_MODEL)
            _pipeline = Pipeline(_text_processor, _grid_processor)
            task_logger.info("Processors initialized/ready for task execution.")
    except Exception as e:
        task_logger.critical(f"FATAL: Failed to initialize core processors in Celery worker: {e}", exc_info=True)
        # You might want to exit the worker or mark it unhealthy here

# --- Lazy Initialization of Hunyuan3D processors ---
def initialize_hunyuan3d_processors():
    """Initialize Hunyuan3D processors if not already initialized"""
    global _hunyuan_i23d_worker, _hunyuan_rembg_worker, _hunyuan_texgen_worker
    global _hunyuan_floater_remove_worker, _hunyuan_degenerate_face_remove_worker, _hunyuan_face_reduce_worker
    
    if not TASK_3D_MODULES_LOADED:
        task_logger.error("Hunyuan3D modules not loaded. Unable to initialize processors.")
        return False
    
    try:
        # Import required modules for 3D processing
        from config import (
            HUNYUAN3D_MODEL_PATH, HUNYUAN3D_SUBFOLDER, HUNYUAN3D_TEXGEN_MODEL_PATH,
            HUNYUAN3D_DEVICE, HUNYUAN3D_ENABLE_FLASHVDM, HUNYUAN3D_COMPILE, HUNYUAN3D_LOW_VRAM_MODE
        )
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        
        if _hunyuan_rembg_worker is None:
            task_logger.info("Initializing Hunyuan3D Background Remover...")
            _hunyuan_rembg_worker = BackgroundRemover()
        
        if _hunyuan_i23d_worker is None:
            task_logger.info(f"Initializing Hunyuan3D Shape Generator with model: {HUNYUAN3D_MODEL_PATH}/{HUNYUAN3D_SUBFOLDER}...")
            _hunyuan_i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN3D_MODEL_PATH,
                subfolder=HUNYUAN3D_SUBFOLDER,
                use_safetensors=True,
                device=HUNYUAN3D_DEVICE
            )
            
            if HUNYUAN3D_ENABLE_FLASHVDM and HUNYUAN3D_DEVICE != 'cpu':
                task_logger.info("Enabling FlashVDM for faster inference...")
                _hunyuan_i23d_worker.enable_flashvdm()
                
            if HUNYUAN3D_COMPILE:
                task_logger.info("Compiling model for faster inference...")
                _hunyuan_i23d_worker.compile()
        
        if _hunyuan_texgen_worker is None and HUNYUAN3D_TEXGEN_MODEL_PATH:
            try:
                task_logger.info(f"Initializing Hunyuan3D Texture Generator with model: {HUNYUAN3D_TEXGEN_MODEL_PATH}...")
                _hunyuan_texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN3D_TEXGEN_MODEL_PATH)
                
                if HUNYUAN3D_LOW_VRAM_MODE:
                    task_logger.info("Enabling CPU offload for texture generator in low VRAM mode...")
                    _hunyuan_texgen_worker.enable_model_cpu_offload()
            except Exception as e:
                task_logger.error(f"Failed to load texture generator: {e}")
                task_logger.warning("Continuing without texture generation capability.")
        
        if _hunyuan_floater_remove_worker is None:
            task_logger.info("Initializing mesh post-processing tools...")
            _hunyuan_floater_remove_worker = FloaterRemover()
            _hunyuan_degenerate_face_remove_worker = DegenerateFaceRemover()
            _hunyuan_face_reduce_worker = FaceReducer()
        
        task_logger.info("Hunyuan3D processors initialized successfully.")
        return True
    
    except Exception as e:
        task_logger.error(f"Failed to initialize Hunyuan3D processors: {e}", exc_info=True)
        return False

# --- Celery Tasks ---

@app.task(name='generate_text_image')
def generate_text_image(prompt: str, width: int, height: int, num_images: int, model_type: str):
    """Celery task to process text prompt and generate images."""
    global _text_processor, _pipeline
    if not TASK_MODULES_LOADED or _pipeline is None:
        # Re-attempt initialization if not ready (e.g., if worker_process_init failed for some reason)
        initialize_processors_for_worker()
        if _pipeline is None: # If still None after re-attempt
            return {"status": "error", "message": "Worker not fully initialized or modules missing."}

    try:
        task_logger.info(f"Task: Processing text prompt: '{prompt[:50]}' with model {model_type}")
        
        # Ensure correct model is used - re-initialize if type changes
        if _text_processor.model_type != model_type:
            task_logger.info(f"Re-initializing TextProcessor to {model_type} for task.")
             # Re-declare global to modify
            _text_processor = TextProcessor(model_type=model_type)
            _pipeline = Pipeline(_text_processor, _grid_processor)

        images = _pipeline.process_text(prompt)
        
        if not images:
            task_logger.error("No images were generated by the pipeline.")
            return {"status": "error", "message": "No images generated."}
        
        # Save images to the OUTPUT_IMAGES_DIR on the worker machine
        image_relative_paths = []
        if len(images) > 1:
            grid_image = create_image_grid(images)
            unique_id = str(uuid.uuid4())[:8]
            grid_filename = f"text_grid_{unique_id}.png"
            grid_path = os.path.join(OUTPUT_IMAGES_DIR, grid_filename)
            grid_image.save(grid_path)
            image_relative_paths.append(grid_filename) # Store just the filename/relative path

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
    if not TASK_MODULES_LOADED or _pipeline is None:
        initialize_processors_for_worker()
        if _pipeline is None:
            return {"status": "error", "message": "Worker not fully initialized or modules missing."}

    try:
        task_logger.info(f"Task: Processing grid data with model {model_type}")

        if _grid_processor.model_type != model_type:
            task_logger.info(f"Re-initializing GridProcessor to {model_type} for task.")
            
            _grid_processor = GridProcessor(model_type=model_type)
            _pipeline = Pipeline(_text_processor, _grid_processor)

        images, grid_viz = _pipeline.process_grid(grid_string)
        
        if not images:
            task_logger.error("No images were generated by the pipeline.")
            return {"status": "error", "message": "No images generated."}

        unique_id = str(uuid.uuid4())[:8]
        viz_filename = f"grid_visualization_{unique_id}.png"
        viz_path = os.path.join(OUTPUT_IMAGES_DIR, viz_filename)
        grid_viz.save(viz_path)
        
        image_filenames = [viz_filename] # Include viz image filename

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
    if not TASK_MODULES_LOADED or _pipeline is None:
        initialize_processors_for_worker()        if _pipeline is None:
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
                if prompt: break
        
        if not prompt:
            results.append(f"Skipping {doc_id}: No valid prompt found.")
            continue
        
        task_logger.info(f"Batch processing prompt: '{prompt[:50]}'")
        try:
            if _text_processor.model_type != model_type:
                
                _text_processor = TextProcessor(model_type=model_type)
                _pipeline = Pipeline(_text_processor, _grid_processor)

            images = _pipeline.process_text(prompt)
            
            if images and images[0]: 
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

# --- Hunyuan3D Tasks ---

@app.task(name='generate_3d_model_from_image', bind=True)
def generate_3d_model_from_image(self, image_path, with_texture=False, output_format='glb'):
    """
    Celery task to generate a 3D model from an image using Hunyuan3D.
    
    Args:
        image_path: Path to the input image
        with_texture: Whether to generate texture for the model
        output_format: Format of the output model file ('glb', 'obj', 'ply', 'stl')
        
    Returns:
        A dictionary with status, message, and paths to the generated files
    """
    if not TASK_3D_MODULES_LOADED:
        return {"status": "error", "message": "Hunyuan3D modules not loaded on this worker."}
    
    # Initialize processors if needed
    if _hunyuan_i23d_worker is None:
        if not initialize_hunyuan3d_processors():
            return {"status": "error", "message": "Failed to initialize Hunyuan3D processors."}
    
    try:
        # Import required modules and config
        from config import (
            OUTPUT_3D_ASSETS_DIR, HUNYUAN3D_REMOVE_BACKGROUND, HUNYUAN3D_STEPS,
            HUNYUAN3D_GUIDANCE_SCALE, HUNYUAN3D_OCTREE_RESOLUTION, HUNYUAN3D_NUM_CHUNKS
        )
        from PIL import Image as PILImage
        import torch
        import time
        
        task_logger.info(f"Starting 3D model generation for image: {image_path}")
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Loading image...'})
        
        # Prepare output paths
        unique_id = str(uuid.uuid4())[:8]
        model_dir = os.path.join(OUTPUT_3D_ASSETS_DIR, f"model_{unique_id}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Load image and metadata
        try:
            image = PILImage.open(image_path).convert('RGB')
        except Exception as e:
            return {"status": "error", "message": f"Failed to load image: {e}"}
        
        # Process timings
        time_meta = {}
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Preprocessing image...'})
          # Remove background if needed
        if HUNYUAN3D_REMOVE_BACKGROUND and _hunyuan_rembg_worker is not None:
            start_time = time.time()
            image = _hunyuan_rembg_worker(image)
            time_meta['remove_background'] = time.time() - start_time
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 40, 'status': 'Generating 3D shape...'})
        
        # Generate 3D shape
        start_time = time.time()
        generator = torch.Generator()
        generator = generator.manual_seed(int(time.time()))
        
        task_logger.info(f"Generating 3D shape with octree resolution: {HUNYUAN3D_OCTREE_RESOLUTION}, steps: {HUNYUAN3D_STEPS}")
        outputs = _hunyuan_i23d_worker(
            image=image,
            num_inference_steps=HUNYUAN3D_STEPS,
            guidance_scale=HUNYUAN3D_GUIDANCE_SCALE,
            generator=generator,
            octree_resolution=HUNYUAN3D_OCTREE_RESOLUTION,
            num_chunks=HUNYUAN3D_NUM_CHUNKS,
            output_type='mesh'
        )
        time_meta['shape_generation'] = time.time() - start_time
        
        # Get the mesh from outputs
        mesh = outputs
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Post-processing mesh...'})
          # Clean up the mesh
        if _hunyuan_floater_remove_worker is not None:
            mesh = _hunyuan_floater_remove_worker(mesh)
        if _hunyuan_degenerate_face_remove_worker is not None:
            mesh = _hunyuan_degenerate_face_remove_worker(mesh)
        
        # Save white (untextured) mesh
        white_mesh_path = os.path.join(model_dir, f"white_mesh.{output_format}")
        mesh.export(white_mesh_path)
        
        result = {
            "status": "success",
            "message": "Generated 3D model successfully",
            "white_mesh_path": white_mesh_path,
            "model_dir": model_dir,
            "time_meta": time_meta
        }
        
        # Apply texture if requested
        if with_texture and _hunyuan_texgen_worker is not None:
            # Update task progress
            self.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Generating texture...'})
            
            task_logger.info("Generating texture for the model...")
            start_time = time.time()
            
            # Create texture
            model_data = {'mesh': mesh, 'generator': generator}
            try:
                outputs = _hunyuan_texgen_worker(image, model_data)
                textured_mesh = outputs
                time_meta['texture_generation'] = time.time() - start_time
                
                # Save textured mesh
                textured_mesh_path = os.path.join(model_dir, f"textured_mesh.{output_format}")
                textured_mesh.export(textured_mesh_path)
                
                result["textured_mesh_path"] = textured_mesh_path
                result["message"] = "Generated 3D model with texture successfully"
                
                task_logger.info(f"Generated textured 3D model: {textured_mesh_path}")
            except Exception as e:
                task_logger.error(f"Texture generation failed: {e}")
                # Continue with untextured model
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete'})
        
        task_logger.info(f"Generated 3D model: {white_mesh_path}")
        return result
    
    except Exception as e:
        task_logger.error(f"Error generating 3D model: {e}", exc_info=True)
        return {"status": "error", "message": f"Task failed: {e}"}

@app.task(name='generate_3d_model_from_prompt', bind=True)
def generate_3d_model_from_prompt(self, text_prompt, with_texture=False, output_format='glb'):
    """
    Celery task to generate a 3D model from a text prompt using Hunyuan3D.
    This pipeline combines text-to-image and image-to-3D.
    
    Args:
        text_prompt: Text description for the model
        with_texture: Whether to generate texture for the model
        output_format: Format of the output model file ('glb', 'obj', 'ply', 'stl')
        
    Returns:
        A dictionary with status, message, and paths to the generated files
    """
    # First, we need to generate an image from text
    if not TASK_3D_MODULES_LOADED or not TASK_2D_MODULES_LOADED:
        return {"status": "error", "message": "Required modules not loaded on this worker."}
    
    try:
        from config import OUTPUT_IMAGES_DIR
        
        task_logger.info(f"Starting text-to-3D pipeline for prompt: '{text_prompt[:50]}...'")
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Initializing processors...'})
        
        # Initialize Hunyuan3D processors if needed
        if _hunyuan_i23d_worker is None:
            if not initialize_hunyuan3d_processors():
                return {"status": "error", "message": "Failed to initialize Hunyuan3D processors."}
        
        # Initialize 2D pipeline processors if needed
        if _pipeline is None:
            initialize_processors_for_worker()
            if _pipeline is None:
                return {"status": "error", "message": "Failed to initialize 2D pipeline processors."}
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Generating image from text...'})
        
        # Generate image from text
        task_logger.info(f"Generating image from text prompt: '{text_prompt[:50]}'")
        images = _pipeline.process_text(text_prompt)
        
        if not images or len(images) == 0:
            return {"status": "error", "message": "Failed to generate image from text prompt."}
        
        # Save the generated image
        unique_id = str(uuid.uuid4())[:8]
        image_path = os.path.join(OUTPUT_IMAGES_DIR, f"text_to_3d_{unique_id}.png")
        images[0].save(image_path)
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Converting image to 3D...'})        # Now generate 3D model from the image
        task_logger.info(f"Generating 3D model from generated image: {image_path}")
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Generating 3D from image...'})
        
        # Call the core 3D generation logic directly
        result = _generate_3d_from_image_core(image_path, with_texture, output_format, 
                                            progress_callback=lambda p, s: self.update_state(
                                                state='PROGRESS', 
                                                meta={'progress': 60 + (p * 0.4), 'status': s}
                                            ))
        
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

@app.task(name='manage_gpu_instance')
def manage_gpu_instance(action="ensure_running", instance_id=None, region=None):
    """
    Celery task to manage the GPU EC2 instance.
    
    Args:
        action: The action to perform ('ensure_running', 'stop', 'status')
        instance_id: The EC2 instance ID to manage
        region: AWS region of the instance
        
    Returns:
        A dictionary with status and message
    """
    try:
        from config import AWS_GPU_INSTANCE_ID, AWS_REGION, AWS_MAX_STARTUP_WAIT_TIME, AWS_EC2_CHECK_INTERVAL
        from aws_manager import get_aws_manager
        
        # Use provided IDs or fall back to config
        instance_id = instance_id or AWS_GPU_INSTANCE_ID
        region = region or AWS_REGION
        
        if not instance_id:
            return {"status": "error", "message": "No GPU instance ID provided or configured."}
        
        ec2_manager = get_aws_manager(instance_id, region)
        
        if action == "ensure_running":
            task_logger.info(f"Ensuring GPU instance {instance_id} is running...")
            success = ec2_manager.ensure_instance_running(
                max_wait_time=AWS_MAX_STARTUP_WAIT_TIME,
                check_interval=AWS_EC2_CHECK_INTERVAL
            )
            
            if success:
                return {"status": "success", "message": f"GPU instance {instance_id} is running."}
            else:
                return {"status": "error", "message": f"Failed to ensure GPU instance {instance_id} is running."}
                
        elif action == "stop":
            task_logger.info(f"Stopping GPU instance {instance_id}...")
            success = ec2_manager.stop_instance()
            
            if success:
                return {"status": "success", "message": f"Stop request sent for GPU instance {instance_id}."}
            else:
                return {"status": "error", "message": f"Failed to stop GPU instance {instance_id}."}
        
        elif action == "status":
            task_logger.info(f"Getting status of GPU instance {instance_id}...")
            info = ec2_manager.get_instance_info()
            
            if info:
                cost_info = ec2_manager.get_instance_cost_estimate()
                return {
                    "status": "success", 
                    "message": f"Instance {instance_id} status retrieved.",
                    "instance_info": {
                        "state": info.state,
                        "instance_type": info.instance_type,
                        "public_ip": info.public_ip,
                        "uptime_hours": info.uptime_hours
                    },
                    "cost_estimate": cost_info
                }
            else:
                return {"status": "error", "message": f"Failed to get status for instance {instance_id}."}
        
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    except Exception as e:
        task_logger.error(f"Error managing GPU instance: {e}", exc_info=True)
        return {"status": "error", "message": f"GPU instance management failed: {e}"}

@app.task(name='cleanup_old_assets')
def cleanup_old_assets(max_age_hours=24):
    """
    Celery task to clean up old generated assets to save disk space.
    
    Args:
        max_age_hours: Maximum age in hours before assets are deleted
        
    Returns:
        A dictionary with cleanup status and statistics
    """
    try:
        from config import OUTPUT_3D_ASSETS_DIR, OUTPUT_IMAGES_DIR
        import time
        from datetime import datetime, timedelta
        
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
        
        # Clean up old temporary images (but keep recent ones)
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
            "files_list": cleaned_files[:10]  # Return first 10 files for reference
        }
        
    except Exception as e:
        task_logger.error(f"Error during asset cleanup: {e}", exc_info=True)
        return {"status": "error", "message": f"Cleanup failed: {e}"}

