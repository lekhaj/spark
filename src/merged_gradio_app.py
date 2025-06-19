# merged_gradio_app.py - Frontend Gradio Application with Development/Production Toggle

import os
import time
import uuid 
import shutil 
import argparse
import logging
import gradio as gr
from PIL import Image, ImageDraw, ImageFont 
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import json 
import httpx 
import base64 
import pymongo # For ObjectId in MongoDB operations (local DB access)

try:
    import uvicorn
except ImportError:
    pass
from datetime import datetime

# --- ALL IMPORTS NOW REFER TO THE NEW CONSOLIDATED STRUCTURE ---
# Ensure config.py is correctly set up with these constants
from config import ( 
    DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL, 
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES,
    OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_3D_ASSETS_DIR, 
    MONGO_DB_NAME, MONGO_BIOME_COLLECTION,
    REDIS_BROKER_URL, # Imported, but only used if USE_CELERY is True and app.backend is used for result fetching
    USE_CELERY # <--- NEW: Flag to control Celery usage
)
from db_helper import MongoDBHelper 

# Global variables to store ID mappings for dropdowns
_prompt_id_mapping = {}
_grid_id_mapping = {}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_biome_logger = logging.getLogger("BiomeInspector") # Defined globally and early

# --- Conditional Imports based on USE_CELERY ---
if USE_CELERY:
    logger.info("Running in PRODUCTION mode (Celery Enabled).")
    # Import Celery task functions to call .delay() on them
    from tasks import generate_text_image as celery_generate_text_image, \
                      generate_grid_image as celery_generate_grid_image, \
                      run_biome_generation as celery_run_biome_generation, \
                      batch_process_mongodb_prompts_task as celery_batch_process_mongodb_prompts_task, \
                      generate_3d_model_from_image as celery_generate_3d_model_from_image, \
                      generate_3d_model_from_prompt as celery_generate_3d_model_from_prompt, \
                      manage_gpu_instance as celery_manage_gpu_instance
    # No direct model processor imports needed here, as they run on the worker.
    # from celery import Celery # Might need this if you want to track task results directly
    # from tasks import app as celery_app_instance # if you need to access backend results
else:
    logger.info("Running in DEVELOPMENT mode (Direct Processing).")
    # Import direct processing functions and their dependencies
    from pipeline.text_processor import TextProcessor 
    from pipeline.grid_processor import GridProcessor 
    from pipeline.pipeline import Pipeline 
    from utils.image_utils import save_image, create_image_grid 
    from text_grid.grid_generator import generate_biome # Direct call for biome generation

    # Global variables for processors in DEV mode
    _dev_text_processor = None
    _dev_grid_processor = None
    _dev_pipeline = None

    # 3D processors (mock implementations for DEV mode)
    def mock_generate_3d_from_image(image_path, with_texture=False, output_format='glb'):
        """Mock 3D generation from image for development mode."""
        logger.info(f"Mock 3D generation from image: {image_path}")
        return f"Mock 3D model generated from {image_path} (texture: {with_texture}, format: {output_format})"
    
    def mock_generate_3d_from_prompt(prompt, with_texture=False, output_format='glb'):
        """Mock 3D generation from text prompt for development mode."""
        logger.info(f"Mock 3D generation from prompt: {prompt}")
        return f"Mock 3D model generated from prompt: '{prompt}' (texture: {with_texture}, format: {output_format})"

    def initialize_dev_processors():
        """Initialize the 2D processors and pipeline for direct execution in DEV mode."""
        global _dev_text_processor, _dev_grid_processor, _dev_pipeline
        logger.info("Initializing 2D processors for direct execution (DEV mode)...")
        _dev_text_processor = TextProcessor(model_type=DEFAULT_TEXT_MODEL)
        _dev_grid_processor = GridProcessor(model_type=DEFAULT_GRID_MODEL) 
        _dev_pipeline = Pipeline(_dev_text_processor, _dev_grid_processor)
        logger.info("2D processors initialized for direct execution.")


# --- Imports for functions that remain local (e.g., UI display, helper functions) ---
# These are functions that do not perform heavy computation and can run on the FastAPI server
# They also provide mocks if the actual modules are not found (for standalone testing)
try:
    from text_grid.structure_registry import get_biome_names, fetch_biome 
    _biome_logger.info("INFO: Loaded actual text_grid.structure_registry.")
except ImportError:
    _biome_logger.warning("WARNING: Could not import text_grid.structure_registry. Using mock functions.")
    
    _mock_biomes_db = {
        "A_mock_forest": {
            "theme": "A lush forest with ancient ruins",
            "structures": ["Tree", "Stone Arch"],
            "grid_data": [[0,1,0],[1,1,1],[0,1,0]],
            "details": "This is a mock forest biome detail."
        },
        "A_mock_desert": {
            "theme": "A vast, arid desert with hidden oases",
            "structures": ["Sand Dune", "Cactus"],
            "grid_data": [[4,4,4],[4,0,4],[4,4,4]],
            "details": "This is a mock desert biome detail."
        }
    }

    def get_biome_names(db_name, collection_name): 
        return list(_mock_biomes_db.keys())

    def fetch_biome(db_name, collection_name, name: str): 
        return _mock_biomes_db.get(name)

    # Mock for generate_biome if running in DEV mode without actual text_grid.grid_generator
    # This mock is for when the *mock* import fails for the `text_grid.grid_generator`
    # and not used directly by the main app in dev mode.
    # The actual `generate_biome` used in DEV mode comes from `text_grid.grid_generator` directly.
    async def generate_biome_mock(theme: str, structure_type_list: list[str]): 
        new_biome_name = f"Generated_Biome_{len(_mock_biomes_db) + 1}"
        _mock_biomes_db[new_biome_name] = {
            "theme": theme,
            "structures": structure_type_list,
            "details": f"Mock details for a biome generated with theme: '{theme}' and structures: {structure_type_list}.",
            "grid_data": [[0,0,0],[0,0,0],[0,0,0]] 
        }
        return f"✅ Mock Biome '{new_biome_name}' generated successfully!"


# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Create output directories (ensure they exist early on the frontend machine)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True) 
os.makedirs(OUTPUT_3D_ASSETS_DIR, exist_ok=True) 

# --- Biome Inspector Functions ---
def _format_data_for_display(data, indent_level=0):
    indent_str = "  " * indent_level
    if isinstance(data, dict):
        formatted_items = []
        for k, v in data.items():
            formatted_value = _format_data_for_display(v, indent_level + 1)
            if isinstance(v, (dict, list)) and '\n' in formatted_value:
                formatted_items.append(f"{indent_str}{k}:\n{formatted_value}")
            else:
                formatted_items.append(f"{indent_str}{k}: {formatted_value}")
        return "\n".join(formatted_items)
    elif isinstance(data, list):
        if all(isinstance(item, list) for item in data) and len(data) > 0 and \
           all(all(isinstance(sub_item, (int, float, str)) for sub_item in sublist) for sublist in data):
            grid_lines = []
            for row in data:
                grid_lines.append(" ".join(map(str, row)))
            return "\n" + "\n".join([f"{indent_str}  {line}" for line in grid_lines])
        else:
            list_items = []
            for item in data:
                list_items.append(f"{indent_str}- {_format_data_for_display(item, 0)}") 
            return "\n" + "\n".join(list_items)
    else:
        return str(data)

def display_selected_biome(db_name: str, collection_name: str, name: str) -> str: 
    if not name:
        _biome_logger.info("No biome selected for display.")
        return "" 
    
    _biome_logger.info(f"Fetching biome details for: '{name}' from DB: '{db_name}', Collection: '{collection_name}'")
    biome = fetch_biome(db_name, collection_name, name) 
    if biome:
        _biome_logger.info(f"Successfully fetched biome '{name}'.")
        return _format_data_for_display(biome)
    else:
        _biome_logger.warning(f"Biome '{name}' not found in registry for DB '{db_name}' and Collection '{collection_name}'.")
        return f"Biome '{name}' not found in the registry for DB '{db_name}' and Collection '{collection_name}'."

async def handler(theme: str, structure_types_str: str, db_name: str, collection_name: str) -> tuple[str, gr.Dropdown]: 
    """
    Handles biome generation, either directly or via Celery.
    """
    _biome_logger.info(f"Received biome generation request for theme: '{theme}', structures: '{structure_types_str}'")
    structure_type_list = [s.strip() for s in structure_types_str.split(',') if s.strip()]

    if not structure_type_list:
        _biome_logger.warning("No structure types provided for biome generation.")
        return "❌ Error: Please provide at least one structure type for biome generation.", \
               gr.update(choices=get_biome_names(db_name, collection_name), value=None) 
    
    if USE_CELERY:
        # Submit task to Celery CPU queue
        task = celery_run_biome_generation.apply_async(
            args=[theme, structure_types_str],
            queue='cpu_tasks'  # Route to CPU instance
        )
        msg = f"✅ Biome generation task submitted to CPU instance (ID: {task.id}). Please refresh 'View Generated Biomes' after a moment to see the new entry."
    else:
        # Direct call for development
        try:
            msg = await generate_biome(theme, structure_type_list) # Directly call the actual function
            _biome_logger.info(f"Biome generation finished directly with message: {msg}")
        except Exception as e:
            _biome_logger.error(f"Error during direct biome generation: {e}", exc_info=True)
            msg = f"❌ Error during direct biome generation: {e}"

    updated_biome_names = get_biome_names(db_name, collection_name) 
    selected_value = updated_biome_names[-1] if updated_biome_names else None
    
    return msg, gr.update(choices=updated_biome_names, value=selected_value)

# --- 3D Generation Functions ---
def submit_3d_from_image_task(image_file, with_texture, output_format, model_type):
    """Submit 3D generation from image task - routes to GPU spot instance"""
    if image_file is None:
        return None, "Error: No image uploaded"
    
    try:
        if USE_CELERY:
            # Ensure GPU spot instance is running before submitting task
            gpu_status_task = celery_manage_gpu_instance.delay("ensure_running")
            
            # Submit the actual 3D generation task to GPU queue with spot instance retry policy
            task = celery_generate_3d_model_from_image.apply_async(
                args=[image_file, with_texture, output_format],
                queue='gpu_tasks',  # Route to GPU spot instance
                retry=True,
                retry_policy={
                    'max_retries': 3,
                    'interval_start': 30,  # Wait for spot instance startup
                    'interval_step': 60,
                    'interval_max': 300,
                }
            )
            return None, f"✅ 3D generation task submitted to GPU spot instance (ID: {task.id}). Processing on 13.203.200.155..."
        else:
            # Direct processing in DEV mode (mock)
            logger.info(f"Mock 3D generation from image: {image_file}")
            result_msg = mock_generate_3d_from_image(image_file, with_texture, output_format)
            return None, f"✅ (DEV Mode Mock) {result_msg}"
    except Exception as e:
        logger.error(f"Error submitting 3D from image task to GPU spot instance: {e}", exc_info=True)
        return None, f"❌ Error: {e}"

def submit_3d_from_prompt_task(prompt, with_texture, output_format, model_type):
    """Submit 3D generation from text prompt task - routes to GPU spot instance"""
    if not prompt:
        return None, None, "Error: No prompt provided"
    
    try:
        if USE_CELERY:
            # Ensure GPU spot instance is running
            gpu_status_task = celery_manage_gpu_instance.delay("ensure_running")
            
            # Submit to GPU queue with spot instance retry policy
            task = celery_generate_3d_model_from_prompt.apply_async(
                args=[prompt, with_texture, output_format],
                queue='gpu_tasks',  # Route to GPU spot instance
                retry=True,
                retry_policy={
                    'max_retries': 3,
                    'interval_start': 30,  # Wait for spot instance startup
                    'interval_step': 60,
                    'interval_max': 300,
                }
            )
            return None, None, f"✅ Text-to-3D task submitted to GPU spot instance (ID: {task.id}). Processing on 13.203.200.155..."
        else:
            # Direct processing in DEV mode (mock)
            logger.info(f"Mock 3D generation from prompt: {prompt}")
            result_msg = mock_generate_3d_from_prompt(prompt, with_texture, output_format)
            return None, None, f"✅ (DEV Mode Mock) {result_msg}"
    except Exception as e:
        logger.error(f"Error submitting text-to-3D task to GPU spot instance: {e}", exc_info=True)
        return None, None, f"❌ Error: {e}"

def manage_gpu_instance_task(action):
    """Manage GPU instance (start/stop/status)"""
    try:
        if USE_CELERY:
            task = celery_manage_gpu_instance.delay(action)
            return f"✅ GPU instance {action} task submitted (ID: {task.id})."
        else:
            return f"✅ (DEV Mode Mock) GPU instance {action} command simulated."
    except Exception as e:
        logger.error(f"Error managing GPU instance: {e}", exc_info=True)
        return f"❌ Error: {e}"

# --- Wrapper functions for image generation (Conditional logic) ---
def process_image_generation_task(prompt_or_grid_content, width, height, num_images, model_type, is_grid_input=False):
    """
    Generic function to handle image generation, routing to Celery or direct processing.
    Returns (image_output, grid_viz_output, message)
    """
    if USE_CELERY:
        if is_grid_input:
            task = celery_generate_grid_image.apply_async(
                args=[prompt_or_grid_content, width, height, num_images, model_type],
                queue='cpu_tasks'  # Route to CPU instance
            )
            return None, None, f"✅ Grid processing task submitted to CPU instance (ID: {task.id}). Image will be saved to '{OUTPUT_IMAGES_DIR}' on the worker."
        else:
            task = celery_generate_text_image.apply_async(
                args=[prompt_or_grid_content, width, height, num_images, model_type],
                queue='cpu_tasks'  # Route to CPU instance
            )
            return None, f"✅ Text-to-image task submitted to CPU instance (ID: {task.id}). Image will be saved to '{OUTPUT_IMAGES_DIR}' on the worker."
    else:
        # Direct processing in DEV mode
        if _dev_pipeline is None: # Initialize if not already
            initialize_dev_processors()

        try:
            if is_grid_input:
                logger.info(f"Directly processing grid: {prompt_or_grid_content[:50]}...")
                images, grid_viz = _dev_pipeline.process_grid(prompt_or_grid_content)
                if not images:
                    return None, None, "No images generated directly."
                
                # Save and return the image for direct display
                img_path = save_image(images[0], f"terrain_direct_{int(time.time())}", "images")
                viz_path = save_image(grid_viz, f"grid_viz_direct_{int(time.time())}", "images")
                return images[0], grid_viz, f"Generated image directly. Saved to {img_path}"
            else:
                logger.info(f"Directly processing text prompt: {prompt_or_grid_content[:50]}...")
                images = _dev_pipeline.process_text(prompt_or_grid_content)
                if not images:
                    return None, "No images generated directly."

                # Save and return the image for direct display
                img_path = save_image(images[0], f"text_direct_{int(time.time())}", "images")
                return images[0], f"Generated image directly. Saved to {img_path}"
        except Exception as e:
            logger.error(f"Error during direct image generation: {e}", exc_info=True)
            if is_grid_input:
                return None, None, f"Error: {e}"
            else:
                return None, f"Error: {e}"

# Specific wrapper functions for Gradio interface
def submit_text_prompt_task(prompt, width, height, num_images, model_type):
    if not prompt:
        return None, "Error: No prompt provided"
    return process_image_generation_task(prompt, width, height, num_images, model_type, is_grid_input=False)

def submit_grid_input_task(grid_string, width, height, num_images, model_type):
    if not grid_string:
        return None, None, "Error: No grid provided"
    return process_image_generation_task(grid_string, width, height, num_images, model_type, is_grid_input=True)

def submit_file_upload_task(file_obj_path, width, height, num_images, text_model_type, grid_model_type):
    if file_obj_path is None:
        return None, None, "Error: No file uploaded"
    
    try:
        with open(file_obj_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        is_grid = True
        non_grid_chars_count = sum(1 for char in content if not (char.isdigit() or char.isspace() or char in '.,-[]'))
        if non_grid_chars_count > len(content) * 0.1: 
            is_grid = False
        
        if is_grid:
            return process_image_generation_task(content, width, height, num_images, grid_model_type, is_grid_input=True)
        else:
            text_result, text_message = process_image_generation_task(content, width, height, num_images, text_model_type, is_grid_input=False)
            return text_result, None, text_message # grid_viz is None for text processing

    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        return None, None, f"Error processing file: {str(e)}"

def create_sample_grid():
    """Create a simple sample grid for demonstration"""
    sample_grid = """
    0 0 0 1 1 1 0 0 0 0
    0 0 1 1 1 1 1 0 0 0
    0 1 1 1 1 1 1 1 0 0
    0 0 1 1 2 2 1 0 0 0
    0 0 0 2 2 2 2 0 0 0
    0 0 0 2 2 2 0 0 0 0
    0 0 3 3 3 3 3 0 0 0
    0 3 3 3 3 3 3 3 0 0
    3 3 3 3 3 3 3 3 3 0
    0 0 0 0 0 0 0 0 0 0
    """
    return sample_grid

# MongoDB Integration Functions (These remain local as they query DB directly, not heavy GPU work)
def get_prompts_from_mongodb(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=100):
    """Retrieve prompts from MongoDB collection"""
    try:
        mongo_helper = MongoDBHelper()
        query = {"$or": [
            {"theme_prompt": {"$exists": True}},
            {"description": {"$exists": True}}
        ]}
        documents = mongo_helper.find_many(db_name, collection_name, query=query, limit=limit) 
        if not documents:
            return [], "No prompts found in the specified collection."
        prompt_items = []
        for doc in documents:
            doc_id = str(doc.get("_id"))
            prompt = doc.get("theme_prompt") or doc.get("description")
            if not prompt and "possible_structures" in doc: 
                for category_key in doc["possible_structures"]:
                    for item_key in doc["possible_structures"][category_key]:
                        if "description" in doc["possible_structures"][category_key][item_key]:
                            prompt = doc["possible_structures"][category_key][item_key]["description"]
                            break
                    if prompt: break
            if prompt: prompt_items.append((doc_id, prompt))
        return prompt_items, f"Found {len(prompt_items)} prompts"
    except Exception as e:
        logger.error(f"MongoDB connection error fetching prompts: {str(e)}")
        return [], f"Error connecting to MongoDB: {str(e)}"

def get_prompts_for_dropdown(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=100):
    """Retrieve prompts from MongoDB and format for dropdown"""
    global _prompt_id_mapping
    prompt_items, status = get_prompts_from_mongodb(db_name, collection_name, limit)
    
    if not prompt_items:
        _prompt_id_mapping = {}
        return gr.update(choices=[], value=None), status
    
    # Create mapping and choices for dropdown
    _prompt_id_mapping = {}
    choices = []
    for doc_id, prompt in prompt_items:
        # Truncate long prompts for display
        display_text = prompt[:80] + "..." if len(prompt) > 80 else prompt
        display_choice = f"{doc_id}: {display_text}"
        choices.append(display_choice)
        _prompt_id_mapping[display_choice] = doc_id
    
    return gr.update(choices=choices, value=choices[0] if choices else None), status

def get_grids_from_mongodb(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=100):
    """Retrieve grid data from MongoDB collection"""
    try:
        mongo_helper = MongoDBHelper()
        query = {"$or": [
            {"grid": {"$exists": True}},
            {"possible_grids.layout": {"$exists": True}}
        ]}
        documents = mongo_helper.find_many(db_name, collection_name, query=query, limit=limit)
        if not documents: return [], "No grids found in the specified collection."
        grid_items = []
        for doc in documents:
            doc_id = str(doc.get("_id"))
            grid_str = ""
            if "grid" in doc:
                grid_content = doc["grid"]
                grid_str = "\n".join([" ".join(map(str, row)) for row in grid_content]) if isinstance(grid_content, list) else str(grid_content)
                grid_items.append((doc_id, grid_str))
            elif "possible_grids" in doc:
                for grid_obj in doc["possible_grids"]:
                    if "layout" in grid_obj:
                        layout_data = grid_obj["layout"]
                        grid_str = "\n".join([" ".join(map(str, row_cell)) for row_cell in layout_data]) if isinstance(layout_data, list) and all(isinstance(row, list) for row in layout_data) else str(layout_data)
                        grid_items.append((f"{doc_id}_{grid_obj.get('grid_id', 'grid')}", grid_str))
        return grid_items, f"Found {len(grid_items)} grids"
    except Exception as e:
        logger.error(f"MongoDB connection error fetching grids: {str(e)}")
        return [], f"Error connecting to MongoDB: {str(e)}"

def get_grids_for_dropdown(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=100):
    """Retrieve grids from MongoDB and format for dropdown"""
    global _grid_id_mapping
    grid_items, status = get_grids_from_mongodb(db_name, collection_name, limit)
    
    if not grid_items:
        _grid_id_mapping = {}
        return gr.update(choices=[], value=None), status
    
    # Create mapping and choices for dropdown
    _grid_id_mapping = {}
    choices = []
    for grid_id, grid_str in grid_items:
        # Truncate long grid strings for display
        display_text = grid_str.replace("\n", " ")[:60] + "..." if len(grid_str) > 60 else grid_str.replace("\n", " ")
        display_choice = f"{grid_id}: {display_text}"
        choices.append(display_choice)
        _grid_id_mapping[display_choice] = grid_id
    
    return gr.update(choices=choices, value=choices[0] if choices else None), status

# This function will now submit a task or process directly
def submit_mongodb_prompt_task(prompt_id, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                             num_images=DEFAULT_NUM_IMAGES, model_type=DEFAULT_TEXT_MODEL):
    """Submits a MongoDB prompt processing task to Celery OR processes directly."""
    logger.info(f"Processing MongoDB prompt task for ID: {prompt_id}")
    
    prompt_content = ""
    try:
        mongo_helper = MongoDBHelper()
        document = mongo_helper.find_one(db_name, collection_name, {"_id": pymongo.results.ObjectId(prompt_id)})
        if document:
            prompt_content = document.get("theme_prompt") or document.get("description") or \
                             (next((item["description"] for category in document.get("possible_structures", {}).values() for item in category.values() if "description" in item), None))
        if not prompt_content:
            return None, f"Error: No prompt content found for ID {prompt_id}."
    except Exception as e:
        logger.error(f"Error fetching prompt content for ID {prompt_id}: {e}")
        return None, f"Error fetching prompt content for ID {prompt_id}: {e}"

    # Use the generic image generation wrapper
    return process_image_generation_task(prompt_content, width, height, num_images, model_type, is_grid_input=False)

def submit_mongodb_prompt_task_from_dropdown(selected_prompt, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                             num_images=DEFAULT_NUM_IMAGES, model_type=DEFAULT_TEXT_MODEL):
    """Wrapper to extract prompt ID from dropdown selection and submit task"""
    global _prompt_id_mapping
    
    if not selected_prompt or selected_prompt not in _prompt_id_mapping:
        return None, "Error: Please select a valid prompt from the dropdown."
    
    prompt_id = _prompt_id_mapping[selected_prompt]
    return submit_mongodb_prompt_task(prompt_id, db_name, collection_name, width, height, num_images, model_type)

# This function will now submit a task or process directly
def submit_mongodb_grid_task(grid_item_id, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                             num_images=DEFAULT_NUM_IMAGES, model_type="stability"):
    """Submits a MongoDB grid processing task to Celery OR processes directly."""
    logger.info(f"Processing MongoDB grid task for item: {grid_item_id}")
    
    grid_content_str = ""
    try:
        parts = grid_item_id.split("_", 1)
        doc_id = parts[0]
        mongo_helper = MongoDBHelper()
        document = mongo_helper.find_one(db_name, collection_name, {"_id": pymongo.results.ObjectId(doc_id)}) 
        if not document: return None, None, f"Error: Document with ID {doc_id} not found."
        
        grid_content = None
        if len(parts) == 1 and "grid" in document:
            grid_content = document["grid"]
        elif len(parts) > 1 and "possible_grids" in document:
            grid_id_suffix = parts[1]
            for grid_obj in document["possible_grids"]:
                if grid_obj.get("grid_id") == grid_id_suffix:
                    if "layout" in grid_obj:
                        grid_content = grid_obj["layout"]
                        break
        
        if not grid_content:
            return None, None, f"Error: Grid content not found for item {grid_item_id}."
        
        grid_content_str = "\n".join([" ".join(map(str, row_cell)) for row_cell in grid_content]) if isinstance(grid_content, list) and all(isinstance(row, list) for row in grid_content) else str(grid_content)
        
    except Exception as e:
        logger.error(f"Error fetching grid content for {grid_item_id}: {e}")
        return None, None, f"Error fetching grid content for {grid_item_id}: {e}"

    # Use the generic image generation wrapper
    return process_image_generation_task(grid_content_str, width, height, num_images, model_type, is_grid_input=True)

def submit_mongodb_grid_task_from_dropdown(selected_grid, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                             num_images=DEFAULT_NUM_IMAGES, model_type="stability"):
    """Wrapper to extract grid ID from dropdown selection and submit task"""
    global _grid_id_mapping
    
    if not selected_grid or selected_grid not in _grid_id_mapping:
        return None, None, "Error: Please select a valid grid from the dropdown."
    
    grid_id = _grid_id_mapping[selected_grid]
    return submit_mongodb_grid_task(grid_id, db_name, collection_name, width, height, num_images, model_type)

# This function will now submit a batch task or process directly
def submit_batch_process_mongodb_prompts_task_ui(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=10, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                                     model_type=DEFAULT_TEXT_MODEL, update_db=False):
    """Submits a batch processing task to Celery OR processes directly."""
    logger.info(f"Processing batch processing task for {limit} prompts from {collection_name}.")
    
    if USE_CELERY:
        task = celery_batch_process_mongodb_prompts_task.apply_async(
            args=[db_name, collection_name, limit, width, height, model_type, update_db],
            queue='cpu_tasks'  # Route to CPU instance
        )
        return f"✅ Batch processing task submitted to CPU instance (ID: {task.id}). Results will be saved to '{OUTPUT_IMAGES_DIR}' on the worker."
    else:
        # Direct batch processing in DEV mode
        try:
            mongo_helper = MongoDBHelper()
            query = {"$or": [
                {"theme_prompt": {"$exists": True}},
                {"description": {"$exists": True}}
            ]}
            prompt_documents = mongo_helper.find_many(db_name, collection_name, query=query, limit=limit)
            
            if not prompt_documents:
                return "No prompts found to process in batch."
            
            results = []
            if _dev_pipeline is None:
                initialize_dev_processors()

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
                
                logger.info(f"Directly batch processing prompt: '{prompt[:50]}'")
                try:
                    # Direct call to generate image
                    images = _dev_pipeline.process_text(prompt)
                    
                    if images:
                        # Save the first image
                        output_path = os.path.join(OUTPUT_IMAGES_DIR, f"batch_{doc_id}.png")
                        images[0].save(output_path)
                        results.append(f"✓ {doc_id}: Generated and saved to {output_path}")
                        
                        # Update database if requested
                        if update_db:
                            try:
                                mongo_helper.update_one(db_name, collection_name, 
                                                      {"_id": doc["_id"]}, 
                                                      {"$set": {"generated_image_path": output_path}})
                                results.append(f"  └─ Updated database record for {doc_id}")
                            except Exception as e:
                                results.append(f"  └─ Failed to update database for {doc_id}: {e}")
                    else:
                        results.append(f"✗ {doc_id}: No images generated")
                        
                except Exception as e:
                    results.append(f"✗ {doc_id}: Error generating image - {e}")
            
            return f"Batch processing completed!\n\n" + "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return f"Error in batch processing: {e}"

# --- FastAPI and Gradio app setup code ---
# Create the Gradio Interface
def build_app():
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1480px;
    }
    """
    
    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
    Integrated 2D Generation Pipeline ({'Celery Enabled' if USE_CELERY else 'Development Mode'})
    </div>
    <div align="center">
    Generate 2D images for terrain and biomes. {'Heavy tasks are offloaded to Celery workers.' if USE_CELERY else 'Tasks are processed directly on this server.'}
    </div>
    """
    
    with gr.Blocks(theme=gr.themes.Base(), title='2D Pipeline', css=custom_css) as demo:
        gr.HTML(title_html)
        
        with gr.Tabs(selected="tab_biome_inspector") as tabs: 
            # Biome Inspector Tab 
            with gr.TabItem("Biome Inspector", id="tab_biome_inspector"):
                gr.Markdown("# Biome Generation Pipeline Inspector")

                with gr.Row():
                    biome_db_name = gr.Textbox(
                        label="Database Name",
                        value=MONGO_DB_NAME, 
                        placeholder="Enter database name"
                    )
                    biome_collection_name = gr.Textbox(
                        label="Collection Name",
                        value=MONGO_BIOME_COLLECTION, 
                        placeholder="Enter collection name"
                    )

                with gr.Row():
                    biome_inspector_theme_input = gr.Textbox(
                        label="Enter Biome Theme",
                        placeholder="e.g., A lush forest with ancient ruins",
                        value="A densely packed, multi-layered cyberpunk city with towering skyscrapers, neon signs, and hidden alleyways."
                    )
                    biome_inspector_structure_types_input = gr.Textbox(
                        label="Enter Structure Types (comma-separated)",
                        value="MegaCorp Tower, Neon Arcade, Data Hub, Slum Dwelling, Skybridge, Rooftop Garden",
                        placeholder="e.g., House, Shop, Tower, Well"
                    )
                
                biome_inspector_generate_button = gr.Button("Generate Biome")
                biome_inspector_output_message = gr.Textbox(label="Generation Status", interactive=False, max_lines=3)

                gr.Markdown("---")

                gr.Markdown("## View Generated Biomes")

                initial_biome_names = []
                try:
                    initial_biome_names = get_biome_names(biome_db_name.value, biome_collection_name.value)
                except Exception as e:
                    _biome_logger.error(f"Error fetching initial biome names for inspector: {e}")
                
                biome_inspector_selector = gr.Dropdown(
                    choices=initial_biome_names,
                    label="Select Biome to View",
                    interactive=True,
                    value=initial_biome_names[-1] if initial_biome_names else None
                )
                
                initial_biome_display_text = ""
                try:
                    if biome_inspector_selector.value: 
                        initial_biome_display_text = display_selected_biome(biome_db_name.value, biome_collection_name.value, biome_inspector_selector.value)
                except Exception as e:
                    _biome_logger.error(f"Error fetching initial biome display details for inspector: {e}")

                biome_inspector_display = gr.Textbox(
                    label="Biome Details",
                    value=initial_biome_display_text, 
                    interactive=False, 
                    lines=20, 
                    max_lines=50, 
                    show_copy_button=True
                )
            
            # Text to Image Tab
            with gr.TabItem("Text to Image", id="tab_text_image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(label="Text Prompt", placeholder="Enter a description of the image you want to generate...")
                        with gr.Row():
                            text_width = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=64, label="Width")
                            text_height = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=64, label="Height")
                        text_num_images = gr.Slider(minimum=1, maximum=4, value=DEFAULT_NUM_IMAGES, step=1, label="Number of Images")
                        text_model = gr.Dropdown(["openai", "stability", "local"], value=DEFAULT_TEXT_MODEL, label="Model")
                        text_submit = gr.Button("Generate Image from Text")
                    
                    with gr.Column(scale=2):
                        text_output = gr.Image(label=f"Generated Image (Check '{OUTPUT_IMAGES_DIR}')", interactive=False) 
                        text_message = gr.Textbox(label="Status", interactive=False)
            
            # Grid to Image Tab
            with gr.TabItem("Grid to Image", id="tab_grid_image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("""
                        ## Grid Format
                        Use numbers to represent different terrain types:
                        - 0: Plain
                        - 1: Forest
                        - 2: Mountain
                        - 3: Water
                        - 4: Desert
                        - 5: Snow
                        - 6: Swamp
                        - 7: Hills
                        - 8: Urban
                        - 9: Ruins
                        """)
                        grid_input = gr.Textbox(label="Grid Data", placeholder="Enter your grid data...", lines=10)
                        sample_button = gr.Button("Load Sample Grid")
                        with gr.Row():
                            grid_width = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=64, label="Width")
                            grid_height = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=64, label="Height")
                        grid_num_images = gr.Slider(minimum=1, maximum=4, value=DEFAULT_NUM_IMAGES, step=1, label="Number of Images")
                        grid_model = gr.Dropdown(["openai", "stability", "local"], value=DEFAULT_GRID_MODEL, label="Model") 
                        grid_submit = gr.Button("Generate Image from Grid")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            grid_output = gr.Image(label=f"Generated Terrain (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                            grid_viz = gr.Image(label=f"Grid Visualization (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                        grid_message = gr.Textbox(label="Status", interactive=False)
            
            # File Upload Tab
            with gr.TabItem("File Upload", id="tab_file"):
                with gr.Row():
                    with gr.Column(scale=3):
                        file_upload = gr.File(label="Upload a text file or grid file", type="filepath") 
                        gr.Markdown("System will automatically detect if the file contains text or grid data")
                        with gr.Row():
                            file_width = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=64, label="Width")
                            file_height = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=64, label="Height")
                        file_num_images = gr.Slider(minimum=1, maximum=4, value=DEFAULT_NUM_IMAGES, step=1, label="Number of Images")
                        with gr.Row():
                            file_text_model = gr.Dropdown(["openai", "stability", "local"], value=DEFAULT_TEXT_MODEL, label="Text Model")
                            file_grid_model = gr.Dropdown(["openai", "stability", "local"], value=DEFAULT_GRID_MODEL, label="Grid Model") 
                        file_submit = gr.Button("Process File")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            file_output = gr.Image(label=f"Generated Image (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                            file_grid_viz = gr.Image(label=f"Grid Visualization (if applicable, Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                        file_message = gr.Textbox(label="Status", interactive=False)
            
            # 3D Generation Tab
            with gr.TabItem("3D Generation", id="tab_3d_generation"):
                gr.Markdown("# 3D Model Generation")
                gr.Markdown(f"Generate 3D models from images or text prompts. {'Tasks are processed on GPU workers via Celery.' if USE_CELERY else 'Tasks are processed directly (mock mode).'}")
                
                with gr.Tabs() as threaded_tabs:
                    # Image to 3D Tab
                    with gr.TabItem("Image to 3D", id="tab_image_to_3d"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                threeded_image_upload = gr.File(
                                    label="Upload Image", 
                                    file_types=["image"],
                                    type="filepath"
                                )
                                gr.Markdown("""
                                **Supported formats:** JPG, PNG, WEBP  
                                **Recommended:** High-contrast images with clear subject matter work best for 3D reconstruction.
                                """)
                                with gr.Row():
                                    threeded_with_texture = gr.Checkbox(
                                        label="Generate with Texture", 
                                        value=True,
                                        info="Include color/texture information in the 3D model"
                                    )
                                    threeded_output_format = gr.Dropdown(
                                        choices=["glb", "obj", "ply"], 
                                        value="glb", 
                                        label="Output Format",
                                        info="GLB: Complete format with textures, OBJ: Geometry only, PLY: Point cloud"
                                    )
                                threeded_model_type = gr.Dropdown(
                                    choices=["hunyuan3d"], 
                                    value="hunyuan3d", 
                                    label="3D Model Type"
                                )
                                threeded_image_submit = gr.Button("Generate 3D Model from Image", variant="primary")
                            
                            with gr.Column(scale=2):
                                threeded_image_output = gr.File(
                                    label=f"Generated 3D Model (Check '{OUTPUT_3D_ASSETS_DIR}')", 
                                    interactive=False
                                )
                                threeded_image_message = gr.Textbox(
                                    label="Status", 
                                    interactive=False,
                                    lines=3
                                )
                                gr.Markdown("""
                                **Download:** Once generated, you can download the 3D model file.  
                                **Viewing:** Use software like Blender, MeshLab, or online 3D viewers to open the model.
                                """)
                    
                    # Text to 3D Tab
                    with gr.TabItem("Text to 3D", id="tab_text_to_3d"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                threeded_text_prompt = gr.Textbox(
                                    label="Text Prompt", 
                                    placeholder="Describe the object you want to generate in 3D (e.g., 'A red sports car', 'A medieval castle')",
                                    lines=3
                                )
                                gr.Markdown("""
                                **Pipeline:** Text → Image → 3D Model  
                                **Tips:** Be descriptive and specific. Mention colors, materials, and key features.
                                """)
                                with gr.Row():
                                    threeded_text_with_texture = gr.Checkbox(
                                        label="Generate with Texture", 
                                        value=True,
                                        info="Include color/texture information in the 3D model"
                                    )
                                    threeded_text_output_format = gr.Dropdown(
                                        choices=["glb", "obj", "ply"], 
                                        value="glb", 
                                        label="Output Format"
                                    )
                                threeded_text_model_type = gr.Dropdown(
                                    choices=["hunyuan3d"], 
                                    value="hunyuan3d", 
                                    label="3D Model Type"
                                )
                                threeded_text_submit = gr.Button("Generate 3D Model from Text", variant="primary")
                            
                            with gr.Column(scale=2):
                                threeded_intermediate_image = gr.Image(
                                    label=f"Intermediate Image (Check '{OUTPUT_IMAGES_DIR}')", 
                                    interactive=False
                                )
                                threeded_text_output = gr.File(
                                    label=f"Generated 3D Model (Check '{OUTPUT_3D_ASSETS_DIR}')", 
                                    interactive=False
                                )
                                threeded_text_message = gr.Textbox(
                                    label="Status", 
                                    interactive=False,
                                    lines=3
                                )
                
                # GPU Instance Management Section
                with gr.Accordion("GPU Instance Management", open=False):
                    gr.Markdown("### AWS GPU Instance Control")
                    gr.Markdown("Manage the GPU instance used for 3D generation tasks.")
                    
                    with gr.Row():
                        gpu_action = gr.Dropdown(
                            choices=["start", "stop", "status"], 
                            value="status", 
                            label="Action"
                        )
                        gpu_submit = gr.Button("Execute GPU Action")
                    
                    gpu_status = gr.Textbox(
                        label="GPU Instance Status", 
                        interactive=False,
                        lines=2
                    )
                    
                    gr.Markdown("""
                    **Note:** GPU instance management is only available in production mode with proper AWS credentials.  
                    - **Start:** Boot up the GPU instance for 3D processing  
                    - **Stop:** Shut down the GPU instance to save costs  
                    - **Status:** Check current instance state and cost estimates  
                    """)
            
            # MongoDB Prompts Tab
            with gr.TabItem("MongoDB", id="tab_mongodb"):
                with gr.Tabs() as mongo_tabs:
                    with gr.TabItem("Text Prompts", id="tab_mongo_text"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                mongo_db_name = gr.Textbox(label="Database Name", value=MONGO_DB_NAME, placeholder="Enter database name")
                                mongo_collection = gr.Textbox(label="Collection Name", value=MONGO_BIOME_COLLECTION, placeholder="Enter collection name")
                                mongo_fetch_btn = gr.Button("Fetch Prompts")
                                with gr.Row():
                                    mongo_width = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=64, label="Width")
                                    mongo_height = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=64, label="Height")
                                mongo_num_images = gr.Slider(minimum=1, maximum=4, value=DEFAULT_NUM_IMAGES, step=1, label="Number of Images")
                                mongo_model = gr.Dropdown(["openai", "stability", "local"], value=DEFAULT_TEXT_MODEL, label="Model")
                            mongo_process_btn = gr.Button("Generate Image", interactive=False)
                    
                        with gr.Column(scale=2):
                            mongo_prompts = gr.Dropdown(label="Select a Prompt", choices=[], interactive=False, allow_custom_value=True)
                            mongo_status = gr.Textbox(label="Status", interactive=False)
                            mongo_output = gr.Image(label=f"Generated Image (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                            mongo_message = gr.Textbox(label="Generation Status", interactive=False)
                        
                    with gr.Accordion("Batch Processing", open=False):
                        with gr.Row():
                            batch_limit = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Prompts to Process")
                            update_db = gr.Checkbox(label="Update MongoDB after processing", value=True)
                        batch_process_btn = gr.Button("Batch Process Prompts")
                        batch_results = gr.Textbox(label="Batch Processing Results", interactive=False, lines=10)
                        
                with gr.TabItem("Grid Data", id="tab_mongo_grid"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            grid_db_name = gr.Textbox(label="Database Name", value=MONGO_DB_NAME, placeholder="Enter database name")
                            grid_collection = gr.Textbox(label="Collection Name", value=MONGO_BIOME_COLLECTION, placeholder="Enter collection name")
                            grid_fetch_btn = gr.Button("Fetch Grids")
                            with gr.Row():
                                grid_width = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=64, label="Width")
                                grid_height = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=64, label="Height")
                            grid_num_images = gr.Slider(minimum=1, maximum=4, value=DEFAULT_NUM_IMAGES, step=1, label="Number of Images")
                            grid_model = gr.Dropdown(["openai", "stability", "local"], value=DEFAULT_GRID_MODEL, label="Model") 
                            grid_process_btn = gr.Button("Generate Image", interactive=False)
                    
                        with gr.Column(scale=2):
                            grid_items = gr.Dropdown(label="Select a Grid", choices=[], interactive=False, allow_custom_value=True)
                            grid_status = gr.Textbox(label="Status", interactive=False)
                            grid_output = gr.Image(label=f"Generated Image (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                            grid_visualization = gr.Image(label=f"Grid Visualization (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                            grid_message = gr.Textbox(label="Generation Status", interactive=False)

        # --- Event Handlers ---

        # Biome Inspector Tab Event Handlers
        biome_inspector_generate_button.click(
            fn=handler, 
            inputs=[biome_inspector_theme_input, biome_inspector_structure_types_input, biome_db_name, biome_collection_name],
            outputs=[biome_inspector_output_message, biome_inspector_selector]
        )
        
        biome_inspector_selector.change(
            fn=display_selected_biome, 
            inputs=[biome_db_name, biome_collection_name, biome_inspector_selector],
            outputs=biome_inspector_display
        )

        # Text to Image Tab
        text_submit.click(
            submit_text_prompt_task, 
            inputs=[text_input, text_width, text_height, text_num_images, text_model],
            outputs=[text_output, text_message]
        )
        
        # Grid to Image Tab
        grid_submit.click(
            submit_grid_input_task, 
            inputs=[grid_input, grid_width, grid_height, grid_num_images, grid_model],
            outputs=[grid_output, grid_viz, grid_message]
        )
        
        # File Upload Tab
        file_submit.click(
            submit_file_upload_task, 
            inputs=[file_upload, file_width, file_height, file_num_images, file_text_model, file_grid_model],
            outputs=[file_output, file_grid_viz, file_message]
        )
        
        # 3D Generation Tab Event Handlers
        threeded_image_submit.click(
            submit_3d_from_image_task,
            inputs=[threeded_image_upload, threeded_with_texture, threeded_output_format, threeded_model_type],
            outputs=[threeded_image_output, threeded_image_message]
        )
        
        threeded_text_submit.click(
            submit_3d_from_prompt_task,
            inputs=[threeded_text_prompt, threeded_text_with_texture, threeded_text_output_format, threeded_text_model_type],
            outputs=[threeded_intermediate_image, threeded_text_output, threeded_text_message]
        )
        
        gpu_submit.click(
            manage_gpu_instance_task,
            inputs=[gpu_action],
            outputs=[gpu_status]
        )
        
        sample_button.click(
            lambda: create_sample_grid(),
            inputs=[],
            outputs=[grid_input]
        )
        
        # MongoDB Prompt tab event handlers
        mongo_fetch_btn.click(
            get_prompts_for_dropdown, 
            inputs=[mongo_db_name, mongo_collection], 
            outputs=[mongo_prompts, mongo_status]
        ).then(
            lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[mongo_prompts, mongo_process_btn]
        )

        mongo_process_btn.click(
            submit_mongodb_prompt_task_from_dropdown, 
            inputs=[
                mongo_prompts, mongo_db_name, mongo_collection, 
                mongo_width, mongo_height, mongo_num_images, mongo_model
            ],
            outputs=[mongo_output, mongo_message] 
        )

        batch_process_btn.click(
            submit_batch_process_mongodb_prompts_task_ui, 
            inputs=[
                mongo_db_name, mongo_collection, batch_limit, 
                mongo_width, mongo_height, mongo_model, update_db
            ],
            outputs=[batch_results] 
        )

        # MongoDB Grid tab event handlers
        grid_fetch_btn.click(
            get_grids_for_dropdown, 
            inputs=[grid_db_name, grid_collection], 
            outputs=[grid_items, grid_status]
        ).then(
            lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[grid_items, grid_process_btn]
        )

        grid_process_btn.click(
            submit_mongodb_grid_task_from_dropdown, 
            inputs=[
                grid_items, grid_db_name, grid_collection, 
                grid_width, grid_height, grid_num_images, grid_model
            ],
            outputs=[grid_output, grid_visualization, grid_message] 
        )
    
    return demo

def create_fastapi_app(gradio_app):
    """Creates a FastAPI app and mounts the Gradio app on it."""
    app = FastAPI()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    return app

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    args = parser.parse_args()
    
    # Create output directories (ensure they exist on the frontend machine)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_3D_ASSETS_DIR, exist_ok=True) 
    
    try:
        # Initialize processors only if in DEV mode
        if not USE_CELERY:
            initialize_dev_processors() # <--- NEW: Call this in DEV mode
        else:
            logger.info("merged_gradio_app.py: FastAPI/Gradio app starting (CPU-only, submitting tasks to Celery).")
        
        logger.info("merged_gradio_app.py: Building Gradio app interface...")
        demo = build_app() 
        logger.info("merged_gradio_app.py: Gradio app interface built.")
        
        logger.info("merged_gradio_app.py: Attempting to launch application server (FastAPI/Uvicorn or Gradio built-in)...")
        logger.info(f"merged_gradio_app.py: Gradio application will be accessible on port: {args.port}")

        try:
            import uvicorn
            app = create_fastapi_app(demo) 
            logger.info(f"merged_gradio_app.py: Launching Uvicorn server at http://{args.host}:{args.port}/")
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            logger.warning("merged_gradio_app.py: Uvicorn not found, falling back to Gradio's built-in server.")
            logger.info(f"merged_gradio_app.py: Launching Gradio built-in server at http://{args.host}:{args.port}/")
            demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    except Exception as e:
        logger.critical(f"merged_gradio_app.py: A critical error occurred during application startup: {str(e)}", exc_info=True)
        print(f"\nFATAL ERROR: Application could not start. Details: {e}")
        print("Please check your environment setup and dependencies.")
        
        # Fallback UI (simplified to only 2D image/grid processing with messages)
        try:
            import gradio as gr
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            # Dummy functions for fallback UI, indicating tasks are not truly run
            def fallback_dummy_task_submit(input_data, *args):
                dummy_image = Image.new('RGB', (256, 256), color=(100, 100, 100))
                draw = ImageDraw.Draw(dummy_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                text = "Task cannot be submitted.\nFull app failed to load."
                text_bbox = draw.textbbox((0,0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.text(((256 - text_width) / 2, (256 - text_height) / 2), text, (255, 255, 255), font=font)
                return dummy_image, "Fallback: App initialization failed. Tasks cannot be processed."

            def fallback_dummy_grid_submit(input_data, *args):
                dummy_image = Image.new('RGB', (256, 256), color=(150, 100, 100))
                draw = ImageDraw.Draw(dummy_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                text = "Grid Task cannot be submitted.\nFull app failed to load."
                text_bbox = draw.textbbox((0,0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.text(((256 - text_width) / 2, (256 - text_height) / 2), text, (255, 255, 255), font=font)
                return dummy_image, dummy_image, "Fallback: App initialization failed. Tasks cannot be processed."

            def fallback_dummy_biome_generate(theme, structures):
                 return "Fallback: Biome generation not available. App initialization failed.", gr.update(choices=[], value=None)

            fallback_demo_app = gr.Blocks(theme=gr.themes.Base())
            with fallback_demo_app:
                gr.HTML(f"""
                <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
                2D Generation Only Mode (Emergency Fallback)
                </div>
                <div align="center" style="color: red; margin-bottom: 20px">
                Full application startup failed due to: {str(e)}
                </div>
                <div align="center">
                <p>This is a simplified interface. For full features, ensure Redis is running and Celery workers are configured and active.</p>
                </div>
                """)
                with gr.Tabs(selected="tab_biome_inspector_fallback") as fallback_tabs:
                    with gr.TabItem("Biome Inspector (Fallback)", id="tab_biome_inspector_fallback"):
                        gr.Markdown("# Biome Generation Pipeline Inspector (Fallback)")
                        gr.HTML("<p>Biome generation and viewing features are limited as the main app failed to load.</p>")
                        fallback_biome_inspector_theme_input = gr.Textbox(
                            label="Enter Biome Theme (Fallback)",
                            placeholder="e.g., A lush forest with ancient ruins"
                        )
                        fallback_biome_inspector_structure_types_input = gr.Textbox(
                            label="Enter Structure Types (comma-separated) (Fallback)",
                            placeholder="e.g., House, Shop, Tower"
                        )
                        fallback_biome_inspector_generate_button = gr.Button("Generate Biome (Fallback)")
                        fallback_biome_inspector_output_message = gr.Textbox(label="Generation Status (Fallback)", interactive=False)

                        fallback_biome_inspector_generate_button.click(
                            fallback_dummy_biome_generate,
                            inputs=[fallback_biome_inspector_theme_input, fallback_biome_inspector_structure_types_input],
                            outputs=[fallback_biome_inspector_output_message, gr.Dropdown(choices=[], value=None)]
                        )
                        gr.HTML("<p>Please resolve the main application errors to access full biome features.</p>")

                    with gr.TabItem("Text to Image (Fallback)"):
                        fallback_text_input = gr.Textbox(label="Text Prompt", placeholder="Enter a description...")
                        fallback_text_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                        fallback_text_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                        fallback_text_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                        fallback_text_model = gr.Dropdown(["openai", "stability", "local"], value="openai", label="Model")
                        fallback_text_output = gr.Image(label="Generated Image")
                        fallback_text_message = gr.Textbox(label="Status", interactive=False)
                        gr.Button("Generate Image").click(
                            fallback_dummy_task_submit, 
                            inputs=[fallback_text_input, fallback_text_width, fallback_text_height, fallback_text_num_images, fallback_text_model], 
                            outputs=[fallback_text_output, fallback_text_message]
                        )
                    with gr.TabItem("Grid to Image (Fallback)"):
                        fallback_grid_input = gr.Textbox(label="Grid Data", lines=10, placeholder="Enter grid data...")
                        fallback_grid_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                        fallback_grid_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                        fallback_grid_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                        fallback_grid_model = gr.Dropdown(["openai", "stability", "local"], value="stability", label="Model")
                        fallback_grid_output = gr.Image(label="Generated Terrain")
                        fallback_grid_viz = gr.Image(label="Grid Visualization")
                        fallback_grid_message = gr.Textbox(label="Status", interactive=False)
                        gr.Button("Generate Grid Image").click(
                            fallback_dummy_grid_submit, 
                            inputs=[fallback_grid_input, fallback_grid_width, fallback_grid_height, fallback_grid_num_images, fallback_grid_model], 
                            outputs=[fallback_grid_output, fallback_grid_viz, fallback_grid_message]
                        )
            logger.info("merged_gradio_app.py: Attempting to launch minimal 2D-only fallback app...")
            logger.info(f"merged_gradio_app.py: Fallback app will be accessible on port: {args.port}")
            fallback_demo_app.launch(server_name=args.host, server_port=args.port, share=args.share)
            logger.info("merged_gradio_app.py: Minimal 2D-only fallback app launched.")
        except Exception as fallback_launch_e:
            logger.critical(f"merged_gradio_app.py: CRITICAL: Failed to launch even the minimal 2D-only fallback app: {fallback_launch_e}", exc_info=True)
            print(f"FATAL: Could not start any part of the application. Please check your Python installation and core dependencies.")

