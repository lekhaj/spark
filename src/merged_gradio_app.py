# merged_gradio_app.py - Frontend Gradio Application with Multiple Image Display

import os
import time
import uuid
import shutil
import argparse
import logging
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import httpx
import base64
import requests
import pymongo # For ObjectId in MongoDB operations (local DB access)
import re
from datetime import datetime # Ensure datetime is imported

# --- Function to handle datetime serialization for JSON ---
def datetime_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat() # Convert datetime objects to ISO 8601 string
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# --- ALL IMPORTS NOW REFER TO THE NEW CONSOLIDATED STRUCTURE ---
from config import (
    DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL,
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES,
    OUTPUT_DIR, OUTPUT_IMAGES_DIR, OUTPUT_3D_ASSETS_DIR,
    MONGO_DB_NAME, MONGO_BIOME_COLLECTION,
    REDIS_BROKER_URL, REDIS_CONFIG, # Enhanced Redis configuration
    USE_CELERY, # <--- NEW: Flag to control Celery usage
    S3_BUCKET_NAME, S3_REGION, USE_S3_STORAGE, S3_3D_ASSETS_PREFIX # S3 configuration
)
from db_helper import MongoDBHelper
from s3_manager import get_s3_manager 

# Global variables to store ID mappings for dropdowns
_prompt_id_mapping = {}
_grid_id_mapping = {}

# Global variable to store image metadata for 3D generation MongoDB updates
_image_metadata_cache = []

# Set up logging
# Setting logging level to DEBUG to capture detailed information, especially for grid issues.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_biome_logger = logging.getLogger("BiomeInspector") # Defined globally and early

# --- Redis Connection Testing ---
def test_redis_connectivity():
    """Test Redis connectivity and log results."""
    try:
        redis_test = REDIS_CONFIG.test_connection()
        logger.info(f"Redis connectivity test - Worker type: {REDIS_CONFIG.worker_type}")
        logger.info(f"Redis write URL: {REDIS_CONFIG.write_url}")
        logger.info(f"Redis read URL: {REDIS_CONFIG.read_url}")
        
        if redis_test.get('write', {}).get('success'):
            logger.info("✅ Redis write connection successful")
        else:
            logger.error(f"❌ Redis write connection failed: {redis_test.get('write', {}).get('error')}")
            
        if redis_test.get('read', {}).get('success'):
            logger.info("✅ Redis read connection successful")
        else:
            logger.error(f"❌ Redis read connection failed: {redis_test.get('read', {}).get('error')}")
            
        return redis_test
    except Exception as e:
        logger.error(f"Redis connectivity test failed: {e}")
        return None

# Test Redis connectivity on startup
if USE_CELERY:
    redis_test_result = test_redis_connectivity()

# --- Conditional Imports based on USE_CELERY ---
if USE_CELERY:
    logger.info("Running in PRODUCTION mode (Celery Enabled).\n")
    from tasks import generate_text_image as celery_generate_text_image, \
                      generate_grid_image as celery_generate_grid_image, \
                      run_biome_generation as celery_run_biome_generation, \
                      batch_process_mongodb_prompts_task as celery_batch_process_mongodb_prompts_task, \
                      generate_3d_model_from_image as celery_generate_3d_model_from_image, \
                      manage_gpu_instance as celery_manage_gpu_instance, \
                      generate_local_image as celery_generate_local_image, \
                      process_local_generation as celery_process_local_generation, \
                      generate_image_sdxl_turbo as celery_generate_image_sdxl_turbo, \
                      batch_generate_images_sdxl_turbo as celery_batch_generate_images_sdxl_turbo
    # No direct model processor imports needed here, as they run on the worker.
    # from celery import Celery # Might need this if you want to track task results directly
    # from tasks import app as celery_app_instance # if you need to access backend results
else:
    logger.info("Running in DEVELOPMENT mode (Direct Processing).\n")
    from pipeline.text_processor import TextProcessor
    from pipeline.grid_processor import GridProcessor
    from pipeline.pipeline import Pipeline
    from utils.image_utils import save_image, create_image_grid
    from text_grid.grid_generator import generate_biome

    _dev_text_processor = None
    _dev_grid_processor = None
    _dev_pipeline = None

    # 3D processors (mock implementations for DEV mode)
    def mock_generate_3d_from_image(image_path, with_texture=False, output_format='glb'):
        """Mock 3D generation from image for development mode."""
        logger.info(f"Mock 3D generation from image: {image_path}")
        return f"Mock 3D model generated from {image_path} (texture: {with_texture}, format: {output_format})"

    def initialize_dev_processors():
        """Initialize the 2D processors and pipeline for direct execution in DEV mode."""
        global _dev_text_processor, _dev_grid_processor, _dev_pipeline
        logger.info("Initializing 2D processors for direct execution (DEV mode)...")
        _dev_text_processor = TextProcessor(model_type=DEFAULT_TEXT_MODEL)
        _dev_grid_processor = GridProcessor(model_type=DEFAULT_GRID_MODEL)
        _dev_pipeline = Pipeline(_dev_text_processor, _dev_grid_processor)
        logger.info("2D processors initialized for direct execution.")


# --- Imports for functions that remain local (e.g., UI display, helper functions) ---
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
            "details": "This is a mock forest biome detail.",
            "image_paths": ["https://example.com/mock_forest.png"],
            "created_at": datetime.now(), # Added for mock to simulate datetime issue
            "possible_structures": {
                "buildings": {
                    "tree_1": {"name": "Ancient Oak", "image_path": "https://example.com/ancient_oak.png"},
                    "ruin_arch": {"name": "Mossy Archway", "attributes": {"image_path": "https://example.com/mossy_archway.png"}}
                }
            },
            "image_path": "https://example.com/forest_biome_main.png" # Top-level image
        },
        "A_mock_desert": {
            "theme": "A vast, arid desert with hidden oases",
            "structures": ["Sand Dune", "Cactus"],
            "grid_data": [[4,4,4],[4,0,4],[4,4,4]],
            "details": "This is a mock desert biome detail.",
            "image_paths": ["https://example.com/mock_desert.png"],
            "created_at": datetime.now(), # Added for mock to simulate datetime issue
            "possible_structures": {
                "buildings": {
                    "dune_1": {"name": "Giant Dune", "image_path": "https://example.com/giant_dune.png"},
                    "cactus_field": {"name": "Spiky Cacti", "attributes": {"image_path": "https://example.com/spiky_cacti.png"}}
                }
            },
            "image_path": "https://example.com/desert_biome_main.png" # Top-level image
        },
        "Whispering Totem Glade": {
            "theme": "A mystical glade with ancient totems",
            "structures": ["Totem", "Mystic Stone"],
            "possible_grids": [
                {"grid_id": "main_layout", "layout": [[0,0,1],[0,1,0],[1,0,0]]},
                {"grid_id": "alt_layout", "layout": [[1,1,0],[1,0,1],[0,1,1]]}
            ],
            "details": "Details for Whispering Totem Glade.",
            "created_at": datetime.now(),
            "possible_structures": {
                "buildings": {
                    "totem_of_whispers": {"name": "Totem of Whispers", "image_path": "https://example.com/totem_whispers.png"},
                    "glowing_stone": {"name": "Glowing Stone", "attributes": {"image_path": "https://example.com/glowing_stone.png"}}
                }
            },
            "image_path": "https://example.com/totem_glade_main.png"
        },
        # Example for Desert Nomad Camp with actual grid and image
        "Desert Nomad Camp": {
            "theme": "A temporary settlement in the vast desert, with tents and sparse vegetation.",
            "structures": ["Tent", "Wagon", "Cactus"],
            "grid": [[4,4,0,0],[4,0,0,4],[0,0,4,4],[0,4,4,0]], # Direct grid example
            "details": "A collection of tents and wagons, suitable for nomadic desert dwellers.",
            "created_at": datetime.now(),
            "possible_structures": {
                "buildings": {
                    "tent_camp": {"name": "Nomad Tent", "image_path": "https://sparkassets.s3.ap-south-1.amazonaws.com/images/mongo_684038b1b4b2f8b37d69e32f_20250611_140425.png"},
                    "supply_wagon": {"name": "Supply Wagon", "attributes": {"image_path": "https://sparkassets.s3.ap-south-1.amazonaws.com/images/sample_wagon.png"}}
                }
            },
            "image_path": "https://sparkassets.s3.ap-south-1.amazonaws.com/images/mongo_684038b1b4b2f8b37d69e32f_20250611_140425.png"
        }
    }

    def get_biome_names(db_name, collection_name):
        return list(_mock_biomes_db.keys())

    def fetch_biome(db_name, collection_name, name: str):
        return _mock_biomes_db.get(name)

    async def generate_biome_mock(theme: str, structure_type_list: list[str]):
        new_biome_name = f"Generated_Biome_{len(_mock_biomes_db) + 1}"
        _mock_biomes_db[new_biome_name] = {
            "theme": theme,
            "structures": structure_type_list,
            "details": f"Mock details for a biome generated with theme: '{theme}' and structures: {structure_type_list}.",
            "grid_data": [[0,0,0],[0,0,0],[0,0,0]],
            "image_paths": [], # Mock for no images
            "created_at": datetime.now() # Added for mock to simulate datetime issue
        }
        return f"✅ Mock Biome '{new_biome_name}' generated successfully!"


# Constants

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create output directories (ensure they exist early on the frontend machine)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_3D_ASSETS_DIR, exist_ok=True)

# --- Biome Inspector Functions ---

def display_selected_biome(db_name: str, collection_name: str, name: str) -> tuple[str, str, list[tuple[str, str]]]:
    """
    Fetches biome details, separates grid data, formats main details as JSON,
    and collects S3 image URLs with captions for display in Gradio Gallery.
    Returns (formatted_json_text, grid_text, list_of_image_tuples_with_captions).
    """
    if not name:
        _biome_logger.info("No biome selected for display.")
        return "", "", []

    _biome_logger.info(f"Fetching biome details for: '{name}' from DB: '{db_name}', Collection: '{collection_name}'")
    biome = fetch_biome(db_name, collection_name, name)

    formatted_json_text = ""
    grid_text = ""
    image_display_items = [] # Changed to store (url, caption) tuples

    if biome:
        _biome_logger.info(f"Successfully fetched biome '{name}'.")
        grid_data_found = None

        _biome_logger.debug(f"Attempting to extract grid for biome '{name}'.")

        # Attempt 1: Check 'grid' key directly
        if "grid" in biome:
            grid_content = biome["grid"]
            _biome_logger.debug(f"Found 'grid' key. Type: {type(grid_content)}, Content (first 100 chars): {str(grid_content)[:100]}")
            
            if isinstance(grid_content, list) and all(isinstance(row, list) for row in grid_content):
                grid_data_found = grid_content
                _biome_logger.info("Extracted grid from 'grid' key (direct list of lists).")
            elif isinstance(grid_content, str): # Handle case where grid might be a string like "[0,1,0]\n[1,0,1]"
                _biome_logger.debug("Content of 'grid' key is a string. Attempting to parse as grid.")
                try:
                    # Robust parsing for string representation of grid
                    parsed_grid = []
                    # Split by newlines to get rows, then strip any leading/trailing whitespace
                    rows = grid_content.strip().split('\n')
                    for row_str in rows:
                        # Remove outer brackets, then split by comma or space to get elements
                        # Using regex split for more robust splitting by comma and/or space
                        import re # <-- Ensure this import is at the top of your file if not already there
                        elements = re.split(r'[, ]+', row_str.strip().replace('[', '').replace(']', ''))
                        # Filter out empty strings that might result from splitting
                        row_list = [int(e.strip()) for e in elements if e.strip()]
                        parsed_grid.append(row_list)

                    if parsed_grid and all(isinstance(row, list) for row in parsed_grid):
                        grid_data_found = parsed_grid
                        _biome_logger.info("Extracted grid from 'grid' key (parsed from string representation).")
                    else:
                        _biome_logger.debug(f"Parsed grid from string is not a list of lists or is empty. Result: {parsed_grid}")
                except Exception as e: # Catch more general exceptions for parsing errors
                    _biome_logger.debug(f"Could not parse 'grid' key content as grid string: {e}")
            else:
                _biome_logger.debug(f"Content of 'grid' key is not a list of lists or a string suitable for parsing. Actual type: {type(grid_content)}")


        # Attempt 2: Check 'possible_grids' if 'grid_data_found' is still None
        if grid_data_found is None and "possible_grids" in biome and isinstance(biome["possible_grids"], list):
            _biome_logger.debug(f"Checking 'possible_grids' for grid data. Number of entries: {len(biome['possible_grids'])}")
            for i, grid_option in enumerate(biome["possible_grids"]):
                _biome_logger.debug(f"  Processing possible_grids[{i}]. Type: {type(grid_option)}, Content (first 100 chars): {str(grid_option)[:100]}")
                
                if isinstance(grid_option, dict) and "layout" in grid_option:
                    layout_data = grid_option["layout"]
                    _biome_logger.debug(f"    Found 'layout' in dict. Type: {type(layout_data)}, Content (first 100 chars): {str(layout_data)[:100]}")
                    if isinstance(layout_data, list) and all(isinstance(row, list) for row in layout_data):
                        grid_data_found = layout_data
                        _biome_logger.info(f"Extracted grid from 'possible_grids' entry {i} with valid 'layout' (list of lists).")
                        break # Found a valid grid, use the first one
                    elif isinstance(layout_data, str): # Handle case where layout might be a string like "[0,1,0]\n[1,0,1]"
                        _biome_logger.debug(f"    'layout' in possible_grids[{i}] is a string. Attempting to parse as grid.")
                        try:
                            # Robust parsing for string representation of grid
                            parsed_layout = []
                            rows = layout_data.strip().split('\n')
                            for row_str in rows:
                                # Remove outer brackets, then split by comma or space to get elements
                                import re # <-- Ensure this import is at the top of your file if not already there
                                elements = re.split(r'[, ]+', row_str.strip().replace('[', '').replace(']', ''))
                                # Filter out empty strings that might result from splitting
                                row_list = [int(e.strip()) for e in elements if e.strip()]
                                parsed_layout.append(row_list)

                            if parsed_layout and all(isinstance(row, list) for row in parsed_layout):
                                grid_data_found = parsed_layout
                                _biome_logger.info(f"Extracted grid from 'possible_grids' entry {i} (parsed from string representation).")
                                break
                            else:
                                _biome_logger.debug(f"    Parsed layout from string is not a list of lists or is empty. Result: {parsed_layout}")
                        except Exception as e:
                            _biome_logger.debug(f"    Could not parse 'layout' in possible_grids[{i}] as grid string: {e}")
                    else:
                        _biome_logger.debug(f"    'layout' in possible_grids[{i}] is not a list of lists or string suitable for parsing. Actual type: {type(layout_data)}")

                elif isinstance(grid_option, list) and all(isinstance(row, list) for row in grid_option):
                    grid_data_found = grid_option
                    _biome_logger.info(f"Extracted grid from 'possible_grids' direct list entry {i} (list of lists).")
                    break # Found a valid grid, use the first one
                else:
                    _biome_logger.debug(f"    possible_grids[{i}] is neither a dict with 'layout' nor a direct list of lists. Skipping.")

        if grid_data_found:
            # ORIGINAL (for space-separated output):
            # grid_text = "\n".join([" ".join(map(str, row)) for row in grid_data_found])

            # CORRECTED LINE (to show as [101,0,0], etc.):
            grid_text = "\n".join([str(row) for row in grid_data_found])
            
            _biome_logger.info(f"Successfully formatted grid data for display. First line: {grid_text.splitlines()[0] if grid_text else 'N/A'}")
        else:
            grid_text = "No valid grid data found or invalid format. Please check debug logs for details."
            _biome_logger.info("Final status: No valid grid data found for display after all checks.")
        # --- END OF MODIFIED SECTION ---

        # Create a copy to modify for JSON display, removing grid keys
        biome_for_json = biome.copy()
        biome_for_json.pop("grid", None) # Remove 'grid' if present
        biome_for_json.pop("possible_grids", None) # Remove 'possible_grids' if present

        try:
            # Use the custom datetime_serializer function for json.dumps
            formatted_json_text = json.dumps(biome_for_json, indent=2, default=datetime_serializer)
        except TypeError as e:
            _biome_logger.error(f"Error converting biome data to JSON: {e}", exc_info=True)
            formatted_json_text = f"Error formatting biome data to JSON: {e}\n{str(biome_for_json)}"

        # --- Image Collection with Captions (Now tuples) ---
        # Top-level biome image
        top_level_image_path = biome.get("image_path")
        if top_level_image_path and isinstance(top_level_image_path, str) and top_level_image_path.startswith("http"):
            biome_name_for_caption = biome.get("name", name) # Use biome 'name' property if available, otherwise the passed 'name'
            image_display_items.append((top_level_image_path, f"Main Biome Image: {biome_name_for_caption}")) # <<< Changed to tuple
            _biome_logger.info(f"Found top-level S3 URL for biome: {top_level_image_path}")

        # Collect image_path from each structure
        possible_structures = biome.get("possible_structures", {})
        buildings = possible_structures.get("buildings", {})

        for struct_id, struct_data in buildings.items():
            img_path = struct_data.get("image_path")
            caption = struct_data.get("type", f"Structure: {type}") # Use description as caption
            
            if not (img_path and isinstance(img_path, str) and img_path.startswith("http")):
                attributes = struct_data.get("attributes", {})
                img_path = attributes.get("image_path")
                if "type" in attributes:
                    caption = attributes["type"] # Use attribute description if available

            if img_path and isinstance(img_path, str) and img_path.startswith("http"):
                image_display_items.append((img_path, caption)) # <<< Changed to tuple
                _biome_logger.info(f"Found S3 URL for structure {struct_id}: {img_path}")
            elif img_path:
                _biome_logger.warning(f"Image path for structure {struct_id} is not a valid S3 URL or is empty: {img_path}")

        if not image_display_items:
            _biome_logger.info(f"No S3 image_paths found for biome '{name}' or its structures.")
        # --- End Image Collection with Captions ---

    else:
        _biome_logger.warning(f"Biome '{name}' not found in the registry for DB '{db_name}' and Collection '{collection_name}'.")
        formatted_json_text = f"Biome '{name}' not found in the registry for DB '{db_name}' and Collection '{collection_name}'."

    # Add a debug log to confirm what URLs are being passed to Gradio Gallery
    _biome_logger.debug(f"Final image_display_items for gallery: {image_display_items}")

    # Return the list of tuples
    return formatted_json_text, grid_text, image_display_items


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
        try:
            msg = await generate_biome(theme, structure_type_list)
            _biome_logger.info(f"Biome generation finished directly with message: {msg}")
        except Exception as e:
            _biome_logger.error(f"Error during direct biome generation: {e}", exc_info=True)
            msg = f"❌ Error during direct biome generation: {e}"

    updated_biome_names = get_biome_names(db_name, collection_name)
    selected_value = updated_biome_names[-1] if updated_biome_names else None

    return msg, gr.update(choices=updated_biome_names, value=selected_value)

# --- 3D Generation Functions ---
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
def process_image_generation_task(prompt_or_grid_content, width, height, num_images, model_type, is_grid_input=False, 
                                doc_id=None, update_collection=None, category_key=None, item_key=None):
    """
    Generic function to handle image generation, routing to Celery or direct processing.
    Now includes 3D-optimized prompt enhancement for better 3D asset generation.
    Routes SDXL Turbo tasks to GPU worker for optimal performance.
    Returns (image_output, grid_viz_output, message)
    
    Args:
        doc_id: MongoDB document ID to update
        update_collection: MongoDB collection name to update  
        category_key: Category key for nested document updates
        item_key: Item key for nested document updates
    """
    # Enhance prompts for 3D generation (only for text prompts, not grids)
    if not is_grid_input and isinstance(prompt_or_grid_content, str):
        enhanced_prompt = enhance_prompt_for_3d_generation(prompt_or_grid_content)
    else:
        enhanced_prompt = prompt_or_grid_content
    
    if USE_CELERY:
        # Route SDXL Turbo tasks to GPU worker
        if model_type == "sdxl-turbo":
            if is_grid_input:
                # For grid input with SDXL, we'll use single prompt for now (could be enhanced to batch later)
                task = celery_generate_image_sdxl_turbo.apply_async(
                    args=[enhanced_prompt],
                    kwargs={
                        'width': width,
                        'height': height,
                        'enhance_prompt': True,
                        'doc_id': doc_id,
                        'update_collection': update_collection,
                        'category_key': category_key,
                        'item_key': item_key
                    },
                    queue='sdxl_tasks'  # Route to GPU instance
                )
                return None, None, f"✅ SDXL Turbo grid processing task submitted to GPU worker (ID: {task.id}). Image will be saved to GPU worker storage.\n🎯 3D-Optimized prompt enhancement applied for better 3D asset generation."
            else:
                task = celery_generate_image_sdxl_turbo.apply_async(
                    args=[enhanced_prompt],
                    kwargs={
                        'width': width,
                        'height': height,
                        'enhance_prompt': True,
                        'doc_id': doc_id,
                        'update_collection': update_collection,
                        'category_key': category_key,
                        'item_key': item_key
                    },
                    queue='sdxl_tasks'  # Route to GPU instance
                )
                return None, f"✅ SDXL Turbo task submitted to GPU worker (ID: {task.id}). High-quality image will be generated on GPU.\n🎯 3D-Optimized prompt used for better 3D asset generation."
        else:
            # Route other models to CPU worker
            if is_grid_input:
                task = celery_generate_grid_image.apply_async(
                    args=[enhanced_prompt, width, height, num_images, model_type],
                    kwargs={
                        'doc_id': doc_id,
                        'update_collection': update_collection,
                        'category_key': category_key,
                        'item_key': item_key
                    },
                    queue='cpu_tasks'  # Route to CPU instance
                )
                return None, None, f"✅ Grid processing task submitted to CPU instance (ID: {task.id}). Image will be saved to '{OUTPUT_IMAGES_DIR}' on the worker."
            else:
                task = celery_generate_text_image.apply_async(
                    args=[enhanced_prompt, width, height, num_images, model_type],
                    kwargs={
                        'doc_id': doc_id,
                        'update_collection': update_collection,
                        'category_key': category_key,
                        'item_key': item_key
                    },
                    queue='cpu_tasks'  # Route to CPU instance
                )
                return None, f"✅ Text-to-image task submitted to CPU instance (ID: {task.id}). Image will be saved to '{OUTPUT_IMAGES_DIR}' on the worker.\n🎯 3D-Optimized prompt used for better 3D asset generation."
    else:
        # Direct processing in DEV mode
        if _dev_pipeline is None:
            initialize_dev_processors()

        try:
            if is_grid_input:
                logger.info(f"Directly processing grid: {enhanced_prompt[:50]}...")
                images, grid_viz = _dev_pipeline.process_grid(enhanced_prompt)
                if not images:
                    return None, None, "No images generated directly."

                img_path = save_image(images[0], f"terrain_direct_{int(time.time())}", "images")
                viz_path = save_image(grid_viz, f"grid_viz_direct_{int(time.time())}", "images")

                return images[0], grid_viz, f"Generated image directly. Saved to {img_path}"
            else:
                logger.info(f"Directly processing enhanced text prompt: {enhanced_prompt[:50]}...")
                images = _dev_pipeline.process_text(enhanced_prompt)
                if not images:
                    return None, "No images generated directly."

                img_path = save_image(images[0], f"text_direct_{int(time.time())}", "images")

                return images[0], f"Generated image directly using 3D-optimized prompt. Saved to {img_path}\n🎯 Enhanced prompt: {enhanced_prompt}"
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
            return text_result, None, text_message

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
    """Retrieve prompts from MongoDB collection - supporting both 'description', 'theme_prompt' and structure descriptions"""
    try:
        mongo_helper = MongoDBHelper()
        # Search for documents with either description, theme_prompt, or possible_structures
        query = {"$or": [
            {"description": {"$exists": True, "$ne": None, "$ne": ""}},
            {"theme_prompt": {"$exists": True, "$ne": None, "$ne": ""}},
            {"possible_structures": {"$exists": True}}
        ]}
        documents = mongo_helper.find_many(db_name, collection_name, query=query, limit=limit) 
        
        if not documents:
            return [], "No documents with prompts found in the specified collection."
        
        prompt_items = []
        for doc in documents:
            doc_id = str(doc.get("_id"))
            
            # 1. Add main theme prompt if exists
            theme_prompt = doc.get("description") or doc.get("theme_prompt")
            if theme_prompt and isinstance(theme_prompt, str):
                prompt_items.append((f"{doc_id}_theme", f"[THEME] {theme_prompt}"))
            
            # 2. Add individual structure descriptions
            if "possible_structures" in doc:
                for category_key, category_data in doc["possible_structures"].items():
                    if isinstance(category_data, dict):
                        for structure_id, structure_data in category_data.items():
                            if isinstance(structure_data, dict):
                                # Check for description in structure data
                                structure_desc = structure_data.get("description")
                                if structure_desc and isinstance(structure_desc, str):
                                    structure_type = structure_data.get("type", structure_id)
                                    prompt_items.append((f"{doc_id}_{structure_id}", f"[{structure_type}] {structure_desc}"))
                                
                                # Check for description in attributes
                                elif "attributes" in structure_data and isinstance(structure_data["attributes"], dict):
                                    attr_desc = structure_data["attributes"].get("description")
                                    if attr_desc and isinstance(attr_desc, str):
                                        structure_type = structure_data.get("type", structure_id)
                                        prompt_items.append((f"{doc_id}_{structure_id}_attr", f"[{structure_type}] {attr_desc}"))
                                
                                # Fallback: use structure type and name if available
                                elif structure_data.get("type"):
                                    structure_type = structure_data.get("type")
                                    structure_name = structure_data.get("name", structure_id)
                                    fallback_desc = f"A {structure_type.lower()} called {structure_name}"
                                    prompt_items.append((f"{doc_id}_{structure_id}_fallback", f"[{structure_type}] {fallback_desc}"))
                                
        return prompt_items, f"Found {len(prompt_items)} prompts (themes + structure descriptions)"
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
    """Submits a MongoDB prompt processing task to Celery OR processes directly.
    Handles both theme prompts and individual structure descriptions:
    MongoDB -> description (prompt) -> image -> S3 bucket -> 3D asset"""
    logger.info(f"Processing MongoDB prompt task for ID: {prompt_id}")
    
    description_content = ""
    try:
        mongo_helper = MongoDBHelper()
        from bson import ObjectId
        
        # Parse the prompt_id to extract document ID and structure ID
        if "_theme" in prompt_id:
            # Theme prompt
            doc_id = prompt_id.replace("_theme", "")
            document = mongo_helper.find_one(db_name, collection_name, {"_id": ObjectId(doc_id)})
            if document:
                description_content = document.get("description") or document.get("theme_prompt")
                
        elif "_" in prompt_id:
            # Structure-specific prompt
            parts = prompt_id.split("_", 1)
            doc_id = parts[0]
            structure_path = parts[1]
            
            document = mongo_helper.find_one(db_name, collection_name, {"_id": ObjectId(doc_id)})
            if document and "possible_structures" in document:
                # Navigate through the structure path to find the description
                for category_key, category_data in document["possible_structures"].items():
                    if isinstance(category_data, dict):
                        for structure_id, structure_data in category_data.items():
                            if structure_path.startswith(structure_id):
                                if isinstance(structure_data, dict):
                                    # Check for description in structure data
                                    if structure_path == structure_id:
                                        description_content = structure_data.get("description")
                                    # Check attribute description
                                    elif structure_path == f"{structure_id}_attr":
                                        if "attributes" in structure_data:
                                            description_content = structure_data["attributes"].get("description")
                                    # Check fallback description
                                    elif structure_path == f"{structure_id}_fallback":
                                        structure_type = structure_data.get("type", "structure")
                                        structure_name = structure_data.get("name", structure_id)
                                        description_content = f"A {structure_type.lower()} called {structure_name}"
                                    
                                    if description_content:
                                        break
                            if description_content:
                                break
                        if description_content:
                            break
        else:
            # Legacy format - try to find document directly
            document = mongo_helper.find_one(db_name, collection_name, {"_id": ObjectId(prompt_id)})
            if document:
                description_content = document.get("description") or document.get("theme_prompt")
        
        if not description_content:
            return None, f"Error: No description content found for ID {prompt_id}."
            
    except Exception as e:
        logger.error(f"Error fetching description content for ID {prompt_id}: {e}")
        return None, f"Error fetching description content for ID {prompt_id}: {e}"

    logger.info(f"Found description: {description_content[:100]}...")
    
    # Extract MongoDB parameters for database update
    doc_id = None
    category_key = None
    item_key = None
    
    if "_theme" in prompt_id:
        # Theme prompt
        doc_id = prompt_id.replace("_theme", "")
    elif "_" in prompt_id:
        # Structure-specific prompt
        parts = prompt_id.split("_", 1)
        doc_id = parts[0]
        structure_path = parts[1]
        
        # For nested structures, extract category_key and item_key
        if document and "possible_structures" in document:
            for cat_key, category_data in document["possible_structures"].items():
                if isinstance(category_data, dict):
                    for struct_id, struct_data in category_data.items():
                        if structure_path.startswith(struct_id):
                            category_key = cat_key
                            item_key = struct_id
                            break
                if category_key and item_key:
                    break
    else:
        # Legacy format
        doc_id = prompt_id
    
    # Use the generic image generation wrapper with 3D-optimized prompt and MongoDB parameters
    return process_image_generation_task(
        description_content, width, height, num_images, model_type, 
        is_grid_input=False, doc_id=doc_id, update_collection=collection_name,
        category_key=category_key, item_key=item_key
    )

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

    # Extract MongoDB parameters for database update
    parts = grid_item_id.split("_", 1)
    doc_id = parts[0]
    
    return process_image_generation_task(
        grid_content_str, width, height, num_images, model_type, 
        is_grid_input=True, doc_id=doc_id, update_collection=collection_name
    )

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
    """Submits a batch processing task to Celery OR processes directly.
    Streamlined workflow: MongoDB -> description (prompt) -> image -> S3 bucket -> 3D asset"""
    logger.info(f"Processing batch description-based prompts from {collection_name} with limit {limit}.")

    if USE_CELERY:
        # Route SDXL Turbo batch tasks to GPU worker for optimal performance
        if model_type == "sdxl-turbo":
            # Get description prompts from MongoDB first
            prompt_items, status = get_prompts_from_mongodb(db_name, collection_name, limit)
            if not prompt_items:
                return f"❌ No description-based prompts found: {status}"
            
            # Extract just the description texts for batch processing
            descriptions = [item[1] for item in prompt_items]  # item[1] is the description content
            
            task = celery_batch_generate_images_sdxl_turbo.apply_async(
                args=[descriptions, width, height, 1],  # Process one image per description
                queue='sdxl_tasks'  # Route to GPU instance
            )
            return f"✅ SDXL Turbo batch processing task submitted to GPU worker (ID: {task.id}). Processing {len(descriptions)} description-based prompts.\n🎯 All prompts will be automatically optimized for 3D asset generation."
        else:
            task = celery_batch_process_mongodb_prompts_task.apply_async(
                args=[db_name, collection_name, limit, width, height, model_type, update_db],
                queue='cpu_tasks'  # Route to CPU instance
            )
            return f"✅ Batch processing task submitted to CPU instance (ID: {task.id}). Processing description-based prompts from MongoDB."
    else:
        # Direct batch processing in DEV mode
        try:
            mongo_helper = MongoDBHelper()
            # Focus on documents with either description or theme_prompt key
            query = {"$or": [
                {"description": {"$exists": True, "$ne": None, "$ne": ""}},
                {"theme_prompt": {"$exists": True, "$ne": None, "$ne": ""}}
            ]}
            prompt_documents = mongo_helper.find_many(db_name, collection_name, query=query, limit=limit)
            
            if not prompt_documents:
                return "No documents with 'description' or 'theme_prompt' key found to process in batch."

            results = []
            if _dev_pipeline is None:
                initialize_dev_processors()

            for doc in prompt_documents:
                doc_id = str(doc.get("_id"))
                
                # Process theme prompt first
                theme_prompt = doc.get("description") or doc.get("theme_prompt")
                if theme_prompt and isinstance(theme_prompt, str):
                    logger.info(f"Directly batch processing theme prompt: '{theme_prompt[:50]}'")
                    try:
                        enhanced_prompt = enhance_prompt_for_3d_generation(theme_prompt)
                        images = _dev_pipeline.process_text(enhanced_prompt)
                        
                        if images:
                            output_path = os.path.join(OUTPUT_IMAGES_DIR, f"batch_theme_{doc_id}.png")
                            images[0].save(output_path)
                            results.append(f"✓ {doc_id}_theme: Generated from theme prompt and saved to {output_path}")
                            
                            if update_db:
                                try:
                                    mongo_helper.update_one(db_name, collection_name, 
                                                          {"_id": doc["_id"]}, 
                                                          {"$set": {"theme_generated_image_path": output_path}})
                                    results.append(f"  └─ Updated database record for theme {doc_id}")
                                except Exception as e:
                                    results.append(f"  └─ Failed to update database for theme {doc_id}: {e}")
                        else:
                            results.append(f"✗ {doc_id}_theme: No images generated from theme prompt")
                    except Exception as e:
                        results.append(f"✗ {doc_id}_theme: Error generating image from theme prompt - {e}")

                # Process structure descriptions
                if "possible_structures" in doc:
                    for category_key, category_data in doc["possible_structures"].items():
                        if isinstance(category_data, dict):
                            for structure_id, structure_data in category_data.items():
                                if isinstance(structure_data, dict):
                                    structure_desc = structure_data.get("description")
                                    if not structure_desc and "attributes" in structure_data:
                                        structure_desc = structure_data["attributes"].get("description")
                                    
                                    if structure_desc and isinstance(structure_desc, str):
                                        logger.info(f"Directly batch processing structure description: '{structure_desc[:50]}'")
                                        try:
                                            enhanced_prompt = enhance_prompt_for_3d_generation(structure_desc)
                                            images = _dev_pipeline.process_text(enhanced_prompt)
                                            
                                            if images:
                                                output_path = os.path.join(OUTPUT_IMAGES_DIR, f"batch_structure_{doc_id}_{structure_id}.png")
                                                images[0].save(output_path)
                                                results.append(f"✓ {doc_id}_{structure_id}: Generated from structure description and saved to {output_path}")
                                                
                                                if update_db:
                                                    try:
                                                        # Update the specific structure with generated image path
                                                        update_path = f"possible_structures.{category_key}.{structure_id}.generated_image_path"
                                                        mongo_helper.update_one(db_name, collection_name, 
                                                                              {"_id": doc["_id"]}, 
                                                                              {"$set": {update_path: output_path}})
                                                        results.append(f"  └─ Updated database record for structure {doc_id}_{structure_id}")
                                                    except Exception as e:
                                                        results.append(f"  └─ Failed to update database for structure {doc_id}_{structure_id}: {e}")
                                            else:
                                                results.append(f"✗ {doc_id}_{structure_id}: No images generated from structure description")
                                        except Exception as e:
                                            results.append(f"✗ {doc_id}_{structure_id}: Error generating image from structure description - {e}")

                # If no prompts found at all for this document
                if not theme_prompt and not any(
                    isinstance(category_data, dict) and 
                    any(isinstance(structure_data, dict) and 
                        (structure_data.get("description") or 
                         (structure_data.get("attributes", {}).get("description") if isinstance(structure_data.get("attributes"), dict) else False))
                        for structure_data in category_data.values())
                    for category_data in doc.get("possible_structures", {}).values()
                ):
                    results.append(f"Skipping {doc_id}: No valid prompt text found in theme or structures.")
            
            return "Batch prompt processing completed!\n\n" + "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in batch description processing: {e}")
            return f"Error in batch description processing: {e}"

# --- S3 and Progress Tracking Helper Functions ---
def get_task_progress(task_id):
    """Get progress of a Celery task."""
    if not USE_CELERY:
        return {"status": "development", "progress": 100}
    
    try:
        from celery.result import AsyncResult
        task = AsyncResult(task_id)
        
        if task.state == 'PENDING':
            return {"status": "pending", "progress": 0, "message": "Task is pending..."}
        elif task.state == 'PROGRESS':
            return {
                "status": "progress", 
                "progress": task.info.get('progress', 0),
                "message": task.info.get('status', 'Processing...')
            }
        elif task.state == 'SUCCESS':
            return {"status": "success", "progress": 100, "result": task.result}
        elif task.state == 'FAILURE':
            return {"status": "error", "progress": 0, "message": str(task.info)}
        else:
            return {"status": task.state.lower(), "progress": 50, "message": f"Task state: {task.state}"}
    except Exception as e:
        logger.error(f"Error getting task progress: {e}")
        return {"status": "error", "progress": 0, "message": f"Error: {e}"}

def get_s3_3d_asset_url(s3_key):
    """Generate S3 URL for 3D asset."""
    if USE_S3_STORAGE and S3_BUCKET_NAME:
        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    return None

def check_s3_3d_asset_exists(source_image_name, output_format="glb"):
    """Check if 3D asset exists in S3 for given source image."""
    if not USE_S3_STORAGE:
        return False, None
    
    try:
        s3_mgr = get_s3_manager()
        if not s3_mgr:
            return False, None
        
        # Generate expected S3 key based on source image name
        base_name = os.path.splitext(source_image_name)[0]
        s3_key = f"{S3_3D_ASSETS_PREFIX}generated/{base_name}.{output_format}"
        
        # Try to get object metadata to check existence
        try:
            s3_mgr.s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            s3_url = get_s3_3d_asset_url(s3_key)
            return True, s3_url
        except:
            return False, None
    except Exception as e:
        logger.error(f"Error checking S3 asset existence: {e}")
        return False, None

# Global variable to store active 3D generation tasks
_active_3d_tasks = {}

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
                initial_selected_biome = None
                try:
                    initial_biome_names = get_biome_names(biome_db_name.value, biome_collection_name.value)
                    if initial_biome_names:
                        initial_selected_biome = initial_biome_names[-1] # Set the last biome as default
                except Exception as e:
                    _biome_logger.error(f"Error fetching initial biome names for inspector: {e}")

                # Calculate initial display values for the selected biome (if any)
                initial_biome_display_text = ""
                initial_grid_display_text = ""
                initial_image_urls_for_gallery_init = [] # This will hold only URLs (strings) for initial value

                if initial_selected_biome:
                    try:
                        # Call the display function to get the initial content in the (url, caption) format
                        temp_json, temp_grid, temp_image_tuples = display_selected_biome(
                            biome_db_name.value,
                            biome_collection_name.value,
                            initial_selected_biome
                        )
                        initial_biome_display_text = temp_json
                        initial_grid_display_text = temp_grid
                        # Extract only URLs for the initial value of gr.Gallery
                        initial_image_urls_for_gallery_init = [item[0] for item in temp_image_tuples]
                    except Exception as e:
                        _biome_logger.error(f"Error fetching initial biome display details for inspector: {e}")


                biome_inspector_selector = gr.Dropdown(
                    choices=initial_biome_names,
                    label="Select Biome to View",
                    interactive=True,
                    value=initial_selected_biome
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        biome_inspector_display = gr.Textbox(
                            label="Biome Details (JSON)",
                            value=initial_biome_display_text, # Use the pre-calculated initial value
                            interactive=False,
                            lines=15,
                            max_lines=30,
                            show_copy_button=True,
                        )
                        biome_grid_display = gr.Textbox(
                            label="Grid Data",
                            value=initial_grid_display_text, # Use the pre-calculated initial value
                            interactive=False,
                            lines=5,
                            max_lines=10,
                            show_copy_button=True,
                            elem_id="biome_grid_display_textbox"
                        )
                    with gr.Column(scale=1):
                        biome_image_display = gr.Gallery(
                            label="Generated Biome Images",
                            # For initial value, Gradio Gallery expects a list of URLs (strings).
                            # For updates, it can take lists of (url, caption) tuples or dicts.
                            value=initial_image_urls_for_gallery_init,
                            interactive=False,
                            height=400,
                            columns=2,
                            rows=-1,
                            object_fit="contain",
                            preview=True,
                        )

            # Text to Image Tab
            with gr.TabItem("Text to Image", id="tab_text_image"):
                gr.Markdown("# Text-to-Image Generation")
                gr.Markdown("Generate images from text descriptions. **All prompts are automatically optimized for 3D asset generation** with added keywords for better 3D model quality.")
                
                # 3D Optimization Info
                with gr.Accordion("🎯 3D Generation Optimization", open=False):
                    gr.Markdown("""
                    ### Automatic Prompt Enhancement for 3D Assets
                    
                    To ensure the best quality 3D models, all text prompts are automatically enhanced with:
                    
                    **Core 3D Enhancements:**
                    - `3d render, photorealistic, clean white background`
                    - `studio lighting, product photography style, sharp details`
                    
                    **Quality Improvements:**
                    - `high resolution, professional lighting, centered composition`
                    - `no shadows on background, isolated object`
                    
                    **Automatic Cleanup:**
                    - Removes conflicting background terms (landscape, indoor, outdoor, etc.)
                    - Optimizes prompt length for better processing
                    
                    This ensures your generated images will work optimally with the 3D model generation feature!
                    
                    ## 🚀 **SDXL Turbo: Local High-Quality Image Generation**
                    
                    **Complete Local 3D Generation Pipeline** - No API keys required!
                    
                    **SDXL Turbo Features**:
                    - **⚡ Ultra-fast generation**: 2-4 inference steps (vs 20-50 for other models)
                    - **🎯 High quality**: State-of-the-art image fidelity optimized for 3D
                    - **💾 Memory efficient**: Designed for 15-20GB VRAM with HunyuanDi-3D
                    - **🔧 3D-optimized**: Automatic prompt enhancement for better 3D assets
                    - **🏠 Local processing**: No API costs, private and secure
                    - **🔄 Seamless integration**: Works perfectly with HunyuanDi-3D
                    
                    **Perfect Local Workflow**: Text → SDXL Turbo → HunyuanDi-3D → 3D Model!
                    """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Text Prompt", 
                            placeholder="Enter a description of the image you want to generate (e.g., 'a red sports car', 'wooden chair', 'ceramic vase')...",
                            lines=3
                        )
                        gr.Markdown("**💡 Tip:** Describe objects clearly for best 3D generation results. Background and environment terms will be automatically optimized.")
                        with gr.Row():
                            text_width = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=64, label="Width")
                            text_height = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=64, label="Height")
                        text_num_images = gr.Slider(minimum=1, maximum=4, value=DEFAULT_NUM_IMAGES, step=1, label="Number of Images")
                        text_model = gr.Dropdown(
                            ["sdxl-turbo"], 
                            value="sdxl-turbo", 
                            label="Model",
                            info="SDXL Turbo: High-quality local GPU image generation optimized for 3D"
                        )
                        text_submit = gr.Button("🚀 Generate Image from Text (3D-Optimized)", variant="primary")

                    with gr.Column(scale=2):
                        text_output = gr.Image(label=f"Generated Image (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                        text_message = gr.Textbox(label="Status", interactive=False, lines=4)

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
                        grid_model = gr.Dropdown(
                            ["sdxl-turbo"], 
                            value="sdxl-turbo", 
                            label="Model",
                            info="SDXL Turbo: High-quality local GPU image generation optimized for 3D"
                        )
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
                            file_text_model = gr.Dropdown(
                                ["sdxl-turbo"], 
                                value="sdxl-turbo", 
                                label="Text Model",
                                info="SDXL Turbo for text processing"
                            )
                            file_grid_model = gr.Dropdown(
                                ["sdxl-turbo"], 
                                value="sdxl-turbo", 
                                label="Grid Model",
                                info="SDXL Turbo for grid processing"
                            )
                        file_submit = gr.Button("Process File")

                    with gr.Column(scale=2):
                        with gr.Row():
                            file_output = gr.Image(label=f"Generated Image (Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                            file_grid_viz = gr.Image(label=f"Grid Visualization (if applicable, Check '{OUTPUT_IMAGES_DIR}')", interactive=False)
                        file_message = gr.Textbox(label="Status", interactive=False)
            
            # 3D Generation Tab
            with gr.TabItem("3D Generation", id="tab_3d_generation"):
                gr.Markdown("# 3D Model Generation")
                gr.Markdown(f"Generate 3D models from images. {'Tasks are processed on GPU workers via Celery.' if USE_CELERY else 'Tasks are processed directly (mock mode).'}")
                
                # S3 Integration Status
                if USE_S3_STORAGE:
                    gr.Markdown(f"""
                    ### 🌐 S3 Cloud Storage Integration Enabled
                    - **Bucket**: `{S3_BUCKET_NAME}` (Region: {S3_REGION})
                    - **Auto-upload**: Generated 3D models are automatically uploaded to S3
                    - **Duplicate check**: Existing models are detected to avoid re-processing
                    - **Download links**: S3 URLs provided for easy access
                    """)
                else:
                    gr.Markdown("### 📁 Local Storage Mode\nModels will be saved locally. Enable S3 integration for cloud storage.")
                    
                # MongoDB Images Section
                with gr.Accordion("MongoDB Images", open=True):
                    gr.Markdown("### Browse and use images from MongoDB database")
                    gr.Markdown("""
                    **Image Sources:**
                    - 🎨 **[THEME]** - Main biome/theme images from document root
                    - 🏗️ **[STRUCTURE]** - Individual structure images from `possible_structures`
                    - 🔄 **Auto-refresh** - Recently generated images will appear here after database updates
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            mongo_img_db_name = gr.Textbox(
                                label="Database Name", 
                                value=MONGO_DB_NAME, 
                                placeholder="Enter database name"
                            )
                            mongo_img_collection = gr.Textbox(
                                label="Collection Name", 
                                value=MONGO_BIOME_COLLECTION, 
                                placeholder="Enter collection name"
                            )
                            fetch_images_btn = gr.Button("🔄 Fetch All Images from MongoDB", variant="secondary")
                            images_status = gr.Textbox(
                                label="Status", 
                                interactive=False,
                                lines=2,
                                placeholder="Click 'Fetch All Images' to load images from database..."
                            )
                        
                        with gr.Column(scale=2):
                            mongodb_images_gallery = gr.Gallery(
                                label="Available Images (Themes + Structures)",
                                show_label=True,
                                elem_id="mongodb_images",
                                columns=3,
                                rows=3,
                                height="300px",
                                interactive=True
                            )
                            # Hidden state to store image URLs
                            mongodb_image_urls = gr.State([])
                            selected_image_url = gr.Textbox(
                                label="Selected Image URL",
                                interactive=False,
                                visible=True,
                                placeholder="Click on an image above to select it"
                            )
                
                # 3D Generation from MongoDB Images
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 3D Generation Settings")
                        gr.Markdown("Configure settings for generating 3D models from selected MongoDB images.")
                        
                        with gr.Row():
                            threeded_with_texture = gr.Checkbox(
                                label="Generate with Texture", 
                                value=True
                            )
                            threeded_output_format = gr.Dropdown(
                                choices=["glb", "obj", "ply"], 
                                value="glb", 
                                label="Output Format"
                            )
                        gr.Markdown("""
                        **Texture**: Include color/texture information in the 3D model  
                        **Format**: GLB (complete with textures), OBJ (geometry only), PLY (point cloud)
                        """, elem_classes=["small-text"])
                        
                        threeded_model_type = gr.Dropdown(
                            choices=["hunyuan3d"], 
                            value="hunyuan3d", 
                            label="3D Model Type"
                        )
                        
                        threeded_generate_btn = gr.Button(
                            "🚀 Generate 3D Model from Selected Image", 
                            variant="primary",
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        threeded_image_output = gr.File(
                            label="Generated 3D Model (Download Link)", 
                            interactive=False
                        )
                        threeded_progress = gr.Textbox(
                            label="Generation Progress",
                            interactive=False,
                            lines=2,
                            placeholder="Progress will appear here during generation..."
                        )
                        threeded_image_message = gr.Textbox(
                            label="Status", 
                            interactive=False,
                            lines=4
                        )
                        gr.Markdown(                        """
                        **How to use:**
                        1. Click "🔄 Fetch All Images" to load images from MongoDB
                        2. Browse theme images (🎨 [THEME]) and structure images (🏗️ [STRUCTURE])
                        3. Click on any image to select it for 3D generation
                        4. Configure 3D generation settings and click "Generate 3D Model"
                        
                        **Image Types:**
                        - **Theme Images**: Main biome/environment images from image generation tasks
                        - **Structure Images**: Individual building/object images from nested structures
                        
                        **S3 Integration:** Models are automatically uploaded to S3 cloud storage for easy access.  
                        **Download:** Use the download link provided after generation to get your 3D model.  
                        **Viewing:** Use software like Blender, MeshLab, or online 3D viewers to open the model.
                        **Database Updates**: Generated 3D asset URLs are automatically saved back to MongoDB.
                        """)
                
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
                        gr.Markdown("### Generate Images from MongoDB Prompts with SDXL Turbo")
                        gr.Markdown("🚀 **SDXL Turbo Integration**: Ultra-fast, high-quality local image generation optimized for 3D assets!")
                        gr.Markdown("🎯 **All prompts are automatically enhanced for 3D asset generation** - perfect for creating images that will work well with the 3D Generation tab!")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                mongo_db_name = gr.Textbox(label="Database Name", value=MONGO_DB_NAME, placeholder="Enter database name")
                                mongo_collection = gr.Textbox(label="Collection Name", value=MONGO_BIOME_COLLECTION, placeholder="Enter collection name")
                                mongo_fetch_btn = gr.Button("Fetch Prompts")
                                gr.Markdown("**🎯 SDXL Turbo Features:**")
                                gr.Markdown("• ⚡ **Ultra-fast**: 2-4 inference steps vs 20-50 for other models")
                                gr.Markdown("• 🏠 **Local processing**: No API costs, completely private")
                                gr.Markdown("• 🎨 **3D-optimized**: Clean backgrounds, perfect lighting for 3D generation")
                                with gr.Row():
                                    mongo_width = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=64, label="Width")
                                    mongo_height = gr.Slider(minimum=256, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=64, label="Height")
                                mongo_num_images = gr.Slider(minimum=1, maximum=4, value=DEFAULT_NUM_IMAGES, step=1, label="Number of Images")
                                # Remove model dropdown - SDXL Turbo is now the default
                                gr.Markdown("**Model**: SDXL Turbo (Local GPU-accelerated generation)")
                            mongo_process_btn = gr.Button("🚀 Generate with SDXL Turbo", interactive=False, variant="primary")

                        with gr.Column(scale=2):
                            mongo_prompts = gr.Dropdown(label="Select a Prompt", choices=[], interactive=False, allow_custom_value=True)
                            mongo_status = gr.Textbox(label="Status", interactive=False)
                            mongo_output = gr.Image(label=f"Generated Image (SDXL Turbo Output)", interactive=False)
                            mongo_message = gr.Textbox(label="Generation Status", interactive=False)

                    with gr.Accordion("Batch Processing with SDXL Turbo", open=False):
                        gr.Markdown("**Batch process multiple prompts using SDXL Turbo for ultra-fast generation**")
                        with gr.Row():
                            batch_limit = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Prompts to Process")
                            update_db = gr.Checkbox(label="Update MongoDB after processing", value=True)
                        batch_process_btn = gr.Button("🚀 Batch Process with SDXL Turbo")
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
                            grid_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                            # Remove model dropdown for grids too - using SDXL Turbo
                            gr.Markdown("**Model**: SDXL Turbo (Optimized for grid-based generation)")
                            grid_process_btn = gr.Button("🚀 Generate with SDXL Turbo", interactive=False)

                        with gr.Column(scale=2):
                            grid_items = gr.Dropdown(label="Select a Grid", choices=[], interactive=False, allow_custom_value=True)
                            grid_status = gr.Textbox(label="Status", interactive=False)
                            grid_output = gr.Image(label=f"Generated Image (SDXL Turbo Output)", interactive=False)
                            grid_visualization = gr.Image(label=f"Grid Visualization", interactive=False)
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
            outputs=[biome_inspector_display, biome_grid_display, biome_image_display]
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
        
        # MongoDB Images functionality
        def fetch_and_update_gallery(db_name, collection_name):
            """Fetch images and update both gallery and URL state"""
            image_items, status = fetch_images_from_mongodb(db_name, collection_name)
            
            # Extract just the URLs for state storage
            urls = [item[0] for item in image_items] if image_items else []
            
            return image_items, status, urls
        
        fetch_images_btn.click(
            fetch_and_update_gallery,
            inputs=[mongo_img_db_name, mongo_img_collection],
            outputs=[mongodb_images_gallery, images_status, mongodb_image_urls]
        )
        
        # Handle image selection from gallery using index
        def on_image_select(evt: gr.SelectData, urls_list):
            """Handle image selection from the MongoDB gallery using index"""
            if evt.index is not None and urls_list and evt.index < len(urls_list):
                selected_url = urls_list[evt.index]
                logger.info(f"Selected image at index {evt.index}: {selected_url}")
                return (
                    gr.update(value=selected_url, visible=True), 
                    gr.update(interactive=True)
                )
            else:
                logger.warning(f"Invalid selection: index={evt.index}, urls_count={len(urls_list) if urls_list else 0}")
                return (
                    gr.update(value="", visible=True), 
                    gr.update(interactive=False)
                )
        
        mongodb_images_gallery.select(
            on_image_select,
            inputs=[mongodb_image_urls],
            outputs=[selected_image_url, threeded_generate_btn]
        )
        
        # Generate 3D model from selected MongoDB image with progress tracking
        def generate_3d_from_mongodb_image_with_progress(image_url, with_texture, output_format, model_type, progress=gr.Progress()):
            """Send MongoDB image URL directly to 3D generation with user-configured settings and progress tracking"""
            if not image_url:
                return None, "❌ No image selected. Please select an image from the MongoDB gallery above.", ""
            
            # Validate URL format
            if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
                logger.error(f"Invalid image URL format: {image_url}")
                return None, f"❌ Invalid URL format: {image_url}", ""
            
            logger.info(f"Submitting MongoDB image URL to 3D generation: {image_url}")
            logger.info(f"Settings: texture={with_texture}, format={output_format}, model={model_type}")
            
            # Find MongoDB metadata for this image URL
            global _image_metadata_cache
            mongodb_metadata = None
            for metadata in _image_metadata_cache:
                if metadata["url"] == image_url:
                    mongodb_metadata = metadata
                    break
            
            if mongodb_metadata:
                logger.info(f"📊 Found MongoDB metadata for image: {mongodb_metadata}")
                logger.info(f"   Doc ID: {mongodb_metadata.get('doc_id')}")
                logger.info(f"   Collection: {mongodb_metadata.get('collection')}")
                logger.info(f"   Type: {mongodb_metadata.get('type')}")
                logger.info(f"   Category Key: {mongodb_metadata.get('category_key')}")
                logger.info(f"   Item Key: {mongodb_metadata.get('item_key')}")
            else:
                logger.warning(f"⚠️ No MongoDB metadata found for image URL: {image_url}")
                # Create basic metadata as fallback
                mongodb_metadata = {
                    "doc_id": None,
                    "collection": "biomes",
                    "type": "unknown",
                    "category_key": None,
                    "item_key": None
                }
            
            # Extract image name from URL for S3 asset checking
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(image_url)
                image_filename = os.path.basename(parsed_url.path)
                
                # Check if 3D asset already exists in S3
                if USE_S3_STORAGE:
                    progress(0.05, desc="Checking existing 3D assets...")
                    asset_exists, existing_s3_url = check_s3_3d_asset_exists(image_filename, output_format)
                    if asset_exists:
                        return existing_s3_url, f"✅ 3D model already exists in S3!\nModel URL: {existing_s3_url}\nDownload the model from the link above.", "✅ Asset found in S3 storage"
                
            except Exception as e:
                logger.warning(f"Could not check for existing S3 assets: {e}")
            
            try:
                if USE_CELERY:
                    progress(0.1, desc="Testing Redis connectivity...")
                    # Test Redis connectivity before submitting task
                    redis_test = REDIS_CONFIG.test_connection('write')
                    if not redis_test.get('write', {}).get('success'):
                        error_msg = redis_test.get('write', {}).get('error', 'Unknown Redis error')
                        logger.error(f"Redis write connection failed: {error_msg}")
                        
                        if "read only replica" in error_msg.lower():
                            return None, "❌ Redis Error: Connected to read-only replica. Please configure a writable Redis master or check Redis configuration.", "❌ Redis Error"
                        else:
                            return None, f"❌ Redis Connection Error: {error_msg}", "❌ Connection Error"
                    
                    progress(0.15, desc="Ensuring GPU instance is running...")
                    # Ensure GPU spot instance is running before submitting task
                    celery_manage_gpu_instance.delay("ensure_running")
                    
                    progress(0.2, desc="Submitting 3D generation task...")
                    # Submit the 3D generation task with the image URL, user settings, and MongoDB metadata
                    task = celery_generate_3d_model_from_image.apply_async(
                        args=[image_url, with_texture, output_format],
                        kwargs={
                            'doc_id': mongodb_metadata.get("doc_id"),
                            'update_collection': mongodb_metadata.get("collection"),
                            'category_key': mongodb_metadata.get("category_key"),
                            'item_key': mongodb_metadata.get("item_key")
                        },
                        queue='gpu_tasks',  # Route to GPU spot instance
                        retry=True,
                        retry_policy={
                            'max_retries': 3,
                            'interval_start': 30,  # Wait for spot instance startup
                            'interval_step': 60,
                            'interval_max': 300,
                        }
                    )
                    
                    # Store task ID for progress tracking
                    task_id = task.id
                    _active_3d_tasks[task_id] = {
                        'image_url': image_url,
                        'settings': {'texture': with_texture, 'format': output_format},
                        'start_time': time.time()
                    }
                    
                    # Monitor task progress
                    progress(0.25, desc="Monitoring task progress...")
                    max_wait_time = 300  # 5 minutes max wait
                    start_time = time.time()
                    last_progress = 0.25
                    
                    while time.time() - start_time < max_wait_time:
                        task_progress = get_task_progress(task_id)
                        current_progress = last_progress + (task_progress.get('progress', 0) / 100) * 0.7  # Scale to 0.25-0.95 range
                        
                        if task_progress['status'] == 'success':
                            progress(0.95, desc="3D generation completed! Processing result...")
                            result = task_progress.get('result', {})
                            
                            # Check if result contains S3 URL
                            if isinstance(result, dict):
                                s3_model_url = result.get('s3_model_url') or result.get('model_s3_url')
                                local_path = result.get('local_path') or result.get('model_local_path')
                                
                                if s3_model_url:
                                    progress(1.0, desc="✅ 3D model ready for download!")
                                    success_msg = f"✅ 3D generation completed successfully!\nS3 URL: {s3_model_url}\nSettings: Texture={with_texture}, Format={output_format}\nProcessed on: {REDIS_CONFIG.gpu_ip}"
                                    return s3_model_url, success_msg, "✅ Generation completed successfully!"
                                elif local_path:
                                    success_msg = f"✅ 3D generation completed successfully!\nLocal Path: {local_path}\nSettings: Texture={with_texture}, Format={output_format}\nProcessed on: {REDIS_CONFIG.gpu_ip}"
                                    return local_path, success_msg, "✅ Generation completed successfully!"
                            
                            # Fallback message if URLs not found in result
                            success_msg = f"✅ 3D generation task completed (ID: {task_id})\nResult: {result}\nSettings: Texture={with_texture}, Format={output_format}\nProcessed on: {REDIS_CONFIG.gpu_ip}"
                            return None, success_msg, "✅ Task completed"
                            
                        elif task_progress['status'] == 'error':
                            error_msg = f"❌ 3D generation failed: {task_progress.get('message', 'Unknown error')}"
                            return None, error_msg, "❌ Generation failed"
                        
                        elif task_progress['status'] == 'progress':
                            progress(current_progress, desc=task_progress.get('message', 'Processing...'))
                            last_progress = current_progress
                        
                        time.sleep(2)  # Check every 2 seconds
                    
                    # Timeout case
                    timeout_msg = f"⏰ 3D generation task submitted but timed out waiting for result (ID: {task_id})\nTask may still be processing. Check the GPU worker logs.\nSettings: Texture={with_texture}, Format={output_format}"
                    return None, timeout_msg, "⏰ Task timeout"
                    
                else:
                    # Direct processing in DEV mode (mock)
                    progress(0.3, desc="Mock processing (DEV mode)...")
                    logger.info(f"Mock 3D generation from MongoDB URL: {image_url}")
                    time.sleep(2)  # Simulate processing time
                    progress(0.8, desc="Finalizing mock result...")
                    result_msg = mock_generate_3d_from_image(image_url, with_texture, output_format)
                    progress(1.0, desc="✅ Mock processing complete!")
                    return None, f"✅ (DEV Mode Mock) {result_msg}\nSettings: Texture={with_texture}, Format={output_format}", "✅ Mock generation completed"
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error submitting 3D generation task with MongoDB URL: {e}", exc_info=True)
                return None, f"❌ Error submitting task: {error_msg}", "❌ Submission error"
        
        threeded_generate_btn.click(
            generate_3d_from_mongodb_image_with_progress,
            inputs=[selected_image_url, threeded_with_texture, threeded_output_format, threeded_model_type],
            outputs=[threeded_image_output, threeded_image_message, threeded_progress]
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
            lambda prompts, db_name, collection, width, height, num_images: 
                submit_mongodb_prompt_task_from_dropdown(prompts, db_name, collection, width, height, num_images, "sdxl-turbo"), 
            inputs=[
                mongo_prompts, mongo_db_name, mongo_collection,
                mongo_width, mongo_height, mongo_num_images
            ],
            outputs=[mongo_output, mongo_message]
        )

        batch_process_btn.click(
            lambda db_name, collection, limit, width, height, update_db:
                submit_batch_process_mongodb_prompts_task_ui(db_name, collection, limit, width, height, "sdxl-turbo", update_db),
            inputs=[
                mongo_db_name, mongo_collection, batch_limit,
                mongo_width, mongo_height, update_db
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
            lambda grid_items, db_name, collection, width, height, num_images:
                submit_mongodb_grid_task_from_dropdown(grid_items, db_name, collection, width, height, num_images, "sdxl-turbo"), 
            inputs=[
                grid_items, grid_db_name, grid_collection,
                grid_width, grid_height, grid_num_images
            ],
            outputs=[grid_output, grid_visualization, grid_message]
        )

    return demo

# --- 3D Prompt Enhancement Function ---
def enhance_prompt_for_3d_generation(prompt):
    """Enhance text prompts for better 3D asset generation"""
    if not prompt or not isinstance(prompt, str):
        return prompt
    
    # Remove problematic background terms that conflict with 3D generation
    background_terms_to_remove = [
        "landscape", "indoor", "outdoor", "scenery", "environment", 
        "background", "setting", "room", "field", "forest", "desert",
        "in a", "on a", "against", "surrounded by"
    ]
    
    enhanced = prompt.strip()
    
    # Remove background terms (case insensitive)
    for term in background_terms_to_remove:
        enhanced = re.sub(rf'\b{term}\b', '', enhanced, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    enhanced = re.sub(r'\s+', ' ', enhanced).strip()
    
    # Add 3D-specific enhancements
    enhancements = [
        "3d render",
        "photorealistic", 
        "clean white background",
        "studio lighting",
        "product photography style",
        "sharp details",
        "high resolution",
        "professional lighting",
        "centered composition",
        "no shadows on background",
        "isolated object"
    ]
    
    # Combine original prompt with enhancements
    enhanced_prompt = f"{enhanced}, {', '.join(enhancements)}"
    
    # Limit length to avoid too long prompts
    if len(enhanced_prompt) > 300:
        enhanced_prompt = enhanced_prompt[:297] + "..."
    
    return enhanced_prompt

# --- 3D Generation Functions ---

def fetch_images_from_mongodb(db_name: str, collection_name: str) -> tuple[list, str]:
    """
    Fetch all documents that have image URLs from MongoDB and return them for display.
    Includes both top-level 'image_path' and nested structure images from 'possible_structures'.
    Returns (list_of_image_tuples, status_message).
    """
    try:
        mongo_helper = MongoDBHelper()
        
        # Query for documents that have either 'image_path' or 'possible_structures' with images
        query = {"$or": [
            {"image_path": {"$exists": True, "$ne": None, "$ne": ""}},
            {"possible_structures": {"$exists": True}}
        ]}
        documents = mongo_helper.find_many(db_name, collection_name, query, limit=50)
        
        image_items = []
        image_metadata = []  # Store metadata for MongoDB updates
        theme_count = 0
        structure_count = 0
        
        for doc in documents:
            doc_id = str(doc.get("_id", "Unknown"))
            doc_name = doc.get("name", doc.get("theme", doc.get("description", "Unnamed")))
            
            # 1. Add top-level theme/biome image
            image_path = doc.get("image_path")
            if image_path and isinstance(image_path, str) and image_path.startswith("http"):
                caption = f"[THEME] {doc_name[:40]}... | ID: {doc_id[:8]}..."
                image_items.append((image_path, caption))
                # Store metadata for theme image
                image_metadata.append({
                    "url": image_path,
                    "doc_id": doc_id,
                    "collection": collection_name,
                    "type": "theme",
                    "category_key": None,
                    "item_key": None
                })
                theme_count += 1
                logger.debug(f"Added theme image: URL={image_path}, Caption={caption}")
            
            # 2. Add structure images from possible_structures
            possible_structures = doc.get("possible_structures", {})
            if isinstance(possible_structures, dict):
                for category_key, category_data in possible_structures.items():
                    if isinstance(category_data, dict):
                        for structure_id, structure_data in category_data.items():
                            if isinstance(structure_data, dict):
                                # Check for direct image_path in structure
                                struct_image_path = structure_data.get("image_path") or structure_data.get("imageUrl")
                                
                                # Also check in attributes if no direct image found
                                if not struct_image_path and "attributes" in structure_data:
                                    attributes = structure_data["attributes"]
                                    if isinstance(attributes, dict):
                                        struct_image_path = attributes.get("image_path") or attributes.get("imageUrl")
                                
                                if struct_image_path and isinstance(struct_image_path, str) and struct_image_path.startswith("http"):
                                    structure_name = structure_data.get("name", structure_id)
                                    structure_type = structure_data.get("type", category_key)
                                    caption = f"[{structure_type.upper()}] {structure_name[:30]}... | Doc: {doc_id[:8]}..."
                                    image_items.append((struct_image_path, caption))
                                    # Store metadata for structure image
                                    image_metadata.append({
                                        "url": struct_image_path,
                                        "doc_id": doc_id,
                                        "collection": collection_name,
                                        "type": "structure",
                                        "category_key": category_key,
                                        "item_key": structure_id
                                    })
                                    structure_count += 1
                                    logger.debug(f"Added structure image: URL={struct_image_path}, Caption={caption}")
                                    logger.debug(f"Structure metadata: doc_id={doc_id}, category={category_key}, item={structure_id}")
        
        # Store metadata globally for use in 3D generation
        global _image_metadata_cache
        _image_metadata_cache = image_metadata
        
        total_images = len(image_items)
        status_msg = f"✅ Found {total_images} images ({theme_count} themes, {structure_count} structures) from {len(documents)} documents in {db_name}.{collection_name}"
        logger.info(status_msg)
        logger.debug(f"Image items structure: {image_items[:2] if image_items else 'No items'}")  # Log first 2 items for debugging
        return image_items, status_msg
        
    except Exception as e:
        error_msg = f"❌ Error fetching images from MongoDB: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [], error_msg

# --- Main Application Startup ---

def main():
    """Main function to start the Gradio application with proper configuration."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Text-to-3D Pipeline - Integrated 2D and 3D Generation")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--disable-3d", action="store_true", help="Disable 3D generation features")
    parser.add_argument("--dev-mode", action="store_true", help="Force development mode (disable Celery)")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Override USE_CELERY if dev mode is requested
    if args.dev_mode:
        global USE_CELERY
        USE_CELERY = False
        logger.info("Development mode forced - Celery disabled")
    
    # Test connections and services
    logger.info("=" * 60)
    logger.info("🚀 Starting Text-to-3D Pipeline Application")
    logger.info("=" * 60)
    
    # Test Redis connectivity if Celery is enabled
    if USE_CELERY:
        logger.info("Testing Redis connectivity...")
        try:
            redis_test = test_redis_connectivity()
            if redis_test and redis_test.get('write', {}).get('success'):
                logger.info("✅ Redis connection successful")
            else:
                logger.warning("⚠️ Redis connection issues - some features may not work")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("⚠️ Continuing without Celery - using direct processing mode")
    
    # Test MongoDB connectivity
    logger.info("Testing MongoDB connectivity...")
    try:
        from db_helper import MongoDBHelper
        mongo_helper = MongoDBHelper()
        # Simple connection test
        test_db = mongo_helper.client[MONGO_DB_NAME]
        result = test_db.command("ping")
        if result.get("ok") == 1:
            logger.info("✅ MongoDB connection successful")
        else:
            logger.warning("⚠️ MongoDB connection issues")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB connection failed: {e}")
        logger.info("📝 MongoDB features will use mock data")
    
    # Check GPU availability
    logger.info("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"✅ GPU available: {gpu_name} ({gpu_count} device{'s' if gpu_count != 1 else ''})")
        else:
            logger.info("📱 No CUDA GPU available - using CPU mode")
    except ImportError:
        logger.info("📱 PyTorch not available - 3D features may be limited")
    
    # Initialize dev processors if not using Celery
    if not USE_CELERY:
        logger.info("Initializing development processors...")
        try:
            initialize_dev_processors()
            logger.info("✅ Development processors initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize dev processors: {e}")
    
    # Build and launch the application
    logger.info("Building Gradio interface...")
    try:
        demo = build_app()
        logger.info("✅ Gradio interface built successfully")
        
        # Create app info message
        mode = "Production (Celery Enabled)" if USE_CELERY else "Development (Direct Processing)"
        info_msg = f"""
🌟 Text-to-3D Pipeline Ready!
📍 Mode: {mode}
🌐 URL: http://{args.host}:{args.port}
🎯 Features: Text→Image→3D, MongoDB Integration, S3 Storage
⚡ SDXL Turbo: Ultra-fast local image generation
🏗️ HunyuanDi-3D: High-quality 3D model generation
        """
        logger.info(info_msg)
        
        # Launch the application
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}", exc_info=True)
        return 1
    
    return 0

# --- Fallback Application for Error Cases ---

def create_fallback_app():
    """Create a minimal fallback app when main app fails to load."""
    try:
        import gradio as gr
        import sys
        
        # Create a simple placeholder image
        def create_placeholder_image():
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                img = Image.new('RGB', (512, 256), color='lightblue')
                draw = ImageDraw.Draw(img)
                
                # Try to load a font, fall back to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except Exception:
                    font = ImageFont.load_default()
                
                text = "Text-to-3D Pipeline"
                # Use textbbox instead of deprecated textsize
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # Fallback for older PIL versions
                    text_width, text_height = 200, 30
                
                x = (512 - text_width) // 2
                y = (256 - text_height) // 2
                draw.text((x, y), text, (255, 255, 255), font=font)
                
                return img
            except Exception:
                # Ultimate fallback - return None
                return None
        
        with gr.Blocks() as fallback_demo:
            gr.Markdown("# Text-to-3D Pipeline - Error Recovery Mode")
            gr.Markdown("The main application encountered errors. Please check your configuration and dependencies.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ## Troubleshooting Steps:
                    
                    1. **Check Dependencies**: Ensure all required packages are installed
                    2. **MongoDB**: Verify MongoDB is running and accessible
                    3. **Redis**: Check Redis server status (if using Celery)
                    4. **GPU**: Verify CUDA/GPU availability for 3D generation
                    5. **Configuration**: Check environment variables in .env file
                    
                    ## Quick Fixes:
                    
                    ```bash
                    # Restart services
                    sudo systemctl restart redis-server mongod
                    
                    # Check ports
                    sudo lsof -ti:7860 | xargs kill -9
                    
                    # Run in development mode
                    python merged_gradio_app.py --dev-mode --debug
                    ```
                    """)
                
                with gr.Column():
                    placeholder_img = create_placeholder_image()
                    if placeholder_img:
                        gr.Image(value=placeholder_img, label="Pipeline Status", interactive=False)
                    
                    gr.Markdown("### System Status")
                    status_text = f"""
                    - **Python**: {sys.version.split()[0]}
                    - **Current Directory**: {os.getcwd()}
                    - **USE_CELERY**: {USE_CELERY}
                    - **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    gr.Code(status_text, language="yaml")
        
        return fallback_demo
    
    except Exception as e:
        print("FATAL: Could not start any part of the application. Please check your Python installation and core dependencies.")
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    try:
        # Try to start the main application
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Critical error in main application: {e}", exc_info=True)
        
        # Try to start fallback application
        logger.info("🔄 Attempting to start fallback application...")
        try:
            fallback_app = create_fallback_app()
            if fallback_app:
                logger.info("🆘 Starting fallback application on port 7861...")
                fallback_app.launch(
                    server_name="0.0.0.0",
                    server_port=7861,
                    share=False,
                    debug=False
                )
            else:
                logger.error("❌ Could not create fallback application")
                sys.exit(1)
        except Exception as fallback_error:
            logger.error(f"❌ Fallback application also failed: {fallback_error}")
            print("FATAL: Complete application failure. Please check your installation.")
            sys.exit(1)
