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
import pymongo # For ObjectId in MongoDB operations (local DB access)

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
    REDIS_BROKER_URL,
    USE_CELERY
)
from db_helper import MongoDBHelper

# Set up logging
# Setting logging level to DEBUG to capture detailed information, especially for grid issues.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_biome_logger = logging.getLogger("BiomeInspector")

# --- CRITICAL ADDITION: Silence pymongo.topology debug logs ---
# Get the specific logger used by pymongo for topology messages
pymongo_logger = logging.getLogger("pymongo.topology")
# Set its level to WARNING or INFO to suppress DEBUG messages
pymongo_logger.setLevel(logging.WARNING)
# You can also set it to logging.INFO if you want to see INFO level messages from pymongo topology

# --- Conditional Imports based on USE_CELERY ---
if USE_CELERY:
    logger.info("Running in PRODUCTION mode (Celery Enabled).\n")
    from tasks import generate_text_image as celery_generate_text_image, \
                      generate_grid_image as celery_generate_grid_image, \
                      run_biome_generation as celery_run_biome_generation, \
                      batch_process_mongodb_prompts_task as celery_batch_process_mongodb_prompts_task
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
        task = celery_run_biome_generation.delay(theme, structure_types_str)
        msg = f"✅ Biome generation task submitted (ID: {task.id}). Please refresh 'View Generated Biomes' after a moment to see the new entry."
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

# --- Wrapper functions for image generation (Conditional logic) ---
def process_image_generation_task(prompt_or_grid_content, width, height, num_images, model_type, is_grid_input=False):
    """
    Generic function to handle image generation, routing to Celery or direct processing.
    Returns (image_output, grid_viz_output, message)
    """
    if USE_CELERY:
        if is_grid_input:
            task = celery_generate_grid_image.delay(prompt_or_grid_content, width, height, num_images, model_type)
            return None, None, f"Task submitted (ID: {task.id}). Image will be saved to '{OUTPUT_IMAGES_DIR}' on the worker."
        else:
            task = celery_generate_text_image.delay(prompt_or_grid_content, width, height, num_images, model_type)
            return None, f"Task submitted (ID: {task.id}). Image will be saved to '{OUTPUT_IMAGES_DIR}' on the worker."
    else:
        # Direct processing in DEV mode
        if _dev_pipeline is None:
            initialize_dev_processors()

        try:
            if is_grid_input:
                logger.info(f"Directly processing grid: {prompt_or_grid_content[:50]}...")
                images, grid_viz = _dev_pipeline.process_grid(prompt_or_grid_content)
                if not images:
                    return None, None, "No images generated directly."

                img_path = save_image(images[0], f"terrain_direct_{int(time.time())}", "images")
                viz_path = save_image(grid_viz, f"grid_viz_direct_{int(time.time())}", "images")

                return images[0], grid_viz, f"Generated image directly. Saved to {img_path}"
            else:
                logger.info(f"Directly processing text prompt: {prompt_or_grid_content[:50]}...")
                images = _dev_pipeline.process_text(prompt_or_grid_content)
                if not images:
                    return None, "No images generated directly."

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
    """Retrieve prompts from MongoDB collection"""
    try:
        mongo_helper = MongoDBHelper()
        query = {"$or": [
            {"theme_prompt": {"$exists": True}},
            {"description": {"$exists": True}}
        ]}
        documents = mongo_helper.find_many(collection_name, query=query, limit=limit)
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

def submit_mongodb_prompt_task(prompt_id, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT,
                             num_images=DEFAULT_NUM_IMAGES, model_type=DEFAULT_TEXT_MODEL):
    """Submits a MongoDB prompt processing task to Celery OR processes directly."""
    logger.info(f"Processing MongoDB prompt task for ID: {prompt_id}")

    prompt_content = ""
    try:
        mongo_helper = MongoDBHelper()
        document = mongo_helper.find_one(collection_name, {"_id": pymongo.results.ObjectId(prompt_id)})
        if document:
            prompt_content = document.get("theme_prompt") or document.get("description") or \
                             (next((item["description"] for category in document.get("possible_structures", {}).values() for item in category.values() if "description" in item), None))
        if not prompt_content:
            return None, f"Error: No prompt content found for ID {prompt_id}."
    except Exception as e:
        logger.error(f"Error fetching prompt content for ID {prompt_id}: {e}")
        return None, f"Error fetching prompt content for ID {prompt_id}: {e}"

    return process_image_generation_task(prompt_content, width, height, num_images, model_type, is_grid_input=False)


def get_grids_from_mongodb(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=100):
    """Retrieve grid data from MongoDB collection"""
    try:
        mongo_helper = MongoDBHelper()
        query = {"$or": [
            {"grid": {"$exists": True}},
            {"possible_grids.layout": {"$exists": True}}
        ]}
        documents = mongo_helper.find_many(collection_name, query=query, limit=limit)
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

def submit_mongodb_grid_task(grid_item_id, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT,
                             num_images=DEFAULT_NUM_IMAGES, model_type="stability"):
    """Submits a MongoDB grid processing task to Celery OR processes directly."""
    logger.info(f"Processing MongoDB grid task for item: {grid_item_id}")

    grid_content_str = ""
    try:
        parts = grid_item_id.split("_", 1)
        doc_id = parts[0]
        mongo_helper = MongoDBHelper()
        document = mongo_helper.find_one(collection_name, {"_id": pymongo.results.ObjectId(doc_id)})
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

    return process_image_generation_task(grid_content_str, width, height, num_images, model_type, is_grid_input=True)


def submit_batch_process_mongodb_prompts_task_ui(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=10, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT,
                                     model_type=DEFAULT_TEXT_MODEL, update_db=False):
    """Submits a batch processing task to Celery OR processes directly."""
    logger.info(f"Processing batch processing task for {limit} prompts from {collection_name}.")

    if USE_CELERY:
        task = celery_batch_process_mongodb_prompts_task.delay(db_name, collection_name, limit, width, height, model_type, update_db)
        return f"Batch processing task submitted (ID: {task.id}). Results will be saved to '{OUTPUT_IMAGES_DIR}' on the worker."
    else:
        # Direct batch processing in DEV mode
        try:
            mongo_helper = MongoDBHelper()
            query = {"$or": [
                {"theme_prompt": {"$exists": True}},
                {"description": {"$exists": True}}
            ]}
            prompt_documents = mongo_helper.find_many(collection_name, query=query, limit=limit)

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
                    images = _dev_pipeline.process_text(prompt)

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
                            mongo_helper.update_by_id(collection_name, doc_id, update_data)
                    else:
                        results.append(f"Failed to generate image for: '{prompt[:30]}' - No image output.")
                except Exception as e:
                    logger.error(f"Error in direct batch processing for {doc_id}: {e}", exc_info=True)
                    results.append(f"Failed to process '{prompt[:30]}...' - Error: {e}")

            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error in overall direct batch processing: {e}", exc_info=True)
            return f"Error during direct batch processing: {e}"


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
                            grid_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
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

        sample_button.click(
            lambda: create_sample_grid(),
            inputs=[],
            outputs=[grid_input]
        )

        # MongoDB Prompt tab event handlers
        mongo_fetch_btn.click(
            get_prompts_from_mongodb,
            inputs=[mongo_db_name, mongo_collection],
            outputs=[mongo_prompts, mongo_status]
        ).then(
            lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[mongo_prompts, mongo_process_btn]
        )

        mongo_process_btn.click(
            submit_mongodb_prompt_task,
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
            get_grids_from_mongodb,
            inputs=[grid_db_name, grid_collection],
            outputs=[grid_items, grid_status]
        ).then(
            lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[grid_items, grid_process_btn]
        )

        grid_process_btn.click(
            submit_mongodb_grid_task,
            inputs=[
                grid_items, grid_db_name, grid_collection,
                grid_width, grid_height, grid_num_images, grid_model
            ],
            outputs=[grid_output, grid_visualization, grid_message]
        )

    return demo

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    args = parser.parse_args()

    # Ensure local output directories still exist for other image generation tasks if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_3D_ASSETS_DIR, exist_ok=True)

    try:
        if not USE_CELERY:
            initialize_dev_processors()
        else:
            logger.info("merged_gradio_app.py: Application starting (CPU-only, submitting tasks to Celery).")

        logger.info("merged_gradio_app.py: Building Gradio app interface...")
        demo = build_app()
        logger.info("merged_gradio_app.py: Gradio app interface built.")

        logger.info("merged_gradio_app.py: Attempting to launch Gradio application...")
        logger.info(f"merged_gradio_app.py: Gradio application will be accessible on port: {args.port}")

        demo.launch(server_name=args.host, server_port=args.port, share=args.share)

    except Exception as e:
        logger.critical(f"merged_gradio_app.py: A critical error occurred during application startup: {str(e)}", exc_info=True)
        print(f"\nFATAL ERROR: Application could not start. Details: {e}")

        # Fallback UI (simplified to only 2D image/grid processing with messages)
        try:
            import gradio as gr
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

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
                draw.text(((256 - text_width) / 2, (256, - text_height) / 2), text, (255, 255, 255), font=font)
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
