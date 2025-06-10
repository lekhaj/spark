# src/merged_gradio_app.py
import os
import time
import uuid
import shutil
import argparse
import logging
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont 
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import json 
import httpx 
import base64 
import re # For robust JSON extraction

try:
    import uvicorn
except ImportError:
    pass
import pymongo
from datetime import datetime

# --- ALL IMPORTS NOW REFER TO THE NEW CONSOLIDATED STRUCTURE ---
from config import ( 
    DEFAULT_TEXT_MODEL, DEFAULT_GRID_MODEL, 
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES,
    OUTPUT_DIR, MONGO_DB_NAME, MONGO_BIOME_COLLECTION, STRUCTURE_TYPES, GRID_DIMENSIONS,
    MONGO_URI, 
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL 
)
from db_helper import MongoDBHelper 
from text_grid.structure_registry import get_biome_names, fetch_biome 
from text_grid.grid_generator import generate_biome 
from pipeline.text_processor import TextProcessor 
from pipeline.grid_processor import GridProcessor 
from pipeline.pipeline import Pipeline 
from terrain.grid_parser import GridParser 
from utils.image_utils import save_image, create_image_grid 
from text_grid import llm as llm_module # Re-added for LLM calls

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Explicitly set 3D support to False as per user request
HAS_3D_SUPPORT = False 

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Create output directories (keeping OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize global variables for processors and pipeline
text_processor = None
grid_processor = None
pipeline = None

def initialize_processors():
    """Initialize the 2D processors and pipeline with default models"""
    global text_processor, grid_processor, pipeline
    logger.info("Initializing 2D processors...")
    text_processor = TextProcessor(model_type=DEFAULT_TEXT_MODEL)
    grid_processor = GridProcessor(model_type=DEFAULT_GRID_MODEL) 
    pipeline = Pipeline(text_processor, grid_processor)
    logger.info("2D processors initialized.")

# Initialize 2D processors on startup
initialize_processors()

# --- Functions from grid_generator for direct biome generation (moved from viewer.py and adapted) ---
async def handle_biome_generation_request(theme: str, structure_types_str: str) -> tuple[str, gr.Dropdown]:
    """
    Handles the biome generation request from the UI.
    Parses the comma-separated structure types and calls generate_biome.
    """
    logger.info(f"Received request for theme: '{theme}', structures: '{structure_types_str}'")
    structure_type_list = [s.strip() for s in structure_types_str.split(',') if s.strip()]
    
    if not structure_type_list:
        logger.warning("No structure types provided by user.")
        return "❌ Error: Please provide at least one structure type for biome generation.", \
               gr.update(choices=get_biome_names(MONGO_DB_NAME, MONGO_BIOME_COLLECTION), value=None)

    msg = await generate_biome(theme, structure_type_list)
    logger.info(f"Biome generation finished with message: {msg}")
    
    updated_biome_names = get_biome_names(MONGO_DB_NAME, MONGO_BIOME_COLLECTION)
    selected_value = updated_biome_names[-1] if updated_biome_names else None
    
    return msg, gr.update(choices=updated_biome_names, value=selected_value)

def display_selected_biome_details(name: str) -> dict:
    """
    Fetches and displays the details of a selected biome.
    fetch_biome is from src.text_grid.structure_registry
    """
    if not name:
        logger.info("No biome selected for display.")
        return {} 
    
    logger.info(f"Fetching biome details for: '{name}'")
    biome = fetch_biome(MONGO_DB_NAME, MONGO_BIOME_COLLECTION, name)
    if biome:
        logger.info(f"Successfully fetched biome '{name}'.")
    else:
        logger.warning(f"Biome '{name}' not found in registry.")
    return biome

async def generate_biome_direct(theme_prompt_text, structure_types_str, save_to_db=True):
    """
    Generates biome data (grid and structures) directly and saves to MongoDB.
    Returns status, generated grid string, and grid visualization.
    """
    if not theme_prompt_text:
        return "Error: Theme prompt cannot be empty.", "", None, None
    if not structure_types_str:
        return "Error: Structure types cannot be empty.", "", None, None

    structure_types = [s.strip() for s in structure_types_str.split(',') if s.strip()]
    if not structure_types:
        return "Error: Please specify at least one structure type.", "", None, None

    try:
        logger.info(f"Generating biome directly for prompt: '{theme_prompt_text}' with types: {structure_types}")
        
        status_message = await generate_biome(theme_prompt_text, structure_types)
        
        mongo_helper = MongoDBHelper()
        latest_biome = mongo_helper.find_one(MONGO_BIOME_COLLECTION,
                                             {"theme_prompt": theme_prompt_text}, 
                                             sort_by=[("created_at", pymongo.DESCENDING)])
        
        grid_str = ""
        grid_viz_image = None

        if latest_biome:
            layout_data = latest_biome.get("possible_grids", [{}])[0].get("layout")
            if layout_data and isinstance(layout_data, list):
                grid_str = "\n".join([" ".join(map(str, row)) for row in layout_data])
                
                grid_parser = GridParser()
                grid_array = np.array(layout_data, dtype=np.int32)
                
                colors = {
                    0: (128, 128, 128),  # Grey for plain
                    1: (34, 139, 34),    # Green for forest
                    2: (139, 69, 19),    # Brown for mountain
                    3: (30, 144, 255),   # Blue for water
                    4: (255, 223, 0),    # Yellow for desert
                    5: (220, 220, 220), # Snow
                    6: (100, 140, 100), # Swamp
                    7: (170, 180, 90), # Hills
                    8: (80, 80, 80), # Urban
                    9: (150, 120, 90) # Ruins
                }
                h, w = grid_array.shape
                scale_factor = 20
                viz_image_array = np.zeros((h * scale_factor, w * scale_factor, 3), dtype=np.uint8)
                for r in range(h):
                    for c in range(w):
                        color = colors.get(grid_array[r, c], (0,0,0)) # Default to black if unknown
                        viz_image_array[r*scale_factor:(r+1)*scale_factor, c*scale_factor:(c+1)*scale_factor] = color
                grid_viz_image = Image.fromarray(viz_image_array)

            return status_message, grid_str, grid_viz_image, theme_prompt_text
        else:
            return status_message + " (Could not retrieve generated biome from DB).", "", None, None

    except Exception as e:
        logger.error(f"Error in generate_biome_direct: {e}")
        return f"Error: {e}", "", None, None

def process_text_prompt(prompt, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, num_images=DEFAULT_NUM_IMAGES, model_type=DEFAULT_TEXT_MODEL):
    """Process text prompt and generate images"""
    global text_processor, pipeline
    
    if not prompt:
        return None, "Error: No prompt provided"
    
    if text_processor is None or text_processor.model_type != model_type:
        try:
            text_processor = TextProcessor(model_type=model_type)
            pipeline = Pipeline(text_processor, grid_processor)
        except Exception as e:
            return None, f"Error initializing {model_type} model: {str(e)}"
    
    try:
        logger.info(f"Processing text prompt: {prompt}")
        images = pipeline.process_text(prompt)
        
        if not images or len(images) == 0:
            return None, "No images were generated"
        
        logger.info(f"Generated {len(images)} images from text prompt")
        
        if len(images) > 1:
            grid_image = create_image_grid(images)
            save_image(grid_image, f"text_grid_{prompt[:20]}")
            return grid_image, f"Generated {len(images)} images from text prompt"
        else:
            save_image(images[0], f"text_image_{prompt[:20]}")
            return images[0], "Generated 1 image from text prompt"
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return None, f"Error: {str(e)}"

def process_grid_input(grid_string, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, num_images=DEFAULT_NUM_IMAGES, model_type="stability"):
    """Process grid data and generate terrain images"""
    global grid_processor, pipeline
    
    if not grid_string:
        return None, None, "Error: No grid provided"
    
    if grid_processor is None or grid_processor.model_type != model_type:
        try:
            grid_processor = GridProcessor(model_type=model_type)
            pipeline = Pipeline(text_processor, grid_processor)
        except Exception as e:
            return None, None, f"Error initializing {model_type} model: {str(e)}"
    
    try:
        logger.info(f"Processing grid")
        images, grid_viz = pipeline.process_grid(grid_string)
        
        if not images or len(images) == 0:
            return None, None, "No images were generated"
        
        logger.info(f"Generated {len(images)} images from grid")
        
        if len(images) > 1:
            grid_image = create_image_grid(images)
            save_image(grid_image, "terrain_grid")
            return grid_image, grid_viz, f"Generated {len(images)} images from grid"
        else:
            save_image(images[0], "terrain_image")
            return images[0], grid_viz, "Generated 1 image from grid"
    
    except Exception as e:
        logger.error(f"Error processing grid: {str(e)}")
        return None, None, f"Error: {str(e)}"

def process_file_upload(file_obj, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, num_images=DEFAULT_NUM_IMAGES, text_model_type=DEFAULT_TEXT_MODEL, grid_model_type="stability"):
    """Process an uploaded file containing text or grid data"""
    if file_obj is None:
        return None, None, "Error: No file uploaded"
    
    try:
        if isinstance(file_obj, str):
            with open(file_obj, 'rb') as f:
                content = f.read().decode("utf-8").strip()
        else:
            content = file_obj.decode("utf-8").strip()
        
        is_grid = True
        non_grid_chars_count = sum(1 for char in content if not (char.isdigit() or char.isspace() or char in '[],'))
        if non_grid_chars_count > len(content) * 0.1: 
            is_grid = False
        
        if is_grid:
            logger.info("File content detected as grid data")
            return process_grid_input(content, width, height, num_images, grid_model_type)
        else:
            logger.info("File content detected as text prompt")
            image, message = process_text_prompt(content, width, height, num_images, text_model_type)
            return image, None, message
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
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

# Removed all 3D Generation Functions

# MongoDB Integration Functions 
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
            prompt = None
            if "theme_prompt" in doc:
                prompt = doc["theme_prompt"]
            elif "description" in doc:
                prompt = doc["description"]
            
            if not prompt and "possible_structures" in doc:
                structures = doc.get("possible_structures", {})
                for category_key in structures:
                    category = structures[category_key]
                    for item_key in category:
                        item = category[item_key]
                        if "description" in item:
                            if not prompt:
                                prompt = item["description"]
                            break
                    if prompt:
                        break
            
            if prompt:
                prompt_items.append((doc_id, prompt))
        
        return prompt_items, f"Found {len(prompt_items)} prompts"
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        return [], f"Error connecting to MongoDB: {str(e)}"

def process_mongodb_prompt(prompt_id, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                             num_images=DEFAULT_NUM_IMAGES, model_type=DEFAULT_TEXT_MODEL):
    """Process a prompt from MongoDB by ID"""
    global text_processor, pipeline
    
    try:
        mongo_helper = MongoDBHelper()
        # Use find_one with ObjectId for direct lookup
        document = mongo_helper.find_one(collection_name, {"_id": pymongo.results.ObjectId(prompt_id)}) 
        
        if not document:
            return None, "Error: Document not found"
        
        prompt = None
        if "theme_prompt" in document:
            prompt = document["theme_prompt"]
        elif "description" in document:
            prompt = document["description"]
        
        if not prompt and "possible_structures" in document:
            structures = document.get("possible_structures", {})
            for category_key in structures:
                category = structures[category_key]
                for item_key in category:
                    item = category[item_key]
                    if "description" in item:
                        if not prompt:
                            prompt = item["description"]
                        break
                if prompt:
                    break
        
        if not prompt:
            return None, "Error: No prompt found in the document"
        
        image, message = process_text_prompt(prompt, width, height, num_images, model_type)
        
        if image is not None and "Error" not in message:
            update = {
                "$set": {
                    "processed": True,
                    "processed_at": datetime.now(),
                    "model_used": model_type
                }
            }
            mongo_helper.update_by_id(collection_name, prompt_id, update)
        
        return image, message
        
    except Exception as e:
        logger.error(f"Error processing MongoDB prompt: {str(e)}")
        return None, f"Error: {str(e)}"

def get_grids_from_mongodb(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=100):
    """Retrieve grid data from MongoDB collection"""
    try:
        mongo_helper = MongoDBHelper()
        
        query = {"$or": [
            {"grid": {"$exists": True}},
            {"possible_grids.layout": {"$exists": True}}
        ]}
        
        documents = mongo_helper.find_many(collection_name, query=query, limit=limit)
        
        if not documents:
            return [], "No grids found in the specified collection."
        
        grid_items = []
        for doc in documents:
            doc_id = str(doc.get("_id"))
            
            if "grid" in doc:
                grid = doc["grid"]
                if isinstance(grid, list):
                    grid_str = "\n".join([" ".join(map(str, row)) for row in grid])
                else:
                    grid_str = str(grid)
                grid_items.append((doc_id, grid_str))
            elif "possible_grids" in doc:
                for grid_obj in doc["possible_grids"]:
                    if "layout" in grid_obj:
                        layout_data = grid_obj["layout"]
                        if isinstance(layout_data, list) and all(isinstance(row, list) for row in layout_data):
                            grid_str = "\n".join([" ".join(map(str, row_cell)) for row_cell in layout_data])
                        else:
                            grid_str = str(layout_data)
                        grid_items.append((f"{doc_id}_{grid_obj.get('grid_id', 'grid')}", grid_str))
            
        return grid_items, f"Found {len(grid_items)} grids"
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        return [], f"Error connecting to MongoDB: {str(e)}"

def process_mongodb_grid(grid_item, db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                             num_images=DEFAULT_NUM_IMAGES, model_type="stability"):
    """Process a grid from MongoDB by ID"""
    global grid_processor, pipeline
    
    try:
        parts = grid_item.split("_", 1)
        doc_id = parts[0]
        
        mongo_helper = MongoDBHelper()
        document = mongo_helper.find_one(collection_name, {"_id": pymongo.results.ObjectId(doc_id)}) 
        
        if not document:
            return None, None, "Error: Document not found"
        
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
            return None, None, "Error: Grid not found in the document or invalid ID format."
        
        if isinstance(grid_content, list) and all(isinstance(row, list) for row in grid_content):
            grid_string_for_processor = "\n".join([" ".join(map(str, row_cell)) for row_cell in grid_content])
        else:
            grid_string_for_processor = str(grid_content)
            
        image, grid_viz, message = process_grid_input(grid_string_for_processor, width, height, num_images, model_type)
        
        if image is not None and grid_viz is not None:
            update = {
                "$set": {
                    "processed": True,
                    "processed_at": datetime.now(),
                    "model_used": model_type
                }
            }
            mongo_helper.update_by_id(collection_name, doc_id, {"$set": update})
        
        return image, grid_viz, message
        
    except Exception as e:
        logger.error(f"Error processing MongoDB grid: {str(e)}")
        return None, None, f"Error: {str(e)}"

def batch_process_mongodb_prompts(db_name=MONGO_DB_NAME, collection_name=MONGO_BIOME_COLLECTION, limit=10, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                                     model_type=DEFAULT_TEXT_MODEL, update_db=False):
    """Batch process multiple prompts from MongoDB"""
    try:
        mongo_helper = MongoDBHelper()
        prompt_items, status = get_prompts_from_mongodb(db_name, collection_name, limit)
        
        if not prompt_items:
            return "No prompts found to process."
        
        results = []
        for doc_id, prompt in prompt_items:
            logger.info(f"Processing prompt: {prompt}")
            image, message = process_text_prompt(prompt, width, height, 1, model_type)
            
            if image is not None and "Error" not in message:
                image_path = os.path.join(OUTPUT_DIR, f"mongo_batch_{doc_id}.png") 
                try:
                    image.save(image_path)
                    results.append(f"Generated image for: {prompt[:30]}... -> {image_path}")
                except Exception as save_e:
                    logger.error(f"Error saving image for {doc_id}: {save_e}")
                    results.append(f"Failed to save image for: {prompt[:30]}... - {save_e}")
                    image_path = None
                
                if update_db:
                    update_data = {
                        "processed": True,
                        "processed_at": datetime.now(),
                        "model_used": model_type
                    }
                    if image_path:
                        update_data["image_path"] = image_path
                    
                    mongo_helper.update_by_id(collection_name, doc_id, {"$set": update_data})
            else:
                results.append(f"Failed to generate image for: {prompt[:30]}... - {message}")
        
        return "\n".join(results)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return f"Error: {str(e)}"

# Create the Gradio Interface
def build_app():
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1480px;
    }
    """
    
    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
    Integrated 2D Generation Pipeline
    </div>
    <div align="center">
    Generate 2D images for terrain and biomes
    </div>
    """
    
    with gr.Blocks(theme=gr.themes.Base(), title='2D Pipeline', css=custom_css) as demo:
        gr.HTML(title_html)
        
        # Only include the specified tabs: Biome Inspector, Text to Image, Grid to Image, File Upload, MongoDB
        with gr.Tabs(selected="tab_biome_inspector") as tabs: 
            # Biome Inspector Tab
            with gr.TabItem("Biome Inspector", id="tab_biome_inspector"):
                gr.Markdown("# Biome Generation Pipeline Inspector")

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
                    initial_biome_names = get_biome_names(MONGO_DB_NAME, MONGO_BIOME_COLLECTION)
                except Exception as e:
                    logger.error(f"Error fetching initial biome names: {e}")
                
                biome_inspector_selector = gr.Dropdown(
                    choices=initial_biome_names,
                    label="Select Biome to View",
                    interactive=True,
                    value=initial_biome_names[-1] if initial_biome_names else None
                )
                
                initial_biome_display_value = {}
                try:
                    if biome_inspector_selector.value: 
                        initial_biome_display_value = display_selected_biome_details(biome_inspector_selector.value)
                except Exception as e:
                    logger.error(f"Error fetching initial biome display details: {e}")

                biome_inspector_display = gr.JSON(
                    label="Biome Details", 
                    value=initial_biome_display_value
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
                        text_output = gr.Image(label="Generated Image")
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
                            grid_output = gr.Image(label="Generated Terrain")
                            grid_viz = gr.Image(label="Grid Visualization")
                        grid_message = gr.Textbox(label="Status", interactive=False)
            
            # File Upload Tab
            with gr.TabItem("File Upload", id="tab_file"):
                with gr.Row():
                    with gr.Column(scale=3):
                        file_upload = gr.File(label="Upload a text file or grid file")
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
                            file_output = gr.Image(label="Generated Image")
                            file_grid_viz = gr.Image(label="Grid Visualization (if applicable)")
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
                            mongo_output = gr.Image(label="Generated Image")
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
                            grid_output = gr.Image(label="Generated Image")
                            grid_visualization = gr.Image(label="Grid Visualization")
                            grid_message = gr.Textbox(label="Generation Status", interactive=False)

        # --- Event Handlers ---

        # Biome Inspector Tab
        biome_inspector_generate_button.click(
            fn=handle_biome_generation_request, 
            inputs=[biome_inspector_theme_input, biome_inspector_structure_types_input],
            outputs=[biome_inspector_output_message, biome_inspector_selector]
        )
        
        biome_inspector_selector.change(
            fn=display_selected_biome_details,
            inputs=biome_inspector_selector,
            outputs=biome_inspector_display
        )

        # Text to Biome (Grid) Tab (This is actually the "Text to Biome" functionality)
        def update_selected_structures(category):
            return ", ".join(STRUCTURE_TYPES.get(category, []))

        # This tab is now just 'Text to Biome'
        tabs.select(
            fn=lambda x: update_selected_structures(biome_structure_types_input.value) if x == "tab_biome_grid" else "",
            inputs=[tabs],
            outputs=[biome_selected_structures_display]
        )

        biome_structure_types_input.change(
            update_selected_structures,
            inputs=[biome_structure_types_input],
            outputs=[biome_selected_structures_display]
        )

        biome_generate_btn.click(
            generate_biome_direct, 
            inputs=[biome_prompt_input, biome_selected_structures_display], 
            outputs=[biome_generation_status, biome_grid_output, biome_grid_viz_output, biome_prompt_input]
        ).then(
            lambda grid_str: gr.update(visible=True) if grid_str else gr.update(visible=False),
            inputs=[biome_grid_output],
            outputs=[biome_grid_to_image_btn]
        )

        biome_grid_to_image_btn.click(
            lambda grid_str: process_grid_input(grid_str, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES, "stability"),
            inputs=[biome_grid_output],
            outputs=[grid_output, grid_viz, grid_message] 
        ).then(
            lambda: gr.update(selected="tab_grid_image"), 
            outputs=[tabs]
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
            process_mongodb_prompt,
            inputs=[
                mongo_prompts, mongo_db_name, mongo_collection,
                mongo_width, mongo_height, mongo_num_images, mongo_model
            ],
            outputs=[mongo_output, mongo_message]
        )

        batch_process_btn.click(
            batch_process_mongodb_prompts,
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
            process_mongodb_grid,
            inputs=[
                grid_items, grid_db_name, grid_collection,
                grid_width, grid_height, grid_num_images, grid_model
            ],
            outputs=[grid_output, grid_visualization, grid_message]
        )
        
        # Set up event handlers for 2D tabs
        text_submit.click(
            process_text_prompt,
            inputs=[text_input, text_width, text_height, text_num_images, text_model],
            outputs=[text_output, text_message]
        )
        
        grid_submit.click(
            process_grid_input,
            inputs=[grid_input, grid_width, grid_height, grid_num_images, grid_model],
            outputs=[grid_output, grid_viz, grid_message]
        )
        
        file_submit.click(
            process_file_upload,
            inputs=[file_upload, file_width, file_height, file_num_images, file_text_model, file_grid_model],
            outputs=[file_output, file_grid_viz, file_message]
        )
        
        sample_button.click(
            lambda: create_sample_grid(),
            inputs=[],
            outputs=[grid_input]
        )
    
    return demo

def create_fastapi_app(gradio_app):
    # Create a FastAPI app
    app = FastAPI()
    
    # Mount the Gradio app
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    return app

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    args = parser.parse_args()
    
    # HAS_3D_SUPPORT is permanently False, no need to update from args
    logger.info("merged_gradio_app.py: 3D generation features are permanently disabled as per current configuration.")

    try:
        initialize_processors() 
        logger.info("merged_gradio_app.py: 2D processors initialized.")
        
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
        
        # Fallback UI (simplified to only 2D image/grid processing)
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            import gradio as gr 

            def fallback_text_processor_dummy(prompt, width, height, num_images, model_type):
                dummy_image = Image.new('RGB', (width, height), color=(70, 130, 180)) # SteelBlue
                draw = ImageDraw.Draw(dummy_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                text = f"Fallback Image for:\n'{prompt[:40]}...'\n(Full app failed to load)"
                text_width, text_height = draw.textsize(text, font=font)
                draw.text(((width - text_width) / 2, (height - text_height) / 2), text, (255, 255, 255), font=font)
                return dummy_image, "Error: Main app failed. Displaying fallback."

            def fallback_grid_processor_dummy(grid_string, width, height, num_images, model_type):
                dummy_image = Image.new('RGB', (width, height), color=(180, 70, 70)) # IndianRed
                draw = ImageDraw.Draw(dummy_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                text = f"Fallback Grid Viz for:\n'{grid_string[:40]}...'\n(Full app failed to load)"
                text_width, text_height = draw.textsize(text, font=font)
                draw.text(((width - text_width) / 2, (height - text_height) / 2), text, (255, 255, 255), font=font)
                return dummy_image, dummy_image, "Error: Main app failed. Displaying fallback."


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
                <p>This is a simplified interface. For full features, resolve the error above.</p>
                </div>
                """)
                # Fallback tabs - keeping only the requested 2D ones
                with gr.Tabs(selected="tab_biome_inspector_fallback") as fallback_tabs:
                    with gr.TabItem("Biome Inspector (Fallback)", id="tab_biome_inspector_fallback"):
                        gr.Markdown("# Biome Generation Pipeline Inspector (Fallback)")
                        gr.HTML("<p>Biome generation and viewing features are limited due to core app failure.</p>")
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
                            lambda t, s: ("Fallback: Biome generation not available in this mode.", gr.update(choices=[], value=None)),
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
                            fallback_text_processor_dummy, 
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
                            fallback_grid_processor_dummy, 
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

