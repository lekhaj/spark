import os
import time
import uuid
import shutil
import argparse
import logging
import gradio as gr
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
try:
    import uvicorn
except ImportError:
    pass
import pymongo
from datetime import datetime
from db_helper import MongoDBHelper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules for 2D image generation
from config import OUTPUT_DIR
from pipeline.text_processor import TextProcessor
from pipeline.grid_processor import GridProcessor
from pipeline.pipeline import Pipeline
from terrain.grid_parser import GridParser
from utils.image_utils import save_image, create_image_grid

# Import modules for 3D generation
try:
    from hy3dgen.shapegen.utils import logger as hy3d_logger
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    HAS_3D_SUPPORT = True
except ImportError as e:
    logger.warning(f"3D generation modules could not be imported: {str(e)}")
    logger.warning("To enable 3D generation, install required system packages:")
    logger.warning("For Ubuntu/Debian: sudo apt-get install libgl1-mesa-glx xvfb")
    logger.warning("For CentOS/RHEL: sudo yum install mesa-libGL")
    HAS_3D_SUPPORT = False

# Constants
SAVE_DIR = "gradio_cache"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_SEED = int(1e7)
HTML_HEIGHT = 650
HTML_WIDTH = 500
HTML_OUTPUT_PLACEHOLDER = f"""
<div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
  <div style='text-align: center; font-size: 16px; color: #6b7280;'>
    <p style="color: #8d8d8d;">Welcome to the Integrated Pipeline!</p>
    <p style="color: #8d8d8d;">No mesh here yet. Generate an image first, then create a 3D model.</p>
  </div>
</div>
"""

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize global variables for processors and pipeline
text_processor = None
grid_processor = None
pipeline = None
rmbg_worker = None
i23d_worker = None
face_reduce_worker = None
texgen_worker = None
HAS_TEXTUREGEN = False
HAS_3D_SUPPORT = False  # Will be updated during import

def initialize_processors():
    """Initialize the processors and pipeline with default models"""
    global text_processor, grid_processor, pipeline
    text_processor = TextProcessor()
    grid_processor = GridProcessor()
    pipeline = Pipeline(text_processor, grid_processor)

def initialize_3d_processors(model_path='tencent/Hunyuan3D-2', subfolder='hunyuan3d-dit-v2-0', device='cuda'):
    """Initialize the 3D processors"""
    global rmbg_worker, i23d_worker, face_reduce_worker, texgen_worker, HAS_TEXTUREGEN, HAS_3D_SUPPORT
    
    if not HAS_3D_SUPPORT:
        logger.warning("3D support is not available. Only 2D image generation will be enabled.")
        return False
    
    try:
        # Initialize background remover
        rmbg_worker = BackgroundRemover()
        
        # Initialize image-to-3D pipeline
        i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        
        # Initialize face reducer
        face_reduce_worker = FaceReducer()
        
        # Try to initialize texture generator if available
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            HAS_TEXTUREGEN = True
        except Exception as e:
            logger.warning(f"Texture generation is disabled due to: {str(e)}")
            HAS_TEXTUREGEN = False
            
        return True
    except Exception as e:
        logger.error(f"Failed to initialize 3D processors: {str(e)}")
        HAS_3D_SUPPORT = False
        return False

# Initialize on startup
initialize_processors()

def process_text_prompt(prompt, width=512, height=512, num_images=1, model_type="openai"):
    """Process text prompt and generate images"""
    global text_processor, pipeline
    
    if not prompt:
        return None, "Error: No prompt provided"
    
    # Update the model type if it's different from current
    if text_processor.model_type != model_type:
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
        
        # Create a grid of images if multiple were generated
        if len(images) > 1:
            grid_image = create_image_grid(images)
            save_image(grid_image, f"text_grid_{prompt[:20]}")
            return grid_image, f"Generated {len(images)} images from text prompt"
        else:
            return images[0], "Generated 1 image from text prompt"
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return None, f"Error: {str(e)}"

def process_grid_input(grid_string, width=512, height=512, num_images=1, model_type="stability"):
    """Process grid data and generate terrain images"""
    global grid_processor, pipeline
    
    if not grid_string:
        return None, None, "Error: No grid provided"
    
    # Update the model type if it's different from current
    if grid_processor.model_type != model_type:
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
        
        # Create a grid of images if multiple were generated
        if len(images) > 1:
            grid_image = create_image_grid(images)
            save_image(grid_image, "terrain_grid")
            return grid_image, grid_viz, f"Generated {len(images)} images from grid"
        else:
            return images[0], grid_viz, "Generated 1 image from grid"
    
    except Exception as e:
        logger.error(f"Error processing grid: {str(e)}")
        return None, None, f"Error: {str(e)}"

def process_file_upload(file_obj, width=512, height=512, num_images=1, text_model_type="openai", grid_model_type="stability"):
    """Process an uploaded file containing text or grid data"""
    if file_obj is None:
        return None, None, "Error: No file uploaded"
    
    try:
        content = file_obj.decode("utf-8").strip()
        
        # Determine if this is a text prompt or a grid
        # (Simple heuristic: if it contains numbers and whitespace primarily, it's a grid)
        is_grid = True
        non_grid_chars = [c for c in content if not (c.isdigit() or c.isspace())]
        if len(non_grid_chars) > len(content) * 0.1:  # More than 10% non-grid chars
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

# 3D Generation Functions
def gen_save_folder(max_size=200):
    """Generate a new save folder and manage old ones"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Get all directory paths
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]

    # If the number of folders exceeds max_size, delete the oldest one
    if len(dirs) >= max_size:
        # Sort by creation time, oldest first
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        logger.info(f"Removed the oldest folder: {oldest_dir}")

    # Generate a new uuid folder name
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    logger.info(f"Created new folder: {new_folder}")

    return new_folder

def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """Export the mesh to a file"""
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path

def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    """Build HTML for the 3D model viewer"""
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = '../Hunyuan3D-2/assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = '../Hunyuan3D-2/assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    logger.info(f'HTML file created: {output_html_path}, relative path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """Generate a random seed or use the provided one"""
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed

def generate_3d_from_image(
    image=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed=False,
    with_texture=True
):
    """Generate a 3D model from an image"""
    global rmbg_worker, i23d_worker, face_reduce_worker, texgen_worker, HAS_3D_SUPPORT
    
    if not HAS_3D_SUPPORT:
        raise gr.Error("3D generation is not supported. Required libraries are missing. Please install libgl1-mesa-glx package.")
    
    if i23d_worker is None:
        success = initialize_3d_processors()
        if not success:
            raise gr.Error("Failed to initialize 3D processors. Please check system requirements.")
    
    if image is None:
        raise gr.Error("Please provide an image for 3D generation.")

    # Convert gradio image to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    seed = int(randomize_seed_fn(seed, randomize_seed))
    
    start_time_0 = time.time()
    save_folder = gen_save_folder()
    stats = {
        'params': {
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        }
    }
    time_meta = {}
    
    # Remove background if needed
    if check_box_rembg or image.mode == "RGB":
        start_time = time.time()
        image = rmbg_worker(image.convert('RGB'))
        time_meta['remove background'] = time.time() - start_time
        
    # Generate 3D shape
    start_time = time.time()
    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    
    outputs = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh'
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info(f"Shape generation took {time.time() - start_time} seconds")
    
    # Export to trimesh
    tmp_start = time.time()
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start
    
    # Record mesh statistics
    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]
    
    # Face reduction
    tmp_time = time.time()
    mesh = face_reduce_worker(mesh)
    logger.info(f"Face reduction took {time.time() - tmp_time} seconds")
    stats['time'] = time_meta
    stats['time']['face reduction'] = time.time() - tmp_time
    
    # Export white mesh
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    
    # Generate textured mesh if requested
    if with_texture and HAS_TEXTUREGEN:
        tmp_time = time.time()
        textured_mesh = texgen_worker(mesh, image)
        logger.info(f"Texture generation took {time.time() - tmp_time} seconds")
        stats['time']['texture generation'] = time.time() - tmp_time
        
        # Export textured mesh
        path_textured = export_mesh(textured_mesh, save_folder, textured=True)
        model_viewer_html_textured = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
        
        stats['time']['total'] = time.time() - start_time_0
        return (
            gr.update(value=path),
            gr.update(value=path_textured),
            model_viewer_html_textured,
            stats,
            seed,
        )
    else:
        stats['time']['total'] = time.time() - start_time_0
        return (
            gr.update(value=path),
            gr.update(value=None),
            model_viewer_html,
            stats,
            seed,
        )



# MongoDB Integration Functions
def get_prompts_from_mongodb(db_name, collection_name, limit=100):
    """Retrieve prompts from MongoDB collection"""
    try:
        mongo_helper = MongoDBHelper()
        
        # Find documents that have theme_prompt or description fields
        query = {"$or": [
            {"theme_prompt": {"$exists": True}},
            {"description": {"$exists": True}}
        ]}
        
        documents = mongo_helper.find_many(db_name, collection_name, query=query, limit=limit)
        
        if not documents:
            return [], "No prompts found in the specified collection."
        
        # Extract prompts and IDs
        prompt_items = []
        for doc in documents:
            doc_id = str(doc.get("_id"))
            # Try different fields that might contain a prompt
            prompt = None
            if "theme_prompt" in doc:
                prompt = doc["theme_prompt"]
            elif "description" in doc:
                prompt = doc["description"]
            
            # Extract nested descriptions if present
            if not prompt and "possible_structures" in doc:
                structures = doc.get("possible_structures", {})
                for category in structures.values():
                    for item in category.values():
                        if "description" in item:
                            if not prompt:  # Take the first description found
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

def process_mongodb_prompt(prompt_id, db_name, collection_name, width=512, height=512, 
                          num_images=1, model_type="openai"):
    """Process a prompt from MongoDB by ID"""
    global text_processor, pipeline
    
    try:
        mongo_helper = MongoDBHelper()
        document = mongo_helper.find_by_id(db_name, collection_name, prompt_id)
        
        if not document:
            return None, "Error: Document not found"
        
        # Try different fields that might contain a prompt
        prompt = None
        if "theme_prompt" in document:
            prompt = document["theme_prompt"]
        elif "description" in document:
            prompt = document["description"]
        
        # Extract nested descriptions if no prompt found
        if not prompt and "possible_structures" in document:
            structures = document.get("possible_structures", {})
            for category in structures.values():
                for item in category.values():
                    if "description" in item:
                        if not prompt:  # Take the first description found
                            prompt = item["description"]
                        break
                if prompt:
                    break
        
        if not prompt:
            return None, "Error: No prompt found in the document"
        
        image, message = process_text_prompt(prompt, width, height, num_images, model_type)
        
        if image is not None and "Error" not in message:
            # Update document in MongoDB to mark as processed
            update = {
                "$set": {
                    "processed": True,
                    "processed_at": datetime.now(),
                    "model_used": model_type
                }
            }
            mongo_helper.update_by_id(db_name, collection_name, prompt_id, update)
        
        return image, message
        
    except Exception as e:
        logger.error(f"Error processing MongoDB prompt: {str(e)}")
        return None, f"Error: {str(e)}"

def get_grids_from_mongodb(db_name, collection_name, limit=100):
    """Retrieve grid data from MongoDB collection"""
    try:
        mongo_helper = MongoDBHelper()
        
        # Find documents that have grid or layout fields
        query = {"$or": [
            {"grid": {"$exists": True}},
            {"possible_grids.layout": {"$exists": True}}
        ]}
        
        documents = mongo_helper.find_many(db_name, collection_name, query=query, limit=limit)
        
        if not documents:
            return [], "No grids found in the specified collection."
        
        # Extract grids and IDs
        grid_items = []
        for doc in documents:
            doc_id = str(doc.get("_id"))
            
            # Check for direct grid field
            if "grid" in doc:
                grid = doc["grid"]
                grid_items.append((doc_id, grid))
            # Check for nested grids in possible_grids array
            elif "possible_grids" in doc:
                for grid_obj in doc["possible_grids"]:
                    if "layout" in grid_obj:
                        # Convert 2D array to string format
                        grid_str = "\n".join([" ".join(map(str, row)) for row in grid_obj["layout"]])
                        grid_items.append((f"{doc_id}_{grid_obj.get('grid_id', 'grid')}", grid_str))
        
        return grid_items, f"Found {len(grid_items)} grids"
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        return [], f"Error connecting to MongoDB: {str(e)}"

def process_mongodb_grid(grid_item, db_name, collection_name, width=512, height=512, 
                        num_images=1, model_type="stability"):
    """Process a grid from MongoDB by ID"""
    global grid_processor, pipeline
    
    try:
        # Split the grid_item which could be "document_id" or "document_id_grid_id"
        parts = grid_item.split("_", 1)
        doc_id = parts[0]
        
        mongo_helper = MongoDBHelper()
        document = mongo_helper.find_by_id(db_name, collection_name, doc_id)
        
        if not document:
            return None, None, "Error: Document not found"
        
        grid = None
        
        # Direct grid field
        if len(parts) == 1 and "grid" in document:
            grid = document["grid"]
        
        # Nested grid in possible_grids
        elif len(parts) > 1 and "possible_grids" in document:
            grid_id = parts[1]
            for grid_obj in document["possible_grids"]:
                if grid_obj.get("grid_id") == grid_id:
                    if "layout" in grid_obj:
                        grid = "\n".join([" ".join(map(str, row)) for row in grid_obj["layout"]])
                        break
        
        if not grid:
            return None, None, "Error: Grid not found in the document"
        
        image, grid_viz, message = process_grid_input(grid, width, height, num_images, model_type)
        
        if image is not None and grid_viz is not None:
            # Update document in MongoDB to mark as processed
            update = {
                "$set": {
                    "processed": True,
                    "processed_at": datetime.now(),
                    "model_used": model_type
                }
            }
            mongo_helper.update_by_id(db_name, collection_name, doc_id, update)
        
        return image, grid_viz, message
        
    except Exception as e:
        logger.error(f"Error processing MongoDB grid: {str(e)}")
        return None, None, f"Error: {str(e)}"

def batch_process_mongodb_prompts(db_name, collection_name, limit=10, width=512, height=512, 
                                 model_type="openai", update_db=False):
    """Batch process multiple prompts from MongoDB"""
    try:
        # First get the prompts
        prompt_items, status = get_prompts_from_mongodb(db_name, collection_name, limit)
        
        if not prompt_items:
            return "No prompts found to process."
        
        results = []
        for doc_id, prompt in prompt_items:
            # Process the prompt
            logger.info(f"Processing prompt: {prompt}")
            image, message = process_text_prompt(prompt, width, height, 1, model_type)
            
            if image is not None and "Error" not in message:
                # Save image
                image_path = os.path.join(OUTPUT_DIR, f"{doc_id}.png")
                image.save(image_path)
                results.append(f"Generated image for: {prompt[:30]}...")
                
                # Update document in MongoDB if requested
                if update_db:
                    mongo_helper = MongoDBHelper()
                    update = {
                        "$set": {
                            "processed": True,
                            "image_path": image_path,
                            "processed_at": datetime.now(),
                            "model_used": model_type
                        }
                    }
                    mongo_helper.update_by_id(db_name, collection_name, doc_id, update)
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
    Integrated 2D-to-3D Generation Pipeline
    </div>
    <div align="center">
    Generate 2D images and convert them to 3D models
    </div>
    """
    
    with gr.Blocks(theme=gr.themes.Base(), title='2D-to-3D Pipeline', css=custom_css) as demo:
        gr.HTML(title_html)
        
        # Display a banner if 3D is not supported
        if not HAS_3D_SUPPORT:
            gr.HTML(f"""
            <div style="background-color: #fff3cd; color: #856404; padding: 10px; margin-bottom: 20px; border-radius: 5px; text-align: center;">
            <strong>Note:</strong> Running in 2D-only mode. 3D generation features are disabled due to missing dependencies.<br>
            To enable 3D features, install the required libraries: <code>sudo apt-get install libgl1-mesa-glx xvfb</code>
            </div>
            """)
        
        with gr.Tabs() as tabs:
            # Text to Image Tab
            with gr.TabItem("Text to Image", id="tab_text"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(label="Text Prompt", placeholder="Enter a description of the image you want to generate...")
                        with gr.Row():
                            text_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                            text_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                        text_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                        text_model = gr.Dropdown(["openai", "stability", "local"], value="openai", label="Model")
                        text_submit = gr.Button("Generate Image from Text")
                    
                    with gr.Column(scale=2):
                        text_output = gr.Image(label="Generated Image")
                        text_message = gr.Textbox(label="Status", interactive=False)
                        text_to_3d_btn = gr.Button("Convert to 3D", visible=False)
            
            # Grid to Image Tab
            with gr.TabItem("Grid to Image", id="tab_grid"):
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
                        """)
                        grid_input = gr.Textbox(label="Grid Data", placeholder="Enter your grid data...", lines=10)
                        sample_button = gr.Button("Load Sample Grid")
                        with gr.Row():
                            grid_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                            grid_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                        grid_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                        grid_model = gr.Dropdown(["openai", "stability", "local"], value="stability", label="Model")
                        grid_submit = gr.Button("Generate Image from Grid")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            grid_output = gr.Image(label="Generated Terrain")
                            grid_viz = gr.Image(label="Grid Visualization")
                        grid_message = gr.Textbox(label="Status", interactive=False)
                        grid_to_3d_btn = gr.Button("Convert to 3D", visible=False)
            
            # File Upload Tab
            with gr.TabItem("File Upload", id="tab_file"):
                with gr.Row():
                    with gr.Column(scale=3):
                        file_upload = gr.File(label="Upload a text file or grid file")
                        gr.Markdown("System will automatically detect if the file contains text or grid data")
                        with gr.Row():
                            file_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                            file_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                        file_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                        with gr.Row():
                            file_text_model = gr.Dropdown(["openai", "stability", "local"], value="openai", label="Text Model")
                            file_grid_model = gr.Dropdown(["openai", "stability", "local"], value="stability", label="Grid Model")
                        file_submit = gr.Button("Process File")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            file_output = gr.Image(label="Generated Image")
                            file_grid_viz = gr.Image(label="Grid Visualization (if applicable)")
                        file_message = gr.Textbox(label="Status", interactive=False)
                        file_to_3d_btn = gr.Button("Convert to 3D", visible=False)
            
            # 3D Generation Tab
            with gr.TabItem("3D Generation", id="tab_3d"):
                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(label="Input Image", type="pil")
                        with gr.Row():
                            steps = gr.Slider(minimum=20, maximum=100, value=50, step=1, label="Steps")
                            guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.1, label="Guidance Scale")
                        with gr.Row():
                            seed = gr.Number(value=1234, label="Seed", precision=0)
                            randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                        with gr.Row():
                            octree_resolution = gr.Slider(minimum=128, maximum=512, value=256, step=16, label="Octree Resolution")
                            num_chunks = gr.Slider(minimum=50000, maximum=500000, value=200000, step=50000, label="Num Chunks")
                        check_box_rembg = gr.Checkbox(label="Remove Background", value=True)
                        with_texture = gr.Checkbox(label="Generate Texture", value=True)
                        gen_3d_btn = gr.Button("Generate 3D Model")
                    
                    with gr.Column(scale=3):
                        with gr.Tabs() as tabs_output:
                            with gr.TabItem("3D Model", id="gen_mesh_panel"):
                                html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER)
                            with gr.TabItem("Statistics", id="stats_panel"):
                                stats_output = gr.JSON(label="Generation Statistics")
                        
                        with gr.Row():
                            file_out = gr.File(label="White Mesh", visible=False)
                            file_out2 = gr.File(label="Textured Mesh", visible=False)
            
            # MongoDB Prompts Tab
            with gr.TabItem("MongoDB", id="tab_mongodb"):
                with gr.Tabs() as mongo_tabs:
                    with gr.TabItem("Text Prompts", id="tab_mongo_text"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                mongo_db_name = gr.Textbox(label="Database Name", value="World_builder", placeholder="Enter database name")
                                mongo_collection = gr.Textbox(label="Collection Name", value="biomes", placeholder="Enter collection name")
                                mongo_fetch_btn = gr.Button("Fetch Prompts")
                                with gr.Row():
                                    mongo_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                                    mongo_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                                mongo_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                                mongo_model = gr.Dropdown(["openai", "stability", "local"], value="openai", label="Model")
                            mongo_process_btn = gr.Button("Generate Image", interactive=False)
                    
                        with gr.Column(scale=2):
                            mongo_prompts = gr.Dropdown(label="Select a Prompt", choices=[], interactive=False, allow_custom_value=True)
                            mongo_status = gr.Textbox(label="Status", interactive=False)
                            mongo_output = gr.Image(label="Generated Image")
                            mongo_message = gr.Textbox(label="Generation Status", interactive=False)
                            mongo_to_3d_btn = gr.Button("Convert to 3D", visible=False)
                        
                    with gr.Accordion("Batch Processing", open=False):
                        with gr.Row():
                            batch_limit = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Prompts to Process")
                            update_db = gr.Checkbox(label="Update MongoDB after processing", value=True)
                        batch_process_btn = gr.Button("Batch Process Prompts")
                        batch_results = gr.Textbox(label="Batch Processing Results", interactive=False, lines=10)
                        
                with gr.TabItem("Grid Data", id="tab_mongo_grid"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            grid_db_name = gr.Textbox(label="Database Name", value="World_builder", placeholder="Enter database name")
                            grid_collection = gr.Textbox(label="Collection Name", value="biomes", placeholder="Enter collection name")
                            grid_fetch_btn = gr.Button("Fetch Grids")
                            with gr.Row():
                                grid_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                                grid_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                            grid_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                            grid_model = gr.Dropdown(["openai", "stability", "local"], value="stability", label="Model")
                            grid_process_btn = gr.Button("Generate Image", interactive=False)
                    
                        with gr.Column(scale=2):
                            grid_items = gr.Dropdown(label="Select a Grid", choices=[], interactive=False, allow_custom_value=True)
                            grid_status = gr.Textbox(label="Status", interactive=False)
                            grid_output = gr.Image(label="Generated Image")
                            grid_visualization = gr.Image(label="Grid Visualization")
                            grid_message = gr.Textbox(label="Generation Status", interactive=False)
                            grid_to_3d_btn = gr.Button("Convert to 3D", visible=False)
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
        ).then(
            lambda: gr.update(visible=HAS_3D_SUPPORT),  # Only show if 3D is supported
            outputs=[mongo_to_3d_btn]
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
        ).then(
            lambda: gr.update(visible=HAS_3D_SUPPORT),  # Only show if 3D is supported
            outputs=[grid_to_3d_btn]
        )

        # Connect MongoDB outputs to 3D generation
        mongo_to_3d_btn.click(
            lambda: gr.update(selected="tab_3d"),
            outputs=[tabs]
        ).then(
            lambda x: x,
            inputs=[mongo_output],
            outputs=[image_input]
        )

        grid_to_3d_btn.click(
            lambda: gr.update(selected="tab_3d"),
            outputs=[tabs]
        ).then(
            lambda x: x,
            inputs=[grid_output],
            outputs=[image_input]
        )     
        # Set up event handlers
        text_submit.click(
            process_text_prompt,
            inputs=[text_input, text_width, text_height, text_num_images, text_model],
            outputs=[text_output, text_message]
        ).then(
            lambda: gr.update(visible=HAS_3D_SUPPORT),  # Only show if 3D is supported
            outputs=[text_to_3d_btn]
        )
        
        grid_submit.click(
            process_grid_input,
            inputs=[grid_input, grid_width, grid_height, grid_num_images, grid_model],
            outputs=[grid_output, grid_viz, grid_message]
        ).then(
            lambda: gr.update(visible=HAS_3D_SUPPORT),  # Only show if 3D is supported
            outputs=[grid_to_3d_btn]
        )
        
        file_submit.click(
            process_file_upload,
            inputs=[file_upload, file_width, file_height, file_num_images, file_text_model, file_grid_model],
            outputs=[file_output, file_grid_viz, file_message]
        ).then(
            lambda: gr.update(visible=HAS_3D_SUPPORT),  # Only show if 3D is supported
            outputs=[file_to_3d_btn]
        )
        
        sample_button.click(
            lambda: create_sample_grid(),
            inputs=[],
            outputs=[grid_input]
        )
        
        # Connect 2D outputs to 3D generation
        text_to_3d_btn.click(
            lambda: gr.update(selected="tab_3d"),
            outputs=[tabs]
        ).then(
            lambda x: x,
            inputs=[text_output],
            outputs=[image_input]
        )
        
        grid_to_3d_btn.click(
            lambda: gr.update(selected="tab_3d"),
            outputs=[tabs]
        ).then(
            lambda x: x,
            inputs=[grid_output],
            outputs=[image_input]
        )
        
        file_to_3d_btn.click(
            lambda: gr.update(selected="tab_3d"),
            outputs=[tabs]
        ).then(
            lambda x: x,
            inputs=[file_output],
            outputs=[image_input]
        )
        
        # 3D generation
        gen_3d_btn.click(
            generate_3d_from_image,
            inputs=[
                image_input, 
                steps, 
                guidance_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                with_texture
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats_output, seed]
        ).then(
            lambda: gr.update(selected="gen_mesh_panel"),
            outputs=[tabs_output]
        )
    
    return demo

def create_fastapi_app(gradio_app):
    # Create a FastAPI app
    app = FastAPI()
    
    # Create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    
    # Copy environment maps from Hunyuan3D-2
    env_maps_src = os.path.join(os.path.dirname(CURRENT_DIR), 'Hunyuan3D-2/assets/env_maps')
    env_maps_dest = os.path.join(static_dir, 'env_maps')
    if os.path.exists(env_maps_src):
        os.makedirs(env_maps_dest, exist_ok=True)
        for file in os.listdir(env_maps_src):
            src_file = os.path.join(env_maps_src, file)
            dst_file = os.path.join(env_maps_dest, file)
            if not os.path.exists(dst_file) and os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
    
    # Mount the Gradio app
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    parser.add_argument('--disable_3d', action='store_true', help='Disable 3D generation functionality')
    args = parser.parse_args()
    
    # Build and launch the Gradio app
    try:
        # Initialize 2D processors
        initialize_processors()
        
        # Initialize 3D processors if 3D support is available and not disabled
        has_3d = False
        if not args.disable_3d and HAS_3D_SUPPORT:
            has_3d = initialize_3d_processors(args.model_path, args.subfolder, args.device)
        
        # Create and launch app
        demo = build_app()
        
        # Display a warning if 3D is not available
        if not has_3d:
            logger.warning("Running in 2D-only mode. 3D generation features will be disabled.")
            logger.warning("To enable 3D features, install the required system packages:")
            logger.warning("For Ubuntu/Debian: sudo apt-get install libgl1-mesa-glx xvfb")
            logger.warning("For CentOS/RHEL: sudo yum install mesa-libGL")
        
        # Launch the app using either FastAPI or Gradio's built-in server
        try:
            # Use FastAPI for better performance and static file handling
            import uvicorn
            app = create_fastapi_app(demo)
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            # Fall back to Gradio's built-in server if uvicorn is not available
            logger.warning("Uvicorn not found, using Gradio's built-in server instead")
            demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    except Exception as e:
        logger.error(f"Error starting app: {str(e)}")
        # Fallback to simple Gradio app without 3D generation if initialization fails
        initialize_processors()
        with gr.Blocks(theme=gr.themes.Base()) as demo:
            gr.HTML(f"""
            <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
            2D Generation Only Mode
            </div>
            <div align="center" style="color: red; margin-bottom: 20px">
            3D generation is disabled due to an error: {str(e)}
            </div>
            <div align="center">
            <p>To enable 3D generation, you may need to install required system libraries:</p>
            <p><code>sudo apt-get install libgl1-mesa-glx xvfb</code></p>
            </div>
            """)
            
            with gr.Tabs():
                # Text to Image Tab
                with gr.TabItem("Text to Image", id="tab_text"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            text_input = gr.Textbox(label="Text Prompt", placeholder="Enter a description of the image you want to generate...")
                            with gr.Row():
                                text_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                                text_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                            text_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                            text_model = gr.Dropdown(["openai", "stability", "local"], value="openai", label="Model")
                            text_submit = gr.Button("Generate Image from Text")
                        
                        with gr.Column(scale=2):
                            text_output = gr.Image(label="Generated Image")
                            text_message = gr.Textbox(label="Status", interactive=False)
                
                # Grid to Image Tab
                with gr.TabItem("Grid to Image", id="tab_grid"):
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
                            """)
                            grid_input = gr.Textbox(label="Grid Data", placeholder="Enter your grid data...", lines=10)
                            sample_button = gr.Button("Load Sample Grid")
                            with gr.Row():
                                grid_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                                grid_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                            grid_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                            grid_model = gr.Dropdown(["openai", "stability", "local"], value="stability", label="Model")
                            grid_submit = gr.Button("Generate Image from Grid")
                        
                        with gr.Column(scale=2):
                            with gr.Row():
                                grid_output = gr.Image(label="Generated Terrain")
                                grid_viz = gr.Image(label="Grid Visualization")
                            grid_message = gr.Textbox(label="Status", interactive=False)
            
            # Set up event handlers for the simplified UI
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
            
            sample_button.click(
                lambda: create_sample_grid(),
                inputs=[],
                outputs=[grid_input]
            )
        
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
