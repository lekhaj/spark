import os
import gradio as gr
import numpy as np
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from config import OUTPUT_DIR
from pipeline.text_processor import TextProcessor
from pipeline.grid_processor import GridProcessor
from pipeline.pipeline import Pipeline
from terrain.grid_parser import GridParser
from utils.image_utils import save_image, create_image_grid

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize global variables for processors and pipeline
text_processor = None
grid_processor = None
pipeline = None

def initialize_processors():
    """Initialize the processors and pipeline with default models"""
    global text_processor, grid_processor, pipeline
    text_processor = TextProcessor()
    grid_processor = GridProcessor()
    pipeline = Pipeline(text_processor, grid_processor)

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

# Create the Gradio Interface
with gr.Blocks(title="Text & Grid to Image Generator") as app:
    gr.Markdown("# Text & Grid to Image Generator")
    gr.Markdown("Generate images from text prompts or terrain grid data")
    
    with gr.Tab("Text to Image"):
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
    
    with gr.Tab("Grid to Image"):
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
    
    with gr.Tab("File Upload"):
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
    
    # Set up event handlers
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

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
