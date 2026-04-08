import os
import gradio as gr
import numpy as np
import logging
from PIL import Image

# Configure logging to track application events and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules from the Image Generator app
from config import OUTPUT_DIR  # Directory for saving generated images
from pipeline.text_processor import TextProcessor  # Processes text prompts
from pipeline.grid_processor import GridProcessor  # Processes grid data
from pipeline.pipeline import Pipeline  # Combines text and grid processors
from terrain.grid_parser import GridParser  # Parses grid input
from utils.image_utils import save_image, create_image_grid  # Utility functions for image handling

# Import Hunyuan3D-2 functionality (assumed to be in a separate module)
# Adjust this import based on your actual Hunyuan3D-2 setup
from hunyuan_app import shape_generation

# Ensure the output directory exists for saving generated files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize global variables for processors and pipeline
text_processor = None
grid_processor = None
pipeline = None

def initialize_processors():
    """Initialize the text and grid processors and the pipeline with default models"""
    global text_processor, grid_processor, pipeline
    text_processor = TextProcessor()  # Create text processor instance
    grid_processor = GridProcessor()  # Create grid processor instance
    pipeline = Pipeline(text_processor, grid_processor)  # Combine processors into pipeline

# Run initialization on startup to set up processors
initialize_processors()

def generate_3d_asset(image):
    """Generate a 3D asset from the provided image using the Hunyuan3D-2 pipeline"""
    # Check if an image was provided
    if image is None:
        return None, "Error: No image provided for 3D generation"
    
    try:
        # Call Hunyuan3D-2's shape_generation function with the input image
        # Assumes it returns file path, HTML for 3D viewer, stats, and seed
        file_out, html_gen_mesh, stats, seed = shape_generation(image=image)
        return html_gen_mesh, stats
    except Exception as e:
        # Log any errors during 3D generation and return error message
        logger.error(f"Error generating 3D asset: {str(e)}")
        return None, f"Error: {str(e)}"

def process_text_prompt(prompt, width=512, height=512, num_images=1, model_type="openai"):
    """Generate images from a text prompt using the Image Generator pipeline"""
    global text_processor, pipeline
    
    # Validate input prompt
    if not prompt:
        return None, "Error: No prompt provided"
    
    # Update model type if it differs from the current one
    if text_processor.model_type != model_type:
        try:
            text_processor = TextProcessor(model_type=model_type)
            pipeline = Pipeline(text_processor, grid_processor)
        except Exception as e:
            return None, f"Error initializing {model_type} model: {str(e)}"
    
    try:
        logger.info(f"Processing text prompt: {prompt}")
        # Generate images from the text prompt
        images = pipeline.process_text(prompt)
        
        # Check if images were generated successfully
        if not images or len(images) == 0:
            return None, "No images were generated"
        
        logger.info(f"Generated {len(images)} images from text prompt")
        # If multiple images, create a grid; otherwise, use the single image
        output_image = create_image_grid(images) if len(images) > 1 else images[0]
        # Save the output image to the output directory
        save_image(output_image, f"text_grid_{prompt[:20]}")
        return output_image, f"Generated {len(images)} images from text prompt"
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return None, f"Error: {str(e)}"

def process_grid_input(grid_string, width=512, height=512, num_images=1, model_type="stability"):
    """Generate terrain images from grid data using the Image Generator pipeline"""
    global grid_processor, pipeline
    
    # Validate grid input
    if not grid_string:
        return None, None, "Error: No grid provided"
    
    # Update model type if it differs from the current one
    if grid_processor.model_type != model_type:
        try:
            grid_processor = GridProcessor(model_type=model_type)
            pipeline = Pipeline(text_processor, grid_processor)
        except Exception as e:
            return None, None, f"Error initializing {model_type} model: {str(e)}"
    
    try:
        logger.info(f"Processing grid")
        # Process grid to generate images and visualization
        images, grid_viz = pipeline.process_grid(grid_string)
        
        # Check if images were generated successfully
        if not images or len(images) == 0:
            return None, None, "No images were generated"
        
        logger.info(f"Generated {len(images)} images from grid")
        # If multiple images, create a grid; otherwise, use the single image
        output_image = create_image_grid(images) if len(images) > 1 else images[0]
        # Save the output image to the output directory
        save_image(output_image, "terrain_grid")
        return output_image, grid_viz, f"Generated {len(images)} images from grid"
    
    except Exception as e:
        logger.error(f"Error processing grid: {str(e)}")
        return None, None, f"Error: {str(e)}"

def process_file_upload(file_obj, width=512, height=512, num_images=1, text_model_type="openai", grid_model_type="stability"):
    """Process an uploaded file containing text or grid data"""
    # Validate file input
    if file_obj is None:
        return None, None, "Error: No file uploaded"
    
    try:
        # Decode the file content as UTF-8 text
        content = file_obj.decode("utf-8").strip()
        # Heuristic to determine if content is grid (mostly digits and whitespace)
        is_grid = len([c for c in content if not (c.isdigit() or c.isspace())]) <= len(content) * 0.1
        
        # Process as grid or text based on heuristic
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
    """Create a sample grid for demonstration purposes"""
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

# Create the Integrated Gradio Interface
with gr.Blocks(title="Integrated Image & 3D Asset Generator") as app:
    # Display title and description
    gr.Markdown("# Integrated Image & 3D Asset Generator")
    gr.Markdown("Generate images from text prompts or terrain grid data, then create 3D assets")
    
    ### Text to Image Tab
    with gr.Tab("Text to Image"):
        with gr.Row():
            with gr.Column(scale=3):
                # Input field for text prompt
                text_input = gr.Textbox(label="Text Prompt", placeholder="Enter a description...")
                # Sliders for image dimensions
                with gr.Row():
                    text_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                    text_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                # Slider for number of images to generate
                text_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                # Dropdown for model selection
                text_model = gr.Dropdown(["openai", "stability", "local"], value="openai", label="Model")
                # Button to generate image
                text_submit = gr.Button("Generate Image")
                # Button to generate 3D asset, initially disabled
                text_3d_submit = gr.Button("Generate 3D Asset", interactive=False)
            
            with gr.Column(scale=2):
                # Output components for image, status, 3D viewer, and stats
                text_output = gr.Image(label="Generated Image")
                text_message = gr.Textbox(label="Status", interactive=False)
                text_3d_viewer = gr.HTML(label="3D Model Viewer")
                text_3d_stats = gr.JSON(label="3D Generation Stats")
    
    ### Grid to Image Tab
    with gr.Tab("Grid to Image"):
        with gr.Row():
            with gr.Column(scale=3):
                # Instructions for grid format
                gr.Markdown("""
                ## Grid Format
                - 0: Plain
                - 1: Forest
                - 2: Mountain
                - 3: Water
                - 4: Desert
                """)
                # Input field for grid data
                grid_input = gr.Textbox(label="Grid Data", placeholder="Enter your grid data...", lines=10)
                # Button to load sample grid
                sample_button = gr.Button("Load Sample Grid")
                # Sliders for image dimensions
                with gr.Row():
                    grid_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                    grid_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                # Slider for number of images
                grid_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                # Dropdown for model selection
                grid_model = gr.Dropdown(["openai", "stability", "local"], value="stability", label="Model")
                # Button to generate image
                grid_submit = gr.Button("Generate Image")
                # Button to generate 3D asset, initially disabled
                grid_3d_submit = gr.Button("Generate 3D Asset", interactive=False)
            
            with gr.Column(scale=2):
                # Output components for image, grid visualization, status, 3D viewer, and stats
                grid_output = gr.Image(label="Generated Terrain")
                grid_viz = gr.Image(label="Grid Visualization")
                grid_message = gr.Textbox(label="Status", interactive=False)
                grid_3d_viewer = gr.HTML(label="3D Model Viewer")
                grid_3d_stats = gr.JSON(label="3D Generation Stats")
    
    ### File Upload Tab
    with gr.Tab("File Upload"):
        with gr.Row():
            with gr.Column(scale=3):
                # Input for file upload
                file_upload = gr.File(label="Upload a text file or grid file")
                # Note about automatic detection
                gr.Markdown("System will detect if the file contains text or grid data")
                # Sliders for image dimensions
                with gr.Row():
                    file_width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                    file_height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                # Slider for number of images
                file_num_images = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                # Dropdowns for model selection
                with gr.Row():
                    file_text_model = gr.Dropdown(["openai", "stability", "local"], value="openai", label="Text Model")
                    file_grid_model = gr.Dropdown(["openai", "stability", "local"], value="stability", label="Grid Model")
                # Button to generate image
                file_submit = gr.Button("Generate Image")
                # Button to generate 3D asset, initially disabled
                file_3d_submit = gr.Button("Generate 3D Asset", interactive=False)
            
            with gr.Column(scale=2):
                # Output components for image, grid visualization, status, 3D viewer, and stats
                file_output = gr.Image(label="Generated Image")
                file_grid_viz = gr.Image(label="Grid Visualization (if applicable)")
                file_message = gr.Textbox(label="Status", interactive=False)
                file_3d_viewer = gr.HTML(label="3D Model Viewer")
                file_3d_stats = gr.JSON(label="3D Generation Stats")
    
    # Set up event handlers for the pipeline
    # Text Tab: Generate image, then enable 3D generation
    text_submit.click(
        # Clear previous 3D outputs to reset the interface
        fn=lambda: (None, None),
        inputs=[],
        outputs=[text_3d_viewer, text_3d_stats]
    ).then(
        # Generate image from text prompt
        fn=process_text_prompt,
        inputs=[text_input, text_width, text_height, text_num_images, text_model],
        outputs=[text_output, text_message]
    ).then(
        # Enable the 3D generation button after image is generated
        fn=lambda: gr.update(interactive=True),
        outputs=[text_3d_submit]
    )
    text_3d_submit.click(
        # Generate 3D asset from the generated image
        fn=generate_3d_asset,
        inputs=[text_output],
        outputs=[text_3d_viewer, text_3d_stats]
    )
    
    # Grid Tab: Generate image and visualization, then enable 3D generation
    grid_submit.click(
        # Clear previous 3D outputs
        fn=lambda: (None, None),
        inputs=[],
        outputs=[grid_3d_viewer, grid_3d_stats]
    ).then(
        # Generate image and visualization from grid input
        fn=process_grid_input,
        inputs=[grid_input, grid_width, grid_height, grid_num_images, grid_model],
        outputs=[grid_output, grid_viz, grid_message]
    ).then(
        # Enable the 3D generation button
        fn=lambda: gr.update(interactive=True),
        outputs=[grid_3d_submit]
    )
    grid_3d_submit.click(
        # Generate 3D asset from the generated image
        fn=generate_3d_asset,
        inputs=[grid_output],
        outputs=[grid_3d_viewer, grid_3d_stats]
    )
    
    # File Tab: Generate image and optional visualization, then enable 3D generation
    file_submit.click(
        # Clear previous 3D outputs
        fn=lambda: (None, None),
        inputs=[],
        outputs=[file_3d_viewer, file_3d_stats]
    ).then(
        # Process uploaded file to generate image
        fn=process_file_upload,
        inputs=[file_upload, file_width, file_height, file_num_images, file_text_model, file_grid_model],
        outputs=[file_output, file_grid_viz, file_message]
    ).then(
        # Enable the 3D generation button
        fn=lambda: gr.update(interactive=True),
        outputs=[file_3d_submit]
    )
    file_3d_submit.click(
        # Generate 3D asset from the generated image
        fn=generate_3d_asset,
        inputs=[file_output],
        outputs=[file_3d_viewer, file_3d_stats]
    )
    
    # Load sample grid for the Grid Tab
    sample_button.click(
        fn=create_sample_grid,
        inputs=[],
        outputs=[grid_input]
    )

# Launch the integrated app
if __name__ == "__main__":
    app.launch(share=True)