import os
import logging
from models.api_client import get_client
from terrain.grid_parser import GridParser
from config import DEFAULT_GRID_MODEL, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES, OUTPUT_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GridProcessor:
    """Process grid data and convert it to terrain images"""
    
    def __init__(self, model_type=DEFAULT_GRID_MODEL):
        """
        Initialize the grid processor
        
        Args:
            model_type (str): The model type to use ('openai', 'stability', or 'local')
        """
        self.model_type = model_type
        self.grid_parser = GridParser()
        try:
            self.client = get_client(model_type)
            logger.info(f"Initialized grid processor with {model_type} model")
        except Exception as e:
            logger.error(f"Error initializing grid processor: {str(e)}")
            raise
    
    def convert_grid_to_image(self, grid_data, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, 
                              num_images=DEFAULT_NUM_IMAGES, save=True, visualize_grid=True):
        """
        Convert a grid of terrain types to an image
        
        Args:
            grid_data (str or list): Either a grid string or a 2D list/array of terrain type IDs
            width (int): Desired image width
            height (int): Desired image height
            num_images (int): Number of images to generate
            save (bool): Whether to save the generated images
            visualize_grid (bool): Whether to create a visualization of the grid
            
        Returns:
            tuple: (List of generated PIL Image objects, Grid visualization Image)
        """
        try:
            # Parse the grid
            if isinstance(grid_data, str):
                self.grid_parser.parse_grid_string(grid_data)
            else:
                self.grid_parser.grid = grid_data
                self.grid_parser.height, self.grid_parser.width = grid_data.shape if hasattr(grid_data, 'shape') else (len(grid_data), len(grid_data[0]))
            
            if self.grid_parser.grid is None:
                logger.error("Failed to parse grid data")
                return [], None
                
            # Generate a prompt from the grid
            prompt = self.grid_parser.get_prompt_from_grid()
            logger.info(f"Generated prompt from grid: {prompt}")
            
            # Generate the terrain image
            images = self.client.generate_image(
                prompt, 
                width=width,
                height=height,
                num_images=num_images
            )
            
            if not images:
                logger.error("No images were generated")
                return [], None
                
            logger.info(f"Generated {len(images)} terrain images from grid")
            
            # Create a visualization of the grid
            grid_viz = None
            if visualize_grid:
                grid_viz = self.grid_parser.visualize_grid()
                
            # Save the images if requested
            if save:
                self._save_images(images, prompt, grid_viz)
                
            return images, grid_viz
            
        except Exception as e:
            logger.error(f"Error converting grid to image: {str(e)}")
            return [], None
    
    def _save_images(self, images, prompt, grid_viz=None):
        """
        Save images to the output directory
        
        Args:
            images (list): List of PIL Image objects
            prompt (str): The generated prompt (used for naming)
            grid_viz (PIL.Image): Grid visualization image
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Create a sanitized version of the prompt for the filename
            import re
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_slug = re.sub(r'[^\w\s-]', '', prompt.lower())
            prompt_slug = re.sub(r'[-\s]+', '_', prompt_slug)
            prompt_slug = prompt_slug[:30]  # Truncate to reasonable length
            
            # Save each generated image
            for i, image in enumerate(images):
                filename = f"grid_{prompt_slug}_{timestamp}_{i+1}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                image.save(filepath)
                logger.info(f"Saved terrain image to {filepath}")
            
            # Save the grid visualization if available
            if grid_viz:
                viz_filename = f"grid_viz_{prompt_slug}_{timestamp}.png"
                viz_filepath = os.path.join(OUTPUT_DIR, viz_filename)
                grid_viz.save(viz_filepath)
                logger.info(f"Saved grid visualization to {viz_filepath}")
                
        except Exception as e:
            logger.error(f"Error saving images: {str(e)}")