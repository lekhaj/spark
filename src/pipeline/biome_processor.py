import os
import json
import logging
from models.api_client import get_client
from terrain.biome_parser import BiomeParser
from config import DEFAULT_GRID_MODEL, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES, OUTPUT_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomeProcessor:
    """Process biome data and convert it to images"""
    
    def __init__(self, model_type=DEFAULT_GRID_MODEL):
        """
        Initialize the biome processor
        
        Args:
            model_type (str): The model type to use ('openai', 'stability', or 'local')
        """
        self.model_type = model_type
        self.biome_parser = BiomeParser()
        try:
            self.client = get_client(model_type)
            logger.info(f"Initialized biome processor with {model_type} model")
        except Exception as e:
            logger.error(f"Error initializing biome processor: {str(e)}")
            raise
    
    def process_biome_file(self, file_path, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT,
                          num_images=DEFAULT_NUM_IMAGES, save=True, visualize_grid=True, grid_index=0):
        """
        Process a biome file and generate images
        
        Args:
            file_path (str): Path to the biome JSON file
            width (int): Desired image width
            height (int): Desired image height
            num_images (int): Number of images to generate
            save (bool): Whether to save the generated images
            visualize_grid (bool): Whether to create a visualization of the grid
            grid_index (int): Index of the grid to use if multiple are available
            
        Returns:
            tuple: (List of generated PIL Image objects, Grid visualization Image)
        """
        try:
            # Create a new BiomeParser instance to prevent issues with reused parsers
            self.biome_parser = BiomeParser()
            
            # Parse the biome file
            biome_data = self.biome_parser.parse_biome_file(file_path)
            if not biome_data:
                logger.error(f"Failed to parse biome file: {file_path}")
                return [], None
            
            return self.process_biome_data(biome_data, width, height, num_images, save, visualize_grid, grid_index)
            
        except Exception as e:
            logger.error(f"Error processing biome file: {str(e)}")
            return [], None
    
    def process_biome_data(self, biome_data, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT,
                          num_images=DEFAULT_NUM_IMAGES, save=True, visualize_grid=True, grid_index=0):
        """
        Process biome data directly and generate images
        
        Args:
            biome_data (dict): Biome data dictionary
            width (int): Desired image width
            height (int): Desired image height
            num_images (int): Number of images to generate
            save (bool): Whether to save the generated images
            visualize_grid (bool): Whether to create a visualization of the grid
            grid_index (int): Index of the grid to use if multiple are available
            
        Returns:
            tuple: (List of generated PIL Image objects, Grid visualization Image)
        """
        try:
            # Parse the biome data if it's a string
            if isinstance(biome_data, str):
                try:
                    biome_data = json.loads(biome_data)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON string")
                    return [], None
            
            # Create a new BiomeParser instance to prevent issues with reused parsers
            self.biome_parser = BiomeParser()
            
            # Parse the biome data
            parsed_data = self.biome_parser.parse_biome_data(biome_data)
            if not parsed_data:
                logger.error("Failed to parse biome data")
                return [], None
            
            # Set the current grid
            grid_layout = self.biome_parser.set_current_grid(grid_index)
            if grid_layout is None:
                logger.error(f"Failed to set grid at index {grid_index}")
                return [], None
                
            # Generate a prompt from the biome
            prompt = self.biome_parser.get_prompt_from_biome()
            logger.info(f"Generated prompt from biome: {prompt}")
            
            # Generate the biome image
            images = self.client.generate_image(
                prompt, 
                width=width,
                height=height,
                num_images=num_images
            )
            
            if not images:
                logger.error("No images were generated")
                return [], None
                
            logger.info(f"Generated {len(images)} biome images")
            
            # Create a visualization of the grid
            grid_viz = None
            if visualize_grid:
                grid_viz = self.biome_parser.visualize_biome_grid()
                
            # Save the images if requested
            if save:
                self._save_images(images, prompt, grid_viz)
                
            return images, grid_viz
            
        except Exception as e:
            logger.error(f"Error processing biome data: {str(e)}")
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
                filename = f"biome_{prompt_slug}_{timestamp}_{i+1}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                image.save(filepath)
                logger.info(f"Saved biome image to {filepath}")
            
            # Save the grid visualization if available
            if grid_viz:
                viz_filename = f"biome_viz_{prompt_slug}_{timestamp}.png"
                viz_filepath = os.path.join(OUTPUT_DIR, viz_filename)
                grid_viz.save(viz_filepath)
                logger.info(f"Saved biome visualization to {viz_filepath}")
                
        except Exception as e:
            logger.error(f"Error saving images: {str(e)}")
