import os
import logging
from models.api_client import get_client
from config import DEFAULT_TEXT_MODEL, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_NUM_IMAGES, OUTPUT_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """Process text prompts and convert them to images"""
    
    def __init__(self, model_type=DEFAULT_TEXT_MODEL):
        """
        Initialize the text processor
        
        Args:
            model_type (str): The model type to use ('openai', 'stability', or 'local')
        """
        self.model_type = model_type
        try:
            self.client = get_client(model_type)
            logger.info(f"Initialized text processor with {model_type} model")
        except Exception as e:
            logger.error(f"Error initializing text processor: {str(e)}")
            raise
    
    def enhance_prompt(self, prompt):
        """
        Enhance a simple text prompt to get better image generation results
        
        Args:
            prompt (str): The basic text prompt
            
        Returns:
            str: Enhanced prompt with additional details
        """
        # Add style enhancements
        enhanced = prompt.strip()
        
        # Don't enhance already detailed prompts
        if len(enhanced.split()) > 15:
            return enhanced
            
        # Add artistic style and quality enhancers if not already present
        style_enhancers = [
            "photorealistic", "detailed", "high resolution", "beautiful",
            "professional photography", "high quality"
        ]
        
        # Check if any style enhancers are already in the prompt
        has_enhancer = any(enhancer in enhanced.lower() for enhancer in style_enhancers)
        
        if not has_enhancer:
            # Add a random style enhancer
            import random
            style = random.choice(style_enhancers)
            enhanced += f", {style}"
        
        return enhanced
    
    def convert_to_image(self, prompt, width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT, num_images=DEFAULT_NUM_IMAGES, save=True):
        """
        Convert a text prompt to an image
        
        Args:
            prompt (str): The text prompt to convert
            width (int): Desired image width
            height (int): Desired image height
            num_images (int): Number of images to generate
            save (bool): Whether to save the generated images
            
        Returns:
            list: List of PIL Image objects
        """
        try:
            # Enhance the prompt for better results
            enhanced_prompt = self.enhance_prompt(prompt)
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            
            # Generate the images
            images = self.client.generate_image(
                enhanced_prompt, 
                width=width,
                height=height,
                num_images=num_images
            )
            
            if not images:
                logger.error("No images were generated")
                return []
                
            logger.info(f"Generated {len(images)} images")
            
            # Save the images if requested
            if save:
                self._save_images(images, prompt)
                
            return images
            
        except Exception as e:
            logger.error(f"Error converting text to image: {str(e)}")
            return []
    
    def _save_images(self, images, prompt):
        """
        Save images to the output directory
        
        Args:
            images (list): List of PIL Image objects
            prompt (str): The original prompt (used for naming)
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
            prompt_slug = prompt_slug[:50]  # Truncate to reasonable length
            
            # Save each image
            for i, image in enumerate(images):
                filename = f"{prompt_slug}_{timestamp}_{i+1}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                image.save(filepath)
                logger.info(f"Saved image to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving images: {str(e)}")