import os
import torch
import logging
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModel:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.pipe = None
        self.model_path = model_path
        
    def load_model(self):
        """Load the diffusion model"""
        try:
            from diffusers import StableDiffusionPipeline
            
            # Use a compact model from Hugging Face
            model_id = "runwayml/stable-diffusion-v1-5"  # Popular free model
            
            logger.info(f"Loading model {model_id}")
            
            # Load the pipeline with low precision to save memory
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Optimize for memory if on CUDA
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def generate_image(self, input_data, width=512, height=512, num_images=1):
        """
        Generate images using the local diffusion model
        
        Args:
            input_data (str): Text prompt or grid description
            width (int): Image width
            height (int): Image height
            num_images (int): Number of images to generate
            
        Returns:
            list: List of PIL Image objects
        """
        try:
            # Load model if not already loaded
            if self.pipe is None:
                success = self.load_model()
                if not success:
                    raise Exception("Failed to load the model")
            
            # Ensure dimensions are multiples of 8 as required by most models
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            logger.info(f"Generating {num_images} image(s) with prompt: {input_data}")
            
            # Run inference
            result = self.pipe(
                input_data,
                num_images_per_prompt=num_images,
                width=width,
                height=height
            )
            
            # Convert to PIL images if not already
            images = []
            for i, image in enumerate(result.images):
                if not isinstance(image, Image.Image):
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
                    images.append(pil_image)
                else:
                    images.append(image)
                    
            logger.info(f"Generated {len(images)} images successfully")
            return images
            
        except Exception as e:
            logger.error(f"Error generating images with local model: {str(e)}")
            # Return a basic colored image to indicate error
            images = []
            for _ in range(num_images):
                # Create a simple gradient image as a fallback
                img = Image.new('RGB', (width, height), color='white')
                images.append(img)
            return images