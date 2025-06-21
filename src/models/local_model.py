import os
import sys
import torch
import logging
from PIL import Image
import numpy as np
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModel:
    """Local image generation model using SDXL Turbo"""
    
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.pipe = None
        self.model_path = model_path
        self.sdxl_worker = None
        
        # Try to import SDXL worker
        try:
            # Add current directory to path for imports
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from sdxl_turbo_worker import SDXLTurboWorker
            self.sdxl_worker = SDXLTurboWorker()
            logger.info("SDXL Turbo worker initialized")
        except ImportError as e:
            logger.warning(f"SDXL Turbo worker not available: {e}")
            logger.info("Falling back to Stable Diffusion v1.5")
        
    def load_model(self):
        """Load the diffusion model"""
        try:
            # Try SDXL Turbo first (if available)
            if self.sdxl_worker:
                logger.info("Loading SDXL Turbo model...")
                success = self.sdxl_worker.load_model()
                if success:
                    logger.info("SDXL Turbo model loaded successfully")
                    return True
                else:
                    logger.warning("Failed to load SDXL Turbo, falling back to SD v1.5")
            
            # Fallback to Stable Diffusion v1.5
            from diffusers import StableDiffusionPipeline
            
            model_id = "runwayml/stable-diffusion-v1-5"
            logger.info(f"Loading fallback model {model_id}")
            
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
            
            logger.info("Fallback model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def generate_image(self, input_data, width=512, height=512, num_images=1):
        """
        Generate images using SDXL Turbo or fallback SD v1.5
        
        Args:
            input_data (str): Text prompt or grid description
            width (int): Image width
            height (int): Image height
            num_images (int): Number of images to generate
            
        Returns:
            list: List of PIL Image objects
        """
        try:
            # Try SDXL Turbo first
            if self.sdxl_worker and self.sdxl_worker.model_loaded:
                logger.info(f"Generating {num_images} image(s) with SDXL Turbo: {input_data}")
                
                images = []
                for i in range(num_images):
                    image, metadata = self.sdxl_worker.generate_image(
                        prompt=input_data,
                        width=width,
                        height=height,
                        num_inference_steps=2,  # Fast generation
                        enhance_prompt=True
                    )
                    
                    if image:
                        images.append(image)
                    else:
                        logger.warning(f"Failed to generate image {i+1}/{num_images}: {metadata}")
                
                if images:
                    logger.info(f"Generated {len(images)} images with SDXL Turbo")
                    return images
                else:
                    logger.warning("SDXL Turbo failed, trying fallback model")
            
            # Fallback to SD v1.5 if SDXL not available or failed
            if self.pipe is None and not self.load_model():
                raise Exception("Failed to load any model")
            
            if self.pipe is None:
                raise Exception("No model available for generation")
            
            # Ensure dimensions are multiples of 8 as required by most models
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            logger.info(f"Generating {num_images} image(s) with SD v1.5: {input_data}")
            
            # Run inference with fallback model
            result = self.pipe(
                input_data,
                num_images_per_prompt=num_images,
                width=width,
                height=height,
                guidance_scale=7.5,
                num_inference_steps=20
            )
            
            # Convert to PIL images if not already
            images = []
            if hasattr(result, 'images'):
                for image in result.images:
                    if isinstance(image, Image.Image):
                        images.append(image)
                    else:
                        # Convert numpy array to PIL Image
                        pil_image = Image.fromarray((image * 255).astype(np.uint8))
                        images.append(pil_image)
            else:
                # Handle case where result is directly a list of images
                for image in result:
                    if isinstance(image, Image.Image):
                        images.append(image)
                    else:
                        pil_image = Image.fromarray((image * 255).astype(np.uint8))
                        images.append(pil_image)
                    
            logger.info(f"Generated {len(images)} images successfully")
            return images
            
        except Exception as e:
            logger.error(f"Error generating images with local model: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            # Return a basic colored image to indicate error
            images = []
            for _ in range(num_images):
                # Create a simple gradient image as a fallback
                img = Image.new('RGB', (width, height), color='white')
                images.append(img)
            return images