"""
SDXL Turbo Image Generation Worker
Optimized for 15-20GB VRAM with memory management and model offloading
"""

import torch
import gc
import logging
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import os
from datetime import datetime
import psutil
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDXLTurboWorker:
    """
    Memory-optimized SDXL Turbo worker for image generation
    Designed to work alongside HunyuanDi-3D on 15-20GB VRAM
    """
    
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.model_loaded = False
        self.lock = threading.Lock()
        self.model_name = "stabilityai/sdxl-turbo"
        
        # Memory management settings
        self.enable_cpu_offload = True
        self.enable_attention_slicing = True
        self.enable_sequential_cpu_offload = True
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
            else:
                gpu_memory = gpu_cached = 0
            
            process = psutil.Process()
            ram_usage = process.memory_info().rss / 1024**3  # GB
            
            return {
                "gpu_allocated": gpu_memory,
                "gpu_cached": gpu_cached,
                "ram_usage": ram_usage
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {"gpu_allocated": 0, "gpu_cached": 0, "ram_usage": 0}
    
    def setup_device(self) -> str:
        """Setup and return the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU with {gpu_memory:.1f}GB VRAM")
        else:
            device = "cpu"
            logger.warning("CUDA not available, using CPU")
        
        self.device = device
        return device
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memory cleared")
    
    def load_model(self) -> bool:
        """Load the SDXL Turbo model with memory optimizations"""
        try:
            with self.lock:
                if self.model_loaded:
                    logger.info("Model already loaded")
                    return True
                
                logger.info("Loading SDXL Turbo model...")
                memory_before = self.get_memory_usage()
                logger.info(f"Memory before loading: {memory_before}")
                
                device = self.setup_device()
                
                # Load model with memory optimizations
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if device == "cuda" else None
                )
                
                if device == "cuda":
                    # Apply memory optimizations
                    if self.enable_attention_slicing:
                        self.pipeline.enable_attention_slicing()
                        logger.info("Enabled attention slicing")
                    
                    if self.enable_sequential_cpu_offload:
                        self.pipeline.enable_sequential_cpu_offload()
                        logger.info("Enabled sequential CPU offload")
                    elif self.enable_cpu_offload:
                        self.pipeline.enable_model_cpu_offload()
                        logger.info("Enabled model CPU offload")
                    else:
                        self.pipeline = self.pipeline.to(device)
                else:
                    self.pipeline = self.pipeline.to(device)
                
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.info(f"xformers not available: {e}")
                
                self.model_loaded = True
                memory_after = self.get_memory_usage()
                logger.info(f"Memory after loading: {memory_after}")
                logger.info("SDXL Turbo model loaded successfully")
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading SDXL Turbo model: {e}")
            self.model_loaded = False
            return False
    
    def unload_model(self):
        """Unload the model to free memory"""
        try:
            with self.lock:
                if self.pipeline is not None:
                    logger.info("Unloading SDXL Turbo model...")
                    del self.pipeline
                    self.pipeline = None
                    self.model_loaded = False
                    self.clear_memory()
                    logger.info("Model unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
    
    def enhance_prompt_for_3d(self, prompt: str) -> str:
        """Enhance prompt for better 3D asset generation"""
        if not prompt:
            return prompt
            
        # Add 3D-friendly keywords
        enhancements = [
            "high quality", "detailed", "professional",
            "clean background", "centered composition",
            "good lighting", "sharp focus"
        ]
        
        # Check if prompt already has these qualities
        prompt_lower = prompt.lower()
        missing_enhancements = [
            enhancement for enhancement in enhancements
            if enhancement not in prompt_lower
        ]
        
        if missing_enhancements:
            # Add missing enhancements
            enhanced = f"{prompt}, {', '.join(missing_enhancements[:3])}"
        else:
            enhanced = prompt
            
        logger.info(f"Enhanced prompt: {enhanced}")
        return enhanced
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,  # SDXL Turbo works well with 1-4 steps
        guidance_scale: float = 0.0,   # SDXL Turbo works without guidance
        seed: Optional[int] = None,
        enhance_prompt: bool = True
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """
        Generate an image using SDXL Turbo
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (optional)
            width: Image width (default 1024)
            height: Image height (default 1024)
            num_inference_steps: Number of denoising steps (1-4 for Turbo)
            guidance_scale: Guidance scale (0.0 for Turbo)
            seed: Random seed for reproducibility
            enhance_prompt: Whether to enhance prompt for 3D generation
            
        Returns:
            Tuple of (generated_image, metadata)
        """
        try:
            if not self.model_loaded:
                if not self.load_model():
                    return None, {"error": "Failed to load model"}
            
            logger.info(f"Generating image with prompt: {prompt}")
            memory_before = self.get_memory_usage()
            
            # Enhance prompt if requested
            if enhance_prompt:
                prompt = self.enhance_prompt_for_3d(prompt)
            
            # Set up generation parameters
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Default negative prompt for better quality
            if negative_prompt is None:
                negative_prompt = (
                    "blurry, low quality, distorted, deformed, "
                    "bad anatomy, low resolution, pixelated"
                )
            
            # Generate image
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="pil"
                )
            
            image = result.images[0] if hasattr(result, 'images') else result[0]
            
            memory_after = self.get_memory_usage()
            
            # Metadata
            metadata = {
                "model": self.model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Image generated successfully")
            return image, metadata
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None, {"error": str(e)}
    
    def generate_and_save(
        self,
        prompt: str,
        output_dir: str,
        filename_prefix: str = "sdxl_turbo",
        **kwargs
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate image and save to file
        
        Returns:
            Tuple of (output_path, metadata)
        """
        try:
            # Generate image
            image, metadata = self.generate_image(prompt, **kwargs)
            
            if image is None:
                return None, metadata
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            
            filename = f"{filename_prefix}_{safe_prompt}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save image
            image.save(output_path, "PNG", quality=95)
            
            metadata["output_path"] = output_path
            metadata["filename"] = filename
            
            logger.info(f"Image saved to: {output_path}")
            return output_path, metadata
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None, {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check worker health status"""
        try:
            memory_info = self.get_memory_usage()
            
            return {
                "status": "healthy" if self.model_loaded else "model_not_loaded",
                "model_loaded": self.model_loaded,
                "device": self.device,
                "memory_usage": memory_info,
                "model_name": self.model_name
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Global worker instance
_sdxl_worker = None

def get_sdxl_worker() -> SDXLTurboWorker:
    """Get or create the global SDXL worker instance"""
    global _sdxl_worker
    if _sdxl_worker is None:
        _sdxl_worker = SDXLTurboWorker()
    return _sdxl_worker

def generate_image_sdxl(
    prompt: str,
    output_dir: str,
    **kwargs
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Convenience function for image generation
    """
    worker = get_sdxl_worker()
    return worker.generate_and_save(prompt, output_dir, **kwargs)

# Test function
if __name__ == "__main__":
    # Test the worker
    worker = SDXLTurboWorker()
    
    print("Testing SDXL Turbo worker...")
    print(f"Health check: {worker.health_check()}")
    
    if worker.load_model():
        image, metadata = worker.generate_image(
            "a beautiful landscape with mountains and lakes",
            num_inference_steps=2
        )
        
        if image:
            print("Image generated successfully!")
            print(f"Metadata: {metadata}")
        else:
            print("Failed to generate image")
    else:
        print("Failed to load model")
