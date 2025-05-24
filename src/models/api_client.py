import os
import requests
import json
import io
import base64
from PIL import Image
import sys
import logging
from src.config import OPENAI_API_KEY, STABILITY_API_KEY, DALLE_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClient:
    """Base class for API clients"""
    def __init__(self, api_key=None):
        self.api_key = api_key

    def generate_image(self, prompt, width=512, height=512, num_images=1):
        """Generate an image based on the prompt"""
        raise NotImplementedError("Subclasses must implement generate_image")


class OpenAIClient(APIClient):
    """Client for OpenAI's DALL-E API"""
    def __init__(self, api_key=None, model=None):
        super().__init__(api_key or OPENAI_API_KEY)
        self.model = model or os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
        if not self.api_key:
            logger.error("OpenAI API key is not set")
            raise ValueError("OpenAI API key is required")

    def generate_image(self, prompt, width=1024, height=1024, num_images=1):
        try:
            size = self._get_compatible_size(width, height)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "n": 1,
                "size": size,
            }
            # Only add these for DALL-E 3
            if self.model == "dall-e-3":
                payload["quality"] = "standard"
                payload["response_format"] = "url"
            logger.info(f"Sending request to OpenAI with payload: {json.dumps(payload)}")
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload
            )
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
            data = response.json()
            images = []
            for item in data.get("data", []):
                image_url = item.get("url")
                if image_url:
                    image_response = requests.get(image_url)
                    image = Image.open(io.BytesIO(image_response.content))
                    images.append(image)
            return images if images else None
        except Exception as e:
            logger.error(f"Error generating image with OpenAI: {str(e)}")
            return None
    
    def _get_compatible_size(self, width, height):
        """
        Convert requested dimensions to a size string compatible with DALL-E 3
        DALL-E 3 only supports: 1024x1024, 1024x1792, or 1792x1024
        """
        # Default to square if dimensions are unusual
        if width == height:
            return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"


class StabilityClient(APIClient):
    """Client for Stability AI's API"""
    def __init__(self, api_key=None):
        super().__init__(api_key or STABILITY_API_KEY)
        if not self.api_key:
            logger.error("Stability API key is not set")
            raise ValueError("Stability API key is required")

    def generate_image(self, prompt, width=512, height=512, num_images=1):
        """Generate an image using Stability AI's model"""
        try:
            engine_id = "stable-diffusion-xl-1024-v1-0"
            api_host = os.getenv('API_HOST', 'https://api.stability.ai')
            
            response = requests.post(
                f"{api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 7,
                    "height": height,
                    "width": width,
                    "samples": min(num_images, 10),  # Maximum 10 images per request
                    "steps": 30,
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Stability API error: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            images = []
            
            for i, image_data in enumerate(data["artifacts"]):
                image_bytes = base64.b64decode(image_data["base64"])
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
                
            return images
            
        except Exception as e:
            logger.error(f"Error generating image with Stability: {str(e)}")
            return None


class LocalModelClient(APIClient):
    """Client for local model inference using Hugging Face models"""
    def __init__(self, model_path=None):
        super().__init__(None)  # No API key needed
        from .local_model import LocalModel
        self.model = LocalModel(model_path)
        logger.info("Local model client initialized")

    def generate_image(self, prompt, width=512, height=512, num_images=1):
        """Generate an image using the local model"""
        return self.model.generate_image(
            prompt, 
            width=width,
            height=height,
            num_images=num_images
        )


def get_client(client_type="openai"):
    """Factory function to get the appropriate client"""
    clients = {
        "openai": OpenAIClient,
        "stability": StabilityClient,
        "local": LocalModelClient
    }
    
    if client_type not in clients:
        logger.error(f"Unknown client type: {client_type}")
        raise ValueError(f"Unknown client type: {client_type}")
    
    try:
        if client_type == "openai":
            # Allow model override via env var
            model = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
            return OpenAIClient(model=model)
        return clients[client_type]()
    except Exception as e:
        logger.error(f"Error creating client of type {client_type}: {str(e)}")
        raise