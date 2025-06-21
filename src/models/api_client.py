import os
import logging

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


class LocalModelClient(APIClient):
    """Client for local SDXL Turbo model inference"""
    def __init__(self, model_path=None):
        super().__init__(None)  # No API key needed
        from .local_model import LocalModel
        self.model = LocalModel(model_path)
        logger.info("Local SDXL Turbo client initialized")

    def generate_image(self, prompt, width=512, height=512, num_images=1):
        """Generate an image using the local SDXL Turbo model"""
        return self.model.generate_image(
            prompt, 
            width=width,
            height=height,
            num_images=num_images
        )


def get_client(client_type="sdxl-turbo"):
    """Factory function to get the local SDXL Turbo client"""
    # Only support local SDXL Turbo implementation
    if client_type in ["sdxl-turbo", "local"]:
        return LocalModelClient()
    else:
        logger.warning(f"Model type {client_type} not supported. Using SDXL Turbo instead.")
        return LocalModelClient()
