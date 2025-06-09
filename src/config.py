import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
DALLE_API_KEY = os.getenv('DALLE_API_KEY')

# Default model settings
DEFAULT_TEXT_MODEL = os.getenv('DEFAULT_TEXT_MODEL', 'openai')  # Options: 'openai', 'stability', 'local'
DEFAULT_GRID_MODEL = os.getenv('DEFAULT_GRID_MODEL', 'stability')  # Options: 'openai', 'stability', 'local'

# Image configuration
DEFAULT_IMAGE_WIDTH = int(os.getenv('DEFAULT_IMAGE_WIDTH', '512'))
DEFAULT_IMAGE_HEIGHT = int(os.getenv('DEFAULT_IMAGE_HEIGHT', '512'))
DEFAULT_NUM_IMAGES = int(os.getenv('DEFAULT_NUM_IMAGES', '1'))

# Local model paths (for future implementation)
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './models/local')

# Output directories
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
OUTPUT_3D_ASSETS_DIR = os.path.join(OUTPUT_DIR, '3d_assets')

class Config:
    API_KEY = os.getenv("API_KEY", "your_api_key_here")
    MODEL_PATH = os.getenv("MODEL_PATH", "path/to/your/model")
    IMAGE_OUTPUT_PATH = os.getenv("IMAGE_OUTPUT_PATH", "output/images")
    GRID_SIZE = int(os.getenv("GRID_SIZE", 10))  # Default grid size
    TERRAIN_TYPES = {
        0: "plain",
        1: "forest",
        2: "mountain",
        3: "water",
        4: "desert",
        # Add more terrain types as needed
    }