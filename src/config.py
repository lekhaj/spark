# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- LLM API Configuration ---
# OpenRouter API Key (Leave as empty string; Canvas will inject if available)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# OpenRouter Base URL
OPENROUTER_BASE_URL = "https://openrouter.ai"

# --- Image Generation Model Defaults ---
DEFAULT_TEXT_MODEL = "stability"  # Default for text-to-image generation (e.g., stability, openai, local)
DEFAULT_GRID_MODEL = "stability"  # Default for grid-to-image generation (e.g., stability, openai, local)

# --- Image Dimensions ---
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_NUM_IMAGES = 1

# --- Output Directories ---
# Directory for saving generated images and 3D assets
OUTPUT_DIR = "generated_assets"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_3D_ASSETS_DIR = os.path.join(OUTPUT_DIR, "3d_assets")

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_3D_ASSETS_DIR, exist_ok=True)

# --- MongoDB Configuration ---
MONGO_URI="mongodb://sagar:KrSiDnSI9m8RgcHE@ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin"
MONGO_DB_NAME="World_builder"
MONGO_BIOME_COLLECTION="biomes" # This will be the main collection for biome data

# --- Grid/Structure Definitions ---
GRID_DIMENSIONS = (10, 10) # 10x10 grid

# Define categories of structures for biome generation
STRUCTURE_TYPES = {
    "Village Structures": [
        "general_housing", # e.g., small huts, houses
        "healing_hut",     # specialized for healing/medicine
        "story_hut",       # for communal gatherings, storytelling
        "herb_pavilion",   # for processing and storing herbs
        "totem_pole",      # spiritual/defensive landmark
        "other_feature",    # for general features not explicitly listed, e.g., "market_stall"
    ],
    "Natural Features": [
        "water",
        "plain",
        "stone_path",
        "mooncap_mushroom_patch", # Example of a very specific feature
        "dense_forest_area", # Placeholder for areas described as dense forest
        "rocky_outcrop", # Placeholder for rocky areas
    ],
    # Add more categories as needed
}

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

