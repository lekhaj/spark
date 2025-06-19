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
OUTPUT_DIR = "./generated_assets"
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

# --- API Configuration ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
DALLE_API_KEY = os.getenv('DALLE_API_KEY')

# --- Local Model Configuration ---
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './models/local')

# --- Celery and Redis Configuration for Distributed EC2 Setup ---
# Redis Configuration for distributed tasks across EC2 instances
# Point to your GPU spot instance as the Redis server
REDIS_BROKER_URL = os.getenv('REDIS_BROKER_URL', 'redis://13.203.200.155:6379/0')
REDIS_RESULT_BACKEND = os.getenv('REDIS_RESULT_BACKEND', 'redis://13.203.200.155:6379/0')
USE_CELERY = os.getenv('USE_CELERY', 'True').lower() == 'true'

# Enhanced task routing for CPU vs GPU workloads across instances
CELERY_TASK_ROUTES = {
    # CPU tasks (stay on CPU instance)
    'generate_text_image': {'queue': 'cpu_tasks'},
    'generate_grid_image': {'queue': 'cpu_tasks'},
    'run_biome_generation': {'queue': 'cpu_tasks'},
    'batch_process_mongodb_prompts_task': {'queue': 'cpu_tasks'},
    
    # GPU tasks (route to GPU spot instance at 13.203.200.155)
    'generate_3d_model_from_image': {'queue': 'gpu_tasks'},
    'generate_3d_model_from_prompt': {'queue': 'gpu_tasks'},
    
    # Infrastructure tasks (can run on either instance)
    'manage_gpu_instance': {'queue': 'infrastructure'},
}

# Worker configuration for different instance types
CPU_WORKER_QUEUES = ['cpu_tasks', 'infrastructure']
GPU_WORKER_QUEUES = ['gpu_tasks']

# Spot instance handling configuration
AWS_GPU_IS_SPOT_INSTANCE = os.getenv('AWS_GPU_IS_SPOT_INSTANCE', 'True').lower() == 'true'
SPOT_INSTANCE_HANDLING_ENABLED = os.getenv('SPOT_INSTANCE_HANDLING_ENABLED', 'True').lower() == 'true'

# Celery retry policies for spot instance tasks
CELERY_SPOT_INSTANCE_RETRY_POLICY = {
    'max_retries': 3,
    'interval_start': 30,  # Wait for spot instance startup
    'interval_step': 60,
    'interval_max': 300,
}

# --- Hunyuan3D Configuration ---
HUNYUAN3D_MODEL_PATH = os.getenv('HUNYUAN3D_MODEL_PATH', 'tencent/Hunyuan3D-2')
HUNYUAN3D_SUBFOLDER = os.getenv('HUNYUAN3D_SUBFOLDER', 'hunyuan3d-dit-v2-mini-turbo')
HUNYUAN3D_TEXGEN_MODEL_PATH = os.getenv('HUNYUAN3D_TEXGEN_MODEL_PATH', 'tencent/Hunyuan3D-2')

# Hunyuan3D Processing Parameters
HUNYUAN3D_STEPS = int(os.getenv('HUNYUAN3D_STEPS', '30'))
HUNYUAN3D_GUIDANCE_SCALE = float(os.getenv('HUNYUAN3D_GUIDANCE_SCALE', '7.5'))
HUNYUAN3D_OCTREE_RESOLUTION = int(os.getenv('HUNYUAN3D_OCTREE_RESOLUTION', '256'))
HUNYUAN3D_NUM_CHUNKS = int(os.getenv('HUNYUAN3D_NUM_CHUNKS', '200000'))

# Hunyuan3D Feature Flags
HUNYUAN3D_REMOVE_BACKGROUND = os.getenv('HUNYUAN3D_REMOVE_BACKGROUND', 'True').lower() == 'true'
HUNYUAN3D_ENABLE_FLASHVDM = os.getenv('HUNYUAN3D_ENABLE_FLASHVDM', 'True').lower() == 'true'
HUNYUAN3D_COMPILE = os.getenv('HUNYUAN3D_COMPILE', 'False').lower() == 'true'
HUNYUAN3D_LOW_VRAM_MODE = os.getenv('HUNYUAN3D_LOW_VRAM_MODE', 'True').lower() == 'true'

# Hunyuan3D Hardware Configuration
HUNYUAN3D_DEVICE = os.getenv('HUNYUAN3D_DEVICE', 'cuda')
HUNYUAN3D_MAX_MEMORY_GB = int(os.getenv('HUNYUAN3D_MAX_MEMORY_GB', '16'))

# --- AWS EC2 Configuration ---
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_GPU_INSTANCE_ID = os.getenv('AWS_GPU_INSTANCE_ID', '')
AWS_GPU_INSTANCE_TYPE = os.getenv('AWS_GPU_INSTANCE_TYPE', 'g4dn.xlarge')

# EC2 Management Settings
AWS_MAX_STARTUP_WAIT_TIME = int(os.getenv('AWS_MAX_STARTUP_WAIT_TIME', '300'))  # 5 minutes
AWS_EC2_CHECK_INTERVAL = int(os.getenv('AWS_EC2_CHECK_INTERVAL', '10'))  # 10 seconds
AWS_AUTO_SHUTDOWN_DELAY = int(os.getenv('AWS_AUTO_SHUTDOWN_DELAY', '300'))  # 5 minutes after last task

# AWS Credentials (optional - can use IAM roles instead)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')

# --- 3D Generation Configuration ---
# Output formats supported
SUPPORTED_3D_FORMATS = ['glb', 'obj', 'ply', 'stl']
DEFAULT_3D_FORMAT = os.getenv('DEFAULT_3D_FORMAT', 'glb')

# Task timeout settings (in seconds)
TASK_TIMEOUT_3D_GENERATION = int(os.getenv('TASK_TIMEOUT_3D_GENERATION', '1800'))  # 30 minutes
TASK_TIMEOUT_2D_GENERATION = int(os.getenv('TASK_TIMEOUT_2D_GENERATION', '300'))   # 5 minutes
TASK_TIMEOUT_EC2_MANAGEMENT = int(os.getenv('TASK_TIMEOUT_EC2_MANAGEMENT', '600')) # 10 minutes

# --- Distributed Worker Configuration ---
# Worker type identification for different EC2 instances
WORKER_TYPE = os.getenv('WORKER_TYPE', 'cpu')  # 'cpu' or 'gpu'

# GPU Spot Instance specific configuration
GPU_SPOT_INSTANCE_IP = os.getenv('GPU_SPOT_INSTANCE_IP', '13.203.200.155')
GPU_INSTANCE_REDIS_PORT = int(os.getenv('GPU_INSTANCE_REDIS_PORT', '6379'))

# Task monitoring and health checks
TASK_HEALTH_CHECK_INTERVAL = int(os.getenv('TASK_HEALTH_CHECK_INTERVAL', '60'))  # 1 minute
SPOT_INSTANCE_CHECK_INTERVAL = int(os.getenv('SPOT_INSTANCE_CHECK_INTERVAL', '300'))  # 5 minutes

# Celery worker configuration based on instance type
if WORKER_TYPE == 'gpu':
    # GPU instance Redis configuration (local)
    REDIS_BROKER_URL = os.getenv('REDIS_BROKER_URL', 'redis://127.0.0.1:6379/0')
    REDIS_RESULT_BACKEND = os.getenv('REDIS_RESULT_BACKEND', 'redis://127.0.0.1:6379/0')
    WORKER_QUEUES = GPU_WORKER_QUEUES
else:
    # CPU instance Redis configuration (points to GPU instance)
    REDIS_BROKER_URL = os.getenv('REDIS_BROKER_URL', f'redis://{GPU_SPOT_INSTANCE_IP}:{GPU_INSTANCE_REDIS_PORT}/0')
    REDIS_RESULT_BACKEND = os.getenv('REDIS_RESULT_BACKEND', f'redis://{GPU_SPOT_INSTANCE_IP}:{GPU_INSTANCE_REDIS_PORT}/0')
    WORKER_QUEUES = CPU_WORKER_QUEUES
