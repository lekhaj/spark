"""
Configuration settings for the text-to-image pipeline
Enhanced with distributed GPU processing capabilities
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
OUTPUT_DIR = BASE_DIR / "output"

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
# --- Path Configuration ---
# Get the directory of the current file (config.py) -> C:\Users\Sagar H V\OneDrive\Desktop\spark\src
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the root of your 'spark' project
# This goes up one level from 'src' to 'spark'
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, os.pardir))

# Define where generated images will be stored on the file system
# This will be C:\Users\Sagar H V\OneDrive\Desktop\spark\src\generated_assets\images
OUTPUT_IMAGES_DIR = os.path.join(CURRENT_FILE_DIR, 'generated_assets', 'images')

# Define general output directory for other assets (e.g., 3D models)
# This will be C:\Users\Sagar H V\OneDrive\Desktop\spark\outputs
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
OUTPUT_3D_ASSETS_DIR = os.path.join(OUTPUT_DIR, '3d_assets')

# Ensure all necessary directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(CURRENT_FILE_DIR, 'generated_assets'), exist_ok=True) # Ensure 'generated_assets' parent exists
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True) # Ensure the 'images' directory exists
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

# --- Celery and Redis Configuration ---
REDIS_BROKER_URL = os.getenv('REDIS_BROKER_URL')
REDIS_RESULT_BACKEND = os.getenv('REDIS_RESULT_BACKEND')


# Celery Task Routing
CELERY_TASK_ROUTES = {
    'run_biome_generation': {'queue': '2d_queue'},
    'batch_process_mongodb_prompts_task': {'queue': '2d_queue'},
    'generate_text_image': {'queue': '2d_queue'},
    'generate_grid_image': {'queue': '2d_queue'},
    'generate_3d_model_from_image': {'queue': '3d_queue'},
    'generate_3d_model_from_prompt': {'queue': '3d_queue'},
    'manage_gpu_instance': {'queue': 'infrastructure_queue'},
}

# --- Hunyuan3D Configuration ---
HUNYUAN3D_MODEL_PATH = os.getenv('HUNYUAN3D_MODEL_PATH', '/tmp/hunyuan3d_models')
HUNYUAN3D_CACHE_DIR = os.getenv('HUNYUAN3D_CACHE_DIR', '/tmp/hunyuan3d_cache')
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
GPU_INSTANCE_ID = os.getenv('GPU_INSTANCE_ID', '')
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

# --- Pipeline Settings ---
USE_CELERY = os.getenv('USE_CELERY', 'True').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# ========================================================================
# NEW: DISTRIBUTED ARCHITECTURE CONFIGURATION
# ========================================================================

# Distributed 3D Processing Configuration
USE_DISTRIBUTED_3D = os.getenv('USE_DISTRIBUTED_3D', 'false').lower() == 'true'
GPU_WORKER_MODE = os.getenv('GPU_WORKER_MODE', 'false').lower() == 'true'

# CPU Instance Configuration (where web interface runs)
CPU_INSTANCE_IP = os.getenv('CPU_INSTANCE_IP', '172.31.6.174')
CPU_INSTANCE_ID = os.getenv('CPU_INSTANCE_ID', '')

# GPU Instance Configuration
GPU_INSTANCE_ENDPOINT = os.getenv('GPU_INSTANCE_ENDPOINT', f'http://localhost:8081')
GPU_INSTANCE_TYPE = os.getenv('GPU_INSTANCE_TYPE', 'g4dn.xlarge')
GPU_SPOT_PRICE = os.getenv('GPU_SPOT_PRICE', '0.50')
USE_SPOT_INSTANCES = os.getenv('USE_SPOT_INSTANCES', 'true').lower() == 'true'

# Auto-scaling Configuration
AUTO_SCALE_GPU = os.getenv('AUTO_SCALE_GPU', 'true').lower() == 'true'
GPU_IDLE_TIMEOUT = int(os.getenv('GPU_IDLE_TIMEOUT', '300'))  # 5 minutes
GPU_STARTUP_TIMEOUT = int(os.getenv('GPU_STARTUP_TIMEOUT', '600'))  # 10 minutes
MAX_GPU_INSTANCES = int(os.getenv('MAX_GPU_INSTANCES', '2'))

# Celery Configuration for Distributed Setup
if USE_DISTRIBUTED_3D:
    # Override Redis URLs to point to CPU instance if we're in GPU worker mode
    if GPU_WORKER_MODE:
        REDIS_BROKER_URL = f'redis://{CPU_INSTANCE_IP}:6379/0'
        REDIS_RESULT_BACKEND = f'redis://{CPU_INSTANCE_IP}:6379/0'
    
    # Enhanced task routing for distributed setup
    CELERY_TASK_ROUTES.update({
        # 3D tasks go to GPU queue
        'generate_3d_model_from_image': {'queue': '3d_queue'},
        'generate_3d_model_from_prompt': {'queue': '3d_queue'},
        'generate_3d_model_with_autoscale': {'queue': '3d_queue'},
        
        # Infrastructure tasks
        'auto_scale_gpu': {'queue': 'infrastructure_queue'},
        'start_gpu_instance': {'queue': 'infrastructure_queue'},
        'stop_gpu_instance': {'queue': 'infrastructure_queue'},
        'check_gpu_health': {'queue': 'infrastructure_queue'},
        
        # 2D tasks stay on CPU
        'generate_text_image': {'queue': '2d_queue'},
        'generate_grid_image': {'queue': '2d_queue'},
        'process_image_grid': {'queue': '2d_queue'},
    })

# S3 Configuration for Asset Sharing
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'text-to-image-pipeline-assets')
S3_REGION = os.getenv('S3_REGION', AWS_REGION)
S3_PREFIX_3D_ASSETS = os.getenv('S3_PREFIX_3D_ASSETS', '3d-assets/')
S3_PREFIX_IMAGES = os.getenv('S3_PREFIX_IMAGES', 'images/')
USE_S3_FOR_ASSETS = os.getenv('USE_S3_FOR_ASSETS', 'true').lower() == 'true'

# Security Configuration
ALLOW_INSTANCE_METADATA_ACCESS = os.getenv('ALLOW_INSTANCE_METADATA_ACCESS', 'true').lower() == 'true'
REQUIRE_API_KEY = os.getenv('REQUIRE_API_KEY', 'false').lower() == 'true'
API_KEY = os.getenv('API_KEY', '')

# Enhanced AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
EC2_KEY_PAIR_NAME = os.getenv('EC2_KEY_PAIR_NAME', 'your-key-pair')
GPU_SECURITY_GROUP_ID = os.getenv('GPU_SECURITY_GROUP_ID', 'sg-your-gpu-sg')
GPU_IAM_ROLE_NAME = os.getenv('GPU_IAM_ROLE_NAME', 'your-gpu-role')
DEEP_LEARNING_AMI_ID = os.getenv('DEEP_LEARNING_AMI_ID', 'ami-0c02fb55956c7d316')

# Monitoring and Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENABLE_TASK_MONITORING = os.getenv('ENABLE_TASK_MONITORING', 'true').lower() == 'true'
HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))  # seconds

# Cost Management
DAILY_COST_LIMIT = float(os.getenv('DAILY_COST_LIMIT', '50.0'))  # USD
COST_ALERT_THRESHOLD = float(os.getenv('COST_ALERT_THRESHOLD', '40.0'))  # USD
ENABLE_COST_ALERTS = os.getenv('ENABLE_COST_ALERTS', 'true').lower() == 'true'

# Performance Configuration
MAX_CONCURRENT_3D_TASKS = int(os.getenv('MAX_CONCURRENT_3D_TASKS', '2'))
TASK_TIMEOUT_3D = int(os.getenv('TASK_TIMEOUT_3D', '1800'))  # 30 minutes
TASK_TIMEOUT_2D = int(os.getenv('TASK_TIMEOUT_2D', '300'))   # 5 minutes

# File Transfer Configuration
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
SUPPORTED_3D_FORMATS = ['glb', 'obj', 'ply', 'stl']

# Development and Debug Settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
MOCK_GPU_PROCESSING = os.getenv('MOCK_GPU_PROCESSING', 'false').lower() == 'true'
SIMULATE_GPU_DELAY = int(os.getenv('SIMULATE_GPU_DELAY', '10'))  # seconds

# ========================================================================
# CONFIGURATION VALIDATION AND SETUP
# ========================================================================

def validate_distributed_config():
    """Validate distributed configuration settings."""
    errors = []
    
    if USE_DISTRIBUTED_3D:
        if not CPU_INSTANCE_IP or CPU_INSTANCE_IP == 'localhost':
            errors.append("CPU_INSTANCE_IP must be set for distributed processing")
        
        if USE_S3_FOR_ASSETS and not S3_BUCKET_NAME:
            errors.append("S3_BUCKET_NAME must be set when using S3 for assets")
        
        if not EC2_KEY_PAIR_NAME or EC2_KEY_PAIR_NAME == 'your-key-pair':
            errors.append("EC2_KEY_PAIR_NAME must be configured")
        
        if not GPU_SECURITY_GROUP_ID or GPU_SECURITY_GROUP_ID.startswith('sg-your'):
            errors.append("GPU_SECURITY_GROUP_ID must be configured")
    
    return errors

def setup_directories():
    """Create necessary directories."""
    directories = [
        OUTPUT_IMAGES_DIR,
        OUTPUT_3D_ASSETS_DIR,
        HUNYUAN3D_CACHE_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_redis_config():
    """Get Redis configuration based on current mode."""
    if GPU_WORKER_MODE:
        return {
            'broker_url': f'redis://{CPU_INSTANCE_IP}:6379/0',
            'result_backend': f'redis://{CPU_INSTANCE_IP}:6379/0'
        }
    else:
        return {
            'broker_url': REDIS_BROKER_URL,
            'result_backend': REDIS_RESULT_BACKEND
        }

def get_processing_mode():
    """Get current processing mode description."""
    if not USE_CELERY:
        return "Local processing (no Celery)"
    elif GPU_WORKER_MODE:
        return f"GPU Worker mode (connecting to {CPU_INSTANCE_IP})"
    elif USE_DISTRIBUTED_3D:
        return "CPU Controller mode (distributed 3D processing)"
    else:
        return "Local Celery mode"

# ========================================================================
# INITIALIZATION
# ========================================================================

# Create output directories
setup_directories()

# Validate configuration if using distributed mode
if USE_DISTRIBUTED_3D:
    config_errors = validate_distributed_config()
    if config_errors and ENVIRONMENT == 'production':
        raise ValueError(f"Configuration errors: {', '.join(config_errors)}")

# Log configuration status
import logging
logger = logging.getLogger(__name__)

logger.info(f"Configuration loaded:")
logger.info(f"  - Environment: {ENVIRONMENT}")
logger.info(f"  - Processing mode: {get_processing_mode()}")
logger.info(f"  - Use Celery: {USE_CELERY}")
logger.info(f"  - Distributed 3D: {USE_DISTRIBUTED_3D}")
logger.info(f"  - GPU Worker Mode: {GPU_WORKER_MODE}")
logger.info(f"  - CPU Instance IP: {CPU_INSTANCE_IP}")
if USE_DISTRIBUTED_3D:
    logger.info(f"  - Auto-scale GPU: {AUTO_SCALE_GPU}")
    logger.info(f"  - Use S3 for assets: {USE_S3_FOR_ASSETS}")
    logger.info(f"  - Max GPU instances: {MAX_GPU_INSTANCES}")

# Export key configurations for easy access
__all__ = [
    # Existing exports
    'REDIS_BROKER_URL', 'REDIS_RESULT_BACKEND', 'USE_CELERY',
    'MONGO_URI', 'MONGO_DB_NAME', 'MONGO_BIOME_COLLECTION',
    'OUTPUT_IMAGES_DIR', 'OUTPUT_3D_ASSETS_DIR',
    'DEFAULT_TEXT_MODEL', 'DEFAULT_GRID_MODEL',
    'AWS_REGION', 'ENVIRONMENT',
    
    # New distributed processing exports
    'USE_DISTRIBUTED_3D', 'GPU_WORKER_MODE', 'CPU_INSTANCE_IP',
    'GPU_INSTANCE_TYPE', 'AUTO_SCALE_GPU', 'USE_S3_FOR_ASSETS',
    'S3_BUCKET_NAME', 'CELERY_TASK_ROUTES',
    'validate_distributed_config', 'get_redis_config', 'get_processing_mode'
]
