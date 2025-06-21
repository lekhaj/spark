# src/config.py
# Updated for Hunyuan3D-2.1 compatibility
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

# --- Worker Type Detection ---
WORKER_TYPE = os.getenv('WORKER_TYPE', 'cpu')  # 'gpu' or 'cpu'
GPU_SPOT_INSTANCE_IP = os.getenv('GPU_SPOT_INSTANCE_IP', '3.109.55.217')

# --- Enhanced Redis Configuration for Read/Write Separation ---
class RedisConfig:
    """Redis configuration manager with separate read/write connections."""
    
    def __init__(self):
        self.worker_type = WORKER_TYPE
        self.gpu_ip = GPU_SPOT_INSTANCE_IP
          # Configure URLs based on worker type
        if self.worker_type == 'gpu':
            # GPU instance: Use local Redis for both read and write
            self.write_url = 'redis://127.0.0.1:6379/0'
            self.read_url = 'redis://127.0.0.1:6379/0'
        else:
            # CPU instance: Connect to GPU instance Redis for both read and write
            self.write_url = f'redis://{self.gpu_ip}:6379/0'
            self.read_url = f'redis://{self.gpu_ip}:6379/0'
            
        # Allow environment override
        self.write_url = os.getenv('REDIS_WRITE_URL', self.write_url)
        self.read_url = os.getenv('REDIS_READ_URL', self.read_url)
        
    @property
    def write_client(self):
        """Get Redis client for write operations."""
        import redis
        return redis.Redis.from_url(self.write_url, socket_timeout=5, socket_connect_timeout=5)
    
    @property  
    def read_client(self):
        """Get Redis client for read operations."""
        import redis
        return redis.Redis.from_url(self.read_url, socket_timeout=5, socket_connect_timeout=5)
    
    def test_connection(self, operation='both'):
        """Test Redis connections.
        
        Args:
            operation (str): 'read', 'write', or 'both'
            
        Returns:
            dict: Connection test results
        """
        import time
        results = {}
        
        if operation in ['write', 'both']:
            try:
                client = self.write_client
                client.ping()
                # Test write capability
                test_key = f"connection_test_{int(time.time())}"
                client.set(test_key, "test_write", ex=10)
                client.delete(test_key)
                results['write'] = {'success': True, 'url': self.write_url}
            except Exception as e:
                results['write'] = {'success': False, 'url': self.write_url, 'error': str(e)}
        
        if operation in ['read', 'both']:
            try:
                client = self.read_client
                client.ping()
                results['read'] = {'success': True, 'url': self.read_url}
            except Exception as e:
                results['read'] = {'success': False, 'url': self.read_url, 'error': str(e)}
        
        return results

# Initialize Redis configuration
REDIS_CONFIG = RedisConfig()

# Celery and Redis Configuration for backward compatibility
REDIS_BROKER_URL = os.getenv('REDIS_BROKER_URL', REDIS_CONFIG.write_url)
REDIS_RESULT_BACKEND = os.getenv('REDIS_RESULT_BACKEND', REDIS_CONFIG.write_url)
USE_CELERY = os.getenv('USE_CELERY', 'True').lower() == 'true'

# Enhanced task routing for CPU vs GPU workloads across instances
CELERY_TASK_ROUTES = {
    # CPU tasks (stay on CPU instance)
    'generate_text_image': {'queue': 'cpu_tasks'},
    'generate_grid_image': {'queue': 'cpu_tasks'},
    'run_biome_generation': {'queue': 'cpu_tasks'},
    'batch_process_mongodb_prompts_task': {'queue': 'cpu_tasks'},
      # GPU tasks (route to GPU spot instance) - with S3 integration
    'generate_3d_model_from_image': {'queue': 'gpu_tasks'},
    'process_image_for_3d_generation': {'queue': 'gpu_tasks'},
    'batch_process_s3_images_for_3d': {'queue': 'gpu_tasks'},
    
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
# Hunyuan3D Model Configuration - Updated for 2.1
# Use local cached model path since we have .ckpt file instead of .safetensors
HUNYUAN3D_MODEL_PATH = os.getenv('HUNYUAN3D_MODEL_PATH', os.path.expanduser('~/.cache/hy3dgen/tencent/Hunyuan3D-2.1'))
HUNYUAN3D_SUBFOLDER = os.getenv('HUNYUAN3D_SUBFOLDER', 'hunyuan3d-dit-v2-1')
HUNYUAN3D_TEXGEN_MODEL_PATH = os.getenv('HUNYUAN3D_TEXGEN_MODEL_PATH', 'tencent/Hunyuan3D-2.1')

# New 2.1 specific configurations
HUNYUAN3D_PAINT_CONFIG_MAX_VIEWS = int(os.getenv('HUNYUAN3D_PAINT_CONFIG_MAX_VIEWS', '6'))
HUNYUAN3D_PAINT_CONFIG_RESOLUTION = int(os.getenv('HUNYUAN3D_PAINT_CONFIG_RESOLUTION', '512'))

# 2.1 PBR Texture Generation Paths
HUNYUAN3D_REALESRGAN_CKPT_PATH = os.getenv('HUNYUAN3D_REALESRGAN_CKPT_PATH', 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth')
HUNYUAN3D_MULTIVIEW_CFG_PATH = os.getenv('HUNYUAN3D_MULTIVIEW_CFG_PATH', 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
HUNYUAN3D_CUSTOM_PIPELINE_PATH = os.getenv('HUNYUAN3D_CUSTOM_PIPELINE_PATH', 'hy3dpaint/hunyuanpaintpbr')

# 2.1 Torchvision compatibility
HUNYUAN3D_APPLY_TORCHVISION_FIX = os.getenv('HUNYUAN3D_APPLY_TORCHVISION_FIX', 'True').lower() == 'true'

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
GPU_SPOT_INSTANCE_IP = os.getenv('GPU_SPOT_INSTANCE_IP', '3.109.55.217')
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

# --- S3 Configuration ---
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'sparkassets')
S3_REGION = os.getenv('S3_REGION', 'ap-south-1')
S3_IMAGES_PREFIX = 'images/'
S3_3D_ASSETS_PREFIX = '3d_assets/'

# S3 Integration settings
USE_S3_STORAGE = os.getenv('USE_S3_STORAGE', 'True').lower() == 'true'
S3_PUBLIC_READ = os.getenv('S3_PUBLIC_READ', 'False').lower() == 'true'
