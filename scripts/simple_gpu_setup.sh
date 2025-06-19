#!/bin/bash
# Simple GPU Instance Setup Script
# This script sets up a minimal GPU worker environment

set -e

echo "🚀 Simple GPU worker setup..."

# Configuration - Use script directory as base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create project structure
echo "📁 Creating directories..."
mkdir -p "$PROJECT_DIR"/{src,scripts,generated_assets/{3d_assets,images}}

# Install system packages
echo "📦 Installing system packages..."
sudo apt update
sudo apt install -y python3 python3-pip redis-server

# Start Redis
echo "🔴 Starting Redis..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Install Python packages globally (simpler approach)
echo "🐍 Installing Python packages..."
pip3 install celery redis python-dotenv pillow

# Create minimal config.py
echo "⚙️ Creating config.py..."
cat > "$PROJECT_DIR/src/config.py" << 'EOF'
import os

# Worker configuration
WORKER_TYPE = os.getenv('WORKER_TYPE', 'gpu')

class RedisConfig:
    def __init__(self):
        self.worker_type = WORKER_TYPE
        self.write_url = 'redis://127.0.0.1:6379/0'
        self.read_url = 'redis://127.0.0.1:6379/0'
        
    @property
    def write_client(self):
        import redis
        return redis.Redis.from_url(self.write_url, socket_timeout=5)
    
    @property  
    def read_client(self):
        import redis
        return redis.Redis.from_url(self.read_url, socket_timeout=5)
    
    def test_connection(self):
        import time
        results = {}
        try:
            client = self.write_client
            client.ping()
            test_key = f"test_{int(time.time())}"
            client.set(test_key, "test", ex=10)
            client.delete(test_key)
            results['write'] = {'success': True, 'url': self.write_url}
            results['read'] = {'success': True, 'url': self.read_url}
        except Exception as e:
            results['write'] = {'success': False, 'error': str(e)}
            results['read'] = {'success': False, 'error': str(e)}
        return results

REDIS_CONFIG = RedisConfig()
REDIS_BROKER_URL = 'redis://127.0.0.1:6379/0'
REDIS_RESULT_BACKEND = 'redis://127.0.0.1:6379/0'

CELERY_TASK_ROUTES = {
    'generate_3d_model_from_image': {'queue': 'gpu_tasks'},
}

# Output directories
OUTPUT_DIR = './generated_assets'
OUTPUT_3D_ASSETS_DIR = os.path.join(OUTPUT_DIR, "3d_assets")
os.makedirs(OUTPUT_3D_ASSETS_DIR, exist_ok=True)
EOF

# Create minimal tasks.py
echo "📝 Creating tasks.py..."
cat > "$PROJECT_DIR/src/tasks.py" << 'EOF'
import os
import logging
from celery import Celery
import uuid
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gpu_worker')

# Simple Celery setup
app = Celery('gpu_tasks', broker='redis://127.0.0.1:6379/0', backend='redis://127.0.0.1:6379/0')

app.conf.update(
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

@app.task(name='generate_3d_model_from_image', bind=True)
def generate_3d_model_from_image(self, image_path, with_texture=False, output_format='glb'):
    """Simple 3D model generation task"""
    
    try:
        logger.info(f"Processing 3D generation for: {image_path}")
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting...'})
        
        # Check if image exists
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image not found: {image_path}"}
        
        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Loading image...'})
        
        # Try to import PIL for basic image handling
        try:
            from PIL import Image
            image = Image.open(image_path)
            logger.info(f"Image loaded: {image.size}")
        except ImportError:
            logger.warning("PIL not available, skipping image validation")
        except Exception as e:
            return {"status": "error", "message": f"Failed to load image: {e}"}
        
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Generating 3D model...'})
        
        # Create output directory
        from config import OUTPUT_3D_ASSETS_DIR
        unique_id = str(uuid.uuid4())[:8]
        model_dir = os.path.join(OUTPUT_3D_ASSETS_DIR, f"model_{unique_id}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Simulate 3D generation (replace this with actual Hunyuan3D when available)
        time.sleep(2)  # Simulate processing time
        
        self.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Saving model...'})
        
        # Create a dummy model file for now
        output_filename = f"model_{unique_id}.{output_format}"
        output_path = os.path.join(model_dir, output_filename)
        
        with open(output_path, 'w') as f:
            f.write(f"# Dummy 3D model generated from {image_path}\n")
            f.write(f"# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# This is a placeholder until Hunyuan3D is properly installed\n")
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Complete!'})
        
        return {
            "status": "success",
            "message": "3D model generated (dummy implementation)",
            "model_path": output_path,
            "model_filename": output_filename,
            "output_format": output_format,
            "note": "This is a placeholder - install Hunyuan3D for actual 3D generation"
        }
        
    except Exception as e:
        logger.error(f"Error in 3D generation: {e}", exc_info=True)
        return {"status": "error", "message": f"Task failed: {e}"}
EOF

# Create simple startup script
echo "📋 Creating startup script..."
cat > "$PROJECT_DIR/scripts/start_gpu_worker.sh" << 'EOF'
#!/bin/bash
set -e

# Get project root directory relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "🚀 Starting simple GPU worker..."

cd "$SRC_DIR"

# Set environment
export WORKER_TYPE=gpu
export REDIS_BROKER_URL="redis://127.0.0.1:6379/0"
export REDIS_RESULT_BACKEND="redis://127.0.0.1:6379/0"

# Test Redis
echo "Testing Redis connection..."
python3 -c "
import redis
try:
    client = redis.Redis.from_url('redis://127.0.0.1:6379/0')
    client.ping()
    print('✅ Redis OK')
except Exception as e:
    print(f'❌ Redis failed: {e}')
    exit(1)
"

# Start worker
echo "Starting Celery worker..."
celery -A tasks worker \
    --loglevel=info \
    --queues=gpu_tasks \
    --hostname=gpu-worker@$(hostname) \
    --concurrency=1 \
    --pool=solo
EOF

chmod +x "$PROJECT_DIR/scripts/start_gpu_worker.sh"

# Test the setup
echo "🧪 Testing setup..."
cd "$PROJECT_DIR/src"

python3 -c "
try:
    from config import REDIS_CONFIG
    results = REDIS_CONFIG.test_connection()
    if results['write']['success']:
        print('✅ Redis configuration OK')
    else:
        print('❌ Redis configuration failed')
except Exception as e:
    print(f'❌ Config test failed: {e}')
"

echo "✅ Simple GPU worker setup complete!"
echo ""
echo "🚀 To start the worker:"
echo "   cd $PROJECT_DIR"
echo "   ./scripts/start_gpu_worker.sh"
echo ""
echo "📝 Note: This uses a dummy 3D implementation."
echo "   Install Hunyuan3D modules for actual 3D generation."
