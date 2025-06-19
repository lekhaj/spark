"""
AWS EC2 Manager for GPU Instance Management
This module provides functionality to manage EC2 instances for Hunyuan3D processing:
- Start/stop GPU instances
- Check instance status and health
- Auto-scaling logic
- Cost optimization features
"""

import boto3
import time
import logging
import requests
import json
from typing import Optional, Dict, Any, List
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class AWSManager:
    def __init__(self, region='us-east-1'):
        """Initialize AWS Manager with enhanced GPU instance management."""
        self.region = region
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)
        
        # GPU Instance Configuration
        self.gpu_instance_id = None
        self.gpu_instance_ip = None
        self.gpu_instance_type = 'g4dn.xlarge'
        self.gpu_spot_price = '0.50'  # Max price per hour
        
        # Set CPU instance info - THIS IS THE SINGLE SOURCE OF TRUTH
        self.cpu_instance_ip = "172.31.6.174"  # Your actual CPU instance IP
        self.cpu_instance_id = self._get_current_instance_id()
        
        logger.info(f"AWS Manager initialized - Region: {region}, CPU IP: {self.cpu_instance_ip}")

    def _get_current_instance_id(self) -> str:
        """Get current CPU instance ID."""
        try:
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id', 
                timeout=5
            )
            return response.text
        except Exception as e:
            logger.warning(f"Could not get instance ID: {e}")
            return "i-cpu-instance"  # fallback

    def start_gpu_instance(self, 
                          instance_type: str = None,
                          spot_price: str = None,
                          use_spot: bool = True) -> Dict[str, Any]:
        """
        Start and configure GPU instance for 3D processing.
        Now properly uses self.cpu_instance_ip everywhere.
        """
        try:
            # Use provided values or defaults
            instance_type = instance_type or self.gpu_instance_type
            spot_price = spot_price or self.gpu_spot_price
            
            logger.info(f"Starting GPU instance: {instance_type}, Spot: {use_spot}")
            logger.info(f"GPU will connect to CPU instance: {self.cpu_instance_ip}")
            
            # Check if GPU instance already running
            existing_instance = self._find_running_gpu_instance()
            if existing_instance:
                self.gpu_instance_id = existing_instance['InstanceId']
                self.gpu_instance_ip = existing_instance.get('PrivateIpAddress')
                
                return {
                    'status': 'success',
                    'message': 'GPU instance already running',
                    'instance_id': self.gpu_instance_id,
                    'instance_ip': self.gpu_instance_ip,
                    'cpu_ip': self.cpu_instance_ip,  # Include for verification
                    'existing': True
                }
            
            # Create user data script (now uses self.cpu_instance_ip properly)
            user_data_script = self._create_gpu_user_data_script()
            
            # Launch configuration
            launch_config = {
                'ImageId': self._get_deep_learning_ami(),
                'InstanceType': instance_type,
                'KeyName': self._get_key_pair_name(),
                'SecurityGroupIds': [self._get_gpu_security_group()],
                'UserData': user_data_script,
                'IamInstanceProfile': {
                    'Name': self._get_gpu_iam_role()
                },
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'Hunyuan3D-GPU-Worker'},
                        {'Key': 'Role', 'Value': '3D-Processing'},
                        {'Key': 'ManagedBy', 'Value': 'Pipeline'},
                        {'Key': 'CPUInstance', 'Value': self.cpu_instance_id or 'unknown'},
                        {'Key': 'CPUInstanceIP', 'Value': self.cpu_instance_ip}  # Track CPU IP
                    ]
                }]
            }
            
            if use_spot:
                # Launch spot instance
                response = self.ec2_client.request_spot_instances(
                    SpotPrice=spot_price,
                    InstanceCount=1,
                    LaunchSpecification=launch_config
                )
                
                spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                logger.info(f"Spot instance requested: {spot_request_id}")
                
                # Wait for spot request fulfillment
                instance_id = self._wait_for_spot_fulfillment(spot_request_id)
                if not instance_id:
                    return {
                        'status': 'error',
                        'message': 'Spot instance request failed or timed out'
                    }
                    
            else:
                # Launch on-demand instance
                response = self.ec2_client.run_instances(
                    MinCount=1,
                    MaxCount=1,
                    **launch_config
                )
                instance_id = response['Instances'][0]['InstanceId']
            
            self.gpu_instance_id = instance_id
            logger.info(f"GPU instance launched: {instance_id}")
            
            # Wait for instance to be running
            self._wait_for_instance_running(instance_id)
            
            # Get instance IP
            self.gpu_instance_ip = self._get_instance_private_ip(instance_id)
            
            # Wait for GPU worker to be ready
            worker_ready = self._wait_for_gpu_worker_ready(timeout=600)  # 10 minutes
            
            result = {
                'status': 'success',
                'message': 'GPU instance started successfully',
                'instance_id': self.gpu_instance_id,
                'instance_ip': self.gpu_instance_ip,
                'cpu_ip': self.cpu_instance_ip,  # Include for verification
                'instance_type': instance_type,
                'worker_ready': worker_ready,
                'existing': False
            }
            
            if not worker_ready:
                result['warning'] = 'Instance started but GPU worker not ready yet'
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to start GPU instance: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Failed to start GPU instance: {str(e)}',
                'cpu_ip': self.cpu_instance_ip  # Include for debugging
            }

    def stop_gpu_instance(self, force: bool = False) -> Dict[str, Any]:
        """
        Stop GPU instance to save costs.
        
        Args:
            force: Force termination even if tasks are running
            
        Returns:
            Dict with status and details
        """
        try:
            # Find GPU instances to stop
            gpu_instances = self._find_all_gpu_instances()
            
            if not gpu_instances:
                return {
                    'status': 'success',
                    'message': 'No GPU instances found to stop'
                }
            
            stopped_instances = []
            
            for instance in gpu_instances:
                instance_id = instance['InstanceId']
                state = instance['State']['Name']
                
                if state in ['stopping', 'stopped', 'terminated']:
                    logger.info(f"Instance {instance_id} already {state}")
                    continue
                
                # Check for running tasks if not forcing
                if not force:
                    has_active_tasks = self._check_active_gpu_tasks(instance_id)
                    if has_active_tasks:
                        logger.warning(f"Instance {instance_id} has active tasks, skipping")
                        continue
                
                # Terminate the instance
                self.ec2_client.terminate_instances(InstanceIds=[instance_id])
                stopped_instances.append(instance_id)
                logger.info(f"Terminated GPU instance: {instance_id}")
            
            # Clear cached instance info
            if self.gpu_instance_id in stopped_instances:
                self.gpu_instance_id = None
                self.gpu_instance_ip = None
            
            return {
                'status': 'success',
                'message': f'Stopped {len(stopped_instances)} GPU instances',
                'stopped_instances': stopped_instances
            }
            
        except Exception as e:
            logger.error(f"Failed to stop GPU instance: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Failed to stop GPU instance: {str(e)}'
            }

    def get_gpu_instance_status(self) -> Dict[str, Any]:
        """
        Check if GPU instance is running and ready.
        
        Returns:
            Dict with detailed GPU instance status
        """
        try:
            gpu_instances = self._find_all_gpu_instances()
            
            if not gpu_instances:
                return {
                    'status': 'stopped',
                    'message': 'No GPU instances found',
                    'instances': [],
                    'worker_ready': False,
                    'total_cost_today': 0.0
                }
            
            instance_details = []
            total_cost = 0.0
            any_worker_ready = False
            
            for instance in gpu_instances:
                instance_id = instance['InstanceId']
                state = instance['State']['Name']
                instance_type = instance['InstanceType']
                launch_time = instance.get('LaunchTime')
                private_ip = instance.get('PrivateIpAddress')
                
                # Check worker status
                worker_ready = False
                if state == 'running' and private_ip:
                    worker_ready = self._check_gpu_worker_health(private_ip)
                    if worker_ready:
                        any_worker_ready = True
                
                # Calculate cost
                instance_cost = self._calculate_instance_cost(instance_type, launch_time)
                total_cost += instance_cost
                
                instance_details.append({
                    'instance_id': instance_id,
                    'state': state,
                    'instance_type': instance_type,
                    'private_ip': private_ip,
                    'launch_time': launch_time.isoformat() if launch_time else None,
                    'worker_ready': worker_ready,
                    'cost_today': instance_cost
                })
            
            # Determine overall status
            if any(i['state'] == 'running' for i in instance_details):
                overall_status = 'running'
            elif any(i['state'] in ['pending', 'starting'] for i in instance_details):
                overall_status = 'starting'
            else:
                overall_status = 'stopped'
            
            return {
                'status': overall_status,
                'message': f'Found {len(gpu_instances)} GPU instances',
                'instances': instance_details,
                'worker_ready': any_worker_ready,
                'total_cost_today': round(total_cost, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Failed to get GPU status: {str(e)}',
                'instances': [],
                'worker_ready': False,
                'total_cost_today': 0.0
            }

    def _create_gpu_user_data_script(self) -> str:
        """Create user data script for GPU instance initialization."""
        # USE SELF.CPU_INSTANCE_IP EVERYWHERE
        script = f'''#!/bin/bash
set -e

# Log all output
exec > /var/log/gpu-initialization.log 2>&1

echo "Starting GPU instance initialization..."
echo "CPU Instance IP: {self.cpu_instance_ip}"

# Update system
apt-get update

# Install basic tools
apt-get install -y git python3-pip redis-tools htop nvtop curl

# Set environment variables using the CPU IP variable
export CPU_INSTANCE_IP="{self.cpu_instance_ip}"
export PYTHONPATH="/tmp/pipeline/src:/tmp/pipeline"

# Create working directory
mkdir -p /tmp/pipeline
cd /tmp/pipeline

# For now, create a minimal worker setup
# Later you'll replace this with actual git clone
mkdir -p src

# Create basic config - USING VARIABLE
cat > src/config.py << 'CONFIGEOF'
import os
REDIS_BROKER_URL = "redis://{self.cpu_instance_ip}:6379/0"
REDIS_RESULT_BACKEND = "redis://{self.cpu_instance_ip}:6379/0"
OUTPUT_3D_ASSETS_DIR = "/tmp/output/3d_assets"
CONFIGEOF

# Create basic hunyuan3d worker
cat > src/hunyuan3d_worker.py << 'WORKEREOF'
import logging
import time
import os
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

def generate_3d_from_image_core(
    image_path: str,
    with_texture: bool = False,
    output_format: str = 'glb',
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """GPU-based 3D generation (placeholder for now)."""
    
    logger.info(f"GPU processing 3D generation: {{image_path}}")
    
    if progress_callback:
        progress_callback(10, "Loading on GPU...")
    
    # Simulate GPU processing
    time.sleep(5)
    
    if progress_callback:
        progress_callback(90, "Finalizing...")
    
    # Create output
    output_dir = "/tmp/output/3d_assets"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gpu_model_{{int(time.time())}}.{{output_format}}")
    
    with open(output_path, 'w') as f:
        f.write(f"# GPU-generated 3D model from {{image_path}}\\n")
        f.write(f"# Format: {{output_format}}, Texture: {{with_texture}}\\n")
    
    if progress_callback:
        progress_callback(100, "Complete!")
    
    return {{
        "status": "success",
        "output_path": output_path,
        "format": output_format,
        "with_texture": with_texture,
        "gpu_processed": True
    }}

def initialize_hunyuan3d_models():
    """Initialize GPU models."""
    logger.info("Initializing GPU models...")
    return True
WORKEREOF

# Create basic tasks module for GPU worker - USING VARIABLE
cat > src/tasks.py << 'TASKSEOF'
import os
import logging
from celery import Celery
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import REDIS_BROKER_URL, REDIS_RESULT_BACKEND
from hunyuan3d_worker import generate_3d_from_image_core

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
app = Celery('gpu_tasks', broker=REDIS_BROKER_URL, backend=REDIS_RESULT_BACKEND)

# Configure Celery
app.conf.update(
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
)

@app.task(name='generate_3d_model_from_image', bind=True)
def generate_3d_model_from_image(self, image_path, with_texture=False, output_format='glb'):
    """GPU Celery task for 3D generation."""
    logger.info(f"GPU worker processing: {{image_path}}")
    
    try:
        result = generate_3d_from_image_core(
            image_path, 
            with_texture, 
            output_format, 
            progress_callback=lambda p, s: self.update_state(
                state='PROGRESS', 
                meta={{'progress': p, 'status': s}}
            )
        )
        return result
    except Exception as e:
        logger.error(f"GPU task error: {{e}}")
        return {{"status": "error", "message": str(e)}}

if __name__ == '__main__':
    print("GPU worker module loaded")
TASKSEOF

# Create __init__.py files
touch src/__init__.py

# Install Python dependencies
pip3 install celery redis boto3 requests pillow numpy

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create environment file - USING VARIABLE
cat > .env << EOF
REDIS_BROKER_URL=redis://{self.cpu_instance_ip}:6379/0
REDIS_RESULT_BACKEND=redis://{self.cpu_instance_ip}:6379/0
USE_CELERY=True
ENVIRONMENT=production
GPU_WORKER_MODE=True
OUTPUT_3D_ASSETS_DIR=/tmp/output/3d_assets
CUDA_VISIBLE_DEVICES=0
CPU_INSTANCE_IP={self.cpu_instance_ip}
EOF

# Create output directories
mkdir -p /tmp/output/3d_assets

# Test GPU
python3 -c "import torch; print(f'CUDA Available: {{torch.cuda.is_available()}}')" || echo "PyTorch/CUDA test failed"

# Test Redis connection to CPU instance - USING VARIABLE
echo "Testing Redis connection to {self.cpu_instance_ip}..."
redis-cli -h {self.cpu_instance_ip} ping || echo "Redis connection failed"

# Wait a bit for everything to settle
sleep 10

# Start Celery worker
echo "Starting Celery worker..."
cd /tmp/pipeline
export PYTHONPATH="/tmp/pipeline/src:/tmp/pipeline"
nohup celery -A src.tasks worker -Q 3d_queue --loglevel=info --hostname=gpu-worker@%h > /var/log/celery-worker.log 2>&1 &

# Also start a worker for general tasks
nohup celery -A src.tasks worker --loglevel=info --hostname=gpu-general@%h > /var/log/celery-general.log 2>&1 &

# Create health check endpoint - USING VARIABLE
cat > /tmp/health_check.py << 'HEALTHEOF'
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import os

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            # Check if Celery worker is running
            try:
                result = subprocess.run(['pgrep', '-f', 'celery.*worker'], 
                                      capture_output=True, text=True)
                worker_running = len(result.stdout.strip()) > 0
                
                # Check GPU
                gpu_available = False
                try:
                    result = subprocess.run(['nvidia-smi'], capture_output=True)
                    gpu_available = result.returncode == 0
                except:
                    pass
                
                # Check Redis connection - USING VARIABLE
                redis_ok = False
                try:
                    result = subprocess.run(['redis-cli', '-h', '{self.cpu_instance_ip}', 'ping'], 
                                          capture_output=True, text=True)
                    redis_ok = 'PONG' in result.stdout
                except:
                    pass
                
                status = {{
                    "status": "healthy" if (worker_running and redis_ok) else "degraded",
                    "worker_running": worker_running,
                    "gpu_available": gpu_available,
                    "redis_connection": redis_ok,
                    "cpu_instance_ip": "{self.cpu_instance_ip}"
                }}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
            except Exception as e:
                error_status = {{"status": "error", "message": str(e)}}
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_status).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress access logs

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
    print("Health check server starting on port 8080...")
    server.serve_forever()
HEALTHEOF

# Start health check server
nohup python3 /tmp/health_check.py > /var/log/health-check.log 2>&1 &

echo "GPU instance initialization completed!"
echo "CPU Instance IP used: {self.cpu_instance_ip}"
echo "Logs available at:"
echo "  - GPU initialization: /var/log/gpu-initialization.log"
echo "  - Celery worker: /var/log/celery-worker.log"
echo "  - Health check: /var/log/health-check.log"
'''
        return script

    def _find_running_gpu_instance(self) -> Optional[Dict[str, Any]]:
        """Find existing running GPU instance."""
        try:
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'instance-state-name', 'Values': ['running']},
                    {'Name': 'tag:Role', 'Values': ['3D-Processing']},
                    {'Name': 'tag:ManagedBy', 'Values': ['Pipeline']}
                ]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    return instance
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding GPU instances: {e}")
            return None

    def _find_all_gpu_instances(self) -> List[Dict[str, Any]]:
        """Find all GPU instances managed by this pipeline."""
        try:
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'stopping', 'stopped']},
                    {'Name': 'tag:Role', 'Values': ['3D-Processing']},
                    {'Name': 'tag:ManagedBy', 'Values': ['Pipeline']}
                ]
            )
            
            instances = []
            for reservation in response['Reservations']:
                instances.extend(reservation['Instances'])
            
            return instances
            
        except Exception as e:
            logger.error(f"Error finding GPU instances: {e}")
            return []

    def _wait_for_spot_fulfillment(self, spot_request_id: str, timeout: int = 300) -> Optional[str]:
        """Wait for spot instance request to be fulfilled."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.ec2_client.describe_spot_instance_requests(
                    SpotInstanceRequestIds=[spot_request_id]
                )
                
                spot_request = response['SpotInstanceRequests'][0]
                state = spot_request['State']
                
                if state == 'active':
                    return spot_request['InstanceId']
                elif state in ['cancelled', 'failed']:
                    logger.error(f"Spot request {spot_request_id} {state}")
                    return None
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error checking spot request: {e}")
                time.sleep(10)
        
        logger.error(f"Spot request {spot_request_id} timed out")
        return None

    def _wait_for_instance_running(self, instance_id: str, timeout: int = 300):
        """Wait for instance to reach running state."""
        waiter = self.ec2_client.get_waiter('instance_running')
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={'Delay': 15, 'MaxAttempts': timeout // 15}
        )

    def _get_instance_private_ip(self, instance_id: str) -> Optional[str]:
        """Get private IP address of instance."""
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            return instance.get('PrivateIpAddress')
        except Exception as e:
            logger.error(f"Error getting instance IP: {e}")
            return None

    def _wait_for_gpu_worker_ready(self, timeout: int = 600) -> bool:
        """Wait for GPU worker to be ready to accept tasks."""
        if not self.gpu_instance_ip:
            return False
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check health endpoint - USING VARIABLE
                response = requests.get(
                    f"http://{self.gpu_instance_ip}:8080/health",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Verify it's connecting to the right CPU instance
                    if data.get('cpu_instance_ip') == self.cpu_instance_ip:
                        logger.info("GPU worker health check passed")
                        return True
                    else:
                        logger.warning(f"GPU worker connected to wrong CPU IP: {data.get('cpu_instance_ip')}")
                    
            except Exception as e:
                logger.debug(f"Waiting for GPU worker: {e}")
            
            time.sleep(30)  # Check every 30 seconds
        
        logger.warning("GPU worker did not become ready within timeout")
        return False

    def _check_gpu_worker_health(self, instance_ip: str) -> bool:
        """Check if GPU worker is healthy."""
        try:
            response = requests.get(
                f"http://{instance_ip}:8080/health",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                # Verify worker is connected to correct CPU instance
                return data.get('cpu_instance_ip') == self.cpu_instance_ip
            return False
        except:
            return False

    # Method to update CPU instance IP if needed
    def update_cpu_instance_ip(self, new_ip: str):
        """Update CPU instance IP and propagate to all configurations."""
        old_ip = self.cpu_instance_ip
        self.cpu_instance_ip = new_ip
        logger.info(f"Updated CPU instance IP from {old_ip} to {new_ip}")
        
        # Update config files, running instances, etc. as needed
        # For now, only updates the in-memory variable

    def get_cpu_instance_ip(self) -> str:
        """Get the current CPU instance IP."""
        return self.cpu_instance_ip

    # ... keep all other existing methods unchanged ...
