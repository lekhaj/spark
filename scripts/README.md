# Worker Startup Scripts Deployment Guide

This directory contains scripts to configure and start Celery workers for distributed task processing between a CPU EC2 instance and a GPU spot instance.

## Files Overview

### Worker Startup Scripts
- **`start_cpu_worker.sh`** - Starts Celery worker for CPU-intensive tasks
- **`start_gpu_worker.sh`** - Starts Celery worker for GPU-intensive tasks
- **`setup_gpu_spot_instance.sh`** - Complete setup script for GPU spot instance

### Environment Files
- **`.env.cpu`** - Environment configuration for CPU instance
- **`.env.gpu`** - Environment configuration for GPU spot instance

## Deployment Steps

### 1. GPU Spot Instance Setup (13.203.200.155)

```bash
# Copy your project to the GPU instance
scp -r /path/to/spark ubuntu@13.203.200.155:~/

# SSH to GPU instance
ssh ubuntu@13.203.200.155

# Run the setup script
cd ~/spark/scripts
sudo bash setup_gpu_spot_instance.sh

# Start the GPU worker
sudo systemctl start celery-gpu-worker

# Check status
sudo systemctl status celery-gpu-worker
```

### 2. CPU Instance Setup

```bash
# Ensure you have the updated code with distributed configuration
cd /home/ubuntu/Shashwat/spark

# Load CPU environment
source .env.cpu

# Test Redis connection to GPU instance
python3 -c "import redis; r = redis.Redis.from_url('redis://13.203.200.155:6379/0'); print('✅ Redis OK' if r.ping() else '❌ Redis Failed')"

# Start CPU worker
./scripts/start_cpu_worker.sh
```

### 3. Running Your Application

```bash
# On CPU instance, start your main application
cd /home/ubuntu/Shashwat/spark/src
python3 merged_gradio_app.py
```

## Testing the Setup

### Test Task Routing

```python
# Test CPU task (should run on CPU instance)
from tasks import generate_text_image
cpu_task = generate_text_image.apply_async(
    args=["a beautiful landscape", 512, 512, 1, "flux"],
    queue='cpu_tasks'
)

# Test GPU task (should run on GPU spot instance)
from tasks import generate_3d_model_from_prompt
gpu_task = generate_3d_model_from_prompt.apply_async(
    args=["a red cube", True, "glb"],
    queue='gpu_tasks'
)

print(f"CPU Task ID: {cpu_task.id}")
print(f"GPU Task ID: {gpu_task.id}")
```

### Monitor Workers

```bash
# On CPU instance
celery -A tasks inspect active --destination=cpu-worker@hostname

# On GPU instance  
celery -A tasks inspect active --destination=gpu-worker@hostname
```

## Troubleshooting

### Redis Connection Issues
```bash
# Test Redis from CPU instance
redis-cli -h 13.203.200.155 ping

# Check Redis logs on GPU instance
sudo journalctl -u redis-server -f
```

### Worker Issues
```bash
# Check worker logs
sudo journalctl -u celery-gpu-worker -f

# Restart workers
sudo systemctl restart celery-gpu-worker
```

### Spot Instance Recovery
```bash
# If spot instance is terminated, launch new instance and run:
cd ~/spark/scripts
sudo bash setup_gpu_spot_instance.sh
```

## Key Benefits

- **Cost Optimization**: GPU tasks only use expensive spot instance
- **Reliability**: CPU tasks continue if spot instance terminates  
- **Auto-Recovery**: Services restart automatically
- **Monitoring**: Built-in health checks and logging
- **Scalability**: Easy to add more workers of each type

## Security Notes

- Redis is configured without authentication for simplicity
- Consider adding Redis AUTH for production use
- Ensure security groups only allow necessary ports
- Use IAM roles instead of access keys when possible
