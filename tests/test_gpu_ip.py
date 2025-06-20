#!/usr/bin/env python3
import os
import sys

# Set up environment for testing
os.environ['WORKER_TYPE'] = 'cpu'
os.environ['GPU_SPOT_INSTANCE_IP'] = '13.201.23.51'
os.environ['USE_CELERY'] = 'True'

# Add src to path
sys.path.insert(0, 'src')

try:
    from config import REDIS_CONFIG
    
    print("=== Redis Configuration Test ===")
    print(f"Worker Type: {REDIS_CONFIG.worker_type}")
    print(f"GPU IP: {REDIS_CONFIG.gpu_ip}")
    print(f"Write URL: {REDIS_CONFIG.write_url}")
    print(f"Read URL: {REDIS_CONFIG.read_url}")
    print()
    
    # Test what would be shown in the UI
    task_id = "test-task-123"
    message = f"✅ 3D generation task submitted to GPU spot instance (ID: {task_id}). Processing on {REDIS_CONFIG.gpu_ip}..."
    print("UI Message Preview:")
    print(message)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
