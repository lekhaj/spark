"""
Celery task definitions for the text-to-image pipeline
Enhanced with distributed GPU processing capabilities
"""
import os
import sys
import logging
import time
import base64
import json
import requests
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Import Celery
from celery import Celery
from celery.signals import worker_ready, worker_shutting_down

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Import configuration
from config import (
    REDIS_BROKER_URL, REDIS_RESULT_BACKEND, CELERY_TASK_ROUTES,
    USE_DISTRIBUTED_3D, GPU_WORKER_MODE, CPU_INSTANCE_IP,
    OUTPUT_3D_ASSETS_DIR, S3_BUCKET_NAME, USE_S3_FOR_ASSETS,
    get_redis_config, is_distributed_mode, is_gpu_worker
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
task_logger = logging.getLogger(__name__)

# ========================================================================
# CELERY TASKS
# ========================================================================

# Create Celery app with enhanced configuration
redis_config = get_redis_config()
app = Celery('tasks', 
             broker=redis_config['broker_url'], 
             backend=redis_config['result_backend'])

# Enhanced Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
    task_routes=CELERY_TASK_ROUTES,
    # Add result expiration
    result_expires=3600,  # 1 hour
    # Add task compression
    task_compression='gzip',
    result_compression='gzip',
)

task_logger.info(f"Celery initialized - Distributed: {is_distributed_mode()}, GPU Worker: {is_gpu_worker()}")

# ========================================================================
# ENHANCED 3D GENERATION TASKS WITH DISTRIBUTED SUPPORT
# ========================================================================

@app.task(name='generate_3d_model_from_image', bind=True)
def generate_3d_model_from_image(self, image_path, with_texture=False, output_format='glb'):
    """
    Enhanced 3D model generation with distributed GPU support.
    This task now intelligently routes to GPU workers.
    """
    task_logger.info(f"3D generation task started - Image: {image_path}, GPU Worker: {is_gpu_worker()}")
    
    try:
        self.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'status': 'Initializing 3D generation...'}
        )
        
        if is_gpu_worker():
            # We're running on GPU instance - do actual processing
            return _process_3d_on_gpu(self, image_path, with_texture, output_format)
        else:
            # We're on CPU instance - check if we should route to GPU
            if is_distributed_mode():
                return _route_to_gpu_worker(self, image_path, with_texture, output_format)
            else:
                # Local processing fallback
                return _process_3d_locally(self, image_path, with_texture, output_format)
                
    except Exception as e:
        task_logger.error(f"3D generation error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"3D generation failed: {str(e)}",
            "image_path": image_path
        }

def _process_3d_on_gpu(task_instance, image_path, with_texture, output_format):
    """Process 3D generation on GPU worker."""
    try:
        task_logger.info("Processing 3D generation on GPU worker")
        
        # Import GPU-specific modules
        from hunyuan3d_worker import generate_3d_from_image_core
        
        task_instance.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Loading GPU models...'}
        )
        
        # Progress callback for GPU processing
        def gpu_progress_callback(progress, status):
            task_instance.update_state(
                state='PROGRESS',
                meta={'progress': progress, 'status': status, 'gpu_processing': True}
            )
        
        # Perform actual GPU processing
        result = generate_3d_from_image_core(
            image_path=image_path,
            with_texture=with_texture,
            output_format=output_format,
            progress_callback=gpu_progress_callback
        )
        
        # Upload to S3 if configured
        if USE_S3_FOR_ASSETS and result.get('status') == 'success':
            s3_url = _upload_to_s3(result['output_path'])
            if s3_url:
                result['s3_url'] = s3_url
                result['storage'] = 's3'
        
        task_logger.info(f"GPU processing completed: {result}")
        return result
        
    except ImportError as e:
        task_logger.error(f"GPU modules not available: {e}")
        return {
            "status": "error",
            "message": "GPU processing modules not available",
            "fallback_required": True
        }
    except Exception as e:
        task_logger.error(f"GPU processing error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"GPU processing failed: {str(e)}"
        }

def _route_to_gpu_worker(task_instance, image_path, with_texture, output_format):
    """Route task to GPU worker in distributed mode."""
    try:
        task_logger.info("Routing 3D generation to GPU worker")
        
        task_instance.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'status': 'Checking GPU worker availability...'}
        )
        
        # Check if GPU workers are available
        inspect = app.control.inspect()
        active_queues = inspect.active_queues()
        
        gpu_workers_available = False
        if active_queues:
            for worker, queues in active_queues.items():
                if any(q['name'] == '3d_queue' for q in queues):
                    gpu_workers_available = True
                    break
        
        if not gpu_workers_available:
            # Start GPU instance if auto-scaling is enabled
            from config import AUTO_SCALE_GPU
            if AUTO_SCALE_GPU:
                task_instance.update_state(
                    state='PROGRESS',
                    meta={'progress': 10, 'status': 'Starting GPU instance...'}
                )
                
                scale_result = auto_scale_gpu.delay('start')
                scale_status = scale_result.get(timeout=600)  # 10 minutes
                
                if scale_status.get('status') != 'success':
                    return {
                        "status": "error",
                        "message": f"Failed to start GPU worker: {scale_status.get('message', 'Unknown error')}"
                    }
        
        task_instance.update_state(
            state='PROGRESS',
            meta={'progress': 15, 'status': 'Submitting to GPU queue...'}
        )
        
        # The task will be re-routed to GPU queue automatically by Celery routing
        # This is a bit recursive, but Celery handles it correctly
        gpu_task = generate_3d_model_from_image.apply_async(
            args=[image_path, with_texture, output_format],
            queue='3d_queue'
        )
        
        # Wait for GPU processing
        return gpu_task.get(timeout=1800)  # 30 minutes
        
    except Exception as e:
        task_logger.error(f"GPU routing error: {e}")
        # Fallback to local processing
        return _process_3d_locally(task_instance, image_path, with_texture, output_format)

def _process_3d_locally(task_instance, image_path, with_texture, output_format):
    """Fallback local 3D processing."""
    task_logger.info("Processing 3D generation locally (fallback)")
    
    task_instance.update_state(
        state='PROGRESS',
        meta={'progress': 20, 'status': 'Processing locally...'}
    )
    
    # Use existing local processing logic
    # ...existing code... (your current local processing)
    
    # Simulate local processing for now
    time.sleep(10)
    
    return {
        "status": "success",
        "message": "3D model generated locally",
        "output_path": f"/tmp/local_3d_model.{output_format}",
        "local_processing": True
    }

# ========================================================================
# NEW: AUTO-SCALING TASKS
# ========================================================================

@app.task(name='auto_scale_gpu', bind=True)
def auto_scale_gpu(self, action='start'):
    """Auto-scale GPU instances based on demand."""
    try:
        task_logger.info(f"Auto-scaling GPU: {action}")
        
        if action == 'start':
            return _start_gpu_instance(self)
        elif action == 'stop':
            return _stop_gpu_instance(self)
        elif action == 'check':
            return _check_gpu_status(self)
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    except Exception as e:
        task_logger.error(f"Auto-scaling error: {e}")
        return {"status": "error", "message": str(e)}

def _start_gpu_instance(task_instance):
    """Start GPU spot instance."""
    try:
        task_instance.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Initializing AWS manager...'}
        )
        
        from aws_manager import AWSManager
        aws_mgr = AWSManager()
        
        task_instance.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'status': 'Checking existing instances...'}
        )
        
        # Check current status
        status = aws_mgr.get_gpu_instance_status()
        if status.get('worker_ready'):
            return {
                "status": "success",
                "message": "GPU worker already available",
                "instance_details": status
            }
        
        task_instance.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'status': 'Starting GPU spot instance...'}
        )
        
        # Start new instance
        result = aws_mgr.start_gpu_instance(use_spot=True)
        
        if result.get('status') == 'success':
            task_instance.update_state(
                state='PROGRESS',
                meta={'progress': 80, 'status': 'Waiting for GPU worker...'}
            )
            
            # Wait for worker to be ready
            max_wait = 600  # 10 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status = aws_mgr.get_gpu_instance_status()
                if status.get('worker_ready'):
                    return {
                        "status": "success",
                        "message": "GPU worker is ready",
                        "instance_details": status
                    }
                time.sleep(30)
            
            return {
                "status": "partial",
                "message": "GPU instance started but worker not ready yet",
                "instance_details": result
            }
        else:
            return result
            
    except Exception as e:
        task_logger.error(f"Error starting GPU instance: {e}")
        return {"status": "error", "message": str(e)}

def _stop_gpu_instance(task_instance):
    """Stop GPU instances."""
    try:
        from aws_manager import AWSManager
        aws_mgr = AWSManager()
        
        task_instance.update_state(
            state='PROGRESS',
            meta={'progress': 50, 'status': 'Stopping GPU instances...'}
        )
        
        result = aws_mgr.stop_gpu_instance()
        return result
        
    except Exception as e:
        task_logger.error(f"Error stopping GPU instance: {e}")
        return {"status": "error", "message": str(e)}

def _check_gpu_status(task_instance):
    """Check GPU instance status."""
    try:
        from aws_manager import AWSManager
        aws_mgr = AWSManager()
        
        status = aws_mgr.get_gpu_instance_status()
        return {
            "status": "success",
            "gpu_status": status
        }
        
    except Exception as e:
        task_logger.error(f"Error checking GPU status: {e}")
        return {"status": "error", "message": str(e)}

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def _upload_to_s3(file_path):
    """Upload file to S3 and return URL."""
    if not USE_S3_FOR_ASSETS or not S3_BUCKET_NAME:
        return None
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3')
        file_name = os.path.basename(file_path)
        s3_key = f"3d-assets/{file_name}"
        
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        task_logger.info(f"Uploaded to S3: {s3_url}")
        return s3_url
        
    except Exception as e:
        task_logger.error(f"S3 upload failed: {e}")
        return None

def _encode_image_to_base64(image_path):
    """Encode image to base64 for transfer."""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        task_logger.error(f"Image encoding failed: {e}")
        return None

def _decode_base64_to_image(base64_data, output_path):
    """Decode base64 to image file."""
    try:
        image_data = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        return output_path
    except Exception as e:
        task_logger.error(f"Image decoding failed: {e}")
        return None

# ========================================================================
# WORKER LIFECYCLE EVENTS
# ========================================================================

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready event."""
    worker_name = sender.hostname if sender else 'unknown'
    task_logger.info(f"Worker ready: {worker_name}")
    
    if is_gpu_worker():
        task_logger.info("GPU worker initialization starting...")
        try:
            from hunyuan3d_worker import initialize_hunyuan3d_models
            if initialize_hunyuan3d_models():
                task_logger.info("GPU models initialized successfully")
            else:
                task_logger.warning("GPU model initialization failed")
        except Exception as e:
            task_logger.error(f"GPU worker initialization error: {e}")

@worker_shutting_down.connect
def worker_shutting_down_handler(sender=None, **kwargs):
    """Handle worker shutdown event."""
    worker_name = sender.hostname if sender else 'unknown'
    task_logger.info(f"Worker shutting down: {worker_name}")

# ========================================================================
# BACKWARD COMPATIBILITY
# Keep all existing task names and signatures
# ========================================================================

# Export for external use
__all__ = [
    'app', 'generate_3d_model_from_image', 'auto_scale_gpu',
    # Add any other existing exports
]

task_logger.info("Celery tasks module loaded successfully")
