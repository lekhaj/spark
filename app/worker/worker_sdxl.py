
import os
import json
import torch
import time
import redis
from diffusers import DiffusionPipeline
from .worker_config import redis_client, s3, BUCKET_NAME, REDIS_HOST, REDIS_PORT


# ------------ Configuration ------------
REDIS_QUEUE = "image_tasks"  # Queue to listen for image tasks
# ---------------------------------------

def initialize_sdxl_pipeline():
    """Initialize and return the SDXL pipeline"""
    print("Loading SDXL model...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention enabled")
    except Exception as e:
        print(f"xformers not available: {e}")
    
    return pipe

def upload_to_s3(local_file_path, s3_key):
    """Upload file to S3 and return public URL"""
    try:
        s3.upload_file(local_file_path, BUCKET_NAME, s3_key)
        # Return public URL
        return f"https://{BUCKET_NAME}.s3.{s3.meta.region_name}.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

def process_image_task(task_data, pipeline):
    """Process a single image generation task"""
    job_id = task_data.get('job_id', 'unknown')
    prompt = task_data.get('prompt', '')
    
    print(f"Processing job {job_id}: {prompt[:60]}...")
    
    try:
        # Generate image with SDXL
        image = pipeline(
            prompt=prompt,
            negative_prompt=task_data.get('negative_prompt', ''),
            width=task_data.get('width', 1024),
            height=task_data.get('height', 1024),
            num_inference_steps=task_data.get('num_inference_steps', 30),
            guidance_scale=task_data.get('guidance_scale', 7.5),
            generator=torch.Generator(device="cuda").manual_seed(int(time.time()))
        ).images[0]

        # Create temporary file
        temp_dir = "/tmp/sdxl_images"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{job_id}.png")
        
        # Save image
        image.save(temp_file, format="PNG")
        print(f"Image generated for job {job_id}")

        # Upload to S3
        s3_key = task_data.get('output_key', f"generated-images/{job_id}/output.png")
        image_url = upload_to_s3(temp_file, s3_key)
        print(f"Image uploaded to S3: {image_url}")

        # Clean up temporary file
        os.remove(temp_file)

        # Prepare success result
        result = {
            "job_id": job_id,
            "status": "completed",
            "image_url": image_url,
            "prompt": prompt,
            "negative_prompt": task_data.get('negative_prompt', ''),
            "dimensions": f"{task_data.get('width', 1024)}x{task_data.get('height', 1024)}",
            "generated_at": time.time(),
            "message": "Image successfully generated and uploaded"
        }

        return result

    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        
        # Prepare error result
        error_result = {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "prompt": prompt,
            "failed_at": time.time(),
            "message": "Failed to generate image"
        }
        
        return error_result

def store_result(redis_client, job_id, result):
    """Store result in Redis with expiration (24 hours)"""
    try:
        result_key = f"result:{job_id}"
        redis_client.setex(result_key, 86400, json.dumps(result))  # 24 hours expiration
        print(f"Result stored for job {job_id}")
    except Exception as e:
        print(f"Failed to store result for job {job_id}: {e}")


def image_worker():
    """Main worker function for image processing"""
    print("=" * 50)
    print("SDXL Image Processing Worker")
    print("=" * 50)
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Queue: {REDIS_QUEUE}")
    print(f"S3 Bucket: {BUCKET_NAME}")
    print("=" * 50)
    try:
        redis_client.ping()
        print("Connected to Redis")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        return

    # Initialize SDXL pipeline
    try:
        sdxl_pipeline = initialize_sdxl_pipeline()
        print("SDXL pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SDXL pipeline: {e}")
        return

    print("Worker ready. Waiting for image tasks...")
    print("Press Ctrl+C to stop the worker")
    print("=" * 50)

    try:
        while True:
            try:
                # Get task from Redis queue (blocking pop with 30 second timeout)
                task_data = redis_client.blpop(REDIS_QUEUE, timeout=30)
                
                if task_data is None:
                    # No task received, continue waiting
                    continue

                # task_data is a tuple: (queue_name, message)
                queue_name, task_json = task_data
                task = json.loads(task_json)
                
                print(f"Received task: {task.get('job_id', 'unknown')}")
                
                # Process the image task
                result = process_image_task(task, sdxl_pipeline)
                
                # Store the result in Redis
                store_result(redis_client, result['job_id'], result)
                
                # Log completion
                if result['status'] == 'completed':
                    print(f"Successfully completed job {result['job_id']}")
                else:
                    print(f"Failed job {result['job_id']}: {result.get('error', 'Unknown error')}")
                
                print("-" * 30)
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse task JSON: {e}")
            except redis.ConnectionError as e:
                print(f"Redis connection error: {e}")
                print("Reconnecting in 5 seconds...")
                time.sleep(5)
                # Reinitialize Redis connection
                # redis_client is already set from worker_config
            except Exception as e:
                print(f"Unexpected error in main loop: {e}")
                time.sleep(5)  # Prevent tight error loops
                
    except KeyboardInterrupt:
        print("\nWorker stopped by user")
    except Exception as e:
        print(f"Fatal error in worker: {e}")

if __name__ == "__main__":
    # Start the image processing worker
    image_worker()