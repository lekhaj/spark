import os
import json
import boto3
import redis
import torch
import time
from diffusers import DiffusionPipeline
from app.services.mongo_service import update_nested_status, check_all_sections_and_update_main_status
# ------------ Configuration ------------
REDIS_HOST = os.getenv("REDIS_HOST", "15.206.99.66")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
S3_REGION = os.getenv("S3_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "sparkassets")
REDIS_QUEUE = "image_tasks"
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
        s3_client = boto3.client("s3", region_name=S3_REGION)
        s3_client.upload_file(local_file_path, BUCKET_NAME, s3_key)
        
        return f"https://{BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

def process_image_task(task_data, sdxl_pipeline):
    """Generates an image from a text prompt and updates MongoDB."""
    job_id = task_data.get("job_id")
    prompt = task_data.get("prompt")
    
    if not prompt:
        return {"job_id": job_id, "status": "failed", "error": "No prompt provided"}

    try:
        print(f"Generating image for job {job_id} with prompt: '{prompt}'")
        image = sdxl_pipeline(prompt=prompt).images[0]
        
        temp_file_path = f"/tmp/{job_id}.png"
        image.save(temp_file_path)
        
        s3_key = f"image_assets/{job_id}.png"
        image_url = upload_to_s3(temp_file_path, s3_key)
        
        if not image_url:
            update_nested_status(task_id=job_id, section_name="image_generation_details", new_status="FAILED")
            return {"job_id": job_id, "status": "failed", "error": "S3 upload failed"}

        # Update MongoDB with the COMPLETED status and the S3 link
        update_success = update_nested_status(
            task_id=job_id,
            section_name="image_generation_details",
            new_status="COMPLETED",
            link=image_url
        )
        
        if update_success:
            check_all_sections_and_update_main_status(job_id)

        os.remove(temp_file_path)

        return {"job_id": job_id, "status": "completed", "url": image_url}

    except Exception as e:
        print(f"Error processing image task {job_id}: {e}")
        update_nested_status(task_id=job_id, section_name="image_generation_details", new_status="FAILED")
        return {"job_id": job_id, "status": "failed", "error": str(e)}

def image_worker():
    """Main worker loop that listens for tasks on Redis."""
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    sdxl_pipeline = initialize_sdxl_pipeline()

    print(f"Listening for tasks on queue: {REDIS_QUEUE}")
    try:
        while True:
            task_data = redis_client.brpop(REDIS_QUEUE, timeout=60)
            
            if not task_data:
                continue

            _, raw_task = task_data
            try:
                task = json.loads(raw_task)
                print(f"Received task: {task.get('job_id', 'unknown')}")
                
                # Immediately update MongoDB to 'PROCESSING' status
                update_nested_status(task_id=task.get("job_id"), section_name="image_generation_details", new_status="PROCESSING")
                
                result = process_image_task(task, sdxl_pipeline)
                
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
                redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
            except Exception as e:
                print(f"Unexpected error in main loop: {e}")
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\nWorker stopped by user")
    except Exception as e:
        print(f"Fatal error in worker: {e}")

if __name__ == "__main__":
    image_worker()
