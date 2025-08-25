# ========================
# File: worker_hunyuan.py
# ========================
# gpu worker
import os
import sys
import time
import redis
import json
import tempfile
import uuid
from datetime import datetime

# Import necessary modules from your project
from app.services.aws_service import upload_to_s3
from app.services.mongo_service import update_task_status
from app.config import db

# Ensure paths are correct for your Hunyuan3D library
sys.path.insert(0, '/opt/Hunyuan3D/hy3dshape')
sys.path.insert(0, '/opt/Hunyuan3D/hy3dpaint')

# shape and paint pipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# Initialize pipelines (this assumes the environment and models are correctly set up)
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))

# Redis connection
r = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379)

def process_task(task_data):
    """Processes a single task from the queue."""
    temp_file = None
    try:
        task = json.loads(task_data)
        task_id = task['id']
        prompt = task['prompt']
        print(f"Processing task {task_id} with prompt: '{prompt}'")
        
    
        textured_mesh = "This is a dummy textured mesh object."

        #  Save the generated mesh to a temporary file
        # This is the crucial step to resolve the 'expected string' error.
        temp_dir = tempfile.gettempdir()
        temp_file_name = f"{task_id}.obj" # or .gltf, .glb, etc.
        temp_file = os.path.join(temp_dir, temp_file_name)
        
        # Placeholder for saving the mesh.
        # You'll need to replace this with the actual code to save your mesh object.
        with open(temp_file, "w") as f:
            f.write(textured_mesh)

        print(f"3D asset saved to temporary file: {temp_file}")
        
        #  Upload the temporary file to S3
        s3_object_name = f"assets/{temp_file_name}"
        s3_link = upload_to_s3(temp_file, s3_object_name)

        if s3_link:
            print(f"Successfully uploaded to S3: {s3_link}")
            # 4. Update the MongoDB task status with the S3 link
            update_task_status(task_id, "completed", s3_link)
            print(f"Completed task {task_id}. Status updated in MongoDB.")
            return True
        else:
            # Handle the case where S3 upload fails
            print(f"Failed to upload task {task_id} to S3.")
            update_task_status(task_id, "failed")
            return False

    except Exception as e:
        print(f"Error processing task {task_id}: {str(e)}")
        # Update MongoDB with a 'failed' status
        update_task_status(task_id, "failed")
        return False
    finally:
        #  Clean up the temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up temporary file: {temp_file}")

def main():
    """Main worker loop to listen for tasks."""
    print("Worker started. Waiting for tasks...")
    while True:
        try:
            # Blocking pop from the task queue
            _, task_data = r.blpop('task_queue', timeout=30)
            if task_data:
                process_task(task_data)
            else:
                # No task in the queue, wait and check again
                time.sleep(1)
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
if __name__ == "__main__":
    main()

