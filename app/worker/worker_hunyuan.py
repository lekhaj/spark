# ===========================
# File: worker_hunyuan.py
# ===========================
import os
import json
import redis
import boto3
import requests
from rembg import remove
from PIL import Image
import trimesh
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from mongo_service import update_nested_status, check_and_update_main_status

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "15.206.99.66")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
MODEL_QUEUE = "model_tasks"

# AWS S3 config
BUCKET_NAME = os.getenv("BUCKET_NAME", "sparkassets")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")
s3 = boto3.client("s3", region_name=S3_REGION)

# Redis client
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Global model caches
shape_pipeline = None
paint_pipeline = None

def download_from_s3_url(s3_url, local_path):
    response = requests.get(s3_url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

def upload_to_s3(local_file, s3_key):
    s3.upload_file(local_file, BUCKET_NAME, s3_key)
    return f"https://{BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"

def remove_bg(input_path, output_path):
    with Image.open(input_path) as img:
        img_no_bg = remove(img)
        img_no_bg.save(output_path)
    return output_path

def process_3d_task(task):
    """
    Main function to process a 3D generation task.
    This function will be triggered by a message queue.
    """
    global shape_pipeline, paint_pipeline
    job_id = task.get("job_id")
    prompt = task.get("prompt")
    image_url = task.get("image_url")
    
    if not job_id or not prompt or not image_url:
        print("Invalid task data received.")
        return {"status": "failed", "error": "Invalid task data"}

    try:
        # Load pipelines once
        if shape_pipeline is None:
            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("hpcai-tech/Hunyuan3D-Shape-Generator")
            shape_pipeline.to(torch.device("cuda"))

        if paint_pipeline is None:
            paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("hpcai-tech/Hunyuan3D-Paint-Generator")
            paint_pipeline.to(torch.device("cuda"))

        # 1. Download image and remove background
        print(f"[GPU] Downloading image for job {job_id}...")
        local_input_img = f"/tmp/{job_id}_input.png"
        local_input_nobg = f"/tmp/{job_id}_nobg.png"
        download_from_s3_url(image_url, local_input_img)
        remove_bg(local_input_img, local_input_nobg)

        # 2. Generate 3D shape
        print(f"[GPU] Generating 3D shape for prompt: '{prompt}'...")
        mesh = shape_pipeline(prompt)
        shape_file = "mesh_output.glb"
        trimesh.exchange.export.export_mesh(mesh, shape_file, file_type='glb')
        print(f"[GPU] Shape generated and saved: {shape_file}")

        # 3. Paint texture
        print("[GPU] Painting texture...")
        textured_mesh = paint_pipeline(mesh, image=local_input_nobg)
        textured_file = "painted_output.glb"
        textured_mesh.export(textured_file)
        print(f"[GPU] Textured mesh saved: {textured_file}")

        # 4. Upload both to S3
        s3_key_untextured = f"3d_assets/{job_id}_mesh.glb"
        s3_key_textured = f"3d_assets/{job_id}_painted.glb"

        url_untextured = upload_to_s3(shape_file, s3_key_untextured)
        url_textured = upload_to_s3(textured_file, s3_key_textured)
        
        # Update MongoDB with textured model URL
        update_success = update_nested_status(
            task_id=job_id,
            section_name="3d_generation_details",
            new_status="COMPLETED",
            link=url_textured
        )
        
        if update_success:
            # Check if the main document status can be updated
            check_and_update_main_status(job_id)

        print(f"[GPU] Uploaded untextured: {url_untextured}")
        print(f"[GPU] Uploaded textured: {url_textured}")
        
        # Cleanup
        os.remove(local_input_img)
        os.remove(local_input_nobg)
        os.remove(shape_file)
        os.remove(textured_file)

        return {"status": "completed", "mesh_url": url_untextured, "painted_url": url_textured}

    except Exception as e:
        print(f"Error processing 3D task {job_id}: {e}")
        return {"status": "failed", "error": str(e)}

def main():
    print(f"[GPU] Worker started. Listening on queue: {MODEL_QUEUE}...")
    while True:
        task_data = r.brpop(MODEL_QUEUE, timeout=30)
        if not task_data:
            print("[GPU] No new tasks. Exiting...")
            break

        _, raw_task = task_data
        task = json.loads(raw_task)

        try:
            print(f"[GPU] job_id={task['job_id']} → status='processing'")
            process_3d_task(task)

        except Exception as e:
            print(f"An error occurred while processing task {task.get('job_id')}: {e}")

if __name__ == "__main__":
    main()