# app.py - Celery Worker
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import os
import boto3
import logging
from botocore.exceptions import ClientError
import trimesh
import io
from celery import Celery, Task

# Configure Celery
# IMPORTANT: Replace '172.31.8.113' with the actual PRIVATE IP of your Redis instance.
redis_ip = "172.31.8.113"
celery_app = Celery('app', broker=f"redis://{redis_ip}:6379/0", backend=f"redis://{redis_ip}:6379/0")

logging.basicConfig(level=logging.INFO)

# --- Utility Functions ---
# Note: These are helper functions, not Celery tasks.

def upload_file_to_s3(file_data, bucket, object_name):
    """
    Uploads file data to an S3 bucket from a BytesIO object.
    Returns the public URL of the file.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_fileobj(file_data, bucket, object_name)
        logging.info(f"Successfully uploaded to s3://{bucket}/{object_name}")
        return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    except ClientError as e:
        logging.error(f"Failed to upload to S3: {e}")
        return None

def decimate_mesh(mesh_data, target_faces=1000):
    """
    Decimates a 3D mesh from a file-like object.
    """
    print(f"\n--- Starting Decimation Step ---")
    print(f"Decimating mesh to approximately {target_faces} faces...")
    
    mesh = trimesh.load(file_obj=io.BytesIO(mesh_data), file_type='glb')
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            print("Warning: Scene contains no geometry. Decimation skipped.")
            return None
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    decimated_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    print(f"Original faces: {len(mesh.faces)}, Decimated faces: {len(decimated_mesh.faces)}")
    
    decimated_data = io.BytesIO()
    decimated_mesh.export(file_obj=decimated_data, file_type='glb')
    decimated_data.seek(0)
    return decimated_data

# --- Celery Tasks ---

@celery_app.task(bind=True)
def generate_2d_image_task(self, text_prompt: str, s3_bucket_name: str, base_filename: str):
    """
    Celery task to generate a 2D image from a text prompt.
    Returns the image data as bytes to be passed to the next task in the chain.
    """
    self.update_state(state='IN_PROGRESS_2D', meta={'status': "Generating 2D image..."})

    # Try to use GPU, fallback to CPU
    try:
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    except Exception:
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")

    image = sd_pipe(text_prompt, num_inference_steps=25).images[0]
    
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    image_data.seek(0)
    
    # Upload the 2D image to S3 and get its public URL
    s3_filename = f"images/{base_filename}.png"
    image_url = upload_file_to_s3(image_data, s3_bucket_name, s3_filename)
    
    # Update state with the image URL, which can be retrieved by the frontend
    self.update_state(state='IN_PROGRESS_3D', meta={'status': "Image generation complete!", 'result': {'image_url': image_url}})
    return image_data.getvalue() # Return the bytes for the next task

@celery_app.task(bind=True)
def generate_3d_from_2d_task(self, image_bytes: bytes, s3_bucket_name: str, base_filename: str):
    """
    Celery task to generate a 3D model from a 2D image.
    Takes image data (bytes) as input from the previous task.
    """
    self.update_state(state='IN_PROGRESS_3D', meta={'status': "Generating 3D model from 2D image..."})
    
    image_2d_input = Image.open(io.BytesIO(image_bytes))
    
    hunyuan_model_id = 'tencent/Hunyuan3D-2mini'
    hunyuan_subfolder = 'hunyuan3d-dit-v2-mini'
    
    try:
        hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            hunyuan_model_id,
            subfolder=hunyuan_subfolder,
            use_safetensors=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception:
        hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(hunyuan_model_id, subfolder=hunyuan_subfolder, use_safetensors=True).to("cpu")
    
    with torch.no_grad():
        mesh = hunyuan_pipeline(image=image_2d_input, num_inference_steps=30)[0]
    
    mesh_data = io.BytesIO()
    mesh.export(file_obj=mesh_data, file_type='glb')
    mesh_data.seek(0)
    
    s3_filename = f"3d_assets/{base_filename}.glb"
    s3_url = upload_file_to_s3(mesh_data, s3_bucket_name, s3_filename)

    self.update_state(state='IN_PROGRESS_DECIMATION', meta={'status': "3D model generation complete!", 'result': {'model_url': s3_url}})
    return mesh_data.getvalue() # Return the bytes for the next task

@celery_app.task(bind=True)
def decimate_3d_task(self, input_3d_bytes: bytes, s3_bucket_name: str, base_filename: str):
    """
    Celery task to decimate a 3D model.
    Takes 3D model data (bytes) as input from the previous task.
    """
    self.update_state(state='IN_PROGRESS_DECIMATION', meta={'status': "Decimating 3D model..."})
    
    try:
        decimated_mesh_data = decimate_mesh(input_3d_bytes, target_faces=1000)
        
        if decimated_mesh_data is not None:
            s3_filename = f"processed/{base_filename}_decimated.glb"
            decimated_url = upload_file_to_s3(decimated_mesh_data, s3_bucket_name, s3_filename)
            
            if decimated_url:
                self.update_state(state='SUCCESS', meta={'status': "Decimation complete!", 'result': {'decimated_url': decimated_url}})
                return {"decimated_url": decimated_url}
            else:
                raise Exception("Failed to upload decimated 3D model to S3.")
        else:
            raise Exception("Decimation failed: No valid mesh produced.")
            
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
