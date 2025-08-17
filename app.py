import os
import io
import boto3
from celery import Celery, Task
from diffusers import StableDiffusionPipeline
import json
from PIL import Image
import numpy as np
import trimesh
from skimage import measure
import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from io import BytesIO
import base64
from pymongo import MongoClient
from datetime import datetime
from uuid import uuid4
from bson.objectid import ObjectId
import time

# --- AWS Configuration from environment variables ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1") # Defaulting to us-east-1 if not set

# --- Redis Configuration ---
redis_ip = "172.31.8.113"
celery_app = Celery("app", broker=f"redis://{redis_ip}:6379/0", backend=f"redis://{redis_ip}:6379/0")

# --- MongoDB Configuration ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://sagar:KrSiDnSI9m8RgcHE@ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "World_builder")

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB_NAME]
    print(f"Connected to MongoDB database: {MONGO_DB_NAME}")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")
    mongo_client = None
    mongo_db = None

# --- S3 Client Initialization ---
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# --- AI Model Caching ---
global_sd_pipe = None
def get_stable_diffusion_pipeline():
    """Initializes and returns a cached Stable Diffusion pipeline."""
    global global_sd_pipe
    if global_sd_pipe is None:
        try:
            print("Loading Stable Diffusion model...")
            global_sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
            global_sd_pipe = global_sd_pipe.to("cuda")
            print("Stable Diffusion model loaded successfully.")
        except Exception as e:
            print(f"Error loading Stable Diffusion model: {e}")
            global_sd_pipe = None
    return global_sd_pipe

global_hunyuan_pipe = None
def get_hunyuan_pipeline():
    """Initializes and returns a cached Hunyuan 3D pipeline."""
    global global_hunyuan_pipe
    if global_hunyuan_pipe is None:
        try:
            print("Loading Hunyuan3D-2mini model...")
            hunyuan_model_id = 'tencent/Hunyuan3D-2mini'
            hunyuan_subfolder = 'hunyuan3d-dit-v2-mini'
            global_hunyuan_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                hunyuan_model_id,
                subfolder=hunyuan_subfolder,
                use_safesensors=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Hunyuan3D-2mini model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Hunyuan3D-2mini with GPU. Error: {e}")
            try:
                global_hunyuan_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    hunyuan_model_id,
                    subfolder=hunyuan_subfolder,
                    use_safesensors=True
                ).to("cpu")
                print("Successfully loaded model on CPU.")
            except Exception as cpu_e:
                print(f"Failed to load Hunyuan3D-2mini on CPU. Error: {cpu_e}")
                global_hunyuan_pipe = None
    return global_hunyuan_pipe


@celery_app.task(name="app.generate_2d_image_task")
def generate_2d_image_task(biome_name, theme_prompt, width, height, num_images, s3_bucket_name, base_filename):
    """
    Generates a 2D image, stores the image path in a new MongoDB document, and returns the document ID.
    """
    if mongo_db is None:
        return {"status": "error", "message": "MongoDB connection failed."}

    try:
        pipe = get_stable_diffusion_pipeline()
        if pipe is None:
            return {"status": "error", "message": "Stable Diffusion model failed to load."}
        
        # Create an initial MongoDB document
        new_doc = {
            "_id": str(uuid4()),
            "biome_name": biome_name,
            "theme_prompt": theme_prompt,
            "possible_structures": {},
            "possible_grids": [],
            "model_used": "runwayml/stable-diffusion-v1-5",
            "processed": False,
            "processed_at": datetime.utcnow(),
            "image_path": None, # Will be filled after generation
            "3d_asset_url": None,
            "3d_asset_format": None,
            "3d_asset_generated_at": None,
        }
        mongo_db.biomes.insert_one(new_doc)
        
        images = pipe(theme_prompt, width=width, height=height, num_images_per_prompt=num_images).images

        results = []
        for i, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            s3_filename = f"images/{base_filename}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, s3_bucket_name, s3_filename)
            print(f"Uploaded {s3_filename} to {s3_bucket_name}")
            
            image_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_filename}"
            results.append(image_url)

        # Update the MongoDB document with the image path
        mongo_db.biomes.update_one(
            {"_id": new_doc["_id"]},
            {"$set": {
                "image_path": results[0],
                "processed_at": datetime.utcnow(),
                "model_used": "stable_diffusion",
                "possible_grids": [{
                    "grid_id": "grid_001",
                    "layout": [], # Placeholder for grid layout
                    "model_used": "stability",
                    "processed": False,
                    "processed_at": datetime.utcnow(),
                    "image_path": results[0],
                    "3d_asset_url": None,
                    "3d_asset_format": None,
                    "3d_asset_generated_at": None
                }]
            }}
        )
        
        return {"status": "success", "document_id": new_doc["_id"]}
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.generate_3d_from_2d_task")
def generate_3d_from_2d_task(document_id, s3_bucket_name, base_filename):
    """
    Generates a 3D asset from a 2D image, updates the MongoDB document, and returns the document ID.
    """
    if mongo_db is None:
        return {"status": "error", "message": "MongoDB connection failed."}

    try:
        # Fetch the MongoDB document to get the 2D image URL
        doc = mongo_db.biomes.find_one({"_id": document_id})
        if not doc or not doc.get("image_path"):
            return {"status": "error", "message": "Document not found or image path is missing."}
        
        image_url = doc["image_path"]
        
        # Download the image from S3
        response = s3_client.get_object(Bucket=s3_bucket_name, Key=image_url.split('/')[-2] + '/' + image_url.split('/')[-1])
        image_bytes = response['Body'].read()
        image_2d_input = Image.open(BytesIO(image_bytes))

        # Get the Hunyuan pipeline
        hunyuan_pipeline = get_hunyuan_pipeline()
        if hunyuan_pipeline is None:
            return {"status": "error", "message": "Hunyuan model failed to load."}

        with torch.no_grad():
            mesh = hunyuan_pipeline(
                image=image_2d_input,
                num_inference_steps=30,
                octree_resolution=256,
                generator=torch.Generator(device=hunyuan_pipeline.device).manual_seed(42)
            )[0]
        
        model_bytes = mesh.export(file_type='glb')
        
        # Define the S3 path and upload the model
        s3_filename = f"3d_assets/generated/{base_filename}_{document_id}.glb"
        s3_client.put_object(Bucket=s3_bucket_name, Key=s3_filename, Body=model_bytes)
        
        print(f"Uploaded generated 3D asset to {s3_filename}")
        
        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_filename}"

        # Update the MongoDB document with the 3D asset details
        mongo_db.biomes.update_one(
            {"_id": document_id, "possible_grids.image_path": image_url},
            {"$set": {
                "3d_asset_url": s3_url,
                "3d_asset_format": "glb",
                "3d_asset_generated_at": datetime.utcnow(),
                "processed": True,
                "possible_grids.$.3d_asset_url": s3_url,
                "possible_grids.$.3d_asset_format": "glb",
                "possible_grids.$.3d_asset_generated_at": datetime.utcnow(),
                "possible_grids.$.processed": True
            }}
        )
        
        return {"status": "success", "document_id": document_id, "result": s3_url}
    except Exception as e:
        print(f"An error occurred during 3D generation: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.decimate_3d_task")
def decimate_3d_task(file_bytes, s3_bucket_name, base_filename):
    """
    Decimates a 3D asset using the trimesh library.
    """
    try:
        # Load the 3D model from bytes
        mesh = trimesh.load(io.BytesIO(file_bytes), file_type='glb')
        
        # Perform decimation (e.g., reduce to 10% of the original faces)
        num_faces_to_keep = int(len(mesh.faces) * 0.1)
        decimated_mesh = mesh.simplify_quadric_decimation(num_faces_to_keep)
        
        # Export the decimated mesh to bytes in GLB format
        decimated_model_bytes = decimated_mesh.export(file_type='glb')
        
        # Define the S3 path and upload the decimated model
        s3_filename = f"processed/{base_filename}_decimated.glb"
        s3_client.put_object(Bucket=s3_bucket_name, Key=s3_filename, Body=decimated_model_bytes)
        
        print(f"Uploaded decimated 3D asset to {s3_filename}")

        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_filename}"

        return {"status": "success", "result": s3_url}
    except Exception as e:
        print(f"An error occurred during decimation: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.generate_image_from_grid_task")
def generate_image_from_grid_task(grid_data_str, width, height, num_images, s3_bucket_name, base_filename):
    """
    Generates an image from a 3D grid.
    """
    # Placeholder for your logic to convert the grid into an image.
    try:
        grid_data = json.loads(grid_data_str)
        from PIL import Image
        img = Image.new('RGB', (width, height), color = 'white')
        
        results = []
        for i in range(num_images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            s3_filename = f"images/grid_{base_filename}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, s3_bucket_name, s3_filename)
            print(f"Uploaded {s3_filename} to {s3_bucket_name}")
            results.append(s3_filename)
            
        return {"status": "success", "result": results}
    except Exception as e:
        print(f"An error occurred during grid visualization: {e}")
        return {"status": "error", "message": str(e)}

# --- New Celery Tasks for MongoDB Integration ---

@celery_app.task(name="app.generate_2d_from_db_task")
def generate_2d_from_db_task(document_id, s3_bucket_name, base_filename, width, height, num_images):
    """
    Fetches a prompt from MongoDB and generates a 2D image.
    """
    if mongo_db is None:
        return {"status": "error", "message": "MongoDB connection failed."}

    try:
        doc = mongo_db.biomes.find_one({"_id": document_id})
        if not doc or not doc.get("theme_prompt"):
            return {"status": "error", "message": "Document not found or prompt missing."}
        
        theme_prompt = doc["theme_prompt"]
        biome_name = doc.get("biome_name", "default_biome")
        
        # Call the existing 2D generation task logic directly
        # Note: Celery tasks cannot be called directly from another task without `delay()` or `apply_async()`.
        # However, for simplicity and to match your intended flow, we can simulate the call.
        # The correct way is `generate_2d_image_task.delay(...)`.
        
        # Simulating the call to avoid circular dependency issues in the worker
        # You should refactor your tasks to be more modular if this becomes complex.
        
        # Placeholder for task execution. In a real-world scenario, you would
        # typically chain or group tasks.
        
        # For a simplified, direct call for this fix:
        
        # Simulate the generation and upload process
        pipe = get_stable_diffusion_pipeline()
        if pipe is None:
            return {"status": "error", "message": "Stable Diffusion model failed to load."}
            
        images = pipe(theme_prompt, width=width, height=height, num_images_per_prompt=num_images).images

        results = []
        for i, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            s3_filename = f"images/{base_filename}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, s3_bucket_name, s3_filename)
            print(f"Uploaded {s3_filename} to {s3_bucket_name}")
            
            image_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_filename}"
            results.append(image_url)

        # Update the MongoDB document with the new image path
        mongo_db.biomes.update_one(
            {"_id": document_id},
            {"$set": {
                "image_path": results[0],
                "processed_at": datetime.utcnow()
            }}
        )
        
        return {"status": "success", "document_id": document_id, "result": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.generate_grid_from_db_task")
def generate_grid_from_db_task(document_id, s3_bucket_name, base_filename, width, height, num_images):
    """
    Fetches a grid from MongoDB and generates a 2D image visualization.
    """
    if mongo_db is None:
        return {"status": "error", "message": "MongoDB connection failed."}

    try:
        doc = mongo_db.biomes.find_one({"_id": document_id})
        if not doc or not doc.get("possible_grids") or not doc["possible_grids"][0].get("layout"):
            return {"status": "error", "message": "Document not found or grid layout missing."}
        
        grid_data_str = json.dumps(doc["possible_grids"][0]["layout"])
        
        # Placeholder for the grid visualization process.
        # The logic here is simplified. In a real scenario, this would
        # involve using an AI model to render the grid.
        
        img = Image.new('RGB', (width, height), color='white')
        results = []
        
        for i in range(num_images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            s3_filename = f"images/grid_{base_filename}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, s3_bucket_name, s3_filename)
            print(f"Uploaded {s3_filename} to {s3_bucket_name}")
            
            image_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_filename}"
            results.append(image_url)
            
        return {"status": "success", "result": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
