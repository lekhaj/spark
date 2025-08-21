import os
import io
import boto3
from celery import Celery
import json
import time
from PIL import Image
import numpy as np
import trimesh
import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from io import BytesIO
import base64
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime
from uuid import uuid4
from bson.objectid import ObjectId
from diffusers import StableDiffusionPipeline


# --- Configuration from Environment Variables ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1") # Defaulting to us-east-1 if not set

# --- Redis Configuration for Celery ---
redis_ip = "172.31.8.113"
celery_app = Celery("app", broker=f"redis://{redis_ip}:6379/0", backend=f"redis://{redis_ip}:6379/0")

# --- MongoDB Configuration ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://sagar:KrSiDnSI9m8RgcHE@ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "World_builder")

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB_NAME]
    mongo_client.admin.command('ping') # Test the connection
    print(f"Connected to MongoDB database: {MONGO_DB_NAME}")
except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    mongo_client = None
    mongo_db = None
except Exception as e:
    print(f"An unexpected error occurred during MongoDB connection: {e}")
    mongo_client = None
    mongo_db = None

# --- S3 Client Initialization ---
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    print("S3 client initialized.")
except Exception as e:
    print(f"S3 client failed to initialize: {e}")
    s3_client = None


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


# --- CELERY TASKS ---

@celery_app.task(name="app.generate_2d_image_task")
def generate_2d_image_task(doc_id, theme_prompt, width=512, height=512, num_images=1):
    """
    Generates a 2D image using Stable Diffusion, uploads it to S3, and updates the MongoDB document.
    """
    try:
        if mongo_db is None:
            raise ConnectionError("MongoDB connection is not established.")
        if s3_client is None:
            raise ConnectionError("S3 client is not initialized.")
            
        pipe = get_stable_diffusion_pipeline()
        if pipe is None:
            raise ValueError("Stable Diffusion model failed to load.")

        print(f"Generating 2D image for document: {doc_id} with prompt: '{theme_prompt}'")
        images = pipe(theme_prompt, width=width, height=height, num_images_per_prompt=num_images).images

        image_urls = []
        for i, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            s3_filename = f"images/{doc_id}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, "sparkassets", s3_filename)
            print(f"Uploaded {s3_filename} to S3.")
            
            image_url = f"s3://sparkassets/{s3_filename}"
            image_urls.append(image_url)

        # Update the MongoDB document with the successful result
        update_data = {
            "status": "COMPLETED",
            "prompt": theme_prompt,
            "model_used": "runwayml/stable-diffusion-v1-5",
            "generated_images": image_urls,
            "timestamp": time.time()
        }
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"image_generation_details": update_data}}
        )
        print(f"Successfully updated document {doc_id} with 2D generation results.")
        return {"status": "success", "document_id": doc_id, "image_urls": image_urls}

    except Exception as e:
        print(f"An error occurred during 2D image generation for doc {doc_id}: {e}")
        # Update MongoDB with the failure status
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"image_generation_details.status": "FAILED", "image_generation_details.error": str(e)}}
        )
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.generate_3d_from_2d_task")
def generate_3d_from_2d_task(doc_id):
    """
    Generates a 3D asset from a 2D image, uploads it to S3, and updates the MongoDB document.
    """
    try:
        if mongo_db is None:
            raise ConnectionError("MongoDB connection is not established.")
        if s3_client is None:
            raise ConnectionError("S3 client is not initialized.")

        # Fetch the MongoDB document to get the 2D image URL
        doc = mongo_db.biomes.find_one({"_id": ObjectId(doc_id)})
        if not doc or not doc.get("image_generation_details", {}).get("generated_images"):
            raise ValueError("Document not found or 2D image path is missing.")
            
        image_url = doc["image_generation_details"]["generated_images"][0]
        s3_key = image_url.replace("s3://sparkassets/", "")
        
        # Download the image from S3
        response = s3_client.get_object(Bucket="sparkassets", Key=s3_key)
        image_bytes = response['Body'].read()
        image_2d_input = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Get the Hunyuan pipeline
        hunyuan_pipeline = get_hunyuan_pipeline()
        if hunyuan_pipeline is None:
            raise ValueError("Hunyuan model failed to load.")

        print(f"Generating 3D model from 2D image for document: {doc_id}")
        with torch.no_grad():
            mesh = hunyuan_pipeline(
                image=image_2d_input,
                num_inference_steps=30,
                octree_resolution=256,
                generator=torch.Generator(device=hunyuan_pipeline.device).manual_seed(42)
            )[0]
        
        model_bytes = mesh.export(file_type='glb')
        
        # Define the S3 path and upload the model
        s3_filename = f"3d_assets/{doc_id}.glb"
        s3_client.put_object(Bucket="sparkassets", Key=s3_filename, Body=model_bytes)
        
        print(f"Uploaded generated 3D asset to {s3_filename}")
        s3_url = f"s3://sparkassets/{s3_filename}"

        # Update the MongoDB document with the 3D asset details
        update_data = {
            "status": "COMPLETED",
            "model_url": s3_url,
            "timestamp": time.time()
        }
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"3d_generation_details": update_data}}
        )
        print(f"Successfully updated document {doc_id} with 3D generation results.")
        return {"status": "success", "document_id": doc_id, "model_url": s3_url}

    except Exception as e:
        print(f"An error occurred during 3D generation for doc {doc_id}: {e}")
        # Update MongoDB with the failure status
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"3d_generation_details.status": "FAILED", "3d_generation_details.error": str(e)}}
        )
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.decimate_3d_task")
def decimate_3d_task(doc_id, file_path, file_bytes):
    """
    Decimates a 3D asset and updates the MongoDB document.
    Note: file_path and file_bytes are passed directly here for demonstration,
    in a production system, you would pass the S3 key.
    """
    try:
        if mongo_db is None:
            raise ConnectionError("MongoDB connection is not established.")
        if s3_client is None:
            raise ConnectionError("S3 client is not initialized.")
            
        print(f"Decimating 3D model for document: {doc_id}")
        
        # Load the 3D model from bytes
        mesh = trimesh.load(io.BytesIO(file_bytes), file_type=file_path.split('.')[-1])
        
        # Perform decimation (e.g., reduce to 10% of the original faces)
        num_faces_to_keep = int(len(mesh.faces) * 0.1)
        decimated_mesh = mesh.simplify_quadric_decimation(num_faces_to_keep)
        
        # Export the decimated mesh to bytes in GLB format
        decimated_model_bytes = decimated_mesh.export(file_type='glb')
        
        # Define the S3 path and upload the decimated model
        s3_filename = f"decimated_assets/{doc_id}.glb"
        s3_client.put_object(Bucket="sparkassets", Key=s3_filename, Body=decimated_model_bytes)
        
        print(f"Uploaded decimated 3D asset to {s3_filename}")
        s3_url = f"s3://sparkassets/{s3_filename}"

        # Update the MongoDB document with the decimated asset details
        update_data = {
            "status": "COMPLETED",
            "model_url": s3_url,
            "timestamp": time.time()
        }
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"decimation_details": update_data}}
        )
        print(f"Successfully updated document {doc_id} with decimation results.")
        return {"status": "success", "document_id": doc_id, "model_url": s3_url}

    except Exception as e:
        print(f"An error occurred during decimation for doc {doc_id}: {e}")
        # Update MongoDB with the failure status
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"decimation_details.status": "FAILED", "decimation_details.error": str(e)}}
        )
        return {"status": "error", "message": str(e)}


@celery_app.task(name="app.generate_image_from_grid_task")
def generate_image_from_grid_task(doc_id, grid_data_str, width=512, height=512, num_images=1):
    """
    Generates an image from a 3D grid, uploads it to S3, and updates the MongoDB document.
    """
    try:
        if mongo_db is None:
            raise ConnectionError("MongoDB connection is not established.")
        if s3_client is None:
            raise ConnectionError("S3 client is not initialized.")

        grid_data = json.loads(grid_data_str)
        # This is a placeholder for your actual grid-to-image conversion logic.
        # Here, it simply creates a white image based on the dimensions.
        print(f"Generating image from grid for document: {doc_id} with data: {grid_data_str}")
        img = Image.new('RGB', (width, height), color='white')
        
        image_urls = []
        for i in range(num_images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            s3_filename = f"images/{doc_id}_grid_viz_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, "sparkassets", s3_filename)
            print(f"Uploaded {s3_filename} to S3.")
            image_urls.append(f"s3://sparkassets/{s3_filename}")

        # Update the MongoDB document
        update_data = {
            "status": "COMPLETED",
            "grid_data": grid_data,
            "model_used": "Simulated Grid Model",
            "generated_images": image_urls,
            "timestamp": time.time()
        }
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"grid_generation_details": update_data}}
        )
        print(f"Successfully updated document {doc_id} with grid visualization results.")
        return {"status": "success", "document_id": doc_id, "image_urls": image_urls}

    except Exception as e:
        print(f"An error occurred during grid visualization for doc {doc_id}: {e}")
        # Update MongoDB with the failure status
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"grid_generation_details.status": "FAILED", "grid_generation_details.error": str(e)}}
        )
        return {"status": "error", "message": str(e)}


