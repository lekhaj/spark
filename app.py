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
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson.objectid import ObjectId
from diffusers import StableDiffusionPipeline

# --- Configuration from Environment Variables ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")  # default ap-south-1

# --- GPU Instance Config ---
GPU_INSTANCE_ID = "i-0e029990527fa2b73"   # 🔥 replace with your EC2 GPU instance ID
GPU_REGION = "ap-south-1"        # 🔥 replace with your region if different

# --- Redis Configuration for Celery ---
redis_ip = "172.31.8.113"
celery_app = Celery("app", broker=f"redis://{redis_ip}:6379/0", backend=f"redis://{redis_ip}:6379/0")

# --- MongoDB Configuration ---
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://sagar:KrSiDnSI9m8RgcHE@ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin"
)
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "World_builder")

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB_NAME]
    mongo_client.admin.command('ping')  # test connection
    print(f"Connected to MongoDB database: {MONGO_DB_NAME}")
except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
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


# --- GPU Instance Control ---
def start_gpu_instance():
    """Starts GPU EC2 instance if not already running."""
    ec2 = boto3.client("ec2", region_name=GPU_REGION)
    response = ec2.describe_instance_status(InstanceIds=[GPU_INSTANCE_ID])
    statuses = response.get("InstanceStatuses", [])

    if not statuses or statuses[0]["InstanceState"]["Name"] != "running":
        print("Starting GPU instance...")
        ec2.start_instances(InstanceIds=[GPU_INSTANCE_ID])
        waiter = ec2.get_waiter("instance_status_ok")
        waiter.wait(InstanceIds=[GPU_INSTANCE_ID])
        print("GPU instance is running now.")
    else:
        print("GPU instance already running.")


def stop_gpu_instance():
    """Stops GPU EC2 instance after task completion."""
    ec2 = boto3.client("ec2", region_name=GPU_REGION)
    print("Stopping GPU instance...")
    ec2.stop_instances(InstanceIds=[GPU_INSTANCE_ID])
    print("GPU instance stopped.")


# --- AI Model Caching ---
global_sd_pipe = None
def get_stable_diffusion_pipeline():
    global global_sd_pipe
    if global_sd_pipe is None:
        try:
            print("Loading Stable Diffusion model...")
            global_sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", use_safetensors=True
            )
        except Exception as e:
            print(f"Error loading Stable Diffusion model: {e}")
            return None

    # lazy device move
    if torch.cuda.is_available():
        global_sd_pipe = global_sd_pipe.to("cuda")
    else:
        global_sd_pipe = global_sd_pipe.to("cpu")
    return global_sd_pipe


global_hunyuan_pipe = None
def get_hunyuan_pipeline():
    global global_hunyuan_pipe
    if global_hunyuan_pipe is None:
        try:
            print("Loading Hunyuan3D-2mini model...")
            hunyuan_model_id = 'tencent/Hunyuan3D-2mini'
            hunyuan_subfolder = 'hunyuan3d-dit-v2-mini'
            global_hunyuan_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                hunyuan_model_id,
                subfolder=hunyuan_subfolder,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Hunyuan3D-2mini model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Hunyuan3D on GPU. Error: {e}")
            try:
                global_hunyuan_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    hunyuan_model_id,
                    subfolder=hunyuan_subfolder
                ).to("cpu")
                print("Loaded Hunyuan3D on CPU fallback.")
            except Exception as cpu_e:
                print(f"Failed to load Hunyuan3D on CPU. Error: {cpu_e}")
                global_hunyuan_pipe = None
    return global_hunyuan_pipe


# --- CELERY TASKS ---
@celery_app.task(name="app.generate_2d_image_task")
def generate_2d_image_task(doc_id, theme_prompt, width=512, height=512, num_images=1):
    try:
        start_gpu_instance()

        pipe = get_stable_diffusion_pipeline()
        if pipe is None:
            raise ValueError("Stable Diffusion model failed to load.")

        print(f"Generating 2D image for doc {doc_id} with prompt: '{theme_prompt}'")
        images = pipe(theme_prompt, width=width, height=height, num_images_per_prompt=num_images).images

        image_urls = []
        for i, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            s3_filename = f"images/{doc_id}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, "sparkassets", s3_filename)
            image_urls.append(f"s3://sparkassets/{s3_filename}")

        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"image_generation_details": {
                "status": "COMPLETED",
                "prompt": theme_prompt,
                "model_used": "runwayml/stable-diffusion-v1-5",
                "generated_images": image_urls,
                "timestamp": time.time()
            }}}
        )
        return {"status": "success", "document_id": doc_id, "image_urls": image_urls}

    except Exception as e:
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"image_generation_details.status": "FAILED", "image_generation_details.error": str(e)}}
        )
        return {"status": "error", "message": str(e)}
    finally:
        stop_gpu_instance()


@celery_app.task(name="app.generate_3d_from_2d_task")
def generate_3d_from_2d_task(doc_id):
    try:
        start_gpu_instance()

        doc = mongo_db.biomes.find_one({"_id": ObjectId(doc_id)})
        if not doc or not doc.get("image_generation_details", {}).get("generated_images"):
            raise ValueError("Document not found or 2D image missing.")

        image_url = doc["image_generation_details"]["generated_images"][0]
        s3_key = image_url.replace("s3://sparkassets/", "")
        response = s3_client.get_object(Bucket="sparkassets", Key=s3_key)
        image_2d_input = Image.open(BytesIO(response['Body'].read())).convert("RGB")

        hunyuan_pipeline = get_hunyuan_pipeline()
        if hunyuan_pipeline is None:
            raise ValueError("Hunyuan model failed to load.")

        with torch.no_grad():
            mesh = hunyuan_pipeline(image=image_2d_input, num_inference_steps=30, octree_resolution=256)[0]

        model_bytes = mesh.export(file_type='glb')
        s3_filename = f"3d_assets/{doc_id}.glb"
        s3_client.put_object(Bucket="sparkassets", Key=s3_filename, Body=model_bytes)
        s3_url = f"s3://sparkassets/{s3_filename}"

        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"3d_generation_details": {
                "status": "COMPLETED",
                "model_url": s3_url,
                "timestamp": time.time()
            }}}
        )
        return {"status": "success", "document_id": doc_id, "model_url": s3_url}

    except Exception as e:
        mongo_db.biomes.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"3d_generation_details.status": "FAILED", "3d_generation_details.error": str(e)}}
        )
        return {"status": "error", "message": str(e)}
    finally:
        stop_gpu_instance()



