import torch
import os
import gc
import logging
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from datetime import datetime
import sys # Import sys module
import shutil # Import shutil for directory removal

# Ensure the current script's directory is in sys.path for local module imports
# This is a more robust way to ensure s3_manager and db_helper are found
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import S3Manager and MongoDBHelper from the provided files
try:
    from s3_manager import S3Manager, get_s3_manager
    from db_helper import MongoDBHelper
    STORAGE_AVAILABLE = True
    print("S3Manager and MongoDBHelper imported successfully.")
except ImportError as e:
    print(f"Warning: Could not import storage modules: {e}. Storage functionality will be skipped.")
    STORAGE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Memory Optimization Helper Function ---
def clear_memory():
    """Clears GPU cache and runs Python garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache.")
    gc.collect()
    logger.info("Ran Python garbage collection.")
    # For good measure, also try to empty the PyTorch cache if using MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache.")


# Step 1: Setup paths
output_dir = "outputs/prompt_to_3d"
os.makedirs(output_dir, exist_ok=True)

# Step 2: Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️ Using device: {device}")

# Step 3: Define your prompt
prompt = "A futuristic red sports car with glowing lights in a showroom"
image_filename = "generated_image.png"
mesh_filename = "generated_model.ply"
image_path = os.path.join(output_dir, image_filename)
mesh_path = os.path.join(output_dir, mesh_filename)

# Step 3.1: Initialize S3 and MongoDB managers
s3_manager = None
db_helper = None
db_name = "World_builder"
image_collection_name = "generated_images"
model_collection_name = "generated_models" # Could be same as images, or separate

if STORAGE_AVAILABLE:
    try:
        # Using the provided connection string for MongoDB
        mongo_connection_string = "mongodb://sagar:KrSiDnSI9m8RgcHE@ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin"
        db_helper = MongoDBHelper(connection_string=mongo_connection_string)
        db_helper.client.admin.command('ping') # Test connection
        logger.info("MongoDB connected successfully.")

        s3_manager = get_s3_manager()
        if s3_manager:
            logger.info("S3 Manager initialized successfully.")
        else:
            logger.warning("S3 Manager could not be initialized. S3 uploads will be skipped.")
            s3_manager = None # Ensure it's None if initialization failed

    except Exception as e:
        logger.error(f"Failed to initialize storage services: {e}. Storage functionality will be skipped.")
        s3_manager = None
        db_helper = None


# Variable to store MongoDB document ID for updating later
mongo_doc_id = None

# Step 4: Generate image from prompt using SDXL Turbo
logger.info("🎨 Generating image from prompt...")
sdxl_pipe = None
try:
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    image = sdxl_pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
                      height=256, width=256).images[0]
    image.save(image_path)
    logger.info(f"✅ Image saved locally at: {os.path.abspath(image_path)}")

    # Step 4.1: Upload image to S3 and store metadata in MongoDB
    if s3_manager and db_helper:
        logger.info("Uploading generated image to S3...")
        s3_upload_result = s3_manager.upload_image(image_path, image_type="prompt_generated")
        if s3_upload_result["status"] == "success":
            logger.info(f"Image uploaded to S3: {s3_upload_result['s3_url']}")
            
            # Store image metadata in MongoDB
            image_metadata = {
                "prompt": prompt,
                "image_local_path": os.path.abspath(image_path),
                "image_s3_key": s3_upload_result.get("s3_key"),
                "image_s3_url": s3_upload_result.get("s3_url"),
                "generated_at": datetime.now().isoformat(),
                "status": "2d_generated",
                "3d_model_s3_key": None,
                "3d_model_s3_url": None
            }
            mongo_doc_id = db_helper.insert_one(db_name, image_collection_name, image_metadata)
            logger.info(f"Image metadata stored in MongoDB with ID: {mongo_doc_id}")
        else:
            logger.error(f"Failed to upload image to S3: {s3_upload_result.get('message')}")
    elif STORAGE_AVAILABLE:
        logger.warning("S3 Manager or MongoDB Helper not initialized. Skipping image upload/metadata storage.")

except Exception as e:
    logger.error(f"Error generating 2D image: {e}")
    sdxl_pipe = None
finally:
    if sdxl_pipe is not None:
        del sdxl_pipe
    clear_memory()

# Only proceed if 2D image generation was successful and local image exists
if os.path.exists(image_path):
    input_image_for_3d = load_image(image_path).convert("RGB")

    # Step 6: Generate 3D model using Hunyuan3D
    logger.info("🧱 Generating 3D mesh from image...")
    hunyuan_pipe = None
    try:
        # Before loading, explicitly clear the cache for tencent/Hunyuan3D-2mini
        # This prevents it from loading the wrong subfolder from a previous cache
        hunyuan_mini_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--tencent--Hunyuan3D-2mini")
        if os.path.exists(hunyuan_mini_cache_path):
            logger.info(f"Clearing Hugging Face cache for {hunyuan_mini_cache_path}...")
            shutil.rmtree(hunyuan_mini_cache_path)
            logger.info("Hugging Face cache cleared for Hunyuan3D-2mini.")

        # CORRECTED: Change the main model ID to "tencent/Hunyuan3D-2mini"
        # and re-add the subfolder argument as per its Hugging Face usage example.
        hunyuan_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini", # CORRECTED MODEL ID
            subfolder="hunyuan3d-dit-v2-mini", # RE-ADDED SUBFOLDER for the mini model
            torch_dtype=torch.float16
        ).to(device)

        output = hunyuan_pipe(input_image_for_3d, output_type="mesh")
        output["mesh"].export(mesh_path)

        logger.info(f"✅ 3D model saved locally at: {os.path.abspath(mesh_path)}")

        # Step 6.1: Upload 3D model to S3 and update metadata in MongoDB
        if s3_manager and db_helper and mongo_doc_id:
            logger.info("Uploading generated 3D model to S3...")
            # Use the base name of the 2D image for consistent naming in S3
            source_image_base_name = os.path.splitext(image_filename)[0]
            s3_upload_result_3d = s3_manager.upload_3d_asset(mesh_path, asset_type="generated_model",
                                                             source_image_name=source_image_base_name)
            if s3_upload_result_3d["status"] == "success":
                logger.info(f"3D model uploaded to S3: {s3_upload_result_3d.get('s3_url') or s3_upload_result_3d.get('main_s3_url')}")
                
                # Update MongoDB document with 3D model S3 details
                from bson.objectid import ObjectId
                update_query = {"_id": ObjectId(mongo_doc_id)}
                update_data = {
                    "$set": {
                        "3d_model_local_path": os.path.abspath(mesh_path),
                        "3d_model_s3_key": s3_upload_result_3d.get("s3_key") or (s3_upload_result_3d.get("uploads")[0].get("s3_key") if s3_upload_result_3d.get("uploads") else None),
                        "3d_model_s3_url": s3_upload_result_3d.get("s3_url") or s3_upload_result_3d.get("main_s3_url"),
                        "status": "3d_generated"
                    }
                }
                modified_count = db_helper.update_one(db_name, image_collection_name, update_query, update_data)
                if modified_count > 0:
                    logger.info(f"MongoDB document {mongo_doc_id} updated with 3D model details.")
                else:
                    logger.warning(f"Failed to update MongoDB document {mongo_doc_id} with 3D model details.")
            else:
                logger.error(f"Failed to upload 3D model to S3: {s3_upload_result_3d.get('message')}")
        elif STORAGE_AVAILABLE:
            logger.warning("S3 Manager, MongoDB Helper, or initial MongoDB ID not available. Skipping 3D model upload/metadata update.")

    except Exception as e:
        logger.error(f"Error generating 3D mesh: {e}")
    finally:
        if hunyuan_pipe is not None:
            del hunyuan_pipe
        clear_memory()
else:
    logger.info("Skipping 3D generation as 2D image was not successfully created.")

logger.info("\nProcess finished.")

# Close MongoDB connection
if db_helper:
    db_helper.close()
    logger.info("MongoDB connection closed.")
