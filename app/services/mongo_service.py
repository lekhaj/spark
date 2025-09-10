# ========================
# File: mongo_service.py
# ========================
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

# --- Configuration from environment variables ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "biomes")

# Global variables to hold the active database client
db_client = None

def get_db_client():
    """Establishes and returns a MongoDB client connection."""
    global db_client
    if db_client is None:
        try:
            db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db_client.admin.command('ping')
            print("Successfully connected to MongoDB.")
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
            db_client = None
    return db_client

def update_nested_status(task_id: str, section_name: str, new_status: str, link: str = None):
    """
    Updates the status and link for a specific nested section within a biome document.

    Args:
        task_id (str): The ID of the biome document to update.
        section_name (str): The name of the nested section (e.g., "3d_generation_details").
        new_status (str): The new status to set (e.g., "COMPLETED").
        link (str): The S3 link for the generated asset.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    client = get_db_client()
    if not client:
        return False

    try:
        biomes_collection = client[MONGO_DB][MONGO_COLLECTION]
        
        # Use dot notation to update the nested fields atomically.
        update_fields = {
            f"{section_name}.status": new_status
        }
        
        if link:
            # Check if the section should have a list or a single URL
            if section_name == "image_generation_details":
                update_fields[f"{section_name}.s3_links"] = [link]
            elif section_name == "3d_generation_details":
                update_fields[f"{section_name}.s3_link"] = link
            elif section_name == "decimated_assets_details":
                update_fields[f"{section_name}.s3_link"] = link
        
        result = biomes_collection.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": update_fields}
        )

        # After updating, check if the overall biome status should be updated
        if result.modified_count > 0 and new_status == "COMPLETED":
            check_all_sections_and_update_main_status(task_id)

        if result.modified_count > 0:
            print(f"Updated document {task_id}: {section_name} status to '{new_status}'")
            return True
        else:
            print(f"No document found or no change made for ID {task_id}.")
            return False

    except (OperationFailure, ConnectionFailure) as e:
        print(f"MongoDB update failed for task {task_id}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during update: {e}")
        return False

def check_all_sections_and_update_main_status(task_id: str):
    """
    Checks the status of all nested sections and updates the main biome status
    to "completed" only if all sections are completed with a valid link/image.
    """
    client = get_db_client()
    if not client:
        return

    try:
        biomes_collection = client[MONGO_DB][MONGO_COLLECTION]
        biome_document = biomes_collection.find_one({"_id": ObjectId(task_id)})

        if not biome_document:
            return

        all_sections_completed = True
        
        # Check image generation
        image_details = biome_document.get("image_generation_details", {})
        if image_details.get("status") != "COMPLETED" or not image_details.get("s3_links"):
            all_sections_completed = False

        # Check 3D generation
        if all_sections_completed:
            model_details = biome_document.get("3d_generation_details", {})
            if model_details.get("status") != "COMPLETED" or not model_details.get("s3_link"):
                all_sections_completed = False

        # Check decimation
        if all_sections_completed:
            decimation_details = biome_document.get("decimation_details", {})
            if decimation_details.get("status") != "COMPLETED" or not decimation_details.get("model_url"):
                all_sections_completed = False

        # If all sections are confirmed complete, update the main status
        if all_sections_completed:
            biomes_collection.update_one(
                {"_id": ObjectId(task_id)},
                {"$set": {"status": "completed"}}
            )
            print(f"Biome {task_id} is now complete!")

    except Exception as e:
        print(f"Error checking and updating main status for task {task_id}: {e}")
