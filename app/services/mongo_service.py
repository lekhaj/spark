# ========================
# File: mongo_service.py
# ========================
from app.config import db
from pymongo.errors import ConnectionFailure, OperationFailure

def ping_db():
    """Pings the database to check the connection."""
    try:
        db.command("ping")
        return True
    except ConnectionFailure:
        return False

def get_data(collection_name: str, limit: int = 5):
    """Retrieves data from a specified collection with a limit."""
    collection = db[collection_name]
    return list(collection.find({}, {"_id": 0}).limit(limit))

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
    try:
        biomes_collection = db["biomes"] # Assuming your collection is named "biomes"
        
        # Use dot notation to update the nested fields atomically.
        update_fields = {
            f"{section_name}.status": new_status
        }
        
        # Check if the section should have a URL or list of images
        if section_name == "image_generation_details":
            update_fields[f"{section_name}.generated_images"] = [link] # Use a list for images
        else:
            update_fields[f"{section_name}.model_url"] = link # Use a string for URLs

        result = biomes_collection.update_one(
            {"_id": task_id},
            {"$set": update_fields}
        )

        # After updating, check if the overall biome status should be updated
        if result.modified_count > 0 and new_status == "COMPLETED":
            check_all_sections_and_update_main_status(task_id)

        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating nested status for task {task_id}: {e}")
        return False

def check_all_sections_and_update_main_status(task_id: str):
    """
    Checks the status of all nested sections and updates the main biome status
    to "completed" only if all sections are completed with a valid link/image.
    """
    try:
        biomes_collection = db["biomes"]
        biome_document = biomes_collection.find_one({"_id": task_id})

        if not biome_document:
            return

        # Define all required sections to check
        sections_to_check = [
            biome_document.get("image_generation_details", {}),
            biome_document.get("3d_generation_details", {}),
            biome_document.get("decimation_details", {})
        ]
        
        all_sections_completed = True
        for section in sections_to_check:
            status = section.get("status")
            
            # For the image section, check the generated_images array
            if "generated_images" in section:
                link_field = section.get("generated_images")
                if status != "COMPLETED" or not (link_field and len(link_field) > 0):
                    all_sections_completed = False
                    break
            # For 3D and decimation, check the model_url string
            elif "model_url" in section:
                link_field = section.get("model_url")
                if status != "COMPLETED" or not link_field:
                    all_sections_completed = False
                    break
            else:
                # If a section is missing or has an unexpected structure, assume not completed
                all_sections_completed = False
                break

        # If all sections are confirmed complete, update the main status
        if all_sections_completed:
            biomes_collection.update_one(
                {"_id": task_id},
                {"$set": {"status": "completed"}}
            )
            print(f"Biome {task_id} is now complete!")

    except Exception as e:
        print(f"Error checking and updating main status for task {task_id}: {e}")

def get_task_by_id(task_id: str):
    """
    Retrieves a single task document by its ID.
    
    Args:
        task_id (str): The ID of the task to retrieve.
        
    Returns:
        dict: The task document, or None if not found.
    """
    try:
        tasks_collection = db["tasks"]
        task = tasks_collection.find_one({"_id": task_id}, {"_id": 0})
        return task
    except Exception as e:
        print(f"Error retrieving task by ID: {e}")
        return None

