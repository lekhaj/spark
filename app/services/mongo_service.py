from app.config import db
from pymongo.errors import ConnectionFailure, OperationFailure

def ping_db():
    """Pings the database to check the connection."""
    return db.command("ping")

def get_data(collection_name: str, limit: int = 5):
    """Retrieves data from a specified collection with a limit."""
    collection = db[collection_name]
    return list(collection.find({}, {"_id": 0}).limit(limit))

def update_task_status(task_id: str, status: str, s3_link: str = None):
    """
    Updates the status and S3 link for a given task ID.
    
    Args:
        task_id (str): The ID of the task to update.
        status (str): The new status ('completed', 'failed', etc.).
        s3_link (str): The S3 link to the generated asset, if applicable.
        
    Returns:
        bool: True if the update was successful, False otherwise.
    """
    try:
        tasks_collection = db["tasks"] # Assuming a 'tasks' collection for tracking
        update_data = {"status": status}
        if s3_link:
            update_data["s3_link"] = s3_link
        
        result = tasks_collection.update_one(
            {"id": task_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating task status: {e}")
        return False

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
        task = tasks_collection.find_one({"id": task_id}, {"_id": 0})
        return task
    except Exception as e:
        print(f"Error retrieving task by ID: {e}")
        return None