import logging
from datetime import datetime
from bson import ObjectId
# from asyncssh import logger
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError
from bson import json_util
try:
    from app.config import settings
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from app.config import settings

class DBConnection:
    client: MongoClient | None = None
    db: Database | None = None

logger_db = logging.getLogger("app.MongoService")
db_connection = DBConnection()
def ping_db():
    if db_connection.db is not None:
        return db_connection.db.command("ping")
    return {"error": "Database not connected"}

def biome_assets_for_task(biome_id: str, status_filter: str = "not complete"):
    """
    Returns a dict of asset objects (buildings, creatures, props, terrain, etc.) for a biome,
    filtered by status (e.g., 'not complete').
    Each key is the asset name, value is a dict with description, name, status, and other info.
    """
    try:
        biome = get_biome(biome_id)
        if not biome or not isinstance(biome, dict):
            return {"error": "Biome not found"}

        # The possible_structures field contains asset categories
        possible_structures = biome.get("possible_structures", {})
        result = {}
        for category in ["buildings", "creatures", "props", "terrain"]:
            assets = possible_structures.get(category, {})
            for asset_name, asset in assets.items():
                # If asset is a dict and has a status field
                if isinstance(asset, dict) and asset.get("status") == status_filter:
                    # Try to get 3D S3 URL and image S3 URL from several possible key names
                        # possible keys for image URL (top-level or within attributes)
                        image_keys = ["image_s3_url", "image_url", "s3_image_url", "s3_image_uri"]
                        s3_3d_keys = ["s3_3d_url", "s3_3d_uri", "3d_s3_url", "s3_model_url"]

                        image_url = None
                        s3_3d_url = None

                        # check top-level keys first
                        for k in image_keys:
                            if k in asset and asset.get(k):
                                image_url = asset.get(k)
                                break
                        for k in s3_3d_keys:
                            if k in asset and asset.get(k):
                                s3_3d_url = asset.get(k)
                                break

                        # then check inside attributes dict if present
                        attrs = asset.get("attributes") if isinstance(asset.get("attributes"), dict) else {}
                        if not image_url:
                            for k in image_keys:
                                if k in attrs and attrs.get(k):
                                    image_url = attrs.get(k)
                                    break
                        if not s3_3d_url:
                            for k in s3_3d_keys:
                                if k in attrs and attrs.get(k):
                                    s3_3d_url = attrs.get(k)
                                    break

                        # Fallback: if still no image_url but attributes contains 'image_url' or 's3_image_url' as nested
                        if not image_url and isinstance(asset.get("attributes"), dict):
                            image_url = asset["attributes"].get("image_url") or asset["attributes"].get("s3_image_url")

                        result[asset_name] = {
                            "name": asset_name,
                            "description": asset.get("description", ""),
                            "status": asset.get("status", ""),
                            "type": asset.get("type", ""),
                            "background_prompt": asset.get("background_prompt", ""),
                            "attributes": asset.get("attributes", {}),
                            "id": asset.get("id", None),
                            "s3_3d_url": s3_3d_url,
                            # provide both fields for compatibility: 'image_url' (legacy) and 'image_s3_url' (preferred)
                            "image_url": image_url,
                            "image_s3_url": image_url,
                        }
        return result
    except Exception as e:
        return {"error": str(e)}
    
def get_db() -> Database | None:
    if db_connection.db is not None:
        return db_connection.db
    try:
        db_connection.client = MongoClient(settings.MONGODB_URL)
        db_connection.client.admin.command('ismaster')
        db_connection.db = db_connection.client[settings.MONGODB_DB_NAME]
        logger_db.info("Database connection established successfully.")
        return db_connection.db

    except PyMongoError as e:
            logger_db.critical(f"CRITICAL: Failed to connect to MongoDB: {e}")
            db_connection.client = None
            db_connection.db = None
            return None

def close_mongo_connection():
    """Closes the connection if it exists."""
    if db_connection.client is not None:
        logger_db.info("Closing MongoDB connection for this process...")
        db_connection.client.close()
        db_connection.client = None
        db_connection.db = None
        logger_db.info("MongoDB connection closed.")

def fetch_recent():
    """Returns a sort order to fetch the most recent documents first."""
    return [("timestamp", -1)]


def get_biome(biome_id: str):
    """Retrieves the source biome document from the 'biomes' collection by _id (primary key). Handles ObjectId and string types."""
    db = get_db()
    if db is None:
        logger_db.error("Database connection is not available.")
        return None
    
    biome_collection = db["biomes"]
    from bson import ObjectId
    try:
        doc = biome_collection.find_one({"_id": ObjectId(biome_id)})
    except Exception:
        doc = biome_collection.find_one({"_id": biome_id})
    if doc is not None:
        return json_util.loads(json_util.dumps(doc))
    # Debug: print all _id values if not found
    all__ids = [str(d.get("_id")) for d in biome_collection.find({}, {"_id": 1})]
    print(f"[DEBUG] No biome found for _id='{biome_id}'. Available _ids: {all__ids}")
    return None

def get_data(collection_name: str, limit: int = 5):
    db = get_db()
    if db is None:
        logger_db.error("Database connection is not available.")
        return []
    collection = db[collection_name]
    cursor = collection.find({}).limit(limit)
    docs = list(cursor)
    # Convert list of BSON docs to JSON-serializable list
    return json_util.loads(json_util.dumps(docs))

def get_data_by_id(collection_name: str, _id: str):
    db = get_db()
    if db is None:
        logger_db.error("Database connection is not available.")
        return None
    collection = db[collection_name]
    doc = collection.find_one({"_id": _id})
    if doc is not None:
        return json_util.loads(json_util.dumps(doc))
    return "biome not found"

def update_or_add_biome_asset(biome_id: str, update_key: str, update_dict: dict):
    """
    Update or add to a nested asset dict in a biome document using a dynamic update key (e.g., 'possible_structures.buildings.market_stall')
    and a dict of attributes to set/merge at that path.
    """
    db = get_db()
    if db is None:
        logger_db.error("Database connection is not available.")
        return {"error": "Database not connected"}
    biome_collection = db["biomes"]
    from bson import ObjectId
    # Try ObjectId, fallback to string
    try:
        filter_ = {"_id": ObjectId(biome_id)}
    except Exception:
        filter_ = {"_id": biome_id}

    # Fetch the current value at the update_key path
    current_doc = biome_collection.find_one(filter_, {update_key: 1})
    # Traverse the nested dict to get the current value
    def get_nested(d, path):
        for k in path:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return {}
        return d if isinstance(d, dict) else {}

    path = update_key.split('.')
    current_value = get_nested(current_doc or {}, path)
    # Merge current_value and update_dict
    merged = {**current_value, **update_dict}
    set_dict = {update_key: merged}
    result = biome_collection.update_one(filter_, {"$set": set_dict})
    if result.matched_count == 0:
        logger_db.error(f"No biome found for _id={biome_id} to update.")
        return {"error": "Biome not found"}
    return {"matched": result.matched_count, "modified": result.modified_count}
def get_biome_asset_update_key_by_job_id(biome_id: str, job_id: str) -> str | None:
    """
    Given a biome ID and a job_id, return the update key path (e.g., 'possible_structures.buildings.market_stall')
    for the asset with the matching 'id' attribute. Returns None if not found.
    """
    biome_data = get_biome(biome_id)
    possible_structures = biome_data.get("possible_structures", {})
    for category in ["buildings", "creatures", "props", "terrain"]:
        assets = possible_structures.get(category, {})
        for key, asset in assets.items():
            if isinstance(asset, dict) and str(asset.get("job_id")) == str(job_id):
                return f"possible_structures.{category}.{key}"
    return None

def update_biome_asset_by_name_or_job_id(biome_id: str, asset_name: str = None, job_id: str = None, update_dict: dict = None):
    """
    Update or add to a biome asset by either name or job_id.
    If asset_name is provided, uses name; if job_id is provided, uses job_id.
    """
    if not update_dict:
        return {"error": "No update_dict provided"}
    update_key = None
    if asset_name:
        update_key = get_biome_asset_update_key(biome_id, asset_name)
    elif job_id:
        update_key = get_biome_asset_update_key_by_job_id(biome_id, job_id)
    if not update_key:
        return {"error": "Asset not found by name or job_id"}
    return update_or_add_biome_asset(biome_id, update_key, update_dict)

def get_biome_asset_update_key(biome_id: str, asset_name: str) -> str | None:
    """
    Given a biome ID and an asset name, return the update key path (e.g., 'possible_structures.buildings.market_stall')
    for the asset with the matching key or 'name' attribute. Returns None if not found.
    """
    biome_data = get_biome(biome_id)
    possible_structures = biome_data.get("possible_structures", {})
    for category in ["buildings", "creatures", "props", "terrain"]:
        assets = possible_structures.get(category, {})
        for key, asset in assets.items():
            if key == asset_name or (isinstance(asset, dict) and asset.get("name") == asset_name):
                return f"possible_structures.{category}.{key}"
    return None

def get_biome_choices_live(database_name, collection_name):
    """
    Fetches all biome names and their IDs from a live MongoDB collection.
    Returns a list of tuples: [(biome_name, doc_id), ...].
    """
    db = get_db()
    if db is None or not database_name or not collection_name:
        return []
    try:
        collection = db[collection_name]
        documents = list(collection.find({}, {"biome_name": 1}))
        choices = [(doc.get("biome_name", "Unknown Biome"), str(doc["_id"])) for doc in documents]
        return choices
    except Exception as e:
        logger_db.error(f"Failed to fetch biomes from collection '{collection_name}': {e}")
        return []
    # Fix: Use sort properly, not as projection
    # biome = biome_collection.find_one({"_id": biome_id})
    # return serialize_mongo_doc(biome)

def get_data(collection_name: str, limit: int = 20):
    db = get_db()                     
    if db is None:
        logger_db.error("Database not connected.")
        return []
    
    collection = db[collection_name]
    # Get documents and serialize them
    documents = list(collection.find({}).limit(limit))
    return [json_util.loads(json_util.dumps(doc)) for doc in documents]

    
def get_assets_by_biome(biome_id: str):
    db = get_db()
    if db is None:
        logger_db.error("Database connection is not available.")
        return []
    
    assets_collection = db["assets"]
    assets_cursor = assets_collection.find({"_id": biome_id})
    assets_list = list(assets_cursor)  
    
    for asset in assets_list:
        if '_id' in asset:
            asset['_id'] = str(asset['_id'])
    
    return assets_list

from typing import List, Dict, Any

def extract_all_images_from_biome(biome_id: str) -> List[Dict[str, Any]]:
    """Extract all images from a specific biome"""
    db = get_db()
    if db is None:
        logger_db.error("Database connection is not available.")
        return []
    
    biome = db["biomes"].find_one({"_id": biome_id})
    if not biome:
        return []
    
    images = []
    image_fields = ['image_url', 's3_image_uri']
    
    def find_images(obj, category=""):
        if isinstance(obj, dict):
            # Check if this object has an image
            for field in image_fields:
                if field in obj and obj[field]:
                    images.append({
                        "url": obj[field],
                        "name": obj.get('description', 'Unnamed'),
                        "type": category,
                        "status": obj.get('status', 'unknown')
                    })
                    break
            
            # Search nested objects
            for key, value in obj.items():
                find_images(value, category or key)
                
        elif isinstance(obj, list):
            for item in obj:
                find_images(item, category)
    
    # Search through all structures
    structures = biome.get('possible_structures', {})
    for category, items in structures.items():
        if isinstance(items, dict):
            for name, details in items.items():
                find_images(details, category)
    
    return images