import logging
from datetime import datetime
from bson import ObjectId
from asyncssh import logger
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError
from app.config import settings

class dbService:
    client: MongoClient | None = None
    db: Database | None = None

logger_db = logging.getLogger("app.MongoService")
db_connection = dbService()

def serialize_mongo_doc(doc):
    """Convert MongoDB document to JSON-serializable format."""
    if doc is None:
        return None
    
    if isinstance(doc, dict):
        result = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = serialize_mongo_doc(value)
            elif isinstance(value, list):
                result[key] = [serialize_mongo_doc(item) if isinstance(item, (dict, ObjectId, datetime)) else item for item in value]
            else:
                result[key] = value
        return result
    elif isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, datetime):
        return doc.isoformat()
    else:
        return doc

def ping_db():
    if db_connection.db is not None:
        return db_connection.db.command("ping")
    return {"error": "Database not connected"}

def get_db():
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

def get_biome(biome_id: str) -> dict | None:
    """Retrieves the source biome document from the 'biomes' collection."""
    db = get_db()
    if db is None:
        logger_db.error("Database connection is not available.")
        return None
    
    biome_collection = db["biomes"]
    # Fix: Use sort properly, not as projection
    biome = biome_collection.find_one({"_id": biome_id})
    return serialize_mongo_doc(biome)

def get_data(collection_name: str, limit: int = 5):
    db = get_db()                     
    if db is None:
        logger_db.error("Database not connected.")
        return []
    
    collection = db[collection_name]
    # Get documents and serialize them
    documents = list(collection.find({}).limit(limit))
    return [serialize_mongo_doc(doc) for doc in documents]

    
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