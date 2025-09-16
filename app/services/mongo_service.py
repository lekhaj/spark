import logging
from asyncssh import logger
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError
from app.config import settings

class dbService:
    client = MongoClient | None = None
    db = Database | None = None

logger_db = logging.getLogger("app.MongoService")
db_connection = dbService()
def ping_db():
    if db_connection.db is not None:
        return db_connection.db.command("ping")
    return {"error": "Database not connected"}

def get_db():
    if db_connection.db is not None:
        return db_connection.db
    try:
        logger_db.info("Database connection not initialized. Initializing now...")
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
    return biome_collection.find_one({"biome_id": biome_id}, fetch_recent())

def get_data(collection_name: str, limit: int = 5):
    db_connection = get_db()
    collection = db_connection.db[collection_name]
    return list(collection.find({}, {"_id": 0}).limit(limit))