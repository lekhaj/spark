def get_biome_choices_live(database_name, collection_name):
    """
    Fetches all biome names and their IDs from a live MongoDB collection.
    Returns a list of tuples: [(biome_name, doc_id), ...].
    """
    client = get_db_client()
    if not client or not database_name or not collection_name:
        return []
    try:
        db = client[database_name]
        collection = db[collection_name]
        documents = list(collection.find({}, {"biome_name": 1}))
        choices = [(doc.get("biome_name", "Unknown Biome"), str(doc["_id"])) for doc in documents]
        return choices
    except Exception as e:
        print(f"Failed to fetch biomes from collection '{collection_name}': {e}")
        return []

import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use the same variable names as the rest of the app
MONGO_URI = os.getenv("MONGODB_URL")
MONGO_DB = os.getenv("MONGODB_DB_NAME")

db_client = None

def get_db_client():
    global db_client
    if db_client is None:
        try:
            db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db_client.admin.command('ping')
            print("Successfully connected to MongoDB.")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            return None
    return db_client
