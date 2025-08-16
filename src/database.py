# src/database.py

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from . import config


def get_logger():
    return logging.getLogger("rpg_fast_api.database")
logger = get_logger()

_client = None
_db = None
_biome_collection = None

def _get_collection():
    """
    Internal function to lazily connect to the database on the first call
    and return the biome collection object.
    """
    global _client, _db, _biome_collection

    if _biome_collection is not None:
        return _biome_collection

    try:
        logger.info("First database access. Initializing MongoDB connection...")
        _client = MongoClient(
            config.MONGO_URL,
            uuidRepresentation='standard'
        )
        _client.admin.command('ping')
        _db = _client[config.MONGO_DB_NAME]
        _biome_collection = _db[config.MONGO_BIOME_COLLECTION]
        logger.info("MongoDB connection successful.")
        return _biome_collection
    except (ConnectionFailure, OperationFailure, ValueError) as e:
        logger.error(f"FATAL: Could not connect to MongoDB: {e}", exc_info=True)
        _biome_collection = None
        return None

def save_biome_document(biome_doc: dict) -> str | None:
    """
    Saves a complete biome document to the database.
    For testing, saves the document to a local file first, then attempts DB save.
    """
    import json
    try:
        biome_name = biome_doc.get("biome_name")
        if not biome_name:
            logger.error("Cannot save document: 'biome_name' is missing.")
            return None
        with open(f"biome_{biome_name}.json", "w", encoding="utf-8") as f:
            json.dump(biome_doc, f, ensure_ascii=False, indent=2)
        logger.info(f"Biome document saved locally to biome_{biome_name}.json for testing.")
    except Exception as e:
        logger.error(f"Failed to save biome document locally: {e}")

    collection = _get_collection()
    if collection is None:
        return None

    try:
        biome_name = biome_doc.get("biome_name")
        if not biome_name:
            logger.error("Cannot save document: 'biome_name' is missing.")
            return None
        
        result = collection.update_one(
            {"biome_name": biome_name},
            {"$set": biome_doc},
            upsert=True
        )
        
        if result.upserted_id:
            logger.info(f"Successfully inserted biome '{biome_name}' with ID: {result.upserted_id}")
            return str(result.upserted_id)
        else:
            logger.info(f"Successfully updated existing biome '{biome_name}'.")
            # For updates, we can confirm by checking modified_count
            return "updated" if result.modified_count > 0 else "no_change"
            
    except Exception as e:
        logger.error(f"Error saving biome document '{biome_doc.get('biome_name')}': {e}", exc_info=True)
        return None


def get_all_biome_names() -> list[str] | None:
    """
    Retrieves a sorted list of all unique biome names from the database.

    Returns:
        list[str] | None: A list of biome names on success, or None on failure.
    """
    collection = _get_collection()
    if not collection:
        return None
    
    try:
        names = collection.distinct("biome_name")
        return sorted(names)
    except Exception as e:
        logger.error(f"Error fetching all biome names: {e}", exc_info=True)
        return None


def get_biome_by_name(name: str) -> dict | None:
    """
    Fetches a single, complete biome document from the database by its name.

    Args:
        name (str): The name of the biome to retrieve.

    Returns:
        dict | None: The biome document if found, otherwise None.
    """
    collection = _get_collection()
    if not collection:
        return None
    
    try:
        biome_doc = collection.find_one({"biome_name": name})
        if biome_doc and '_id' in biome_doc:
            # Convert the BSON ObjectId to a string for JSON serialization
            biome_doc['_id'] = str(biome_doc['_id'])
        return biome_doc
    except Exception as e:
        logger.error(f"Error fetching biome by name '{name}': {e}", exc_info=True)
        return None