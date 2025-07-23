# src/text_grid/structure_registry.py
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import json
import uuid
import os # Added for path logging

# Import directly from the top-level config.py
# Alias MONGO_BIOME_COLLECTION to COLLECTION_NAME as it's used that way below
from config import MONGO_URI, MONGO_DB_NAME, MONGO_BIOME_COLLECTION, USE_CELERY as COLLECTION_NAME, STRUCTURE_TYPES, GRID_DIMENSIONS, USE_CELERY

logger = logging.getLogger(__name__)

# Initialize MongoDB client - these are module-level variables
client = None
db = None
structure_collection = None

# Added for debugging: Log the path of the loaded file
logger.info(f"Loading structure_registry.py from: {os.path.abspath(__file__)}")

try:
    # Use the variables imported directly from config
    client = MongoClient(MONGO_URI, uuidRepresentation='standard')
    db = client[MONGO_DB_NAME] # Use MONGO_DB_NAME directly
    structure_collection = db[COLLECTION_NAME] # Use COLLECTION_NAME directly
    logger.info("MongoDB connection established successfully in structure_registry.")
    db.command('ping') 
    logger.info("MongoDB ping successful in structure_registry.")
except ConnectionFailure as e:
    logger.error(f"MongoDB connection failed in structure_registry: {e}")
    client = None
    db = None
    structure_collection = None
except OperationFailure as e:
    logger.error(f"MongoDB authentication failed or operation error in structure_registry: {e}")
    client = None
    db = None
    structure_collection = None
except Exception as e:
    logger.error(f"An unexpected error occurred during MongoDB initialization in structure_registry: {e}")
    client = None
    db = None
    structure_collection = None


def get_next_structure_id():
    """
    Retrieves the next available ID for a new structure.
    """
    if structure_collection is not None: 
        try:
            pipeline = [
                {"$unwind": "$possible_structures.buildings"},
                {"$project": {"_id": 0, "struct_id": {"$toInt": "$possible_structures.buildings._id"}}},
                {"$group": {"_id": None, "max_id": {"$max": "$struct_id"}}}
            ]
            result = list(structure_collection.aggregate(pipeline))
            if result and result[0]['max_id'] is not None:
                return result[0]['max_id'] + 1
            else:
                return 101 
        except Exception as e:
            logger.warning(f"Could not query max structure ID from database: {e}. Defaulting to 101.")
            return 101
    return 101 # Default fallback if DB is not connected or initialized

def register_new_structures(new_structures: dict, theme_prompt: str, biome_name: str, layout: list):
    """
    Registers new structure definitions and associates them with a biome in the database.
    This function's role is mostly to facilitate saving the *full biome document*.
    """
    if structure_collection is None:
        logger.error("Cannot register new structures: MongoDB not connected.")
        raise ConnectionFailure("MongoDB not connected. Cannot register structures.")
    logger.info(f"Structures for biome '{biome_name}' are part of the main biome document. No separate registration needed here.")


# THIS IS THE CRITICAL CHANGE FOR 'get_biome_names()' TAKES 0 ARGUMENTS ERROR
def get_biome_names(db_name: str, collection_name: str): 
    """
    Retrieves a list of all existing biome names from the database.
    """
    # Added for debugging: Log call arguments
    logger.info(f"get_biome_names called with db_name='{db_name}', collection_name='{collection_name}'")

    if structure_collection is None:
        logger.error("Cannot fetch biome names: MongoDB not connected within structure_registry.")
        return []
    try:
        collection = db[MONGO_BIOME_COLLECTION]
        names = collection.distinct("biome_name")
        return sorted(list(names))
    except Exception as e:
        logger.error(f"Error fetching biome names: {e}")
        return []

# THIS IS THE CRITICAL CHANGE FOR 'fetch_biome()' IF IT HAD A SIMILAR ERROR
def fetch_biome(db_name: str, collection_name: str, name: str): 
    """
    Fetches a specific biome document by its name.
    """
    # Added for debugging: Log call arguments
    logger.info(f"fetch_biome called with db_name='{db_name}', collection_name='{collection_name}', name='{name}'")

    if structure_collection is None:
        logger.error("Cannot fetch biome: MongoDB not connected within structure_registry.")
        return None
    try:
        collection = db[MONGO_BIOME_COLLECTION]
        biome = collection.find_one({"biome_name": name})
        if biome:
            if '_id' in biome:
                biome['_id'] = str(biome['_id'])
            if "possible_grids" in biome and biome["possible_grids"]:
                for grid_info in biome["possible_grids"]:
                    if "layout" in grid_info and isinstance(grid_info["layout"], list):
                        grid_info["layout"] = "\n".join([f"[{', '.join(map(str, (cell for cell in row)))}]" for row in grid_info["layout"]])
            return biome
        return None
    except Exception as e:
        logger.error(f"Error fetching biome '{name}': {e}")
        return None

def insert_biome(biome_document: dict):
    """
    Inserts a complete biome document into the database.
    """
    if structure_collection is None:
        logger.error("Cannot insert biome: MongoDB not connected.")
        raise ConnectionFailure("MongoDB not connected. Cannot insert biome.")
    try:
        result = structure_collection.insert_one(biome_document)
        logger.info(f"Biome '{biome_document.get('biome_name', 'Unnamed')}' inserted with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        logger.error(f"Error inserting biome document: {e}")
        raise
