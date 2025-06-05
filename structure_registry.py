import logging
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME, DB_USER, DB_PASSWORD, DB_HOST
import urllib.parse
from utils import generate_biome_name_from_prompt
structure_collection = ["structures"]

logging.basicConfig(filename="log.txt", level=logging.INFO)

encoded_pw = urllib.parse.quote_plus(DB_PASSWORD)
MONGO_URI = f"mongodb://{DB_USER}:{encoded_pw}@{DB_HOST}/{DB_NAME}?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
structure_collection = db["biomes"]



def insert_biome(doc):
    try:
        result = structure_collection.insert_one(doc)
        logging.info(f"[DB] Inserted biome: {doc['biome_name']} ID: {result.inserted_id}")
        print(f"✅ Biome '{doc['biome_name']}' generated and saved.")
    except Exception as e:
        logging.error(f"[DB] Failed to insert biome: {e}")
        print(f"[DB] ❌ Biome insertion failed: {e}")


def get_biome_names():
    return [doc["biome_name"] for doc in collection.find({}, {"biome_name": 1})]


def get_next_structure_id():
    all_ids = []
    for doc in collection.find({}, {"possible_structures": 1}):
        for cat in doc.get("possible_structures", {}).values():
            all_ids.extend(map(int, cat.keys()))
    return max(all_ids, default=0) + 1


def get_structures_by_theme(theme):
    result = {}
    for doc in collection.find({"theme_prompt": theme}, {"possible_structures": 1}):
        for cat, entries in doc.get("possible_structures", {}).items():
            if cat not in result:
                result[cat] = {}
            for sid, info in entries.items():
                result[cat][sid] = info
    return result
def fetch_biome(name):
    from bson.json_util import dumps
    doc = collection.find_one({"biome_name": name}, {"_id": 0})
    return doc if doc else None

def get_next_structure_id():
    max_id = 0
    for doc in structure_collection.find({}, {"_id": 0}):
        for cat in doc.get("structures", {}).values():
            for sid in cat.keys():
                max_id = max(max_id, int(sid))
    return max_id + 1

def get_matching_structures(theme):
    return list(structure_collection.find({"theme_prompt": theme}, {"_id": 0}))

def register_new_structures(structures, theme_prompt, biome_name,  layout):
    structure_doc = {
        "biome_name": biome_name,
        "theme_prompt": theme_prompt,
        "possible_structures": { "buildings": structures },
        "possible_grids": [{
        "grid_id": f"{biome_name.replace(' ', '_')}_grid_1",
        "layout": [[int(val) for val in row] for row in layout]

     }]
    }

    try:
        result = structure_collection.insert_one(structure_doc)
        logging.info(f"[DB] Inserted structures for '{theme_prompt}'. ID: {result.inserted_id}")
    except Exception as e:
        logging.error(f"[DB] Failed to insert structures: {e}")
        print(f"[DB] ❌ Insertion failed: {e}")


