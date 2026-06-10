import nltk
nltk.download('punkt', quiet=True, download_dir='/home/ubuntu/nltk_data')
nltk.download('punkt_tab', quiet=True, download_dir='/home/ubuntu/nltk_data')
from text_grid import llm, utils
from text_grid.grid_placement_logic import generate_grid_from_hints
from datetime import datetime
import json

from text_grid.placement_suggestor import get_biome_generation_hints
from text_grid import structure_registry  # Use the robust MongoDB integration

theme_prompt = "A densely packed, multi-layered cyberpunk city with towering skyscrapers, neon signs, and hidden alleyways."
structure_types = [
    "MegaCorp Tower", "Neon Arcade", "Data Hub", "Slum Dwelling",
    "Skybridge", "Rooftop Garden", "Underground Market"
]

# import requests

# def call_llm_server(prompt):
#     response = requests.post("http://localhost:8000/generate/", json={"prompt": prompt})
#     return response.json()["result"]

# # Usage:
# llm_output = call_llm_server(your_prompt)

# 1. Call the LLM for structure definitions
llm_output = llm.call_llm_for_structure_definitions(theme_prompt, structure_types)
print("LLM Output (Structure Definitions):\n", llm_output)

if not llm_output:
    print("LLM did not return any output. Exiting.")
    exit(1)

print("[DEBUG] llm_output type:", type(llm_output))
print("[DEBUG] llm_output (first 500 chars):", str(llm_output)[:500])
# 2. Parse structure definitions
try:
    structured, type_to_id = utils.build_structure_definitions(llm_output, start_id=101)
    print("Parsed Structure Definitions:\n", json.dumps(structured, indent=2))
except Exception as e:
    print("[ERROR] Failed to parse structure definitions:", e)
    import traceback
    traceback.print_exc()
    structured, type_to_id = {"error": str(e)}, {}

print("[INFO] Requesting placement hints from LLM...")
try:
    placement_hints = get_biome_generation_hints(structured, theme_prompt)
    print("Parsed Placement Hints (LLM):\n", json.dumps(placement_hints, indent=2))
except Exception as e:
    print("[ERROR] Failed to get placement hints from LLM:", e)
    import traceback
    traceback.print_exc()
    placement_hints = {"error": str(e)}

# 6. Generate grid from hints (use generate_grid_from_hints)
try:
    grid = generate_grid_from_hints(placement_hints, structured)
    print("Generated Grid:\n", grid)
except Exception as e:
    print("[ERROR] Failed to generate grid:", e)
    import traceback
    traceback.print_exc()
    grid = []



# 7. Assemble biome JSON (matching MongoDB schema, sanitize booleans)

import uuid
from bson import ObjectId

def sanitize_bools(obj):
    if isinstance(obj, dict):
        return {k: sanitize_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_bools(v) for v in obj]
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    else:
        return obj


# Generate a new ObjectId for the document
doc_id = ObjectId()
biome_name = placement_hints.get("biome_name_suggestion") or utils.generate_biome_name_from_prompt(theme_prompt)
grid_id = f"{biome_name.replace(' ', '_')}_grid_{uuid.uuid4().hex[:8]}"

# Sanitize all attributes in structured and add structureId
def sanitize_structures(structs, doc_id):
    sanitized = {}
    for sid, struct in structs.items():
        struct_copy = dict(struct)
        if "attributes" in struct_copy:
            struct_copy["attributes"] = sanitize_bools(struct_copy["attributes"])
        if "adjacent_environmental_objects" in struct_copy:
            struct_copy["adjacent_environmental_objects"] = sanitize_bools(struct_copy["adjacent_environmental_objects"])
        # Add structureId field as <doc_id>_<sid>
        struct_copy["structureId"] = f"{str(doc_id)}_{sid}"
        # Add status field as "yet to start"
        struct_copy["status"] = "yet to start"
        sanitized[sid] = struct_copy
    return sanitized

sanitized_structured = sanitize_structures(structured, doc_id)

possible_structures = {
    "buildings": sanitized_structured
}

# Ensure grid is a serializable 2D list (not a numpy array or other type)
def ensure_serializable_grid(grid):
    if hasattr(grid, "tolist"):
        return grid.tolist()
    elif isinstance(grid, list):
        return [ensure_serializable_grid(row) for row in grid]
    else:
        return grid

possible_grids = [{
    "grid_id": grid_id,
    "layout": ensure_serializable_grid(grid)
}]

biome_data = {
    "_id": doc_id,
    "biome_name": biome_name,
    "theme_prompt": theme_prompt,
    "possible_structures": possible_structures,
    "possible_grids": possible_grids
}
print("Biome JSON to store:\n", json.dumps(biome_data, indent=2, default=str))

# 8. Store in MongoDB (test connection first, then insert)
from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB_NAME, MONGO_BIOME_COLLECTION

print("[INFO] Testing MongoDB connection...")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_BIOME_COLLECTION]
    # Try a ping
    client.admin.command('ping')
    print("[SUCCESS] Connected to MongoDB and pinged server.")
    # Try inserting the document directly
    result = collection.insert_one(biome_data)
    print("[SUCCESS] Inserted document with ID:", result.inserted_id)
except Exception as e:
    print("[ERROR] Could not connect to MongoDB or insert document:", e)