
import pymongo
from config import MONGO_URI, MONGO_DB_NAME, MONGO_BIOME_COLLECTION
MONGO_URI = MONGO_URI
DB_NAME = MONGO_DB_NAME
COLLECTION_NAME = MONGO_BIOME_COLLECTION

client = pymongo.MongoClient(MONGO_URI)
col = client[DB_NAME][COLLECTION_NAME]

def update_structure_statuses():
    for doc in col.find({}):
        updated = False
        buildings = doc.get("possible_structures", {}).get("buildings", {})
        for key, struct in buildings.items():
            # 3D asset present
            if struct.get("asset_3d_url") or struct.get("asset_3d_format"):
                if struct.get("status") != "3D asset generated":
                    struct["status"] = "3D asset generated"
                    updated = True
            # Image present but not 3D
            elif struct.get("imageUrl") or struct.get("local_image_path"):
                if struct.get("status") != "Image generated":
                    struct["status"] = "Image generated"
                    updated = True
            # No image or 3D asset
            else:
                if struct.get("status") != "yet to start":
                    struct["status"] = "yet to start"
                    updated = True
        if updated:
            col.update_one({"_id": doc["_id"]}, {"$set": {"possible_structures.buildings": buildings}})
            print(f"Updated {doc.get('biome_name', doc['_id'])}")

import time
# ...existing code...

if __name__ == "__main__":
    while True:
        update_structure_statuses()
        print("Done. Sleeping for 60 seconds...")
        time.sleep(60)