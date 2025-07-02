#!/usr/bin/env python3
import boto3
import pymongo
from urllib.parse import urlparse
from bson import ObjectId

# ——— CONFIG ———
MONGO_URI      = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"
DB_NAME        = "World_builder"
COLLECTION     = "biomes"

# If you only want to process one biome, set its _id here (as a string).
# Otherwise set to None to process all biomes.
TARGET_BIOME_ID = None  # e.g. "68409a05d25a861ab725a324"

# ——— SETUP ———
mongo = pymongo.MongoClient(MONGO_URI)
coll  = mongo[DB_NAME][COLLECTION]
s3    = boto3.client('s3')

def delete_s3_object(s3_url):
    """Parse an S3 HTTPS URL and delete the corresponding object."""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc.split('.')[0]
    key    = parsed.path.lstrip('/')
    print(f" → Deleting s3://{bucket}/{key}")
    s3.delete_object(Bucket=bucket, Key=key)

def main():
    # Build the query
    query = {}
    if TARGET_BIOME_ID:
        query["_id"] = ObjectId(TARGET_BIOME_ID)
    # Only look at documents that have buildings at all
    query["possible_structures.buildings"] = {"$exists": True}

    for doc in coll.find(query):
        biome_id = doc["_id"]
        buildings = doc.get("possible_structures", {}).get("buildings", {})
        print(f"\nBiome {biome_id}: found {len(buildings)} buildings")

        for bkey, bld in buildings.items():
            url = bld.get("model3dUrl")
            if not url:
                continue

            print(f" - Building {bkey} has model3dUrl")

            # Delete from S3
            try:
                delete_s3_object(url)
            except Exception as e:
                print(f"   ! Failed to delete {url}: {e}")
                continue

            # Optionally: unset model3dUrl and decimated_3d_asset, and reset status
            coll.update_one(
                {"_id": biome_id},
                {
                    "$unset": {
                        f"possible_structures.buildings.{bkey}.model3dUrl": "",
                        f"possible_structures.buildings.{bkey}.decimated_3d_asset": "",
                        f"possible_structures.buildings.{bkey}.poly_before": "",
                        f"possible_structures.buildings.{bkey}.poly_after": ""
                    },
                    "$set": {
                        f"possible_structures.buildings.{bkey}.status": "3D asset generated"
                    }
                }
            )

            print(f"   ✓ Updated MongoDB building {bkey}")

if __name__ == "__main__":
    main()
