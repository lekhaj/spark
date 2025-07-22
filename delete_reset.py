#!/usr/bin/env python3
import boto3
import pymongo
from urllib.parse import urlparse
from bson import ObjectId

# ——— CONFIG ———
MONGO_URI      = "mongodb://ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017"
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
    query["possible_structures.buildings"] = {"$exists": True}

    for doc in coll.find(query):
        biome_id = doc["_id"]
        buildings = doc.get("possible_structures", {}).get("buildings", {})
        print(f"\nBiome {biome_id}: found {len(buildings)} buildings")

        for bkey, bld in buildings.items():
            urls_to_delete = []

            model3dUrl = bld.get("model3dUrl")
            if model3dUrl:
                urls_to_delete.append(model3dUrl)

            decimatedUrl = bld.get("decimated_3d_asset")
            if decimatedUrl:
                urls_to_delete.append(decimatedUrl)

            if not urls_to_delete:
                print(f" - Building {bkey}: No 3D URLs to delete")
                continue

            print(f" - Building {bkey}: Deleting {len(urls_to_delete)} S3 assets")

            for url in urls_to_delete:
                try:
                    delete_s3_object(url)
                except Exception as e:
                    print(f"   ! Failed to delete {url}: {e}")

            # Unset all 3D fields in MongoDB, reset status
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

            print(f"   ✓ Reset MongoDB fields for building {bkey}")

if __name__ == "__main__":
    main()
