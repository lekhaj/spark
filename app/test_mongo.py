# ========================
# File: test_mongo.py
# This file should be placed in the root of the 'spark' directory,
# not inside the 'app' directory.
# ========================
import asyncio
from uuid import uuid4
from pymongo import MongoClient
import os
from app.config import db
from app.services.mongo_service import update_nested_status, check_all_sections_and_update_main_status, get_task_by_id

# Let's set up a test document to work with.
# This mimics the initial task document created by your FastAPI app.

async def run_test():
    """Runs a complete test of the MongoDB update functions."""
    print("Starting MongoDB update test...")

    # Create a unique ID for our test biome
    test_id = str(uuid4())
    print(f"Creating a test biome with ID: {test_id}")

    # Connect to MongoDB and create a test document
    try:
        biomes_collection = db["biomes"]
        biomes_collection.insert_one({
            "_id": test_id,
            "biome_name": "Test_Biome.png",
            "status": "PENDING",
            "image_generation_details": {
                "status": "PENDING",
                "prompt": "",
                "model_used": "Simulated AI Model",
                "generated_images": []
            },
            "3d_generation_details": {
                "status": "PENDING",
                "model_url": ""
            },
            "decimation_details": {
                "status": "PENDING",
                "model_url": ""
            },
            "timestamp": 123456789
        })
        print("Test biome created successfully.")

    except Exception as e:
        print(f"Error creating test biome: {e}")
        return

    # Simulate a successful image generation
    print("\nSimulating image generation...")
    image_link = f"https://my-s3-bucket.com/images/{test_id}.png"
    await update_nested_status(test_id, "image_generation_details", "COMPLETED", image_link)
    print("Image section updated.")

    # Simulate 3D model generation
    print("\nSimulating 3D model generation...")
    model_3d_link = f"https://my-s3-bucket.com/models/{test_id}.glb"
    await update_nested_status(test_id, "3d_generation_details", "COMPLETED", model_3d_link)
    print("3D model section updated.")

    # Simulate decimation process
    print("\nSimulating decimation...")
    decimation_link = f"https://my-s3-bucket.com/decimated/{test_id}_decimated.glb"
    await update_nested_status(test_id, "decimation_details", "COMPLETED", decimation_link)
    print("Decimation section updated.")

    # Final check of the document
    print("\nChecking final biome status...")
    final_doc = await get_task_by_id(test_id)
    print(f"Final document status: {final_doc['status']}")

    # Clean up the test document
    try:
        biomes_collection.delete_one({"_id": test_id})
        print(f"Test biome {test_id} deleted successfully.")
    except Exception as e:
        print(f"Error deleting test biome: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())

