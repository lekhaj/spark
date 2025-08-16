from pdb import pm
import motor.motor_asyncio
from src.config import MONGO_URL, MONGO_DB_NAME
import asyncio
import pymongo as pm
import pymongo_schema as ps

async def test_connection():
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        db = client[MONGO_DB_NAME]
        # Try to list collections as a test
        collections = await db.list_collection_names()
        print(f"Connected to DB. Collections: {collections}")
        return True
    except Exception as e:
        print(f"DB connection failed: {e}")
        return False
    finally:
        client.close()
async def get_schema():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    schema = {}
    try:
        collections = await client[MONGO_DB_NAME].list_collection_names()
        for collection in collections:
            schema[collection] = await client[MONGO_DB_NAME][collection].find_one()
    except Exception as e:
        print(f"Error fetching schema: {e}")
    finally:
        client.close()
    return schema

async def actual_schema():
    client = pm.MongoClient(MONGO_URL)
    db = client[MONGO_DB_NAME]
    schema = ps.extract.database_schema(db)
    print(schema)

if __name__ == "__main__":
    # asyncio.run(test_connection())
    # schema = asyncio.run(get_schema())
    # if schema:
    #     print("Schema fetched successfully:")
    #     for collection, document in schema.items():
    #         print(f" - {collection}: {document}")
    asyncio.run(actual_schema())