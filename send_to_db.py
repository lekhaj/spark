import json
import urllib.parse
from pymongo import MongoClient

# MongoDB Atlas credentials
DB_USER = "admin"
DB_PASSWORD = "xtYJX17zyaQlHIxS"
DB_HOST = "cluster0.erd0ikm.mongodb.net"
DB_NAME = "world_builder"

# URL encode the password and construct URI
encoded_pw = urllib.parse.quote_plus(DB_PASSWORD)
MONGO_URI = f"mongodb+srv://{DB_USER}:{encoded_pw}@{DB_HOST}/{DB_NAME}?retryWrites=true&w=majority"

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db["biomes"]

# Load JSON data
with open("biome_final.json", "r", encoding="utf-8") as f:
    biome_doc = json.load(f)

# Insert into MongoDB
result = collection.insert_one(biome_doc)
print(f"✅ Document inserted with ID: {result.inserted_id}")