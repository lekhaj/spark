#!/usr/bin/env python3
"""
Debug script for MongoDB image functionality.
Tests MongoDB connection and image path queries.
"""

import sys
import os
sys.path.append('/home/ubuntu/Shashwat/spark/src')

def test_mongodb_connection():
    """Test MongoDB connection and image query"""
    print("🧪 Testing MongoDB Image Fetching")
    print("=" * 50)
    
    try:
        from db_helper import MongoDBHelper
        
        # Initialize MongoDB connection
        print("🔗 Connecting to MongoDB...")
        mongo_helper = MongoDBHelper()
        
        # Test database listing
        print("📋 Available databases:")
        databases = mongo_helper.list_databases()
        for db in databases:
            print(f"  - {db}")
        
        # Test with the available database
        db_name = "World_builder"  # Use the available database
        collection_name = "biomes"
        
        print(f"\n🔍 Checking collection '{collection_name}' in database '{db_name}'...")
        
        # Check if collection exists
        collections = mongo_helper.list_collections(db_name)
        print(f"📂 Available collections in {db_name}:")
        for coll in collections:
            print(f"  - {coll}")
        
        if collection_name not in collections:
            print(f"⚠️  Collection '{collection_name}' not found. Trying other collections...")
            if collections:
                collection_name = collections[0]
                print(f"🔄 Using collection '{collection_name}' instead")
            else:
                print("❌ No collections found in database")
                return False
        
        # Query for documents with image_path
        print(f"\n🖼️  Querying for documents with 'image_path' field...")
        query = {"image_path": {"$exists": True, "$ne": None, "$ne": ""}}
        documents = mongo_helper.find_many(db_name, collection_name, query, limit=5)
        
        print(f"📊 Found {len(documents)} documents with image_path")
        
        for i, doc in enumerate(documents):
            doc_id = str(doc.get("_id", "Unknown"))
            image_path = doc.get("image_path", "No path")
            name = doc.get("name", doc.get("theme", doc.get("prompt", "Unnamed")))
            
            print(f"\n  Document {i+1}:")
            print(f"    ID: {doc_id}")
            print(f"    Name/Theme: {name}")
            print(f"    Image Path: {image_path}")
            print(f"    Valid URL: {isinstance(image_path, str) and image_path.startswith('http')}")
        
        print(f"\n✅ MongoDB connection and query test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during MongoDB test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mongodb_connection()
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")
    sys.exit(0 if success else 1)
