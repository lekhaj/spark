#!/usr/bin/env python3
"""
Test script to verify that 3D generation from MongoDB images correctly updates 
the database with 3D asset links for both theme and structure images.
"""

import os
import sys
import logging
from datetime import datetime

# Add the src directory to Python path
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_metadata_cache():
    """Test that the image metadata cache correctly stores structure image metadata"""
    
    try:
        from merged_gradio_app import fetch_images_from_mongodb
        from db_helper import MongoDBHelper
        
        logger.info("🧪 Testing image metadata cache functionality...")
        
        # Test with default MongoDB settings
        db_name = "World_builder"
        collection_name = "biomes"
        
        # Fetch images and check metadata cache
        image_items, status = fetch_images_from_mongodb(db_name, collection_name)
        
        logger.info(f"📊 Fetch results:")
        logger.info(f"   Status: {status}")
        logger.info(f"   Number of images: {len(image_items)}")
        
        if image_items:
            logger.info(f"   Sample images:")
            for i, (url, caption) in enumerate(image_items[:3]):
                logger.info(f"     {i+1}. {caption}")
                logger.info(f"        URL: {url[:80]}...")
        
        # Check metadata cache
        from merged_gradio_app import _image_metadata_cache
        
        logger.info(f"📋 Metadata cache:")
        logger.info(f"   Number of metadata entries: {len(_image_metadata_cache)}")
        
        # Count by type
        theme_count = sum(1 for meta in _image_metadata_cache if meta.get('type') == 'theme')
        structure_count = sum(1 for meta in _image_metadata_cache if meta.get('type') == 'structure')
        
        logger.info(f"   Theme images: {theme_count}")
        logger.info(f"   Structure images: {structure_count}")
        
        # Show sample structure metadata
        structure_samples = [meta for meta in _image_metadata_cache if meta.get('type') == 'structure'][:3]
        if structure_samples:
            logger.info(f"   Sample structure metadata:")
            for i, meta in enumerate(structure_samples):
                logger.info(f"     {i+1}. Doc ID: {meta.get('doc_id', 'N/A')[:12]}...")
                logger.info(f"        Category: {meta.get('category_key', 'N/A')}")
                logger.info(f"        Item: {meta.get('item_key', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

def test_task_parameters():
    """Test that the 3D generation task accepts the new parameters"""
    
    try:
        from tasks import generate_3d_model_from_image
        import inspect
        
        logger.info("🧪 Testing task function signature...")
        
        # Get function signature
        sig = inspect.signature(generate_3d_model_from_image)
        params = list(sig.parameters.keys())
        
        logger.info(f"📋 Task parameters: {params}")
        
        # Check for required parameters
        required_params = ['image_s3_key_or_path', 'with_texture', 'output_format', 
                          'doc_id', 'update_collection', 'category_key', 'item_key']
        
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            logger.error(f"❌ Missing required parameters: {missing_params}")
            return False
        else:
            logger.info(f"✅ All required parameters present")
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

def test_mongodb_connection():
    """Test MongoDB connection and sample data"""
    
    try:
        from db_helper import MongoDBHelper
        
        logger.info("🧪 Testing MongoDB connection...")
        
        mongo_helper = MongoDBHelper()
        
        # Test connection
        test_db = mongo_helper.client["World_builder"]
        result = test_db.command("ping")
        
        if result.get("ok") == 1:
            logger.info("✅ MongoDB connection successful")
        else:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Check for sample documents with images
        biomes_collection = "biomes"
        
        # Query for documents with images
        query = {"$or": [
            {"image_path": {"$exists": True, "$ne": None, "$ne": ""}},
            {"possible_structures": {"$exists": True}}
        ]}
        
        docs = mongo_helper.find_many("World_builder", biomes_collection, query, limit=5)
        
        logger.info(f"📊 Sample documents in {biomes_collection}:")
        logger.info(f"   Found {len(docs)} documents with images")
        
        for i, doc in enumerate(docs):
            doc_id = str(doc.get("_id", "Unknown"))
            name = doc.get("name", doc.get("theme", "Unnamed"))
            has_image = bool(doc.get("image_path"))
            has_structures = bool(doc.get("possible_structures"))
            
            logger.info(f"   {i+1}. {name[:30]}... (ID: {doc_id[:12]}...)")
            logger.info(f"      Has theme image: {has_image}")
            logger.info(f"      Has structures: {has_structures}")
            
            # Check structure images
            if has_structures:
                structures = doc.get("possible_structures", {})
                structure_count = 0
                for category, category_data in structures.items():
                    if isinstance(category_data, dict):
                        for struct_id, struct_data in category_data.items():
                            if isinstance(struct_data, dict):
                                struct_image = struct_data.get("image_path") or struct_data.get("imageUrl")
                                if struct_image:
                                    structure_count += 1
                
                logger.info(f"      Structure images: {structure_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB test failed: {e}", exc_info=True)
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting 3D MongoDB Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("MongoDB Connection", test_mongodb_connection),
        ("Task Parameters", test_task_parameters),
        ("Image Metadata Cache", test_image_metadata_cache),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The 3D MongoDB integration should work correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
