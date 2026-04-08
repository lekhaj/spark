# 3D Generation MongoDB Integration Fix

## Issue
The 3D generation functionality was not updating MongoDB documents correctly when generating 3D assets from structure images. While theme images at the document root level were working, nested structure images from the `possible_structures` field were not getting their 3D asset links updated.

## Root Cause
The Celery task for 3D generation (`generate_3d_model_from_image`) was missing the `category_key` and `item_key` parameters needed to update nested structure fields in MongoDB documents.

## Changes Made

### 1. Updated Task Function Signature
**File:** `src/tasks.py`

```python
# Before
def generate_3d_model_from_image(self, image_s3_key_or_path, with_texture=False, output_format='glb', doc_id=None, update_collection=None):

# After  
def generate_3d_model_from_image(self, image_s3_key_or_path, with_texture=False, output_format='glb', doc_id=None, update_collection=None, category_key=None, item_key=None):
```

### 2. Fixed MongoDB Update Logic
**File:** `src/tasks.py`

The task now correctly handles both theme and structure image updates:

```python
if category_key and item_key:
    # Update nested field for structure images - add 3D asset link
    update_path = f"possible_structures.{category_key}.{item_key}.asset_3d_url"
    update_data = {
        update_path: model_s3_url,
        f"possible_structures.{category_key}.{item_key}.asset_3d_generated_at": datetime.now(),
        f"possible_structures.{category_key}.{item_key}.asset_3d_format": output_format
    }
else:
    # Update root-level for theme images
    update_data = {
        "asset_3d_url": model_s3_url,
        "asset_3d_generated_at": datetime.now(),
        "asset_3d_format": output_format
    }
```

### 3. Updated Gradio App Task Call
**File:** `src/merged_gradio_app.py`

The Gradio app now passes the correct MongoDB metadata parameters:

```python
task = celery_generate_3d_model_from_image.apply_async(
    args=[image_url, with_texture, output_format],
    kwargs={
        'doc_id': mongodb_metadata.get("doc_id"),
        'update_collection': mongodb_metadata.get("collection"),
        'category_key': mongodb_metadata.get("category_key"),  # NEW
        'item_key': mongodb_metadata.get("item_key")           # NEW
    },
    queue='gpu_tasks'
)
```

### 4. Enhanced Metadata Logging
**File:** `src/merged_gradio_app.py`

Added detailed logging to help debug metadata capture:

```python
if mongodb_metadata:
    logger.info(f"📊 Found MongoDB metadata for image: {mongodb_metadata}")
    logger.info(f"   Doc ID: {mongodb_metadata.get('doc_id')}")
    logger.info(f"   Collection: {mongodb_metadata.get('collection')}")
    logger.info(f"   Type: {mongodb_metadata.get('type')}")
    logger.info(f"   Category Key: {mongodb_metadata.get('category_key')}")
    logger.info(f"   Item Key: {mongodb_metadata.get('item_key')}")
```

## Database Schema Impact

### Structure Images
Structure images will now have these new fields added when 3D assets are generated:

```json
{
  "possible_structures": {
    "settlements": {
      "village_001": {
        "name": "Medieval Village",
        "image_path": "https://s3-url-to-image.jpg",
        "asset_3d_url": "https://s3-url-to-3d-model.glb",        // NEW
        "asset_3d_generated_at": "2025-01-07T10:30:00Z",        // NEW  
        "asset_3d_format": "glb"                                 // NEW
      }
    }
  }
}
```

### Theme Images  
Theme images will continue to have root-level 3D asset fields:

```json
{
  "name": "Forest Biome",
  "image_path": "https://s3-url-to-theme-image.jpg",
  "asset_3d_url": "https://s3-url-to-3d-model.glb",            // NEW
  "asset_3d_generated_at": "2025-01-07T10:30:00Z",            // NEW
  "asset_3d_format": "glb"                                     // NEW
}
```

## How It Works

1. **Image Fetch**: `fetch_images_from_mongodb()` collects both theme and structure images, storing metadata in `_image_metadata_cache`

2. **Metadata Storage**: For each image, metadata includes:
   - `doc_id`: MongoDB document ID
   - `collection`: Collection name  
   - `type`: "theme" or "structure"
   - `category_key`: Structure category (e.g., "settlements") - null for themes
   - `item_key`: Structure item ID (e.g., "village_001") - null for themes

3. **3D Generation**: When user selects an image and generates 3D:
   - Metadata is looked up from cache using image URL
   - Task is called with correct MongoDB parameters
   - Database is updated with 3D asset link in the appropriate location

## Testing

Run the test script to verify functionality:

```bash
python test_3d_mongodb_integration.py
```

This will test:
- MongoDB connection and sample data
- Task function parameters
- Image metadata cache functionality

## Benefits

✅ **Structure Images**: Now properly update MongoDB with 3D asset links  
✅ **Theme Images**: Continue to work as before  
✅ **Nested Updates**: Correctly updates nested document fields  
✅ **Metadata Tracking**: Comprehensive logging for debugging  
✅ **Backwards Compatible**: Existing functionality preserved  

## Verification

To verify the fix is working:

1. Generate 3D models from structure images in the Gradio app
2. Check MongoDB documents - structure items should have new `asset_3d_url` fields
3. Monitor logs for detailed metadata information
4. Run the test script to validate the setup

The 3D generation pipeline now correctly supports the full round-trip workflow for both theme and structure images stored in MongoDB.
