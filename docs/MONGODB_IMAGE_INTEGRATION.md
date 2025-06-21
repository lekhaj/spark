# MongoDB Image Integration for 3D Generation

## Overview
This enhancement adds functionality to the Gradio app to fetch, display, and use images from MongoDB database for 3D model generation.

## New Features Added

### 1. MongoDB Images Section in 3D Generation Tab
- **Location**: Added as an accordion at the top of the "3D Generation" tab
- **Purpose**: Browse and select images stored in MongoDB for 3D model generation

### 2. New UI Components
- **Database/Collection Selection**: Text inputs to specify MongoDB database and collection
- **Fetch Images Button**: Retrieves all documents with "image_path" field from MongoDB
- **Images Gallery**: Displays fetched images in a grid layout with captions
- **Status Display**: Shows operation status and results
- **Selected Image URL**: Shows the URL of the currently selected image
- **Generate 3D Button**: Creates 3D models from selected images with custom settings

### 3. Simplified MongoDB-Only Workflow
The interface focuses exclusively on MongoDB images, providing a streamlined experience:
- **No Manual File Uploads**: Removes complexity of multiple input methods
- **Direct URL Processing**: Sends MongoDB URLs directly to GPU workers
- **Custom Settings**: Users can configure texture, format, and model type before generation

#### `fetch_images_from_mongodb(db_name: str, collection_name: str)`
- Queries MongoDB for documents containing "image_path" field
- Returns list of image tuples (url, caption) and status message
- Limits results to 50 images for performance
- Creates descriptive captions using document ID and name/theme/prompt

#### `download_and_prepare_image_for_3d(image_url: str)`
- Downloads image from URL to temporary local file
- Validates image format and converts to RGB if needed
- Returns local file path and status message
- Handles errors gracefully with detailed error messages

### 4. Event Handlers
- **Fetch Images**: Connects "Fetch Images from MongoDB" button to `fetch_images_from_mongodb()`
- **Image Selection**: Handles gallery image selection and updates UI state
- **Use Image**: Downloads selected image and sets it in the file upload component

## Workflow

1. **Browse Images**: 
   - User enters database and collection name
   - Clicks "Fetch Images from MongoDB"
   - Images are displayed in gallery with captions

2. **Select Image**:
   - User clicks on an image in the gallery
   - Selected image URL is displayed and stored
   - Generate button becomes active

3. **Configure and Generate**:
   - User configures 3D generation settings (texture, format, model type)
   - Clicks "� Generate 3D Model from Selected Image"
   - MongoDB URL is sent directly to GPU worker for processing

## Technical Details

### MongoDB-Only Processing
- **Direct URL Transmission**: MongoDB image URLs are sent directly to Celery workers
- **GPU-Side Download**: Images are downloaded on the GPU worker, not the frontend
- **No Local Storage**: Eliminates temporary file management on the frontend
- **Efficient Bandwidth**: Only metadata travels to frontend, actual images downloaded on worker

## Technical Details

### Database Query
- Uses query: `{"image_path": {"$exists": True, "$ne": None, "$ne": ""}}`
- Fetches documents that have non-empty image_path field
- Supports HTTP/HTTPS URLs only

### Image Processing
- Downloads images to temporary files
- Converts images to RGB format for compatibility
- Validates image format using PIL
- Cleans up temporary files appropriately

### UI Integration
- Seamlessly integrates with existing 3D generation workflow
- Maintains existing functionality for direct file uploads
- Provides clear status feedback to users

## Dependencies
- `requests`: For downloading images from URLs
- `PIL (Pillow)`: For image validation and processing
- `pymongo`: For MongoDB database operations (already present)
- `tempfile`: For temporary file management

## Error Handling
- Network errors during image download
- Invalid image formats
- MongoDB connection issues
- Missing or malformed image URLs
- UI state management errors

All errors are handled gracefully with user-friendly error messages displayed in the status components.

## Troubleshooting

### Error: "No connection adapters were found"
This error occurs when the gallery selection returns a complex object instead of a simple URL string. The fix implemented includes:

1. **Index-based Selection**: Uses the gallery selection index to retrieve the URL from a separate state variable
2. **Improved URL Extraction**: Handles complex gallery selection objects with multiple fallback approaches
3. **Better Debugging**: Added visible URL display field to help users see what image is selected

### Database Configuration
- **Default Database**: "World_builder" (configure in `src/config.py`)
- **Default Collection**: "biomes" 
- **Query**: Documents with non-empty "image_path" field
- **Validation**: Only HTTP/HTTPS URLs are processed

### Common Issues
- **MongoDB Connection**: Ensure MongoDB server is accessible at the configured URL
- **Image URLs**: Verify images are accessible via HTTP/HTTPS
- **Gallery Selection**: Click directly on images in the gallery, not empty areas
- **File Formats**: Supported formats are JPG, PNG, WEBP (automatically converted to RGB)

### Gradio Version Compatibility
- **Issue**: `Button.__init__() got an unexpected keyword argument 'info'`
- **Cause**: The `info` parameter is not supported in older Gradio versions
- **Fix**: Removed all `info` parameters and replaced with descriptive Markdown text
- **Components Fixed**: Button, Checkbox, Dropdown components in 3D generation section
