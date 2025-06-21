# S3-Based 3D Asset Generation Workflow

This document explains how the pipeline now handles image processing and 3D asset generation using S3 storage and MongoDB integration.

## Overview

The system has been enhanced to support a cloud-native workflow where:

1. **Images are stored in S3** instead of local file system
2. **GPU workers fetch images from S3** for processing
3. **Generated 3D assets are uploaded to S3** 
4. **MongoDB is updated with S3 URLs** and processing status
5. **Frontend receives S3 links** for displaying results

## Architecture Changes

### Before (Local Storage)
```
Frontend → Local Image → GPU Processing → Local 3D Assets → Database (local paths)
```

### After (S3 Integration)
```
Frontend → S3 Image → GPU Processing → S3 3D Assets → Database (S3 URLs)
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# S3 Configuration
S3_BUCKET_NAME=your-3d-assets-bucket
S3_REGION=us-east-1
USE_S3_STORAGE=True
S3_PUBLIC_READ=False

# AWS Credentials (or use IAM roles)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### S3 Bucket Structure

The system organizes files in S3 as follows:

```
your-bucket/
├── images/
│   ├── generated/         # AI-generated images
│   ├── uploaded/          # User-uploaded images
│   └── text_to_3d/        # Images from text-to-3D pipeline
└── 3d_assets/
    ├── generated/         # Generated 3D models
    ├── text_to_3d/        # Models from text-to-3D pipeline
    └── model/             # Other model types
```

## New Tasks

### 1. `process_image_for_3d_generation`

Processes an image from S3 and generates 3D assets.

```python
from s3_workflow_helpers import submit_s3_image_for_3d_generation

result = submit_s3_image_for_3d_generation(
    image_s3_key="images/uploaded/my_object.png",
    doc_id="mongodb_document_id",
    collection_name="3d_models",
    with_texture=True,
    output_format="glb"
)
```

### 2. `batch_process_s3_images_for_3d`

Batch processes multiple S3 images.

```python
from s3_workflow_helpers import submit_batch_s3_images_for_3d_generation

result = submit_batch_s3_images_for_3d_generation(
    image_s3_keys=[
        "images/uploaded/object1.png",
        "images/uploaded/object2.png",
        "images/uploaded/object3.png"
    ],
    with_texture=True,
    output_format="glb"
)
```

## API Changes

### Enhanced 3D Generation Tasks

The existing tasks have been enhanced with S3 support:

#### `generate_3d_model_from_image`

```python
# Now accepts S3 keys or local paths
task_result = generate_3d_model_from_image.apply_async(
    args=[
        "s3://bucket/images/uploaded/object.png",  # S3 key
        True,  # with_texture
        "glb", # output_format
        "mongodb_doc_id",  # doc_id (optional)
        "collection_name"  # collection (optional)
    ],
    queue="gpu_tasks"
)
```

## MongoDB Integration

### Document Updates

When processing completes, MongoDB documents are automatically updated with:

```json
{
  "model_generated": true,
  "model_generated_at": "2025-01-15T10:30:00Z",
  "model_format": "glb",
  "model_with_texture": true,
  "model_status": "success",
  "model_s3_url": "https://bucket.s3.region.amazonaws.com/3d_assets/generated/model.glb",
  "model_local_path": "/tmp/model_abc123/model.glb",
  "image_s3_url": "https://bucket.s3.region.amazonaws.com/images/uploaded/object.png"
}
```

### Manual Updates

You can also manually update documents with S3 links:

```python
from s3_workflow_helpers import update_mongodb_with_s3_links

result = update_mongodb_with_s3_links(
    doc_id="document_id",
    collection_name="models",
    image_s3_url="https://...",
    model_s3_url="https://...",
    additional_data={"processing_method": "gpu_enhanced"}
)
```

## Integration Examples

### Upload and Process Workflow

```python
from s3_workflow_helpers import upload_image_and_submit_for_3d_generation

# Upload local image and submit for 3D generation
result = upload_image_and_submit_for_3d_generation(
    local_image_path="/path/to/uploaded/image.png",
    doc_id="user_upload_123",
    collection_name="user_models",
    with_texture=True,
    output_format="glb"
)

if result["status"] == "success":
    task_id = result["task_id"]
    image_url = result["image_s3_url"]
    
    # Monitor progress and get results
    # (see monitoring section below)
```

### Direct S3 Processing

```python
from s3_workflow_helpers import submit_s3_image_for_3d_generation

result = submit_s3_image_for_3d_generation(
    image_s3_key="images/uploaded/my_object.png",
    doc_id="mongodb_document_id",
    collection_name="3d_models",
    with_texture=True,
    output_format="glb"
)
```

### Batch Processing

```python
from s3_workflow_helpers import submit_batch_s3_images_for_3d_generation

result = submit_batch_s3_images_for_3d_generation(
    image_s3_keys=[
        "images/uploaded/object1.png",
        "images/uploaded/object2.png",
        "images/uploaded/object3.png"
    ],
    with_texture=True,
    output_format="glb"
)
```

### Progress Monitoring

```python
from s3_workflow_helpers import get_task_status

def monitor_task(task_id):
    while True:
        status = get_task_status(task_id)
        
        if status["ready"]:
            if status["successful"]:
                result = status["result"]
                model_url = result.get("s3_model_url")
                # Display success to user
                break
            else:
                # Handle error
                break
        else:
            # Show progress
            progress = status.get("progress", {})
            # Update UI with progress
            
        time.sleep(5)
```

## Deployment

### GPU Worker Setup

Start GPU workers to handle 3D generation tasks:

```bash
# On GPU instance
python -m celery worker --app=tasks --queues=gpu_tasks --loglevel=info
```

### CPU Worker Setup

Start CPU workers for other tasks:

```bash
# On CPU instance  
python -m celery worker --app=tasks --queues=cpu_tasks --loglevel=info
```

### Task Routing

Tasks are automatically routed based on configuration in `config.py`:

- `cpu_tasks`: Text/image generation, biome processing
- `gpu_tasks`: 3D model generation, S3-based processing
- `infrastructure`: Instance management

## Security Considerations

### S3 Permissions

Ensure your S3 bucket has appropriate permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::ACCOUNT:role/gpu-worker-role"},
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket/*"
    }
  ]
}
```

### Access Control

- Use IAM roles instead of access keys when possible
- Consider signed URLs for temporary access
- Implement proper authentication for API endpoints

## Error Handling

### Common Issues

1. **S3 Access Denied**: Check AWS credentials and bucket permissions
2. **Task Timeout**: Increase task time limits for large 3D models
3. **MongoDB Connection**: Verify connection string and authentication

### Monitoring

Monitor task execution with:

```bash
# Check Celery status
celery -A tasks inspect active

# Monitor task results
celery -A tasks events
```

## Performance Optimization

### Parallel Processing

The system supports parallel processing of multiple images:

```python
# Process multiple images in parallel
task_ids = []
for image_key in image_s3_keys:
    result = submit_s3_image_for_3d_generation(image_key)
    task_ids.append(result["task_id"])

# Monitor all tasks
for task_id in task_ids:
    monitor_task_progress(task_id)
```

### Caching

- Consider implementing model caching for repeated processing
- Use S3 Transfer Acceleration for faster uploads
- Implement Redis caching for frequently accessed metadata

## Testing and Validation

### Quick API Test

```python
from s3_workflow_helpers import submit_s3_image_for_3d_generation, get_task_status

# Submit test image for processing
result = submit_s3_image_for_3d_generation(
    image_s3_key="images/test/sample.jpg",
    doc_id="test_doc_123",
    collection_name="test_models",
    with_texture=False,
    output_format="glb"
)

# Monitor task
if result["status"] == "submitted":
    task_id = result["task_id"]
    status = get_task_status(task_id)
    print(f"Task status: {status}")
```

### Naming Convention Validation

The system ensures exact filename matching:

- **Source**: `red_car.jpg` → **3D Asset**: `red_car.glb`
- **Source**: `blue_house.png` → **3D Asset**: `blue_house.obj`  
- **Source**: `my_object.jpeg` → **3D Asset**: `my_object.glb`

Only the file extension changes, maintaining perfect traceability.

## Migration from Local Storage

To migrate existing local workflows:

1. Upload existing images to S3 using the AWS CLI or S3 upload utilities
2. Update database records with S3 URLs using the helper functions
3. Modify frontend to use new S3-based API endpoints  
4. Test with a subset of data before full migration

## Testing the Integration

### Verify S3 Configuration

```python
from s3_manager import get_s3_manager

s3_mgr = get_s3_manager()
if s3_mgr:
    print("✅ S3 Manager initialized successfully")
else:
    print("❌ S3 Manager initialization failed")
```

### Test MongoDB Updates

```python
from s3_workflow_helpers import update_mongodb_with_s3_links

result = update_mongodb_with_s3_links(
    doc_id="test_doc",
    collection_name="test_collection", 
    status="pending",
    additional_data={"test": True}
)
print(f"Update result: {result}")
```

## Support

For issues with the S3 integration:

1. Check AWS CloudTrail logs for S3 access issues
2. Verify Celery worker logs for task execution problems
3. Monitor MongoDB for update failures
4. Use the provided example scripts for testing
