# S3 Image Support for 3D Generation

This document describes how to use AWS S3 stored images for 3D model generation with Hunyuan3D.

## Overview

The 3D generation pipeline now supports fetching images directly from AWS S3 buckets instead of requiring local file storage. This enables:

- Processing images stored in cloud storage
- Simplified workflow for web applications
- Better integration with AWS-based pipelines
- Automatic cleanup of temporary files

## Supported S3 URL Formats

The system automatically detects and supports the following S3 URL formats:

1. **S3 Protocol URLs**
   ```
   s3://bucket-name/path/to/image.jpg
   ```

2. **Virtual-hosted Style URLs**
   ```
   https://bucket-name.s3.region.amazonaws.com/path/to/image.jpg
   https://bucket-name.s3.amazonaws.com/path/to/image.jpg
   ```

3. **Path Style URLs**
   ```
   https://s3.region.amazonaws.com/bucket-name/path/to/image.jpg
   https://s3.amazonaws.com/bucket-name/path/to/image.jpg
   ```

## Configuration

Add the following environment variables to configure S3 support:

```bash
# S3 Configuration
S3_IMAGES_BUCKET=your-bucket-name
S3_REGION=us-east-1
S3_DOWNLOAD_TIMEOUT=60
S3_TEMP_DIR=/tmp/s3_images
S3_CLEANUP_TEMP_FILES=True

# AWS Credentials (optional if using IAM roles)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

## AWS Credentials Setup

The system supports multiple ways to provide AWS credentials:

### 1. Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-east-1
```

### 2. AWS CLI Configuration
```bash
aws configure
```

### 3. IAM Roles (Recommended for EC2)
When running on EC2, attach an IAM role with S3 read permissions.

### 4. AWS Credentials File
Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key
region = us-east-1
```

## Required Permissions

The AWS credentials/role must have the following S3 permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name/*",
                "arn:aws:s3:::your-bucket-name"
            ]
        }
    ]
}
```

## Supported Image Formats

The following image formats are supported for S3 downloads:

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`
- `.webp`

## Usage Examples

### Basic Usage with S3 URL

```python
from src.hunyuan3d_worker import generate_3d_from_image_core

# Generate 3D model from S3 image
result = generate_3d_from_image_core(
    image_path="s3://my-bucket/images/example.jpg",
    with_texture=False,
    output_format='glb'
)

if result['status'] == 'success':
    print(f"3D model generated: {result['model_path']}")
```

### Using with Celery Tasks

```python
from src.tasks import generate_3d_model_from_image

# Submit task with S3 image URL
task = generate_3d_model_from_image.delay(
    image_path="https://my-bucket.s3.amazonaws.com/images/example.png",
    with_texture=True,
    output_format='glb'
)

result = task.get()
print(f"Result: {result}")
```

### Mixed Local and S3 Images

```python
# The same function works with both local paths and S3 URLs
local_image = "/path/to/local/image.jpg"
s3_image = "s3://bucket/remote/image.png"

# Both will work seamlessly
result1 = generate_3d_from_image_core(local_image)
result2 = generate_3d_from_image_core(s3_image)
```

## How It Works

1. **URL Detection**: The system automatically detects if the provided path is an S3 URL
2. **Download**: If it's an S3 URL, the image is downloaded to a temporary directory
3. **Processing**: The local temporary file is used for 3D generation
4. **Cleanup**: The temporary file is automatically deleted after processing

## Error Handling

The system provides detailed error messages for common issues:

- **Missing boto3**: Install with `pip install boto3`
- **Invalid credentials**: Check AWS credential configuration
- **Bucket not found**: Verify bucket name and permissions
- **Object not found**: Check the object key/path
- **Network issues**: Check internet connectivity and S3 endpoint accessibility

## Performance Considerations

- **Download Time**: Large images will take longer to download
- **Temporary Storage**: Ensure sufficient disk space in the temporary directory
- **Bandwidth**: Consider image size for bandwidth usage
- **Cleanup**: Temporary files are automatically cleaned up, but monitor disk usage

## Troubleshooting

### Common Issues

1. **"NoCredentialsError"**
   - Solution: Configure AWS credentials using one of the methods above

2. **"NoSuchBucket"**
   - Solution: Check bucket name and ensure it exists in the specified region

3. **"AccessDenied"**
   - Solution: Verify IAM permissions for S3 access

4. **"boto3 not available"**
   - Solution: Install boto3 with `pip install boto3`

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('hunyuan3d_worker').setLevel(logging.DEBUG)
```

## Security Best Practices

1. **Use IAM Roles**: Prefer IAM roles over access keys when running on EC2
2. **Least Privilege**: Grant only necessary S3 permissions
3. **Bucket Policies**: Use bucket policies to restrict access
4. **Encryption**: Consider using S3 server-side encryption
5. **Access Logging**: Enable S3 access logging for audit trails

## Integration Examples

### Web Application Integration

```python
# Flask/FastAPI endpoint example
@app.post("/generate-3d")
async def generate_3d_model(s3_image_url: str):
    if not is_s3_url(s3_image_url):
        return {"error": "Invalid S3 URL"}
    
    task = generate_3d_model_from_image.delay(s3_image_url)
    return {"task_id": task.id}
```

### Batch Processing

```python
# Process multiple S3 images
s3_urls = [
    "s3://bucket/image1.jpg",
    "s3://bucket/image2.png",
    "s3://bucket/image3.jpg"
]

tasks = []
for url in s3_urls:
    task = generate_3d_model_from_image.delay(url)
    tasks.append(task)

# Wait for all tasks to complete
results = [task.get() for task in tasks]
```

## Cost Considerations

- **S3 Requests**: Each image download counts as a GET request
- **Data Transfer**: Consider data transfer costs for large images
- **Storage**: Temporary local storage is used during processing

## Future Enhancements

Potential future improvements:

- Support for S3 presigned URLs
- Caching of frequently accessed images
- Support for other cloud storage providers (Azure Blob, Google Cloud Storage)
- Direct streaming processing without temporary files
