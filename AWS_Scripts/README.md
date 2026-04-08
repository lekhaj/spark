# S3 Upload Script

This script uploads the contents of the local output directories to AWS S3:
- `/output/images` → S3 `images/` directory
- `/output/3d_assets` → S3 `3DModels/` directory

## Files in this Directory

- `s3_upload.py`: Main script to upload files to S3
- `credentials_guide.md`: Guide for setting up AWS credentials on EC2 instances
- `ec2_bootstrap.sh`: Bootstrap script for configuring new EC2 instances
- `requirements.txt`: Required Python packages

## Prerequisites

1. AWS credentials configured on your system using one of these methods:
   - AWS CLI (`aws configure`)
   - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
   - IAM role (if running on EC2 instance)

2. Required Python packages:
   ```
   pip install boto3
   ```
   or simply:
   ```
   pip install -r requirements.txt
   ```

## Usage

Basic usage with bucket name as argument:
```
python s3_upload.py --bucket your-bucket-name
```

Using a specific AWS profile:
```
python s3_upload.py --bucket your-bucket-name --profile your-profile-name
```

Setting the bucket via environment variable:
```
export S3_BUCKET_NAME=your-bucket-name
python s3_upload.py
```

### Advanced Options

Synchronize mode (only upload new or changed files):
```
python s3_upload.py --bucket your-bucket-name --sync
```

Dry run mode (preview what would be uploaded without making changes):
```
python s3_upload.py --bucket your-bucket-name --dry-run
```

Specify a region:
```
python s3_upload.py --bucket your-bucket-name --region us-west-2
```

Disable progress bar for file uploads:
```
python s3_upload.py --bucket your-bucket-name --no-progress
```

All options can be combined as needed.

## Example

```bash
# Upload all output files to your S3 bucket
python s3_upload.py --bucket red-panda-assets
```

The script will maintain the directory structure when uploading to S3.

## EC2 Instance Setup

For EC2 instances, we recommend using IAM roles instead of hard-coded credentials:

1. Create an IAM role with S3 permissions
2. Attach the IAM role to your EC2 instance
3. Run the script without any credential parameters

See `credentials_guide.md` for detailed setup instructions and security best practices.

You can also use `ec2_bootstrap.sh` as a User Data script when launching a new EC2 instance to automatically set up the upload mechanism.
