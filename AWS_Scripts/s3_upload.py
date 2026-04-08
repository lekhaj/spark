#!/usr/bin/env python3
"""
S3 Upload Script

This script uploads the contents of the local /output/images directory to AWS S3 /images directory
and /output/3d_assets to AWS S3 3DModels/ directory.

Usage:
    python s3_upload.py [--bucket BUCKET_NAME] [--sync] [--dry-run]

Requirements:
    - AWS credentials configured via AWS CLI, environment variables, or IAM role
    - boto3 library installed
"""

import os
import sys
import logging
import argparse
import boto3
import hashlib
import time
from botocore.exceptions import ClientError, EndpointConnectionError
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"s3_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger()

# Base directory path
BASE_DIR = Path(__file__).resolve().parent.parent

# Output directories to sync
OUTPUT_PATHS = {
    'images': {
        'local_path': BASE_DIR / 'output' / 'images',
        's3_prefix': 'images/'
    },
    '3d_assets': {
        'local_path': BASE_DIR / 'output' / '3d_assets',
        's3_prefix': '3DModels/'
    }
}

# Maximum number of worker threads for parallel uploads
MAX_WORKERS = 10

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upload output files to S3')
    parser.add_argument('--bucket', type=str, default=os.environ.get('S3_BUCKET_NAME'),
                        help='S3 bucket name (can also be set via S3_BUCKET_NAME environment variable)')
    parser.add_argument('--profile', type=str, default=None,
                        help='AWS profile name to use')
    parser.add_argument('--region', type=str, default=os.environ.get('AWS_DEFAULT_REGION'),
                        help='AWS region (can also be set via AWS_DEFAULT_REGION environment variable)')
    parser.add_argument('--sync', action='store_true',
                        help='Only upload new or modified files (compare ETags)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be uploaded without actually uploading')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar for file uploads')
    return parser.parse_args()

def get_s3_client(profile_name=None, region_name=None):
    """Create and return an S3 client."""
    try:
        session_args = {}
        client_args = {}
        
        if profile_name:
            session_args['profile_name'] = profile_name
        
        if region_name:
            client_args['region_name'] = region_name
            
        session = boto3.Session(**session_args)
        s3_client = session.client('s3', **client_args)
        
        # Test the connection by listing buckets
        s3_client.list_buckets()
        return s3_client
    
    except EndpointConnectionError:
        logger.error("Could not connect to S3 endpoint. Check your internet connection.")
        sys.exit(1)
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDenied':
            logger.error("Access denied. Check your AWS credentials and permissions.")
        elif e.response['Error']['Code'] == 'InvalidAccessKeyId':
            logger.error("Invalid AWS access key. Check your credentials.")
        elif e.response['Error']['Code'] == 'SignatureDoesNotMatch':
            logger.error("Signature verification failed. Check your AWS secret key.")
        else:
            logger.error(f"Failed to create S3 client: {str(e)}")
        sys.exit(1)

def calculate_etag(file_path):
    """Calculate S3 ETag for a file (a simple MD5 hash for non-multipart uploads)."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def check_file_exists(s3_client, bucket, s3_key, local_file_path=None):
    """Check if a file exists in S3 and compare ETags if needed."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        s3_etag = response.get('ETag', '').strip('"')
        
        if local_file_path:
            # Compare ETags to see if content has changed
            local_etag = calculate_etag(local_file_path)
            if local_etag == s3_etag:
                return True, "IDENTICAL"
            else:
                return True, "MODIFIED"
        return True, "EXISTS"
    
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False, "NOT_FOUND"
        elif e.response['Error']['Code'] == '403':
            logger.error(f"Access denied to s3://{bucket}/{s3_key} - Check permissions")
            return None, "ACCESS_DENIED"
        else:
            logger.warning(f"Error checking s3://{bucket}/{s3_key}: {e.response['Error']['Code']}")
            return None, f"ERROR: {e.response['Error']['Code']}"

def upload_file(s3_client, file_path, bucket, s3_key, sync=False, dry_run=False):
    """Upload a file to S3 bucket."""
    try:
        if sync:
            exists, status = check_file_exists(s3_client, bucket, s3_key, file_path)
            if exists and status == "IDENTICAL":
                logger.info(f"Skipping {file_path} (already up-to-date in S3)")
                return True

        if dry_run:
            if sync:
                action = "Would upload (new/modified)" if status != "IDENTICAL" else "Would skip (identical)"
            else:
                action = "Would upload"
            logger.info(f"{action}: {file_path} → s3://{bucket}/{s3_key}")
            return True
        
        # Proceed with upload
        logger.info(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket, s3_key)
        return True
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            logger.error(f"Access denied uploading {file_path} - Check your permissions")
        elif error_code == 'NoSuchBucket':
            logger.error(f"Bucket {bucket} does not exist")
        elif error_code == 'SlowDown' or error_code == 'ThrottlingException':
            logger.warning(f"S3 throttling detected, retrying {file_path} after delay")
            time.sleep(2)  # Back off before retry
            return upload_file(s3_client, file_path, bucket, s3_key, sync, dry_run)
        else:
            logger.error(f"Failed to upload {file_path}: {str(e)}")
        return False

def upload_directory(s3_client, local_dir, bucket, s3_prefix, sync=False, dry_run=False, show_progress=True):
    """Upload all files from a local directory to S3."""
    if not local_dir.exists():
        logger.warning(f"Local directory {local_dir} does not exist. Skipping.")
        return False

    # Collect all files first
    all_files = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}{relative_path}"
            all_files.append((local_path, s3_key))
    
    success_count = 0
    error_count = 0
    
    # Use tqdm for progress reporting if requested
    file_iter = tqdm(all_files, desc=f"Uploading {local_dir.name}", unit="file") if all_files and show_progress else all_files
    
    # For small number of files, process sequentially
    if len(all_files) <= 5:
        for local_path, s3_key in file_iter:
            if upload_file(s3_client, local_path, bucket, s3_key, sync, dry_run):
                success_count += 1
            else:
                error_count += 1
    else:
        # For larger numbers of files, use parallel uploads with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(all_files))) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(upload_file, s3_client, local_path, bucket, s3_key, sync, dry_run): (local_path, s3_key) 
                for local_path, s3_key in all_files
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                    if show_progress:
                        file_iter.update()
                except Exception as e:
                    logger.error(f"Unexpected error uploading {file_info[0]}: {str(e)}")
                    error_count += 1
                    if show_progress:
                        file_iter.update()
    
    status = "Dry run complete" if dry_run else "Directory upload complete"
    logger.info(f"{status}. {success_count} files processed, {error_count} errors.")
    return error_count == 0

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Verify bucket name is provided
    if not args.bucket:
        logger.error("S3 bucket name must be provided either via --bucket argument or S3_BUCKET_NAME environment variable")
        sys.exit(1)
    
    # Log script mode
    mode = []
    if args.dry_run:
        mode.append("DRY RUN")
    if args.sync:
        mode.append("SYNC")
    mode_str = " + ".join(mode) if mode else "STANDARD"
    logger.info(f"Running S3 upload in {mode_str} mode")
    
    # Get S3 client
    s3_client = get_s3_client(args.profile, args.region)
    
    # Upload each directory
    overall_success = True
    for name, path_info in OUTPUT_PATHS.items():
        logger.info(f"Processing {name} directory")
        success = upload_directory(
            s3_client, 
            path_info['local_path'], 
            args.bucket, 
            path_info['s3_prefix'],
            sync=args.sync,
            dry_run=args.dry_run,
            show_progress=not args.no_progress
        )
        if not success:
            overall_success = False
    
    # Final status message
    if args.dry_run:
        logger.info("Dry run completed. No files were modified.")
    elif overall_success:
        logger.info("All directories uploaded successfully!")
    else:
        logger.warning("Upload completed with some errors. Check the log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
