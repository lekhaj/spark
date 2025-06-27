"""
S3 Manager Module for Image and 3D Asset Storage
This module provides functionality to upload and download images and 3D assets to/from S3
"""

import os
import boto3
import logging
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

class S3Manager:
    """Manages S3 operations for images and 3D assets."""
    
    def __init__(self, bucket_name: Optional[str] = None, region: str = 'ap-south-1'):
        """
        Initialize S3 Manager.
        
        Args:
            bucket_name: S3 bucket name (uses environment variable if not provided)
            region: AWS region
        """
        self.bucket_name = bucket_name or os.environ.get('S3_BUCKET_NAME')
        self.region = region
        self.s3_client = None
        
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME must be provided or set as environment variable")
        
        try:
            self._initialize_s3_client()
            logger.info(f"S3 Manager initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def _initialize_s3_client(self):
        """Initialize boto3 S3 client"""
        try:
            self.s3_client = boto3.client('s3', region_name=self.region)
            # Test the connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"S3 bucket '{self.bucket_name}' does not exist")
            else:
                logger.error(f"AWS S3 client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing S3 client: {e}")
            raise
    
    def upload_file(self, local_file_path: str, s3_key: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file to S3.
        
        Args:
            local_file_path: Path to local file
            s3_key: S3 object key (path in bucket)
            content_type: MIME content type
            
        Returns:
            Dict with upload result
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            if self.s3_client is None:
                raise Exception("S3 client not initialized")
                
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"Successfully uploaded {local_file_path} to s3://{self.bucket_name}/{s3_key}")
            
            return {
                "status": "success",
                "s3_key": s3_key,
                "s3_url": s3_url,
                "bucket": self.bucket_name
            }
            
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path} to S3: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def download_file(self, s3_key: str, local_file_path: str) -> Dict[str, Any]:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_file_path: Local path to save file
            
        Returns:
            Dict with download result
        """
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            if self.s3_client is None:
                raise Exception("S3 client not initialized")
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
            
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_file_path}")
            
            return {
                "status": "success",
                "local_path": local_file_path,
                "s3_key": s3_key
            }
            
        except Exception as e:
            logger.error(f"Failed to download s3://{self.bucket_name}/{s3_key}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def upload_image(self, image_path: str, image_type: str = "generated") -> Dict[str, Any]:
        """
        Upload an image to S3 with proper naming convention.
        
        Args:
            image_path: Local path to image
            image_type: Type of image (generated, processed, etc.)
            
        Returns:
            Dict with upload result including S3 URL
        """
        try:
            filename = os.path.basename(image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"images/{image_type}/{timestamp}_{filename}"
            
            # Set appropriate content type for images
            content_type = "image/png" if filename.lower().endswith('.png') else "image/jpeg"
            
            return self.upload_file(image_path, s3_key, content_type)
            
        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {e}")
            return {"status": "error", "message": str(e)}
    
    def upload_3d_asset(self, asset_path: str, asset_type: str = "model", source_image_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a 3D asset to S3 with proper naming convention.
        
        Args:
            asset_path: Local path to 3D asset file or directory
            asset_type: Type of asset (model, texture, etc.)
            source_image_name: Name of source image to match (without extension)
            
        Returns:
            Dict with upload result including S3 URL(s)
        """
        try:
            if os.path.isfile(asset_path):
                # Single file upload
                original_filename = os.path.basename(asset_path)
                original_ext = os.path.splitext(original_filename)[1]
                
                # Use source image name if provided, keeping exact same name but changing extension
                if source_image_name:
                    # Remove any extension from source image name and add 3D file extension
                    base_name = os.path.splitext(source_image_name)[0]
                    filename = f"{base_name}{original_ext}"
                else:
                    filename = original_filename
                
                # No timestamp prefix - use exact filename
                s3_key = f"3d_assets/{asset_type}/{filename}"
                
                # Set content type based on file extension
                content_type = self._get_3d_content_type(filename)
                
                return self.upload_file(asset_path, s3_key, content_type)
                
            elif os.path.isdir(asset_path):
                # Directory upload (for multi-file 3D assets)
                results = []
                original_dir_name = os.path.basename(asset_path)
                
                # Use source image name for directory if provided
                if source_image_name:
                    base_name = os.path.splitext(source_image_name)[0]
                    dir_name = f"{base_name}_3d_assets"
                else:
                    dir_name = original_dir_name
                
                # No timestamp prefix for directory
                
                for root, dirs, files in os.walk(asset_path):
                    for file in files:
                        local_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file_path, asset_path)
                        s3_key = f"3d_assets/{asset_type}/{dir_name}/{relative_path}"
                        
                        content_type = self._get_3d_content_type(file)
                        result = self.upload_file(local_file_path, s3_key, content_type)
                        results.append(result)
                
                # Return combined results
                success_count = sum(1 for r in results if r.get("status") == "success")
                
                if success_count == len(results):
                    return {
                        "status": "success",
                        "message": f"Uploaded {success_count} files",
                        "uploads": results,
                        "main_s3_url": results[0].get("s3_url") if results else None
                    }
                else:
                    return {
                        "status": "partial",
                        "message": f"Uploaded {success_count}/{len(results)} files",
                        "uploads": results
                    }
            else:
                return {"status": "error", "message": "Asset path does not exist"}
                
        except Exception as e:
            logger.error(f"Failed to upload 3D asset {asset_path}: {e}")
            return {"status": "error", "message": str(e)}
    
    def download_image(self, s3_key: str, local_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Download an image from S3.
        
        Args:
            s3_key: S3 key for the image
            local_dir: Local directory to save image (uses temp if not provided)
            
        Returns:
            Dict with download result
        """
        try:
            if local_dir is None:
                local_dir = tempfile.mkdtemp()
            
            filename = os.path.basename(s3_key)
            local_path = os.path.join(local_dir, filename)
            
            return self.download_file(s3_key, local_path)
            
        except Exception as e:
            logger.error(f"Failed to download image {s3_key}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_3d_content_type(self, filename: str) -> str:
        """Get appropriate content type for 3D files."""
        ext = filename.lower().split('.')[-1]
        content_types = {
            'glb': 'model/gltf-binary',
            'gltf': 'model/gltf+json',
            'obj': 'text/plain',
            'mtl': 'text/plain',
            'ply': 'application/octet-stream',
            'stl': 'application/octet-stream',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def get_signed_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a signed URL for temporary access to S3 object.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            
        Returns:
            Signed URL string or None if failed
        """
        try:
            if self.s3_client is None:
                return None
                
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {s3_key}: {e}")
            return None
    
    def get_filename_from_s3_key(self, s3_key: str) -> str:
        """
        Extract filename from S3 key.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Filename without path
        """
        return os.path.basename(s3_key)
    
    def get_base_name_from_s3_key(self, s3_key: str) -> str:
        """
        Extract base filename (without extension) from S3 key.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Base filename without extension
        """
        filename = self.get_filename_from_s3_key(s3_key)
        return os.path.splitext(filename)[0]

def get_s3_manager() -> Optional[S3Manager]:
    """Get S3 manager instance."""
    try:
        return S3Manager()
    except Exception as e:
        logger.error(f"Failed to create S3 manager: {e}")
        return None
