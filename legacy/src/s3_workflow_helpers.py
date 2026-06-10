"""
Helper functions for S3-based 3D asset generation workflow
"""

import logging
from typing import Dict, Any, Optional, List
from celery import current_app

logger = logging.getLogger(__name__)

def submit_s3_image_for_3d_generation(
    image_s3_key: str,
    doc_id: Optional[str] = None,
    collection_name: Optional[str] = None,
    with_texture: bool = False,
    output_format: str = 'glb',
    queue: str = 'gpu_tasks'
) -> Dict[str, Any]:
    """
    Submit an S3 image for 3D generation processing.
    
    Args:
        image_s3_key: S3 key of the image to process
        doc_id: MongoDB document ID to update (optional)
        collection_name: MongoDB collection name (optional)
        with_texture: Whether to generate textures
        output_format: Output format (glb, obj, etc.)
        queue: Celery queue to use
        
    Returns:
        Dict with task submission result
    """
    try:
        from tasks import app as celery_app
        
        processing_options = {
            'with_texture': with_texture,
            'output_format': output_format
        }
        
        task = celery_app.send_task(
            'process_image_for_3d_generation',
            args=[image_s3_key, doc_id, collection_name, processing_options],
            queue=queue
        )
        
        logger.info(f"Submitted S3 image {image_s3_key} for 3D generation (task: {task.id})")
        
        # Set status to pending in MongoDB if doc_id provided
        if doc_id and collection_name:
            pending_result = set_processing_status_pending(
                doc_id=doc_id,
                collection_name=collection_name,
                task_id=task.id,
                processing_type="image_to_3d"
            )
            
            if pending_result.get("status") != "success":
                logger.warning(f"Failed to set pending status: {pending_result.get('message')}")
        
        return {
            "status": "submitted",
            "task_id": task.id,
            "image_s3_key": image_s3_key,
            "processing_options": processing_options,
            "doc_id": doc_id
        }
        
    except Exception as e:
        logger.error(f"Failed to submit S3 image for 3D generation: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def submit_batch_s3_images_for_3d_generation(
    image_s3_keys: List[str],
    with_texture: bool = False,
    output_format: str = 'glb',
    queue: str = 'gpu_tasks'
) -> Dict[str, Any]:
    """
    Submit multiple S3 images for batch 3D generation processing.
    
    Args:
        image_s3_keys: List of S3 keys for images to process
        with_texture: Whether to generate textures
        output_format: Output format (glb, obj, etc.)
        queue: Celery queue to use
        
    Returns:
        Dict with batch task submission result
    """
    try:
        from tasks import app as celery_app
        
        processing_options = {
            'with_texture': with_texture,
            'output_format': output_format
        }
        
        task = celery_app.send_task(
            'batch_process_s3_images_for_3d',
            args=[image_s3_keys, processing_options],
            queue=queue
        )
        
        logger.info(f"Submitted {len(image_s3_keys)} S3 images for batch 3D generation (task: {task.id})")
        
        return {
            "status": "submitted",
            "task_id": task.id,
            "image_count": len(image_s3_keys),
            "processing_options": processing_options
        }
        
    except Exception as e:
        logger.error(f"Failed to submit batch S3 images for 3D generation: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a Celery task.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Dict with task status information
    """
    try:
        from celery.result import AsyncResult
        
        result = AsyncResult(task_id, app=current_app)
        
        status_info = {
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None,
        }
        
        if result.ready():
            if result.successful():
                status_info["result"] = result.result
            elif result.failed():
                status_info["error"] = str(result.result)
        else:
            # Task is still running, get progress info
            if hasattr(result, 'info') and result.info:
                status_info["progress"] = result.info
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        }

def upload_image_and_submit_for_3d_generation(
    local_image_path: str,
    doc_id: Optional[str] = None,
    collection_name: Optional[str] = None,
    with_texture: bool = False,
    output_format: str = 'glb',
    image_type: str = "uploaded"
) -> Dict[str, Any]:
    """
    Upload a local image to S3 and submit it for 3D generation.
    
    Args:
        local_image_path: Path to local image file
        doc_id: MongoDB document ID to update (optional)
        collection_name: MongoDB collection name (optional)
        with_texture: Whether to generate textures
        output_format: Output format (glb, obj, etc.)
        image_type: Type of image for S3 organization
        
    Returns:
        Dict with upload and submission result
    """
    try:
        from s3_manager import get_s3_manager
        
        # Upload image to S3
        s3_mgr = get_s3_manager()
        if s3_mgr is None:
            return {"status": "error", "message": "S3 manager not available"}
        
        upload_result = s3_mgr.upload_image(local_image_path, image_type)
        if upload_result.get("status") != "success":
            return {
                "status": "error",
                "message": f"Failed to upload image to S3: {upload_result.get('message')}"
            }
        
        image_s3_key = upload_result["s3_key"]
        image_s3_url = upload_result["s3_url"]
        
        # Submit for 3D generation
        submission_result = submit_s3_image_for_3d_generation(
            image_s3_key=image_s3_key,
            doc_id=doc_id,
            collection_name=collection_name,
            with_texture=with_texture,
            output_format=output_format
        )
        
        if submission_result.get("status") == "submitted":
            return {
                "status": "success",
                "message": "Image uploaded to S3 and submitted for 3D generation",
                "image_s3_key": image_s3_key,
                "image_s3_url": image_s3_url,
                "task_id": submission_result["task_id"],
                "processing_options": submission_result["processing_options"]
            }
        else:
            return {
                "status": "error",
                "message": f"Image uploaded but 3D generation submission failed: {submission_result.get('message')}",
                "image_s3_key": image_s3_key,
                "image_s3_url": image_s3_url
            }
            
    except Exception as e:
        logger.error(f"Failed to upload and submit image for 3D generation: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def update_mongodb_with_s3_links(
    doc_id: str,
    collection_name: str,
    image_s3_url: Optional[str] = None,
    model_s3_url: Optional[str] = None,
    status: str = "pending",
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update MongoDB document with S3 links and processing status.
    
    Args:
        doc_id: MongoDB document ID
        collection_name: Collection name
        image_s3_url: S3 URL of the processed image
        model_s3_url: S3 URL of the generated 3D model
        status: Processing status (pending, processing, completed, failed)
        additional_data: Additional data to update
        
    Returns:
        Dict with update result
    """
    try:
        from db_helper import MongoDBHelper
        from config import MONGO_DB_NAME
        from datetime import datetime
        
        mongo_helper = MongoDBHelper()
        
        update_data = {
            "s3_processing_updated_at": datetime.now(),
            "status": status
        }
        
        if status == "pending":
            update_data["s3_processing_started"] = True
            update_data["s3_processing_started_at"] = datetime.now()
            # Note: For generated models, status stays "pending" as requested
        elif status == "completed":
            update_data["s3_processing_completed"] = True
            update_data["s3_processing_completed_at"] = datetime.now()
        elif status == "failed":
            update_data["s3_processing_failed"] = True
            update_data["s3_processing_failed_at"] = datetime.now()
            
        if image_s3_url:
            update_data["image_s3_url"] = image_s3_url
            
        if model_s3_url:
            update_data["model_s3_url"] = model_s3_url
            
        if additional_data:
            update_data.update(additional_data)
        
        result = mongo_helper.update_by_id(MONGO_DB_NAME, collection_name, doc_id, update_data)
        
        logger.info(f"Updated MongoDB document {doc_id} with S3 links and status: {status}")
        
        return {
            "status": "success",
            "updated_records": result,
            "doc_id": doc_id,
            "update_data": update_data
        }
        
    except Exception as e:
        logger.error(f"Failed to update MongoDB with S3 links: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def set_processing_status_pending(
    doc_id: str,
    collection_name: str,
    task_id: str,
    processing_type: str = "3d_generation"
) -> Dict[str, Any]:
    """
    Set processing status to pending when task is submitted.
    
    Args:
        doc_id: MongoDB document ID
        collection_name: Collection name
        task_id: Celery task ID
        processing_type: Type of processing (3d_generation, batch_processing, etc.)
        
    Returns:
        Dict with update result
    """
    try:
        from db_helper import MongoDBHelper
        from config import MONGO_DB_NAME
        from datetime import datetime
        
        mongo_helper = MongoDBHelper()
        
        update_data = {
            "status": "pending",
            "task_id": task_id,
            "task_submitted_at": datetime.now(),
            "processing_type": processing_type,
            "last_updated": datetime.now()
        }
        
        result = mongo_helper.update_by_id(MONGO_DB_NAME, collection_name, doc_id, update_data)
        
        logger.info(f"Set status to pending for MongoDB document {doc_id} (task: {task_id})")
        
        return {
            "status": "success",
            "updated_records": result,
            "doc_id": doc_id,
            "task_id": task_id
        }
        
    except Exception as e:
        logger.error(f"Failed to set pending status: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
