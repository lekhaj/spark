# models.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    image_path: str  
    output_key: str  
    status: TaskStatus = TaskStatus.QUEUED
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    result_url: Optional[str] = None