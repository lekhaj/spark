# models.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from typing import Dict
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

class SubAssetOutput(BaseModel):
    """The final, canonical output for a single generated sub-asset."""
    s3_image_uri: str
    s3_3d_uri: str
    
class Assets(BaseModel):
    id : str = Field(alias="_id",description="Unique identifier for the asset")
    Asset_name : Dict[str,Dict[str, SubAssetOutput]] =Field(
        default_factory=dict,
        description="A nested dictionary mapping: { asset_type -> { sub_asset_id -> output } }"
    )
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "652c8259-aa17-4cdc-af15-518c30fc3d08",
                "asset_types": {
                    "buildings": {
                        "sha256_hash_of_building_description": {
                            "s3_image_uri": "s3://.../building.png",
                            "s3_3d_uri": "s3://.../building.glb"
                        }
                    }
                }
            }
        }
