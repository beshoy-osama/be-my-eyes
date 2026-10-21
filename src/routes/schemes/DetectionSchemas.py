from pydantic import BaseModel, Field
from typing import List, Optional
from models.enums import PositionEnums


class DetectionRequest(BaseModel):
    """Request body for the processing endpoints."""
    file_id: str = Field(..., min_length=1, description="file_id returned by the upload endpoint")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold (defaults to DEFAULT_CONFIDENCE)",
    )


class DetectedObject(BaseModel):
    """Detected object with position and confidence"""
    object: str = Field(..., min_length=1, description="Object class name")
    position: PositionEnums = Field(..., description="Object position in image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")


class DetectionResult(BaseModel):
    """Output of the YOLO detection module."""
    objects: List[DetectedObject] = Field(default_factory=list, description="List of detected objects")
    total_objects: int = Field(..., ge=0, description="Total number of objects detected")
    confidence_threshold: float = Field(..., ge=0.0, le=1.0, description="Confidence threshold used")
    image_width: int = Field(..., ge=0, description="Image width in pixels")
    image_height: int = Field(..., ge=0, description="Image height in pixels")
    model_name: str = Field(..., description="YOLO model used for detection")
    processing_time: float = Field(..., ge=0, description="Detection time in seconds")
