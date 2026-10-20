from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class Position(str, Enum):
    """Valid position values"""
    left = "left"
    center = "center"
    right = "right"


class DetectedObject(BaseModel):
    """Detected object with position and confidence"""
    object: str = Field(..., min_length=1, description="Object class name")
    position: Position = Field(..., description="Object position in image")
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


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
