"""Response schemas for each processing stage exposed by the processes route."""

from pydantic import BaseModel, Field

from ..routes.schemes.DetectionSchemas import DetectionResult
from .vlm_schemas import CaptionResult
from .tts_schemas import SpeechResult


class DetectionResponse(BaseModel):
    """Stage 1: detection only."""
    signal: str = Field(..., description="Response signal")
    project_id: str
    file_id: str
    detection: DetectionResult
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")


class DetectionCaptionResponse(DetectionResponse):
    """Stage 2: detection + VLM caption."""
    caption: CaptionResult


class DetectionCaptionSpeechResponse(DetectionCaptionResponse):
    """Stage 3: detection + VLM caption + text-to-speech."""
    speech: SpeechResult

