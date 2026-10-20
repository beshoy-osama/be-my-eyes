from pydantic import BaseModel, Field
from typing import Optional


class ProcessRequest(BaseModel):
    """Request body for the processing endpoints."""
    file_id: str = Field(..., min_length=1, description="file_id returned by the upload endpoint")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold (defaults to DEFAULT_CONFIDENCE)",
    )
    language: str = Field("en", description="Speech language code (used by the TTS stage)")

