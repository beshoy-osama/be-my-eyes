from pydantic import BaseModel, Field
from typing import Optional


class SpeechResult(BaseModel):
    """Output of the text-to-speech module."""
    file_name: str = Field(..., description="Generated audio file name")
    file_path: str = Field(..., description="Generated audio file path on the server")
    file_size: int = Field(..., ge=0, description="Audio file size in bytes")
    language: str = Field(..., description="Speech language code")
    cached: bool = Field(False, description="Whether the audio was served from cache")
    processing_time: float = Field(..., ge=0, description="Speech generation time in seconds")
    url: Optional[str] = Field(None, description="URL to stream/download the audio file")

