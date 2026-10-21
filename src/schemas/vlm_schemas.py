from pydantic import BaseModel, Field


class CaptionResult(BaseModel):
    """Output of the VLM captioning module."""
    caption: str = Field(..., min_length=1, description="Accessibility-focused caption for visually impaired users")
    model_name: str = Field(..., description="VLM model used to generate the caption")
    processing_time: float = Field(..., ge=0, description="Caption generation time in seconds")

