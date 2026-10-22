from pydantic import BaseModel, Field

class UploadImage(BaseModel):
    """Request model for processing an uploaded file."""
    file_id: str = Field(..., description="Unique identifier for the uploaded file.")
    