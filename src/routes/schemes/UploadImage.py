from pydantic import BaseModel, Field
from typing import List, Optional

class UploadImage(BaseModel):
    """Request model for processing an uploaded file."""
    file_id: str
    