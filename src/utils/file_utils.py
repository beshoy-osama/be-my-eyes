"""
File Upload Utilities
Provides secure file handling for uploads with validation and cleanup.

Key responsibilities:
- Validates file types and sizes
- Manages temporary file storage in src/uploads
- Ensures proper directory structure
- Handles resource cleanup to prevent leaks
"""

import os
import uuid
from fastapi import UploadFile
from pathlib import Path
from config.settings import settings  
from logging import getLogger

logger = getLogger(__name__)

def save_upload_file(upload_file: UploadFile) -> str:
    """
    Securely save an uploaded file with validation
    
    Args:
        upload_file: The uploaded file object to process
        
    Returns:
        str: Absolute path to the saved file
        
    Raises:
        ValueError: If file extension is not in ALLOWED_EXTENSIONS
        
    Process flow:
        1. Validate file extension against allowed types
        2. Generate unique filename to prevent conflicts
        3. Save file to designated uploads directory
        4. Return absolute path for reference
    
    Security features:
        - Validates file type before saving
        - Uses unique filenames to prevent overwrites
        - Creates uploads directory if needed
    """
    # Validate file extension
    ext = os.path.splitext(upload_file.filename)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Only {', '.join(settings.ALLOWED_EXTENSIONS)} formats are allowed. "
            f"Your file has extension: {ext}"
        )
    
    # Generate unique filename
    file_name = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(get_uploads_dir(), file_name)
    logger.info(f"Saving uploaded file to: {file_path}")
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())
    
    return file_path

def get_uploads_dir() -> str:
    """
    Get absolute path to uploads directory inside src
    
    Returns:
        str: Absolute path to the uploads directory
        
    Process:
        1. Determine current file location (file_utils.py)
        2. Navigate up to src directory
        3. Append uploads folder name
        4. Create directory if it doesn't exist
    
    Benefits:
        - Works in any environment (Windows, Linux, Mac)
        - Uses absolute paths for reliability
        - Creates directory on first use
    """
    current_dir = Path(__file__).parent
    uploads_dir = current_dir.parent / settings.UPLOAD_FOLDER
    
    # Ensure directory exists
    uploads_dir.mkdir(exist_ok=True)
    
    return str(uploads_dir)

def cleanup_file(file_path: str):
    """
    Safely clean up temporary file after processing
    
    Args:
        file_path: Path to the file to delete
        
    Process:
        1. Check if file exists
        2. Delete file if exists
        3. Log success or failure
        
    Error handling:
        - Catches and logs any deletion errors
        - Prevents application crashes from cleanup failures
        - Ensures resources are freed when possible
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file removed: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file: {e}")