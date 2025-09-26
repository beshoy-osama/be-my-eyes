"""
API Endpoints Router
Implements the main API endpoints for object detection and speech delivery.

This module:
- Defines the core API endpoints for the Object Detection service
- Handles file uploads and processing
- Serves generated speech files
- Implements proper error handling and resource management
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from src.config.settings import get_settings
from src.schemas.detection_schemas import ErrorResponse
from src.controllers import ObjectDetectionController
from src.utils.file_utils import save_upload_file, cleanup_file
from src.models.tts.tts_model import TTSModel

import os
import logging
import time

logger = logging.getLogger(__name__)
settings = get_settings()
folder_dir = TTSModel().output_dir  # Get TTS output directory from model

# Track startup time for health checks
startup_time = time.time()

# Initialize FastAPI with comprehensive documentation
app = FastAPI(
    title="BE-MY-EYES service",
    description="for describing surroundings for visually impaired users",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """
    Process image and return object detection results with accessibility features
    
    Args:
        file: Image file to process (".png",".jpg",".jpeg")
        confidence: Minimum confidence threshold (0.0-1.0)
    
    Returns:
        JSON response containing:
        - Object detection results
        - Accessibility-focused caption
        - Speech file information (if available)
    
    Process flow:
        1. Save uploaded file to temporary storage
        2. Process detection through ObjectDetectionController
        3. Return structured response with results
        4. Clean up temporary files (always)
    
    Error handling:
        - 500 Internal Server Error for processing failures
        - File validation handled in file_utils
    """
    file_path = None
    try:
        # Save uploaded file to temporary storage
        file_path = save_upload_file(file)
        
        # Process detection through controller
        controller = ObjectDetectionController()
        result = await controller.detect_objects(file_path, confidence)
        
        return result
        
    except Exception as e:
        # Handle all errors and return 500 response
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up temporary files
        if file_path:
            cleanup_file(file_path)

@app.get("/api/speech/{file_name}")
async def get_speech(file_name: str):
    """
    Download generated speech file for accessibility
    
    Args:
        file_name: Name of the speech file to download
    
    Returns:
        FileResponse with audio/mpeg content type
    
    Process flow:
        1. Construct full path to speech file
        2. Verify file exists
        3. Return file as audio stream
    
    Error handling:
        - 404 Not Found if file doesn't exist
    """
    # Construct absolute path to speech file
    file_path = os.path.join(folder_dir, file_name)
    
    # Verify file exists before serving
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Speech file not found")
    
    # Return file with proper media type
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=file_name
    )