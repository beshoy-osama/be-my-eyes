"""
Object Detection Controller
Implements the Controller layer of MVC architecture for object detection.

This controller handles:
- API request processing
- Business logic coordination via services
- Response formatting and validation
- Error handling with detailed logging

Key features:
- Async operations for better performance
- Clear separation of concerns (MVC pattern)
- Comprehensive error logging
- Validation and response standardization
"""

import logging
import time
import os
from src.schemas.detection_schemas import DetectionResponse
from src.services.detection_service import DetectionService

logger = logging.getLogger(__name__)

class ObjectDetectionController:
    """Main controller for object detection operations"""
    
    def __init__(self):
        """Initialize the controller with service dependency"""
        self.service = DetectionService()
        logger.info("ObjectDetectionController initialized with DetectionService")
    
    async def detect_objects(self, image_path: str, min_confidence: float = 0.5) -> DetectionResponse:
        """
        Main detection endpoint handler
        
        Args:
            image_path: Path to the image file for detection
            min_confidence: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            DetectionResponse: Standardized API response object
            
        Raises:
            Any exceptions are caught and converted to standardized error response
        """
        start_time = time.time()
        
        try:
            # Execute detection pipeline through service layer
            objects, caption, speech_file = await self.service.organize_services(
                image_path, 
                min_confidence
            )
            
            # Apply confidence threshold filtering
            filtered_objects = [
                obj for obj in objects 
                if obj.confidence >= min_confidence
            ]
            
            # Prepare speech file information for response
            speech_info = None
            if speech_file and os.path.exists(speech_file):
                speech_info = {
                    "file_path": speech_file,
                    "file_size": os.path.getsize(speech_file)
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return successful response with detection results
            return DetectionResponse(
                success=True,
                caption=caption, 
                speech=speech_info, 
                objects=filtered_objects,
                total_objects=len(filtered_objects),
                original_count=len(objects),
                processing_time=processing_time
            )
            
        except Exception as e:
            # Log detailed error information with traceback
            logger.error(f"Detection failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            
            # Return standardized error response
            return DetectionResponse(
                success=False,
                error=str(e),
                objects=[],
                total_objects=0,
                processing_time=processing_time
            )