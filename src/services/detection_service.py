"""
Detection Service Layer
Implements the service layer that coordinates detection workflow between YOLO, VLM, and TTS.

Key responsibilities:
- Orchestrates the full detection pipeline (YOLO → VLM → TTS)
- Manages error cases gracefully
- Provides consistent data structures for controllers
- Handles service coordination and data flow

This service is the central component of the MVC architecture, sitting between:
- Models (YOLO, VLM, TTS) - which handle specific technical tasks
- Controllers - which handle API requests and responses

The service layer:
- Is responsible for business logic coordination
- Does not handle direct API interactions
- Ensures data flows correctly between components
- Handles service-specific error cases
"""

from src.models.yolo.yolo_model import YOLOModelManager
from src.models.vlm.vlm_model import VLMModelManager
from src.services.tts_service import TTSService
from src.config.settings import get_settings
from logging import getLogger

logger = getLogger(__name__)

settings = get_settings()

class DetectionService:
    """Main service layer for object detection workflow"""
    
    def __init__(self):
        """Initialize all required services for detection pipeline"""
        self.yolo = YOLOModelManager()
        self.vlm = VLMModelManager()
        self.tts = TTSService()
    
    async def organize_services(self, image_path: str, confidence: float) -> tuple:
        """
        Execute full detection pipeline:
        1. Run YOLO object detection
        2. Generate VLM caption (if API key available)
        3. Create TTS audio (if caption available)
        
        Args:
            image_path: Path to input image for detection
            confidence: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            tuple: (objects, caption, speech_file)
                - objects: List of detected objects
                - caption: Accessibility-focused description (or None)
                - speech_file: Path to generated audio file (or None)
        
        Process flow:
            1. YOLO: Detect objects in image
            2. VLM: Generate natural language caption for visually impaired users
            3. TTS: Convert caption to speech audio file
            4. Return results in consistent format for controllers
        
        Error handling:
            - VLM errors are logged but don't break the workflow
            - TTS errors are logged but don't break the workflow
            - YOLO errors will propagate to controller (critical path)
        """
        # Step 1: Object detection using YOLO
        objects = self.yolo.detect_objects_yolo(image_path, confidence)
        
        # Step 2: Generate caption using VLM (if API key is configured)
        caption = None
        if settings.OPENROUTER_API_KEY:
            try:
                caption = await self.vlm.generate_caption(image_path, objects)
            except Exception as e:
                logger.error(f"VLM Error: {str(e)}")
        
        # Step 3: Generate speech audio (if caption is available)
        speech_file = None
        if caption:
            try:
                speech_file = self.tts.get_speech(caption)
            except Exception as e:
                logger.error(f"TTS failed: {str(e)}")
        
        # Return results in a consistent format
        return objects, caption, speech_file