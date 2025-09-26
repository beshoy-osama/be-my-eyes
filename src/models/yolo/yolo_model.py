"""
YOLO Model Manager
Implements the Model layer for YOLO object detection with caching.

Key responsibilities:
- Manages YOLO model loading and caching
- Processes detection results into structured objects
- Handles position determination and description generation
- Provides confidence-based filtering
"""

from PIL import Image
import logging
from ultralytics import YOLO
from src.config.settings import settings
from src.schemas.detection_schemas import DetectedObject, Position
import os

logger = logging.getLogger(__name__)

class YOLOModelManager:
    """
    Manager for YOLO models with built-in caching for performance optimization
    
    This class handles:
    - Model loading with cache management
    - Object detection execution
    - Position determination of detected objects
    - Structured result formatting
    """
    
    def __init__(self):
        """Initialize the model manager with empty cache"""
        self._models_cache = {}
    
    def load_model(self, model_name: str = None) -> YOLO:
        """
        Load YOLO model with caching mechanism
        
        Args:
            model_name: Name of the model file to load (default: settings.YOLO_MODEL_NAME)
            
        Returns:
            YOLO: The loaded YOLO model instance
            
        Raises:
            RuntimeError: If model loading fails
        """
        model_name = model_name or settings.YOLO_MODEL_NAME
        
        # Check cache first to avoid redundant loading
        if model_name not in self._models_cache:
            try:
                model_path = self._get_model_path(model_name)
                self._models_cache[model_name] = YOLO(model_path)
                logger.info(f"YOLO model '{model_name}' loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model '{model_name}': {e}")
                raise RuntimeError(f"Model loading failed: {str(e)}")
        
        return self._models_cache[model_name]
    
    def _get_model_path(self, model_name: str) -> str:
        """
        Find model path in multiple possible locations
        
        Args:
            model_name: Name of the model file
            
        Returns:
            str: Full path to model file, or model_name if not found locally
        """
        # Check multiple possible paths for the model
        possible_paths = [
            model_name,
            f"models/yolo/{model_name}",
            f"src/models/yolo/{model_name}",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "yolo", model_name)
        ]
        
        # Return first existing path
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no local path found, return model name (will trigger download)
        logger.warning(f"Model file '{model_name}' not found locally, YOLO will attempt to download")
        return model_name
    
    def detect_objects_yolo(self, image_path: str, confidence: float = 0.5) -> tuple:
        """
        Detect objects in image and return structured results
        
        Args:
            image_path: Path to the image file for detection
            confidence: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            tuple: (list of DetectedObject)
        """
        # Ensure model is loaded
        model = self.load_model()
        img = Image.open(image_path).convert("RGB")
        img_width, _ = img.size
        
        # Run detection
        results = model(img)
        objects = []
        
        for box in results[0].boxes:
            # Parse bounding box coordinates
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            
            # Determine object position
            x_center = (x_min + x_max) / 2
            position = self._determine_position(x_center, img_width)
            
            # Create validated object
            detected_obj = DetectedObject(
                object=model.names[int(box.cls[0])],
                position=position,
                confidence=float(box.conf[0]),
            )
            objects.append(detected_obj)
        
        return objects
    
    def _determine_position(self, x_center: float, img_width: int) -> str:
        """
        Determine object position in image (left/center/right)
        
        Args:
            x_center: X-coordinate of object's center
            img_width: Width of the image
            
        Returns:
            str: Position category (left/center/right)
        """
        if x_center < img_width / 3:
            return Position.left.value
        elif x_center > 2 * img_width / 3:
            return Position.right.value
        return Position.center.value