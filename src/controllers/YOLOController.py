import logging
import os
import threading
import time
from typing import Any

from PIL import Image
from ultralytics import YOLO

from schemas.detection_schemas import DetectedObject, DetectionResult, Position
from .BaseController import BaseController

logger = logging.getLogger(__name__)


class YOLOController(BaseController):
    """Controller for running object detection with YOLO."""

    def __init__(self):
        super().__init__()
        self.model_path: str = None
        self.client: YOLO = None
        self.device: str = "cpu"
        self._lock = threading.Lock()

    def set_model_path(self, model_name: str) -> str:
        """Resolve YOLO model path from name and verify file exists on disk."""
        if not model_name:
            raise ValueError("A YOLO model name is required")

        # Resolve path using BaseController's yolo_model_dir
        if os.path.isabs(model_name) and os.path.exists(model_name):
            model_path = model_name
        else:
            model_path = os.path.join(
                self.yolo_model_dir,
                os.path.basename(model_name),
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model file '{model_name}' was not found at path: '{model_path}'"
            )

        self.model_path = model_path
        return self.model_path

    def get_model_path(self) -> str:
        """Return the resolved model path, setting default from configuration if not yet set."""
        if not self.model_path:
            self.set_model_path(self.app_settings.YOLO_MODEL_NAME)
        return self.model_path

    def get_model(self) -> Any:
        """Thread-safe getter for the loaded YOLO client."""
        if self.client is None:
            with self._lock:
                if self.client is None:
                    resolved_path = self.get_model_path()
                    print(f"[{time.strftime('%X')}] 1. Loading YOLO model weights from: {resolved_path}...")
                    start_load = time.perf_counter()
                    
                    self.device = "cpu"
                    self.client = YOLO(resolved_path)
                    print(f"[{time.strftime('%X')}] 2. YOLO weights loaded successfully in {time.perf_counter() - start_load:.2f}s!")
        return self.client


    def preprocess_image(self, image_path: str) -> tuple[Any, int, int]:
        image = Image.open(image_path).convert("RGB")
        return image, image.width, image.height


    def postprocess_predictions(self, predictions, image_width: int) -> list[DetectedObject]:
        objects: list[DetectedObject] = []
        if not predictions or len(predictions) == 0:
            return objects

        model_client = self.get_model()
        first_pred = predictions[0]

        if hasattr(first_pred, "boxes") and first_pred.boxes is not None:
            for box in first_pred.boxes:
                x_min, _, x_max, _ = box.xyxy[0].tolist()
                x_center = (x_min + x_max) / 2
                class_id = int(box.cls[0])

                objects.append(
                    DetectedObject(
                        object=model_client.names[class_id],
                        position=self._determine_position(x_center, image_width),
                        confidence=float(box.conf[0]),
                    )
                )

        return objects

    
    def _validate_confidence(self, confidence: float) -> None:
        if not (
            self.app_settings.MIN_CONFIDENCE <= confidence <= self.app_settings.MAX_CONFIDENCE
        ):
            raise ValueError(
                "Confidence must be between "
                f"{self.app_settings.MIN_CONFIDENCE} and "
                f"{self.app_settings.MAX_CONFIDENCE}"
            )

        
    def predict(self, image_path: str, confidence: float | None = None) -> DetectionResult:
        """Run YOLO prediction with handled validation and thread safety."""
        start_time = time.perf_counter()

        selected_confidence = (
            self.app_settings.DEFAULT_CONFIDENCE if confidence is None else confidence
        )
        self._validate_confidence(selected_confidence)

        client = self.get_model()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: '{image_path}'")

        print(f"[{time.strftime('%X')}] Reading image file: {image_path}")

        rgb_image, image_width, image_height = self.preprocess_image(image_path)

        print(f"[{time.strftime('%X')}] Executing YOLO inference on {self.device}...")

        start_infer = time.perf_counter()

        predictions = client.predict(
                source=rgb_image,
                conf=selected_confidence,
                imgsz=640,
                device=self.device,
                verbose=False,
            )

        print(f"[{time.strftime('%X')}] Inference completed in {time.perf_counter() - start_infer:.2f}s")

        objects = self.postprocess_predictions(predictions, image_width)
        
        print(f"[{time.strftime('%X')}] Detection finished: found {len(objects)} objects (Total request time: {time.perf_counter() - start_time:.2f}s)")

        return DetectionResult(
            objects=objects,
            total_objects=len(objects),
            confidence_threshold=selected_confidence,
            image_width=image_width,
            image_height=image_height,
            model_name=self.app_settings.YOLO_MODEL_NAME,
            processing_time=time.perf_counter() - start_time,
        )


    @staticmethod
    def _determine_position(x_center: float, image_width: int) -> Position:
        if x_center < image_width / 3:
            return Position.left
        if x_center > (2 * image_width) / 3:
            return Position.right
        return Position.center

