import logging
import os
import threading
import time
from typing import Any

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from schemas import DetectedObject, DetectionResult
from models.enums import PositionEnums
from .BaseController import BaseController

logger = logging.getLogger(__name__)


class YOLOService(BaseController):
    """Controller for running object detection with YOLO."""

    def __init__(self):
        super().__init__()
        self.model_path: str = None
        self.client: YOLO = None
        self.device: str = self.app_settings.device
        self._lock = threading.Lock()

    def set_model_path(self, model_name: str) -> str:
        """Resolve YOLO model path from name and verify file exists on disk."""
        if not model_name or not model_name.strip():
            logger.error("Attempted to set an empty YOLO model name")
            raise ValueError("A YOLO model name is required")

        model_name = model_name.strip()

        # Resolve path using BaseController's yolo_model_dir
        if os.path.isabs(model_name) and os.path.isfile(model_name):
            model_path = model_name
        else:
            model_path = os.path.join(
                self.yolo_model_dir,
                os.path.basename(model_name),
            )

        if not os.path.isfile(model_path):
            logger.error(f"YOLO model file '{model_name}' was not found at path: '{model_path}'")
            raise FileNotFoundError(
                f"YOLO model file '{model_name}' was not found at path: '{model_path}'"
            )

        with self._lock:
            if self.model_path != model_path:
                self.model_path = model_path
                self.client = None  # Reset client so new model weights load on next get_model()
                logger.info(f"YOLO model path set to: '{self.model_path}'")

        return self.model_path

    def get_model_path(self) -> str:
        """Return the resolved model path, setting default from configuration if not yet set."""
        if not self.model_path:
            self.set_model_path(self.app_settings.YOLO_MODEL_NAME)

        return self.model_path

    def get_model(self) -> Any:
        """Thread-safe getter for the loaded YOLO client."""
        if self.client is None:
            resolved_path = self.get_model_path()
            with self._lock:
                if self.client is None:
                    if not os.path.isfile(resolved_path):
                        logger.error(f"YOLO model file was not found at path: '{resolved_path}'")
                        raise FileNotFoundError(
                            f"YOLO model file was not found at path: '{resolved_path}'"
                        )

                    try:
                        logger.info(f"Loading YOLO weights from '{resolved_path}'...")
                        self.client = YOLO(resolved_path)
                        logger.info("YOLO weights loaded successfully")
                    except Exception as e:
                        logger.exception(f"Failed to load YOLO model from '{resolved_path}': {e}")
                        raise

        return self.client

    def preprocess_image(self, image_path: str) -> tuple[Image.Image, int, int]:
        """Open an image file and convert it to RGB format, returning the image and its dimensions."""
        try:
            with Image.open(image_path) as img:
                rgb_image = img.convert("RGB")
                width, height = rgb_image.width, rgb_image.height
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Invalid or corrupted image file at '{image_path}': {e}")
            raise ValueError(f"Invalid or corrupted image file: '{image_path}'") from e

        if width <= 0 or height <= 0:
            logger.error(f"Image at '{image_path}' has invalid dimensions: {width}x{height}")
            raise ValueError(f"Image has invalid dimensions: {width}x{height}")

        return rgb_image, width, height

    def postprocess_predictions(self, predictions: Any, image_width: int) -> list[DetectedObject]:
        objects: list[DetectedObject] = []
        if not predictions or len(predictions) == 0:
            logger.info("YOLO returned no prediction results")
            return objects

        first_pred = predictions[0]
        class_names = getattr(first_pred, "names", None) or self.get_model().names

        if hasattr(first_pred, "boxes") and first_pred.boxes is not None:
            for box in first_pred.boxes:
                x_min, _, x_max, _ = box.xyxy[0].tolist()
                x_center = (x_min + x_max) / 2
                class_id = int(box.cls[0])
                conf_score = float(box.conf[0])

                object_name = class_names.get(class_id) if isinstance(class_names, dict) else None
                if not object_name:
                    logger.warning(f"Unknown class_id {class_id} in YOLO predictions; skipping box")
                    continue

                objects.append(
                    DetectedObject(
                        object=object_name,
                        position=self._determine_position(x_center, image_width),
                        confidence=conf_score,
                    )
                )

        return objects

    def predict(self, image_path: str, confidence: float | None = None) -> DetectionResult:
        """Run YOLO prediction with handled validation and thread safety."""
        start_time = time.perf_counter()

        if not image_path or not isinstance(image_path, str) or not image_path.strip():
            logger.error("Empty or invalid image_path passed to predict()")
            raise ValueError("A valid image_path is required")

        if not os.path.isfile(image_path):
            logger.error(f"Image file not found at path: '{image_path}'")
            raise FileNotFoundError(f"Image not found at path: '{image_path}'")

        confidence = (
            self.app_settings.DEFAULT_CONFIDENCE if confidence is None else confidence
        )
        if confidence is None or not (0.0 < confidence <= 1.0):
            logger.error(f"Invalid confidence threshold: {confidence}. Must be in (0.0, 1.0]")
            raise ValueError(f"Confidence threshold must be between 0.0 (exclusive) and 1.0, got {confidence}")

        client = self.get_model()

        logger.info(f"Reading image file: {image_path}")
        rgb_image, image_width, image_height = self.preprocess_image(image_path)

        logger.info(f"Executing YOLO inference on device='{self.device}' with confidence={confidence}...")
        start_infer = time.perf_counter()

        try:
            with self._lock:
                predictions = client.predict(
                    source=rgb_image,
                    conf=confidence,
                    imgsz=640,
                    device=self.device,
                    verbose=False,
                )
        except Exception as e:
            logger.exception(f"YOLO inference failed for image '{image_path}': {e}")
            raise

        infer_duration = time.perf_counter() - start_infer
        logger.info(f"Inference completed in {infer_duration:.2f}s")

        objects = self.postprocess_predictions(predictions, image_width)
        total_duration = time.perf_counter() - start_time

        logger.info(
            f"Detection finished: found {len(objects)} objects (Total time: {total_duration:.2f}s)"
        )

        return DetectionResult(
            objects=objects,
            total_objects=len(objects),
            confidence_threshold=confidence,
            image_width=image_width,
            image_height=image_height,
            model_name=os.path.basename(self.model_path) if self.model_path else self.app_settings.YOLO_MODEL_NAME,
            processing_time=total_duration,
        )


    @staticmethod
    def _determine_position(x_center: float, image_width: int) -> PositionEnums:
        if x_center < image_width / 3:
            return PositionEnums.left
        if x_center > (2 * image_width) / 3:
            return PositionEnums.right
        return PositionEnums.center

