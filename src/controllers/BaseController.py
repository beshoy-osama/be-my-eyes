import os
import random
import string
from helpers import get_settings


class BaseController:
    """Base controller providing common configuration and filesystem paths."""

    def __init__(self):
        self.app_settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(__file__))  # src directory

        self.files_dir = os.path.join(self.base_dir, "assets/files")
        self.yolo_model_dir = os.path.join(self.base_dir, "assets/models")
        self.yolo_model_path = os.path.join(
            self.yolo_model_dir,
            self.app_settings.YOLO_MODEL_NAME,
        )
        self.tts_output_dir = os.path.join(self.base_dir, "assets/tts_output")

    @staticmethod
    def generate_random_string(length: int = 12) -> str:
        """Generate a random alphanumeric string."""
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
