import asyncio
import base64
import mimetypes
import time
from collections.abc import Iterable
from typing import Any

from routes.schemes.DetectionSchemas import DetectedObject
from schemas.vlm_schemas import CaptionResult
from .BaseController import BaseController


class VLMController(BaseController):
    """Controller for requesting scene descriptions from the vision-language model."""

    def __init__(self, model_name: str | None = None, timeout: float = 60.0):
        super().__init__()
        self.model_name = model_name or self.app_settings.VLM_MODEL_NAME
        self.api_url = self.app_settings.VLM_API_URL
        self.api_key = self.app_settings.OPENROUTER_API_KEY.get_secret_value()
        self.timeout = timeout

    def set_model_name(self, model_name: str) -> str:
        """Update the target VLM model identifier."""
        if not model_name:
            raise ValueError("A VLM model name is required")
        self.model_name = model_name
        return self.model_name

    def get_model_name(self) -> str:
        """Return the target VLM model identifier."""
        return self.model_name

    async def generate_caption(
        self,
        image_path: str,
        objects: Iterable[DetectedObject],
    ) -> CaptionResult:
        """Prepare the vision prompt, send HTTP request to VLM API, and return CaptionResult."""
        start_time = time.perf_counter()

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

        image_data_url = self._encode_image(image_path)
        objects_description = self._describe_objects(objects)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_prompt(objects_description),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
        }

        result = await asyncio.to_thread(self._send_request, payload)
        choices = result.get("choices") or []
        caption = choices[0].get("message", {}).get("content") if choices else None
        if not caption or not caption.strip():
            raise RuntimeError("The VLM returned an empty caption")

        return CaptionResult(
            caption=caption.strip(),
            model_name=self.model_name,
            processing_time=time.perf_counter() - start_time,
        )

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send HTTP POST request to OpenRouter VLM endpoint."""
        import requests

        if not self.api_url:
            raise ValueError("VLM API URL is not configured")

        response = requests.post(
            self.api_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        return f"data:{mime_type};base64,{encoded_image}"

    @staticmethod
    def _describe_objects(objects: Iterable[DetectedObject]) -> str:
        descriptions = []
        for detected_object in objects:
            position = getattr(
                detected_object.position,
                "value",
                detected_object.position,
            )
            descriptions.append(
                f"{detected_object.object} on the {position} "
                f"({detected_object.confidence:.0%} confidence)"
            )
        return ", ".join(descriptions) or "No objects were confidently detected"

    @staticmethod
    def _build_prompt(objects_description: str) -> str:
        return (
            "Describe this image for a visually impaired user in one natural "
            "paragraph with no headings or bullet points. Start with the most "
            "prominent elements, describe people and important objects, include "
            "their relative positions, and infer the likely context carefully. "
            "Keep the description under five sentences. "
            f"Detected elements: {objects_description}."
        )
