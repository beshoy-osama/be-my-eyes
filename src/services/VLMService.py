import asyncio
import base64
import logging
import mimetypes
import os
import time
from collections.abc import Iterable
from typing import Any

import httpx

from schemas import DetectedObject, CaptionResponse
from models.prompts import _build_system_prompt, _build_user_prompt
from .BaseController import BaseController

logger = logging.getLogger(__name__)


class VLMService(BaseController):
    """Service for requesting scene descriptions from the vision-language model."""

    def __init__(self, timeout: float = 60.0):
        super().__init__()
        self.model_name: str
        self.api_url: httpx.URL
        self.api_key: str
        self.timeout: float = timeout
        self.client: httpx.AsyncClient | None = None

    async def set_model(
        self,
        model_name: str,
        api_url: str,
        api_key: str,
    ) -> str:
        """Validate and configure VLM model name, API URL, and API key."""
        if not model_name or not model_name.strip():
            logger.error("Attempted to set an empty VLM model name")
            raise ValueError("A VLM model name is required")

        if not api_url or not api_url.strip():
            logger.error("Attempted to set an empty VLM API URL")
            raise ValueError("A VLM API URL is required")

        if not api_key or not api_key.strip():
            logger.error("Attempted to set an empty VLM API key")
            raise ValueError("A VLM API key is required")

        self.model_name = model_name.strip()
        self.api_url = httpx.URL(api_url.strip().rstrip("/"))
        self.api_key = api_key.strip()

        if self.client is not None:
            await self.client.aclose()
            self.client = None  # Reset client so new configuration loads on next get_model()

        logger.info(f"VLM model configured: '{self.model_name}'")
        return self.model_name

    def get_model_name(self) -> str:
        """Return the configured model name, setting defaults from configuration if not yet set."""
        if not self.model_name or not self.api_url or not self.api_key:
            self.model_name = self.app_settings.VLM_MODEL_NAME.strip()
            self.api_url = httpx.URL(self.app_settings.VLM_API_URL.strip().rstrip("/"))
            self.api_key = self.app_settings.OPENROUTER_API_KEY.get_secret_value().strip()

        if not self.model_name or not self.api_url or not self.api_key:
            raise ValueError("VLM model name, API URL, and API key must be set")

        return self.model_name

    def get_model(self) -> httpx.AsyncClient:
        """Getter for the initialized VLM HTTP client."""
        if self.client is None:
            self.get_model_name()
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            logger.info(f"Initialized VLM HTTP client for model '{self.model_name}'")

        return self.client

    async def aclose(self) -> None:
        """Gracefully close the underlying HTTP connections."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    def preprocess_inputs(
        self,
        image_path: str,
        objects: Iterable[DetectedObject],
    ) -> dict[str, Any]:
        """Encode image, format detected objects, and build the VLM request payload."""
        image_data_url = self._encode_image(image_path)
        objects_description = self._describe_objects(objects)

        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": _build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _build_user_prompt(objects_description),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ],
        }

    def postprocess_response(self, result: dict[str, Any]) -> str:
        """Extract and validate the generated caption text from the VLM API response."""
        choices = result.get("choices") or []
        message_content = choices[0].get("message", {}).get("content") if choices else None

        if isinstance(message_content, list):
            caption = " ".join(
                part.get("text", "")
                for part in message_content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif isinstance(message_content, str):
            caption = message_content
        else:
            caption = None

        if not caption or not caption.strip():
            logger.error("The VLM returned an empty caption")
            raise RuntimeError("The VLM returned an empty caption")

        return caption.strip()

    async def generate_caption(
        self,
        image_path: str,
        objects: Iterable[DetectedObject],
    ) -> CaptionResponse:
        """Run VLM caption generation with validation, preprocessing, and postprocessing."""
        start_time = time.perf_counter()

        if not image_path or not isinstance(image_path, str) or not image_path.strip():
            logger.error("Empty or invalid image_path passed to generate_caption()")
            raise ValueError("A valid image_path is required")

        if not os.path.isfile(image_path):
            logger.error(f"Image file not found at path: '{image_path}'")
            raise FileNotFoundError(f"Image not found at path: '{image_path}'")

        client = self.get_model()

        logger.info(f"Preparing image and prompt for VLM: {image_path}")
        payload = await asyncio.to_thread(self.preprocess_inputs, image_path, objects)

        logger.info(f"Sending caption request to VLM model '{self.model_name}'...")
        start_infer = time.perf_counter()

        result = await self._send_request(client, payload)

        infer_duration = time.perf_counter() - start_infer
        logger.info(f"VLM response received in {infer_duration:.2f}s")

        caption = self.postprocess_response(result)
        actual_model= result.get("model")
        total_duration = time.perf_counter() - start_time

        logger.info(
            f"Caption generated successfully by '{actual_model}' (Total time: {total_duration:.2f}s)"
        )

        return CaptionResponse(
            caption=caption,
            model_name=actual_model,
            processing_time=total_duration
        )

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send asynchronous HTTP POST request using httpx connection pool."""
        try:
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"VLM API returned status {exc.response.status_code}: {exc.response.text}"
            )
            raise RuntimeError(
                f"VLM API returned status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error(f"VLM network request failed: {exc}")
            raise RuntimeError(f"VLM network request failed: {str(exc)}") from exc

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
