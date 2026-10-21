import os
import time
import uuid

from schemas.tts_schemas import SpeechResult
from .BaseController import BaseController


class TextToSpeechController(BaseController):
    """Controller for generating and caching text-to-speech audio files using gTTS."""

    def __init__(self, default_language: str = "en"):
        super().__init__()
        self.default_language = default_language
        self.cache: dict[tuple[str, str], str] = {}
        os.makedirs(self.tts_output_dir, exist_ok=True)

    def synthesize_file(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        slow: bool = False,
    ) -> str:
        """Create gTTS client and save synthesized audio directly to output_path."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        from gtts import gTTS

        client = gTTS(text=text.strip(), lang=language, slow=slow)
        client.save(output_path)
        return output_path

    def generate_speech(
        self,
        text: str,
        language: str = "en",
        use_cache: bool = True,
    ) -> SpeechResult:
        """Validate text, manage caching, create the audio file, and return SpeechResult."""
        start_time = time.perf_counter()

        normalized_text = text.strip() if text else ""
        if not normalized_text:
            raise ValueError("Text cannot be empty")

        cache_key = (normalized_text, language)
        cached_path = self.cache.get(cache_key)
        if use_cache and cached_path and os.path.exists(cached_path):
            return self._build_result(cached_path, language, True, start_time)

        output_path = os.path.join(
            self.tts_output_dir,
            f"{uuid.uuid4()}.mp3",
        )

        result_path = self.synthesize_file(
            text=normalized_text,
            output_path=output_path,
            language=language,
        )

        if use_cache:
            self.cache[cache_key] = result_path
        return self._build_result(result_path, language, False, start_time)

    def get_speech(
        self,
        text: str,
        language: str = "en",
        use_cache: bool = True,
    ) -> SpeechResult:
        return self.generate_speech(text, language, use_cache)

    def clear_cache(self) -> None:
        self.cache.clear()

    def get_speech_file_path(self, file_name: str) -> str | None:
        """Return the path of a generated audio file, or None if it does not exist."""
        safe_name = os.path.basename(file_name)
        if not safe_name or safe_name != file_name or not safe_name.endswith(".mp3"):
            return None

        file_path = os.path.join(self.tts_output_dir, safe_name)
        return file_path if os.path.isfile(file_path) else None

    @staticmethod
    def _build_result(
        file_path: str,
        language: str,
        cached: bool,
        start_time: float,
    ) -> SpeechResult:
        return SpeechResult(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            language=language,
            cached=cached,
            processing_time=time.perf_counter() - start_time,
        )
