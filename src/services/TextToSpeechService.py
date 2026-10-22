import logging
import os
import time
from typing import Any

from gtts import gTTS

from schemas.tts_schemas import SpeechResult
from .BaseController import BaseController

logger = logging.getLogger(__name__)


class TextToSpeechService(BaseController):
    """Service for generating and caching text-to-speech audio files using gTTS."""

    def __init__(self, default_language: str = "en"):
        super().__init__()
        self.default_language = default_language
        self.cache: dict[tuple[str, str, str], str] = {}
        self.pipeline_cache: dict[tuple[str, str, float, str], dict[str, Any]] = {}
        os.makedirs(self.tts_output_dir, exist_ok=True)
        logger.info(
            f"TTS service configured: default_language='{self.default_language}', "
            f"output_dir='{self.tts_output_dir}'"
        )

    def build_speech_output_path(self, project_id: str, file_id: str) -> tuple[str, str]:
        """Build the output .mp3 path inside tts_output/<project_id>/ using the image's base name."""
        safe_project_id = os.path.basename(str(project_id).strip())
        if not safe_project_id:
            logger.error("Attempted to build speech path with an empty project_id")
            raise ValueError("A valid project_id is required")

        safe_file_id = os.path.basename(str(file_id).strip())
        stem, _ = os.path.splitext(safe_file_id)
        if not stem:
            logger.error("Attempted to build speech path with an empty file_id")
            raise ValueError("A valid file_id is required")

        project_tts_dir = os.path.join(self.tts_output_dir, safe_project_id)
        os.makedirs(project_tts_dir, exist_ok=True)

        file_name = f"{stem}.mp3"
        output_path = os.path.join(project_tts_dir, file_name)
        logger.info(f"Resolved TTS output path: '{output_path}'")
        return output_path, file_name

    def synthesize_file(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        slow: bool = False,
    ) -> str:
        """Create gTTS client and save synthesized audio directly to output_path."""
        if not text or not text.strip():
            logger.error("Attempted to synthesize speech from empty text")
            raise ValueError("Text cannot be empty")

        start_synth = time.perf_counter()
        client = gTTS(text=text.strip(), lang=language, slow=slow)
        client.save(output_path)
        synth_duration = time.perf_counter() - start_synth
        logger.info(f"gTTS synthesis completed in {synth_duration:.2f}s")
        return output_path

    def generate_speech(
        self,
        text: str,
        project_id: str,
        file_id: str,
        language: str | None = None,
        use_cache: bool = True,
    ) -> SpeechResult:
        """Validate text, manage caching, create the audio file in project's tts_output folder, and return SpeechResult."""
        start_time = time.perf_counter()

        normalized_text = text.strip() if text else ""
        if not normalized_text:
            logger.error("Empty text passed to generate_speech()")
            raise ValueError("Text cannot be empty")

        lang = (language or self.default_language).strip()
        output_path, _ = self.build_speech_output_path(project_id=project_id, file_id=file_id)

        cache_key = (str(project_id).strip(), os.path.basename(output_path), lang)
        cached_path = self.cache.get(cache_key)
        if use_cache and cached_path and os.path.isfile(cached_path):
            logger.info(f"Serving cached TTS audio from: '{cached_path}'")
            return self._build_result(cached_path, lang, True, start_time)

        logger.info(f"Synthesizing TTS audio (lang='{lang}') to: '{output_path}'...")
        result_path = self.synthesize_file(
            text=normalized_text,
            output_path=output_path,
            language=lang,
        )

        if use_cache:
            self.cache[cache_key] = result_path

        result = self._build_result(result_path, lang, False, start_time)
        logger.info(
            f"TTS generation finished: saved '{result.file_name}' "
            f"({result.file_size} bytes, Total time: {result.processing_time:.2f}s)"
        )
        return result

    def _build_pipeline_cache_key(
        self,
        project_id: str,
        file_id: str,
        confidence: float | None,
        language: str | None,
    ) -> tuple[str, str, float, str]:
        safe_project_id = os.path.basename(str(project_id).strip())
        safe_file_id = os.path.basename(str(file_id).strip())
        conf = (
            self.app_settings.DEFAULT_CONFIDENCE if confidence is None else float(confidence)
        )
        lang = (language or self.default_language).strip()
        return (safe_project_id, safe_file_id, conf, lang)

    def get_cached_pipeline(
        self,
        project_id: str,
        file_id: str,
        confidence: float | None = None,
        language: str | None = None,
    ) -> dict[str, Any] | None:
        """Return cached (detection, caption, speech) if available and MP3 file still exists on disk."""
        start_time = time.perf_counter()
        key = self._build_pipeline_cache_key(project_id, file_id, confidence, language)
        cached_entry = self.pipeline_cache.get(key)
        if not cached_entry:
            return None

        speech: SpeechResult = cached_entry["speech"]
        if not os.path.isfile(speech.file_path):
            self.pipeline_cache.pop(key, None)
            return None

        cached_speech = self._build_result(
            file_path=speech.file_path,
            language=speech.language,
            cached=True,
            start_time=start_time,
        )
        logger.info(
            f"Serving cached pipeline result for file '{file_id}' in project '{project_id}' "
            f"(audio: '{cached_speech.file_path}')"
        )
        return {
            "detection": cached_entry["detection"],
            "caption": cached_entry["caption"],
            "speech": cached_speech,
        }

    def set_cached_pipeline(
        self,
        project_id: str,
        file_id: str,
        confidence: float | None,
        language: str | None,
        detection: Any,
        caption: Any,
        speech: SpeechResult,
    ) -> None:
        """Store (detection, caption, speech) in pipeline cache."""
        key = self._build_pipeline_cache_key(project_id, file_id, confidence, language)
        self.pipeline_cache[key] = {
            "detection": detection,
            "caption": caption,
            "speech": speech,
        }

    def get_speech(
        self,
        text: str,
        project_id: str,
        file_id: str,
        language: str | None = None,
        use_cache: bool = True,
    ) -> SpeechResult:
        return self.generate_speech(
            text=text,
            project_id=project_id,
            file_id=file_id,
            language=language,
            use_cache=use_cache,
        )

    def clear_cache(self) -> None:
        self.cache.clear()
        self.pipeline_cache.clear()

    def get_speech_file_path(self, file_name: str, project_id: str) -> str | None:
        """Return the path of a generated audio file inside tts_output/<project_id>/, or None if not found."""
        safe_project_id = os.path.basename(str(project_id).strip())
        safe_name = os.path.basename(str(file_name).strip())
        if not safe_project_id or not safe_name or safe_name != file_name.strip():
            return None

        stem, ext = os.path.splitext(safe_name)
        if not stem:
            return None

        mp3_name = safe_name if ext.lower() == ".mp3" else f"{stem}.mp3"
        file_path = os.path.join(self.tts_output_dir, safe_project_id, mp3_name)
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
            url=None,  # URL will be set in the route handler
        )

