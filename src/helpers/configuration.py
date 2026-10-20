from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES : List[str]
    FILE_MAX_SIZE : int

    FILE_DEFAULT_CHUNK_SIZE: int


    DEFAULT_CONFIDENCE: float
    MIN_CONFIDENCE: float 
    MAX_CONFIDENCE: float


    OPENROUTER_API_KEY: SecretStr = SecretStr("")
    VLM_MODEL_NAME: str 
    VLM_API_URL: str

    YOLO_MODEL_NAME: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


def get_settings() -> Settings:
    """Return the same configuration instance to all app layers."""
    return Settings()
