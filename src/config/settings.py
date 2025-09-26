from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Main application settings"""

# File Upload Configuration
    MAX_FILE_SIZE: int
    ALLOWED_EXTENSIONS: list[str]
    CLEANUP_INTERVAL: int
    UPLOAD_FOLDER: str 

# Detection Configuration
    DEFAULT_CONFIDENCE: float
    MIN_CONFIDENCE: float
    MAX_CONFIDENCE: float

# CORS Configuration
    CORS_ORIGINS: List[str] 
    CORS_METHODS: List[str] 
    CORS_HEADERS: List[str] 
    CORS_CREDENTIALS: bool 

# VLM Settings 
    OPENROUTER_API_KEY: SecretStr
    VLM_MODEL_NAME: str 
    VLM_API_URL: str

# YOLO Configuration
    YOLO_MODEL_NAME: str 


    # TTS Settings
    TTS_OUTPUT_DIR: str = "tts_output" 


    model_config= SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )
        
settings = Settings()

@lru_cache()
def get_settings():
    return settings