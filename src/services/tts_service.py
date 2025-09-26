"""
Text-to-Speech Service
Implements the service layer for text-to-speech conversion with caching.

Key responsibilities:
- Coordinates between TTS model and application logic
- Manages audio file caching for performance optimization
- Handles error cases gracefully
- Provides consistent interface for controllers

This service follows the MVC architecture as the service layer between:
- Models (TTSModel): Handles actual speech generation
- Controllers: Handle API requests and responses
"""

from src.models.tts.tts_model import TTSModel

class TTSService:
    """Service layer for text-to-speech conversion with caching"""
    
    def __init__(self):
        """Initialize TTS service with model and cache"""
        self.tts = TTSModel()
        self.cache = {}
        # Initialize with empty cache to store generated audio files
    
    def get_speech(self, text: str, use_cache: bool = True) -> str:
        """
        Get speech file path (from cache or generate new)
        
        Args:
            text: Text to convert to speech
            use_cache: Whether to use cached results (default: True)
            
        Returns:
            str: Path to audio file
            
        Process flow:
            1. Check if text is in cache (if caching enabled)
            2. If found, return cached file path
            3. If not found, generate new speech file
            4. Store in cache if caching enabled
            5. Return file path
            
        Error handling:
            - Catches any exceptions during speech generation
            - Re-raises as RuntimeError with descriptive message
        """
        # Check cache first if caching is enabled
        if use_cache and text in self.cache:
            return self.cache[text]
        
        try:
            # Generate new speech file
            file_path = self.tts.generate_speech(text)
            
            # Store in cache if requested
            if use_cache:
                self.cache[text] = file_path
                
            return file_path
        except Exception as e:
            # Handle any errors during speech generation
            raise RuntimeError(f"TTS Service Error: {str(e)}")
    
    def clear_cache(self):
        """Clear all cached audio files from memory
        
        This method:
            - Resets the in-memory cache
            - Does not delete physical files from disk
            - Helps manage memory usage when cache grows large
        """
        self.cache.clear()