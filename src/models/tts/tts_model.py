"""
Text-to-Speech Model
Implements the Model layer for text-to-speech conversion.

Key responsibilities:
- Manages audio file storage and naming
- Handles text-to-speech conversion using gTTS
- Provides error handling for speech generation
- Ensures proper directory structure for audio files
"""

import os
import uuid
from gtts import gTTS
from config.settings import settings

class TTSModel:
    """Text-to-Speech Model implementation with audio file management"""
    
    def __init__(self):
        """Initialize TTS model with output directory setup"""
        self.output_dir = self.get_tts_output_dir()
        # Initialize with absolute path to ensure correct directory
    
    def get_tts_output_dir(self) -> str:
        """
        Get absolute path to tts_output directory inside src
        
        Returns:
            str: Absolute path to the tts_output directory
        """
        # Get absolute path to current file (tts_model.py)
        current_file_path = os.path.abspath(__file__)
        
        # Navigate up 3 levels to reach project root (src directory)
        # __file__ -> tts_model.py -> models/tts/ -> models/ -> src/
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        
        # Create absolute path to tts_output
        output_dir = os.path.join(src_dir, settings.TTS_OUTPUT_DIR)
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir

    def generate_speech(self, text: str) -> str:
        """
        Convert text to speech audio file
        
        Args:
            text: Input text to convert to speech
            
        Returns:
            str: Path to the generated audio file
            
        Raises:
            ValueError: If input text is empty
            RuntimeError: If speech generation fails
            
        Process:
            1. Validate input text
            2. Generate unique filename
            3. Convert text to speech using gTTS
            4. Save audio file to output directory
            5. Return file path
        """
        # Validate input text
        if not text:
            raise ValueError("Empty text cannot be converted to speech")
        
        # Generate unique filename
        file_name = f"{uuid.uuid4()}.mp3"
        file_path = os.path.join(self.output_dir, file_name)
        
        try:
            # Convert text to speech using gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save the audio file
            tts.save(file_path)
            
            return file_path
        
        except Exception as e:
            # Handle any errors during speech generation
            raise RuntimeError(f"Speech generation failed: {str(e)}")