"""
Vision Language Model (VLM) Manager
Implements the Model layer for vision-language processing with OpenRouter API.

Key responsibilities:
- Manages communication with OpenRouter's Llama Vision API
- Handles image encoding and prompt engineering
- Provides error handling for external API calls
- Integrates with the project's configuration system
"""

import base64
import requests
import asyncio
import logging
from typing import Optional
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

class VLMModelManager:
    """Manager for Vision Language Model with OpenRouter API integration"""
    
    def __init__(self):
        """Initialize VLM manager with API configuration"""
        self.api_key = settings.OPENROUTER_API_KEY.get_secret_value()
        self.api_url = settings.VLM_API_URL
        self.model_name = settings.VLM_MODEL_NAME
        # Initialize with project configuration values
    
    async def generate_caption(self, image_path: str, objects_desc: str) -> Optional[str]:
        """
        Generate accessibility-focused caption using Llama Vision
        
        Args:
            image_path: Path to the image file for caption generation
            objects_desc: YOLO-generated object description (used in prompt)
            
        Returns:
            Optional[str]: Generated caption or None if:
                - API key is missing
                - API call fails
                - No valid response is received
        
        Process flow:
            1. Validate API key availability
            2. Encode image to base64 for API request
            3. Construct optimized prompt for visually impaired users
            4. Send request to OpenRouter API
            5. Extract and return caption from response
        
        Error handling:
            - Logs all API errors at ERROR level
            - Returns None on failure (safe for controller layer)
        """
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. Skipping VLM caption generation.")
            return None
        
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Construct accessibility-focused prompt
            prompt_text = (
                "Describe this image for a visually impaired user in one smooth, continuous paragraph — no headings, no bullet points, no line breaks. "
                "Use natural, conversational language that paints a clear mental picture. "
                "Start with the most prominent elements (highest confidence), then describe others in order of importance. "
                "Mention people: how many, where they are (left/center/right), and what they might be doing or feeling. "
                "Describe key objects (like tables, signs, decorations) and their placement. "
                "Suggest the context — is it a celebration? A meal? A meeting? — based on what’s visible. "
                "Keep it under 5 sentences, but rich in useful sensory and spatial details. "
                f"Key elements in order of prominence: {objects_desc}."
            )

            # Prepare API request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ]
            }

            # Set up request headers with authentication
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Execute API call in background thread to avoid blocking
            response = await asyncio.to_thread(
                requests.post, 
                self.api_url, 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract caption from response if available
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            
            return None

        except Exception as e:
            # Log detailed error information for debugging
            logger.error(f"VLM Error: {e}")
            return None