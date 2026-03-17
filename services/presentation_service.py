"""
Presentation Service for SlideBron Integration
Handles presentation generation requests.
"""
import logging
import os
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class PresentationService:
    """Service for generating presentations using SlideBron-style logic."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the presentation service.
        
        Args:
            output_dir: Directory to save generated presentations
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        logger.info(f"PresentationService initialized with output_dir={output_dir}")
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def create_presentation_request(
        self,
        topic: str,
        content: Optional[str] = None,
        num_slides: int = 7,
        theme: str = "professional",
        enable_search: bool = True,
        enable_images: bool = True
    ) -> Dict:
        """
        Create a presentation generation request.
        
        Args:
            topic: Presentation topic
            content: Optional content to base slides on
            num_slides: Number of slides to generate
            theme: Visual theme (professional, modern, minimal, dark)
            enable_search: Enable web search for additional research
            enable_images: Include stock images
            
        Returns:
            Request configuration dictionary
        """
        request = {
            "topic": topic,
            "content": content,
            "num_slides": num_slides,
            "theme": theme,
            "enable_search": enable_search,
            "enable_images": enable_images,
            "output_dir": self.output_dir
        }
        
        logger.info(f"Created presentation request for topic: {topic}")
        return request
    
    def get_available_themes(self) -> List[str]:
        """Get list of available presentation themes."""
        return ["professional", "modern", "minimal", "dark", "cyber"]
