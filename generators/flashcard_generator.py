"""
Flashcard Generator using AI
Generates flashcards from text content using LLM.
"""
import logging
from typing import List, Dict
import json

logger = logging.getLogger(__name__)


class FlashcardGenerator:
    """AI-powered flashcard generator."""
    
    def __init__(self, llm):
        """
        Initialize the flashcard generator.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        logger.info("FlashcardGenerator initialized")
    
    def generate_from_text(self, text: str, num_cards: int = 10, topic: str = None) -> List[Dict]:
        """
        Generate flashcards from text content.
        
        Args:
            text: Source text to generate flashcards from
            num_cards: Number of flashcards to generate
            topic: Optional topic for the flashcards
            
        Returns:
            List of flashcard dictionaries with 'question' and 'answer'
        """
        prompt = f"""Generate {num_cards} high-quality flashcards from the following text. 
Each flashcard should have a clear question and a concise answer.
Focus on key concepts, definitions, and important facts.

Format your response as a JSON array of objects with 'question' and 'answer' fields.

Text:
{text}

Generate exactly {num_cards} flashcards in this JSON format:
[
  {{"question": "What is...", "answer": "..."}},
  {{"question": "How does...", "answer": "..."}}
]
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            # Look for JSON array in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                flashcards = json.loads(json_str)
                logger.info(f"Generated {len(flashcards)} flashcards from text")
                return flashcards
            else:
                logger.warning("Could not parse flashcards from LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            return []
    
    def generate_from_conversation(self, messages: List[Dict], num_cards: int = 10) -> List[Dict]:
        """
        Generate flashcards from a conversation.
        
        Args:
            messages: List of conversation messages
            num_cards: Number of flashcards to generate
            
        Returns:
            List of flashcard dictionaries
        """
        # Extract text from conversation
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        ])
        
        return self.generate_from_text(conversation_text, num_cards)
