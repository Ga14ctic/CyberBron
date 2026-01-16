"""
Flashcard Service for Flashcard Management
Handles CRUD operations and spaced repetition tracking for flashcards.
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class FlashcardService:
    """Service for managing flashcards and spaced repetition."""
    
    def __init__(self, flashcards_dir: str = "flashcards"):
        """
        Initialize the flashcard service.
        
        Args:
            flashcards_dir: Directory to store flashcard files
        """
        self.flashcards_dir = flashcards_dir
        self.flashcards_file = os.path.join(flashcards_dir, "flashcards.json")
        
        self._ensure_flashcards_dir()
        logger.info(f"FlashcardService initialized with flashcards_dir={flashcards_dir}")
    
    def _ensure_flashcards_dir(self):
        """Ensure flashcards directory exists."""
        if not os.path.exists(self.flashcards_dir):
            os.makedirs(self.flashcards_dir)
            logger.info(f"Created flashcards directory: {self.flashcards_dir}")
    
    def _load_flashcards(self) -> List[Dict]:
        """Load all flashcards from storage."""
        if os.path.exists(self.flashcards_file):
            try:
                with open(self.flashcards_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading flashcards: {e}")
                return []
        return []
    
    def _save_flashcards(self, flashcards: List[Dict]):
        """Save flashcards to storage."""
        try:
            with open(self.flashcards_file, 'w', encoding='utf-8') as f:
                json.dump(flashcards, f, indent=4, ensure_ascii=False)
            logger.debug("Saved flashcards to storage")
        except Exception as e:
            logger.error(f"Error saving flashcards: {e}")
            raise
    
    def create_flashcard(
        self,
        question: str,
        answer: str,
        deck: str = "General",
        topic: Optional[str] = None,
        source: str = "manual"
    ) -> Dict:
        """
        Create a new flashcard.
        
        Args:
            question: Question/front of card
            answer: Answer/back of card
            deck: Deck name
            topic: Optional topic tag
            source: Source of flashcard (manual, generated, notes)
            
        Returns:
            Created flashcard object
        """
        flashcards = self._load_flashcards()
        
        flashcard = {
            "id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "deck": deck,
            "topic": topic,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "last_reviewed": None,
            "review_count": 0,
            "difficulty": "medium",  # easy, medium, hard
            "next_review": None
        }
        
        flashcards.append(flashcard)
        self._save_flashcards(flashcards)
        
        logger.info(f"Created flashcard in deck '{deck}' (id={flashcard['id']})")
        return flashcard
    
    def get_flashcard(self, card_id: str) -> Optional[Dict]:
        """Get a specific flashcard by ID."""
        flashcards = self._load_flashcards()
        for card in flashcards:
            if card.get("id") == card_id:
                return card
        return None
    
    def get_all_flashcards(self) -> List[Dict]:
        """Get all flashcards."""
        return self._load_flashcards()
    
    def get_flashcards_by_deck(self, deck: str) -> List[Dict]:
        """Get all flashcards in a specific deck."""
        flashcards = self._load_flashcards()
        return [card for card in flashcards if card.get("deck") == deck]
    
    def get_flashcards_by_topic(self, topic: str) -> List[Dict]:
        """Get all flashcards for a specific topic."""
        flashcards = self._load_flashcards()
        return [card for card in flashcards if card.get("topic") == topic]
    
    def update_flashcard(
        self,
        card_id: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        deck: Optional[str] = None,
        topic: Optional[str] = None
    ) -> Optional[Dict]:
        """Update an existing flashcard."""
        flashcards = self._load_flashcards()
        
        for card in flashcards:
            if card.get("id") == card_id:
                if question is not None:
                    card["question"] = question
                if answer is not None:
                    card["answer"] = answer
                if deck is not None:
                    card["deck"] = deck
                if topic is not None:
                    card["topic"] = topic
                
                self._save_flashcards(flashcards)
                logger.info(f"Updated flashcard: {card_id}")
                return card
        
        logger.warning(f"Flashcard not found: {card_id}")
        return None
    
    def delete_flashcard(self, card_id: str) -> bool:
        """Delete a flashcard."""
        flashcards = self._load_flashcards()
        original_count = len(flashcards)
        
        flashcards = [card for card in flashcards if card.get("id") != card_id]
        
        if len(flashcards) < original_count:
            self._save_flashcards(flashcards)
            logger.info(f"Deleted flashcard: {card_id}")
            return True
        
        logger.warning(f"Flashcard not found for deletion: {card_id}")
        return False
    
    def record_review(self, card_id: str, difficulty: str):
        """
        Record a flashcard review for spaced repetition.
        
        Args:
            card_id: ID of flashcard
            difficulty: User-rated difficulty (easy, medium, hard)
        """
        flashcards = self._load_flashcards()
        
        for card in flashcards:
            if card.get("id") == card_id:
                card["last_reviewed"] = datetime.now().isoformat()
                card["review_count"] = card.get("review_count", 0) + 1
                card["difficulty"] = difficulty
                
                # Simple spaced repetition: next review based on difficulty
                # This is a simplified algorithm
                from datetime import timedelta
                now = datetime.now()
                if difficulty == "easy":
                    next_review = now + timedelta(days=7)
                elif difficulty == "medium":
                    next_review = now + timedelta(days=3)
                else:  # hard
                    next_review = now + timedelta(days=1)
                
                card["next_review"] = next_review.isoformat()
                
                self._save_flashcards(flashcards)
                logger.info(f"Recorded review for flashcard {card_id}: {difficulty}")
                return
        
        logger.warning(f"Flashcard not found for review: {card_id}")
    
    def get_due_flashcards(self, deck: Optional[str] = None) -> List[Dict]:
        """
        Get flashcards that are due for review.
        
        Args:
            deck: Optional deck filter
            
        Returns:
            List of due flashcards
        """
        flashcards = self._load_flashcards()
        now = datetime.now()
        
        due_cards = []
        for card in flashcards:
            # Include if no next_review set or if next_review is in the past
            next_review = card.get("next_review")
            if next_review is None or datetime.fromisoformat(next_review) <= now:
                if deck is None or card.get("deck") == deck:
                    due_cards.append(card)
        
        return due_cards
    
    def get_all_decks(self) -> List[str]:
        """Get list of all unique decks."""
        flashcards = self._load_flashcards()
        decks = set(card.get("deck", "General") for card in flashcards)
        return sorted(list(decks))
    
    def get_deck_stats(self, deck: str) -> Dict:
        """
        Get statistics for a deck.
        
        Args:
            deck: Deck name
            
        Returns:
            Dictionary with deck statistics
        """
        cards = self.get_flashcards_by_deck(deck)
        due_cards = self.get_due_flashcards(deck)
        
        return {
            "total_cards": len(cards),
            "due_cards": len(due_cards),
            "reviewed_cards": len([c for c in cards if c.get("review_count", 0) > 0]),
            "mastered_cards": len([c for c in cards if c.get("difficulty") == "easy"])
        }
