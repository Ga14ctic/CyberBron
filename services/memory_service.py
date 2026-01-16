"""
Memory Service for Long-term Memory Management
Handles persistent storage of user preferences, topics, and learning progress.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing long-term memory across sessions."""
    
    def __init__(self, memory_dir: str = "memory"):
        """
        Initialize the memory service.
        
        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = memory_dir
        self.preferences_file = os.path.join(memory_dir, "preferences.json")
        self.topics_file = os.path.join(memory_dir, "topics.json")
        self.progress_file = os.path.join(memory_dir, "progress.json")
        
        self._ensure_memory_dir()
        logger.info(f"MemoryService initialized with memory_dir={memory_dir}")
    
    def _ensure_memory_dir(self):
        """Ensure memory directory exists."""
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)
            logger.info(f"Created memory directory: {self.memory_dir}")
    
    def _load_json(self, filepath: str) -> Dict:
        """Load JSON file or return empty dict if not exists."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                return {}
        return {}
    
    def _save_json(self, filepath: str, data: Dict):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logger.debug(f"Saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to {filepath}: {e}")
    
    # === Preferences Management ===
    
    def get_preferences(self) -> Dict:
        """Get user preferences."""
        return self._load_json(self.preferences_file)
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        prefs = self.get_preferences()
        prefs[key] = value
        prefs["updated_at"] = datetime.now().isoformat()
        self._save_json(self.preferences_file, prefs)
        logger.info(f"Set preference: {key}={value}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a specific preference."""
        prefs = self.get_preferences()
        return prefs.get(key, default)
    
    # === Topics Management ===
    
    def record_topic(self, topic: str):
        """Record that a topic was studied."""
        topics = self._load_json(self.topics_file)
        
        if topic not in topics:
            topics[topic] = {
                "first_studied": datetime.now().isoformat(),
                "last_studied": datetime.now().isoformat(),
                "study_count": 1
            }
        else:
            topics[topic]["last_studied"] = datetime.now().isoformat()
            topics[topic]["study_count"] = topics[topic].get("study_count", 0) + 1
        
        self._save_json(self.topics_file, topics)
        logger.info(f"Recorded topic: {topic}")
    
    def get_topics(self) -> Dict:
        """Get all studied topics."""
        return self._load_json(self.topics_file)
    
    def get_frequent_topics(self, limit: int = 5) -> List[str]:
        """Get most frequently studied topics."""
        topics = self.get_topics()
        sorted_topics = sorted(
            topics.items(),
            key=lambda x: x[1].get("study_count", 0),
            reverse=True
        )
        return [topic for topic, _ in sorted_topics[:limit]]
    
    def get_recent_topics(self, limit: int = 5) -> List[str]:
        """Get recently studied topics."""
        topics = self.get_topics()
        sorted_topics = sorted(
            topics.items(),
            key=lambda x: x[1].get("last_studied", ""),
            reverse=True
        )
        return [topic for topic, _ in sorted_topics[:limit]]
    
    # === Progress Tracking ===
    
    def update_progress(self, category: str, data: Dict):
        """
        Update learning progress for a category.
        
        Args:
            category: Category name (e.g., "flashcards", "quizzes", "notes")
            data: Progress data to store
        """
        progress = self._load_json(self.progress_file)
        
        if category not in progress:
            progress[category] = {}
        
        progress[category].update(data)
        progress[category]["updated_at"] = datetime.now().isoformat()
        
        self._save_json(self.progress_file, progress)
        logger.info(f"Updated progress for category: {category}")
    
    def get_progress(self, category: Optional[str] = None) -> Dict:
        """
        Get learning progress.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Progress data for category or all progress
        """
        progress = self._load_json(self.progress_file)
        if category:
            return progress.get(category, {})
        return progress
    
    # === Conversation Summaries ===
    
    def save_conversation_summary(self, conversation_id: str, summary: str):
        """Save a summary of a conversation."""
        prefs = self.get_preferences()
        
        if "conversation_summaries" not in prefs:
            prefs["conversation_summaries"] = {}
        
        prefs["conversation_summaries"][conversation_id] = {
            "summary": summary,
            "created_at": datetime.now().isoformat()
        }
        
        self._save_json(self.preferences_file, prefs)
        logger.info(f"Saved summary for conversation: {conversation_id}")
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get summary for a specific conversation."""
        prefs = self.get_preferences()
        summaries = prefs.get("conversation_summaries", {})
        summary_data = summaries.get(conversation_id)
        return summary_data.get("summary") if summary_data else None
    
    # === Facts and Corrections ===
    
    def add_learned_fact(self, fact: str, source: str = "user"):
        """Add a learned fact or correction."""
        prefs = self.get_preferences()
        
        if "learned_facts" not in prefs:
            prefs["learned_facts"] = []
        
        prefs["learned_facts"].append({
            "fact": fact,
            "source": source,
            "learned_at": datetime.now().isoformat()
        })
        
        self._save_json(self.preferences_file, prefs)
        logger.info(f"Added learned fact: {fact[:50]}...")
    
    def get_learned_facts(self) -> List[Dict]:
        """Get all learned facts."""
        prefs = self.get_preferences()
        return prefs.get("learned_facts", [])
