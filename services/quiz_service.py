"""
Quiz Service for Quiz Generation and Grading
Handles quiz management and scoring.
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class QuizService:
    """Service for managing quizzes and quiz results."""
    
    def __init__(self, flashcards_dir: str = "flashcards"):
        """
        Initialize the quiz service.
        Uses the same directory as flashcards for consistency.
        
        Args:
            flashcards_dir: Directory to store quiz files
        """
        self.quiz_dir = flashcards_dir
        self.quiz_file = os.path.join(flashcards_dir, "quizzes.json")
        self.results_file = os.path.join(flashcards_dir, "quiz_results.json")
        
        self._ensure_quiz_dir()
        logger.info(f"QuizService initialized with quiz_dir={flashcards_dir}")
    
    def _ensure_quiz_dir(self):
        """Ensure quiz directory exists."""
        if not os.path.exists(self.quiz_dir):
            os.makedirs(self.quiz_dir)
            logger.info(f"Created quiz directory: {self.quiz_dir}")
    
    def _load_quizzes(self) -> List[Dict]:
        """Load all quizzes from storage."""
        if os.path.exists(self.quiz_file):
            try:
                with open(self.quiz_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading quizzes: {e}")
                return []
        return []
    
    def _save_quizzes(self, quizzes: List[Dict]):
        """Save quizzes to storage."""
        try:
            with open(self.quiz_file, 'w', encoding='utf-8') as f:
                json.dump(quizzes, f, indent=4, ensure_ascii=False)
            logger.debug("Saved quizzes to storage")
        except Exception as e:
            logger.error(f"Error saving quizzes: {e}")
            raise
    
    def _load_results(self) -> List[Dict]:
        """Load quiz results from storage."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading quiz results: {e}")
                return []
        return []
    
    def _save_results(self, results: List[Dict]):
        """Save quiz results to storage."""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.debug("Saved quiz results to storage")
        except Exception as e:
            logger.error(f"Error saving quiz results: {e}")
            raise
    
    def create_quiz(
        self,
        title: str,
        questions: List[Dict],
        topic: Optional[str] = None,
        difficulty: str = "medium"
    ) -> Dict:
        """
        Create a new quiz.
        
        Args:
            title: Quiz title
            questions: List of question objects
            topic: Optional topic
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            Created quiz object
        """
        quizzes = self._load_quizzes()
        
        quiz = {
            "id": str(uuid.uuid4()),
            "title": title,
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions,
            "created_at": datetime.now().isoformat(),
            "times_taken": 0
        }
        
        quizzes.append(quiz)
        self._save_quizzes(quizzes)
        
        logger.info(f"Created quiz: {title} (id={quiz['id']})")
        return quiz
    
    def get_quiz(self, quiz_id: str) -> Optional[Dict]:
        """Get a specific quiz by ID."""
        quizzes = self._load_quizzes()
        for quiz in quizzes:
            if quiz.get("id") == quiz_id:
                return quiz
        return None
    
    def get_all_quizzes(self) -> List[Dict]:
        """Get all quizzes."""
        return self._load_quizzes()
    
    def get_quizzes_by_topic(self, topic: str) -> List[Dict]:
        """Get quizzes for a specific topic."""
        quizzes = self._load_quizzes()
        return [quiz for quiz in quizzes if quiz.get("topic") == topic]
    
    def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz."""
        quizzes = self._load_quizzes()
        original_count = len(quizzes)
        
        quizzes = [quiz for quiz in quizzes if quiz.get("id") != quiz_id]
        
        if len(quizzes) < original_count:
            self._save_quizzes(quizzes)
            logger.info(f"Deleted quiz: {quiz_id}")
            return True
        
        logger.warning(f"Quiz not found for deletion: {quiz_id}")
        return False
    
    def submit_quiz_result(
        self,
        quiz_id: str,
        answers: Dict[int, Any],
        score: float,
        total_questions: int
    ) -> Dict:
        """
        Submit quiz results.
        
        Args:
            quiz_id: ID of the quiz
            answers: Dictionary mapping question index to user answer
            score: Score achieved
            total_questions: Total number of questions
            
        Returns:
            Result object
        """
        results = self._load_results()
        
        result = {
            "id": str(uuid.uuid4()),
            "quiz_id": quiz_id,
            "answers": answers,
            "score": score,
            "total_questions": total_questions,
            "percentage": (score / total_questions * 100) if total_questions > 0 else 0,
            "completed_at": datetime.now().isoformat()
        }
        
        results.append(result)
        self._save_results(results)
        
        # Update quiz stats
        quizzes = self._load_quizzes()
        for quiz in quizzes:
            if quiz.get("id") == quiz_id:
                quiz["times_taken"] = quiz.get("times_taken", 0) + 1
                self._save_quizzes(quizzes)
                break
        
        logger.info(f"Submitted quiz result for quiz {quiz_id}: {score}/{total_questions}")
        return result
    
    def get_quiz_results(self, quiz_id: Optional[str] = None) -> List[Dict]:
        """
        Get quiz results.
        
        Args:
            quiz_id: Optional filter by quiz ID
            
        Returns:
            List of quiz results
        """
        results = self._load_results()
        
        if quiz_id:
            return [r for r in results if r.get("quiz_id") == quiz_id]
        
        return results
    
    def get_quiz_statistics(self, quiz_id: str) -> Optional[Dict]:
        """
        Get statistics for a specific quiz.
        
        Args:
            quiz_id: Quiz ID
            
        Returns:
            Statistics dictionary or None
        """
        results = self.get_quiz_results(quiz_id)
        
        if not results:
            return None
        
        scores = [r.get("percentage", 0) for r in results]
        
        return {
            "times_taken": len(results),
            "average_score": sum(scores) / len(scores),
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "last_taken": results[-1].get("completed_at") if results else None
        }
    
    def grade_answer(
        self,
        question_type: str,
        user_answer: Any,
        correct_answer: Any
    ) -> bool:
        """
        Grade a single answer.
        
        Args:
            question_type: Type of question (multiple_choice, true_false, short_answer)
            user_answer: User's answer
            correct_answer: Correct answer
            
        Returns:
            True if correct, False otherwise
        """
        if question_type in ["multiple_choice", "true_false"]:
            return str(user_answer).strip().lower() == str(correct_answer).strip().lower()
        
        elif question_type == "short_answer":
            # For short answer, do a case-insensitive comparison
            # In a real implementation, you'd want more sophisticated matching
            return str(user_answer).strip().lower() == str(correct_answer).strip().lower()
        
        return False
