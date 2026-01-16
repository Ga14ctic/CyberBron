"""
Quiz Generator using AI
Generates quizzes from text content using LLM.
"""
import logging
from typing import List, Dict
import json

logger = logging.getLogger(__name__)


class QuizGenerator:
    """AI-powered quiz generator."""
    
    def __init__(self, llm):
        """
        Initialize the quiz generator.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        logger.info("QuizGenerator initialized")
    
    def generate_quiz(
        self,
        text: str,
        num_questions: int = 10,
        difficulty: str = "medium",
        topic: str = None
    ) -> List[Dict]:
        """
        Generate quiz questions from text content.
        
        Args:
            text: Source text to generate quiz from
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            topic: Optional topic for the quiz
            
        Returns:
            List of question dictionaries
        """
        prompt = f"""Generate {num_questions} {difficulty} difficulty quiz questions from the following text.
Include a mix of multiple choice, true/false, and short answer questions.

For multiple choice questions, provide 4 options.
For true/false questions, the answer should be "true" or "false".
For short answer questions, provide a brief correct answer.

Format your response as a JSON array of question objects.

Text:
{text}

Generate exactly {num_questions} questions in this JSON format:
[
  {{
    "type": "multiple_choice",
    "question": "What is...",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A) Option 1",
    "explanation": "Brief explanation..."
  }},
  {{
    "type": "true_false",
    "question": "Statement to verify...",
    "correct_answer": "true",
    "explanation": "Brief explanation..."
  }},
  {{
    "type": "short_answer",
    "question": "Describe...",
    "correct_answer": "Brief answer...",
    "explanation": "Brief explanation..."
  }}
]
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                questions = json.loads(json_str)
                logger.info(f"Generated {len(questions)} quiz questions from text")
                return questions
            else:
                logger.warning("Could not parse quiz questions from LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return []
    
    def grade_short_answer(self, user_answer: str, correct_answer: str) -> tuple:
        """
        Use AI to grade a short answer question.
        
        Args:
            user_answer: User's answer
            correct_answer: Expected correct answer
            
        Returns:
            Tuple of (is_correct: bool, feedback: str)
        """
        prompt = f"""Grade the following short answer question.
Determine if the user's answer is correct or captures the main idea.
Be lenient with wording differences if the core concept is correct.

Correct Answer: {correct_answer}
User's Answer: {user_answer}

Respond with JSON:
{{
  "is_correct": true/false,
  "feedback": "Brief feedback explaining the grading..."
}}
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result.get("is_correct", False), result.get("feedback", "")
            else:
                # Fallback to simple comparison
                is_correct = user_answer.strip().lower() in correct_answer.strip().lower()
                return is_correct, "Auto-graded based on keyword matching."
                
        except Exception as e:
            logger.error(f"Error grading answer: {e}")
            is_correct = user_answer.strip().lower() in correct_answer.strip().lower()
            return is_correct, "Error during AI grading, using simple comparison."
