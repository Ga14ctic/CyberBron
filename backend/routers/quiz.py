"""
Quiz router - Handle quiz creation, generation, and attempts
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import logging
from typing import Optional, Dict

from ..database import get_db, User, Quiz, QuizAttempt
from ..schemas import (
    QuizCreate, QuizResponse, QuizGenerateRequest,
    QuizAttemptCreate, QuizAttemptResponse,
    SuccessResponse, PaginatedResponse
)
from ..utils.auth import get_current_user
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/quizzes", response_model=PaginatedResponse)
async def list_quizzes(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    difficulty: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all quizzes for the current user.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **difficulty**: Filter by difficulty (easy, medium, hard)
    """
    try:
        query = select(Quiz).where(Quiz.user_id == current_user.id)
        
        if difficulty:
            query = query.where(Quiz.difficulty == difficulty)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        total = result.scalar()
        
        # Apply pagination
        query = query.order_by(Quiz.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        quizzes = result.scalars().all()
        
        quiz_responses = [QuizResponse.from_orm(quiz) for quiz in quizzes]
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=quiz_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing quizzes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list quizzes"
        )


@router.post("/quizzes", response_model=QuizResponse, status_code=status.HTTP_201_CREATED)
async def create_quiz(
    quiz_data: QuizCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new quiz manually.
    """
    try:
        # Convert questions to dict format for JSON storage
        questions_dict = [q.dict() for q in quiz_data.questions]
        
        new_quiz = Quiz(
            user_id=current_user.id,
            title=quiz_data.title,
            questions=questions_dict,
            difficulty=quiz_data.difficulty
        )
        
        db.add(new_quiz)
        await db.commit()
        await db.refresh(new_quiz)
        
        logger.info(f"Created quiz '{quiz_data.title}' for user {current_user.username}")
        
        return QuizResponse.from_orm(new_quiz)
        
    except Exception as e:
        logger.error(f"Error creating quiz: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create quiz"
        )


@router.post("/quizzes/generate")
async def generate_quiz(
    request: QuizGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a quiz from content using AI.
    
    - **content**: Source content (min 100 characters)
    - **num_questions**: Number of questions (5-50)
    - **difficulty**: Difficulty level (easy, medium, hard)
    - **title**: Quiz title
    """
    try:
        if request.num_questions > settings.MAX_QUIZ_QUESTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot generate more than {settings.MAX_QUIZ_QUESTIONS} questions"
            )
        
        # TODO: Integrate with AI service to generate quiz
        # For now, return a placeholder response
        logger.info(f"Quiz generation requested by user {current_user.username}: {request.num_questions} questions")
        
        return SuccessResponse(
            success=True,
            message="Quiz generation initiated. This feature requires AI integration.",
            data={
                "title": request.title,
                "num_questions": request.num_questions,
                "difficulty": request.difficulty
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quiz"
        )


@router.get("/quizzes/{quiz_id}", response_model=QuizResponse)
async def get_quiz(
    quiz_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific quiz by ID.
    """
    try:
        result = await db.execute(
            select(Quiz).where(
                Quiz.id == quiz_id,
                Quiz.user_id == current_user.id
            )
        )
        quiz = result.scalar_one_or_none()
        
        if not quiz:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quiz not found"
            )
        
        return QuizResponse.from_orm(quiz)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quiz: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quiz"
        )


@router.post("/quizzes/{quiz_id}/attempt", response_model=QuizAttemptResponse)
async def submit_quiz_attempt(
    quiz_id: int,
    attempt_data: QuizAttemptCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a quiz attempt and get score.
    
    - **answers**: Dictionary mapping question index to user answer
    """
    try:
        # Get the quiz
        result = await db.execute(
            select(Quiz).where(
                Quiz.id == quiz_id,
                Quiz.user_id == current_user.id
            )
        )
        quiz = result.scalar_one_or_none()
        
        if not quiz:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Quiz not found"
            )
        
        # Grade the quiz
        questions = quiz.questions
        score = 0
        max_score = len(questions)
        
        for idx, question in enumerate(questions):
            user_answer = attempt_data.answers.get(idx)
            correct_answer = question.get("correct_answer")
            
            if user_answer and str(user_answer).strip().lower() == str(correct_answer).strip().lower():
                score += 1
        
        percentage = (score / max_score * 100) if max_score > 0 else 0
        
        # Save attempt
        attempt = QuizAttempt(
            user_id=current_user.id,
            quiz_id=quiz_id,
            answers=attempt_data.answers,
            score=score,
            max_score=max_score
        )
        
        db.add(attempt)
        await db.commit()
        await db.refresh(attempt)
        
        logger.info(f"Quiz attempt submitted by {current_user.username}: {score}/{max_score}")
        
        return QuizAttemptResponse(
            id=attempt.id,
            quiz_id=quiz_id,
            score=score,
            max_score=max_score,
            percentage=percentage,
            completed_at=attempt.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting quiz attempt: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit quiz attempt"
        )


@router.get("/quizzes/attempts", response_model=PaginatedResponse)
async def get_quiz_attempts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    quiz_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's quiz attempt history.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **quiz_id**: Optional filter by specific quiz
    """
    try:
        query = select(QuizAttempt).where(QuizAttempt.user_id == current_user.id)
        
        if quiz_id:
            query = query.where(QuizAttempt.quiz_id == quiz_id)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        total = result.scalar()
        
        # Apply pagination
        query = query.order_by(QuizAttempt.completed_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        attempts = result.scalars().all()
        
        attempt_responses = []
        for attempt in attempts:
            percentage = (attempt.score / attempt.max_score * 100) if attempt.max_score > 0 else 0
            attempt_responses.append(
                QuizAttemptResponse(
                    id=attempt.id,
                    quiz_id=attempt.quiz_id,
                    score=attempt.score,
                    max_score=attempt.max_score,
                    percentage=percentage,
                    completed_at=attempt.completed_at
                )
            )
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=attempt_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error getting quiz attempts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quiz attempts"
        )
