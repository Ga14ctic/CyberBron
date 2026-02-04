"""
Flashcard router - Handle flashcard CRUD and spaced repetition
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta, timezone
import logging
from typing import Optional

from ..database import get_db, User, Flashcard
from ..schemas import (
    FlashcardCreate, FlashcardUpdate, FlashcardResponse,
    FlashcardReview, FlashcardGenerateRequest,
    SuccessResponse, PaginatedResponse
)
from ..utils.auth import get_current_user
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/flashcards", response_model=PaginatedResponse)
async def list_flashcards(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    deck: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all flashcards for the current user.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **deck**: Filter by deck name
    """
    try:
        query = select(Flashcard).where(Flashcard.user_id == current_user.id)
        
        if deck:
            query = query.where(Flashcard.deck == deck)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        total = result.scalar()
        
        # Apply pagination
        query = query.order_by(Flashcard.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        flashcards = result.scalars().all()
        
        flashcard_responses = [FlashcardResponse.from_orm(card) for card in flashcards]
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=flashcard_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing flashcards: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list flashcards"
        )


@router.post("/flashcards", response_model=FlashcardResponse, status_code=status.HTTP_201_CREATED)
async def create_flashcard(
    flashcard_data: FlashcardCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new flashcard.
    """
    try:
        new_flashcard = Flashcard(
            user_id=current_user.id,
            deck=flashcard_data.deck,
            question=flashcard_data.question,
            answer=flashcard_data.answer,
            next_review=datetime.now(timezone.utc),
            interval_days=1,
            ease_factor=250,  # 2.5 * 100
            review_count=0
        )
        
        db.add(new_flashcard)
        await db.commit()
        await db.refresh(new_flashcard)
        
        logger.info(f"Created flashcard in deck '{flashcard_data.deck}' for user {current_user.username}")
        
        return FlashcardResponse.from_orm(new_flashcard)
        
    except Exception as e:
        logger.error(f"Error creating flashcard: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create flashcard"
        )


@router.get("/flashcards/due")
async def get_due_flashcards(
    deck: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get flashcards that are due for review.
    
    - **deck**: Optional deck filter
    - **limit**: Maximum number of cards to return
    """
    try:
        query = select(Flashcard).where(
            Flashcard.user_id == current_user.id,
            Flashcard.next_review <= datetime.now(timezone.utc)
        )
        
        if deck:
            query = query.where(Flashcard.deck == deck)
        
        query = query.order_by(Flashcard.next_review.asc()).limit(limit)
        
        result = await db.execute(query)
        flashcards = result.scalars().all()
        
        return [FlashcardResponse.from_orm(card) for card in flashcards]
        
    except Exception as e:
        logger.error(f"Error getting due flashcards: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get due flashcards"
        )


@router.post("/flashcards/{flashcard_id}/review", response_model=FlashcardResponse)
async def review_flashcard(
    flashcard_id: int,
    review: FlashcardReview,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a flashcard review and update spaced repetition schedule.
    
    - **difficulty**: User rating (easy, medium, hard)
    """
    try:
        result = await db.execute(
            select(Flashcard).where(
                Flashcard.id == flashcard_id,
                Flashcard.user_id == current_user.id
            )
        )
        flashcard = result.scalar_one_or_none()
        
        if not flashcard:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Flashcard not found"
            )
        
        # Update review count
        flashcard.review_count += 1
        
        # Calculate next review interval using simplified SM-2 algorithm
        # Note: ease_factor is stored as int (value * 100) in database
        ease_factor = flashcard.ease_factor / 100.0
        
        if review.difficulty == "hard":
            ease_factor = max(1.3, ease_factor - 0.2)
            flashcard.interval_days = max(1, flashcard.interval_days // 2)
        elif review.difficulty == "medium":
            flashcard.interval_days = max(1, int(flashcard.interval_days * ease_factor))
        else:  # easy
            ease_factor = min(2.5, ease_factor + 0.1)
            flashcard.interval_days = max(1, int(flashcard.interval_days * ease_factor * 1.3))
        
        flashcard.ease_factor = int(ease_factor * 100)
        
        # Set next review date
        flashcard.next_review = datetime.now(timezone.utc) + timedelta(days=flashcard.interval_days)
        
        await db.commit()
        await db.refresh(flashcard)
        
        logger.info(f"Reviewed flashcard {flashcard_id}: {review.difficulty}, next in {flashcard.interval_days} days")
        
        return FlashcardResponse.from_orm(flashcard)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing flashcard: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to review flashcard"
        )


@router.put("/flashcards/{flashcard_id}", response_model=FlashcardResponse)
async def update_flashcard(
    flashcard_id: int,
    flashcard_data: FlashcardUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a flashcard.
    """
    try:
        result = await db.execute(
            select(Flashcard).where(
                Flashcard.id == flashcard_id,
                Flashcard.user_id == current_user.id
            )
        )
        flashcard = result.scalar_one_or_none()
        
        if not flashcard:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Flashcard not found"
            )
        
        if flashcard_data.deck is not None:
            flashcard.deck = flashcard_data.deck
        if flashcard_data.question is not None:
            flashcard.question = flashcard_data.question
        if flashcard_data.answer is not None:
            flashcard.answer = flashcard_data.answer
        
        await db.commit()
        await db.refresh(flashcard)
        
        logger.info(f"Updated flashcard {flashcard_id} for user {current_user.username}")
        
        return FlashcardResponse.from_orm(flashcard)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating flashcard: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update flashcard"
        )


@router.delete("/flashcards/{flashcard_id}", response_model=SuccessResponse)
async def delete_flashcard(
    flashcard_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a flashcard.
    """
    try:
        result = await db.execute(
            select(Flashcard).where(
                Flashcard.id == flashcard_id,
                Flashcard.user_id == current_user.id
            )
        )
        flashcard = result.scalar_one_or_none()
        
        if not flashcard:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Flashcard not found"
            )
        
        await db.delete(flashcard)
        await db.commit()
        
        logger.info(f"Deleted flashcard {flashcard_id} for user {current_user.username}")
        
        return SuccessResponse(
            success=True,
            message="Flashcard deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting flashcard: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete flashcard"
        )


@router.post("/flashcards/generate")
async def generate_flashcards(
    request: FlashcardGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate flashcards from content using AI.
    
    - **content**: Source content (min 50 characters)
    - **num_cards**: Number of cards to generate (1-20)
    - **deck**: Deck name for generated cards
    """
    try:
        if request.num_cards > settings.MAX_FLASHCARDS_PER_REQUEST:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot generate more than {settings.MAX_FLASHCARDS_PER_REQUEST} flashcards at once"
            )
        
        # TODO: Integrate with AI service to generate flashcards
        # For now, return a placeholder response
        logger.info(f"Flashcard generation requested by user {current_user.username}: {request.num_cards} cards")
        
        return SuccessResponse(
            success=True,
            message=f"Flashcard generation initiated. This feature requires AI integration.",
            data={"num_cards": request.num_cards, "deck": request.deck}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating flashcards: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate flashcards"
        )


@router.get("/flashcards/stats")
async def get_flashcard_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get flashcard statistics for spaced repetition dashboard.
    
    Returns:
    - Total flashcards
    - Due today count
    - Mastered count (high ease factor)
    - Review streak
    - Average ease factor
    """
    try:
        from datetime import datetime, timezone
        
        # Total flashcards
        total_query = select(func.count()).where(Flashcard.user_id == current_user.id)
        result = await db.execute(total_query)
        total = result.scalar()
        
        # Due today
        now = datetime.now(timezone.utc)
        due_query = select(func.count()).where(
            Flashcard.user_id == current_user.id,
            Flashcard.next_review <= now
        )
        result = await db.execute(due_query)
        due_today = result.scalar()
        
        # Mastered cards (ease_factor >= 250, review_count >= 5)
        mastered_query = select(func.count()).where(
            Flashcard.user_id == current_user.id,
            Flashcard.ease_factor >= 250,
            Flashcard.review_count >= 5
        )
        result = await db.execute(mastered_query)
        mastered = result.scalar()
        
        # Average ease factor
        avg_query = select(func.avg(Flashcard.ease_factor)).where(
            Flashcard.user_id == current_user.id
        )
        result = await db.execute(avg_query)
        avg_ease = result.scalar() or 250
        
        # Reviewed today count
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        reviewed_today_query = select(func.count()).where(
            Flashcard.user_id == current_user.id,
            Flashcard.updated_at >= today_start
        )
        result = await db.execute(reviewed_today_query)
        reviewed_today = result.scalar()
        
        logger.info(f"Retrieved flashcard stats for user {current_user.username}")
        
        return {
            "total_flashcards": total,
            "due_today": due_today,
            "mastered": mastered,
            "reviewed_today": reviewed_today,
            "average_ease_factor": round(avg_ease / 100, 2),
            "decks": []  # TODO: Add deck-specific stats
        }
        
    except Exception as e:
        logger.error(f"Error getting flashcard stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get flashcard statistics"
        )
