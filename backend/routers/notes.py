"""
Notes router - Handle note CRUD operations
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
import logging
from typing import Optional, List

from ..database import get_db, User, Note
from ..schemas import (
    NoteCreate, NoteUpdate, NoteResponse, 
    SuccessResponse, PaginatedResponse
)
from ..utils.auth import get_current_user
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/notes", response_model=PaginatedResponse)
async def list_notes(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    folder: Optional[str] = None,
    tag: Optional[str] = None,
    source: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all notes for the current user with pagination and filtering.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **folder**: Filter by folder
    - **tag**: Filter by tag
    - **source**: Filter by source (manual, conversation, import)
    """
    try:
        # Build query
        query = select(Note).where(Note.user_id == current_user.id)
        
        # Apply filters
        if folder:
            query = query.where(Note.folder == folder)
        if source:
            query = query.where(Note.source == source)
        if tag:
            # For JSON column, we need to use contains
            query = query.where(Note.tags.contains([tag]))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        total = result.scalar()
        
        # Apply pagination
        query = query.order_by(Note.updated_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        notes = result.scalars().all()
        
        # Convert to response models
        note_responses = [NoteResponse.from_orm(note) for note in notes]
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=note_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing notes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list notes"
        )


@router.post("/notes", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(
    note_data: NoteCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new note.
    """
    try:
        new_note = Note(
            user_id=current_user.id,
            title=note_data.title,
            content=note_data.content,
            folder=note_data.folder,
            tags=note_data.tags,
            source=note_data.source
        )
        
        db.add(new_note)
        await db.commit()
        await db.refresh(new_note)
        
        logger.info(f"Created note '{new_note.title}' for user {current_user.username}")
        
        return NoteResponse.from_orm(new_note)
        
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create note"
        )


@router.get("/notes/search")
async def search_notes(
    q: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Search notes by title and content.
    
    - **q**: Search query
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    """
    try:
        search_pattern = f"%{q}%"
        
        # Search in title and content
        query = select(Note).where(
            Note.user_id == current_user.id,
            or_(
                Note.title.ilike(search_pattern),
                Note.content.ilike(search_pattern)
            )
        )
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        total = result.scalar()
        
        # Apply pagination
        query = query.order_by(Note.updated_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        notes = result.scalars().all()
        
        note_responses = [NoteResponse.from_orm(note) for note in notes]
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=note_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search notes"
        )


@router.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific note by ID.
    """
    try:
        result = await db.execute(
            select(Note).where(
                Note.id == note_id,
                Note.user_id == current_user.id
            )
        )
        note = result.scalar_one_or_none()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        return NoteResponse.from_orm(note)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get note"
        )


@router.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: int,
    note_data: NoteUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a note.
    """
    try:
        result = await db.execute(
            select(Note).where(
                Note.id == note_id,
                Note.user_id == current_user.id
            )
        )
        note = result.scalar_one_or_none()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # Update fields if provided
        if note_data.title is not None:
            note.title = note_data.title
        if note_data.content is not None:
            note.content = note_data.content
        if note_data.folder is not None:
            note.folder = note_data.folder
        if note_data.tags is not None:
            note.tags = note_data.tags
        
        await db.commit()
        await db.refresh(note)
        
        logger.info(f"Updated note {note_id} for user {current_user.username}")
        
        return NoteResponse.from_orm(note)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating note: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update note"
        )


@router.delete("/notes/{note_id}", response_model=SuccessResponse)
async def delete_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a note.
    """
    try:
        result = await db.execute(
            select(Note).where(
                Note.id == note_id,
                Note.user_id == current_user.id
            )
        )
        note = result.scalar_one_or_none()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        await db.delete(note)
        await db.commit()
        
        logger.info(f"Deleted note {note_id} for user {current_user.username}")
        
        return SuccessResponse(
            success=True,
            message="Note deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete note"
        )


@router.post("/notes/{note_id}/summarize")
async def summarize_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    AI-powered summarization of note content.
    Returns a concise summary of the note.
    """
    try:
        result = await db.execute(
            select(Note).where(
                Note.id == note_id,
                Note.user_id == current_user.id
            )
        )
        note = result.scalar_one_or_none()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # TODO: Integrate with Ollama LLM for summarization
        # For now, return placeholder
        summary = f"Summary of '{note.title}': [AI summarization coming soon]"
        
        logger.info(f"Summarized note {note_id} for user {current_user.username}")
        
        return {
            "note_id": note_id,
            "title": note.title,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to summarize note"
        )


@router.post("/notes/{note_id}/expand")
async def expand_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    AI-powered expansion of note content.
    Adds more detail, context, and explanations to the note.
    """
    try:
        result = await db.execute(
            select(Note).where(
                Note.id == note_id,
                Note.user_id == current_user.id
            )
        )
        note = result.scalar_one_or_none()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # TODO: Integrate with Ollama LLM for expansion
        # For now, return placeholder
        expanded_content = f"{note.content}\n\n[AI expansion coming soon - will add detailed explanations, examples, and context]"
        
        logger.info(f"Expanded note {note_id} for user {current_user.username}")
        
        return {
            "note_id": note_id,
            "title": note.title,
            "original_length": len(note.content),
            "expanded_content": expanded_content,
            "expanded_length": len(expanded_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error expanding note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to expand note"
        )


@router.post("/notes/{note_id}/generate-quiz")
async def generate_quiz_from_note(
    note_id: int,
    num_questions: int = Query(10, ge=5, le=20),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a quiz from note content using AI.
    
    - **note_id**: ID of the note to generate quiz from
    - **num_questions**: Number of questions to generate (5-20)
    
    Returns quiz questions in format ready for quiz system.
    """
    try:
        result = await db.execute(
            select(Note).where(
                Note.id == note_id,
                Note.user_id == current_user.id
            )
        )
        note = result.scalar_one_or_none()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # TODO: Integrate with Ollama LLM for quiz generation
        # For now, return placeholder
        from datetime import datetime, timezone
        
        quiz_data = {
            "title": f"Quiz: {note.title}",
            "questions": [
                {
                    "type": "multiple_choice",
                    "question": "Sample question based on note content?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option A",
                    "explanation": "This is a sample question. Real quiz generation coming soon."
                }
            ] * num_questions,
            "note_id": note_id,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Generated quiz from note {note_id} for user {current_user.username}")
        
        return quiz_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating quiz from note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quiz from note"
        )
