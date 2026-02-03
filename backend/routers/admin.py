"""
Admin router - Handle administrative operations
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging
from typing import Optional

from ..database import get_db, User, Note, Flashcard, Quiz, QuizAttempt, Conversation
from ..schemas import UserResponse, SuccessResponse, PaginatedResponse
from ..utils.auth import get_current_admin_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/admin/users", response_model=PaginatedResponse)
async def list_all_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = None,
    is_admin: Optional[bool] = None,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users in the system (admin only).
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **is_active**: Filter by active status
    - **is_admin**: Filter by admin status
    """
    try:
        query = select(User)
        
        if is_active is not None:
            query = query.where(User.is_active == is_active)
        if is_admin is not None:
            query = query.where(User.is_admin == is_admin)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        total = result.scalar()
        
        # Apply pagination
        query = query.order_by(User.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        users = result.scalars().all()
        
        user_responses = [UserResponse.from_orm(user) for user in users]
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=user_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.put("/admin/users/{user_id}/activate", response_model=UserResponse)
async def toggle_user_activation(
    user_id: int,
    activate: bool,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Activate or deactivate a user account (admin only).
    
    - **user_id**: ID of user to modify
    - **activate**: True to activate, False to deactivate
    """
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-deactivation
        if user.id == current_user.id and not activate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )
        
        user.is_active = activate
        await db.commit()
        await db.refresh(user)
        
        action = "activated" if activate else "deactivated"
        logger.info(f"User {user.username} {action} by admin {current_user.username}")
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling user activation: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )


@router.put("/admin/users/{user_id}/admin", response_model=UserResponse)
async def toggle_admin_status(
    user_id: int,
    make_admin: bool,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Grant or revoke admin privileges (admin only).
    
    - **user_id**: ID of user to modify
    - **make_admin**: True to grant admin, False to revoke
    """
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-demotion
        if user.id == current_user.id and not make_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot revoke your own admin privileges"
            )
        
        user.is_admin = make_admin
        await db.commit()
        await db.refresh(user)
        
        action = "granted" if make_admin else "revoked"
        logger.info(f"Admin privileges {action} for user {user.username} by {current_user.username}")
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling admin status: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update admin status"
        )


@router.get("/admin/stats")
async def get_system_statistics(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get system-wide statistics (admin only).
    
    Returns counts for users, notes, flashcards, quizzes, etc.
    """
    try:
        # Count users
        result = await db.execute(select(func.count(User.id)))
        total_users = result.scalar()
        
        result = await db.execute(select(func.count(User.id)).where(User.is_active == True))
        active_users = result.scalar()
        
        result = await db.execute(select(func.count(User.id)).where(User.is_admin == True))
        admin_users = result.scalar()
        
        # Count notes
        result = await db.execute(select(func.count(Note.id)))
        total_notes = result.scalar()
        
        # Count flashcards
        result = await db.execute(select(func.count(Flashcard.id)))
        total_flashcards = result.scalar()
        
        # Count quizzes
        result = await db.execute(select(func.count(Quiz.id)))
        total_quizzes = result.scalar()
        
        # Count quiz attempts
        result = await db.execute(select(func.count(QuizAttempt.id)))
        total_quiz_attempts = result.scalar()
        
        # Count conversations
        result = await db.execute(select(func.count(Conversation.id)))
        total_conversations = result.scalar()
        
        statistics = {
            "users": {
                "total": total_users,
                "active": active_users,
                "inactive": total_users - active_users,
                "admins": admin_users
            },
            "content": {
                "notes": total_notes,
                "flashcards": total_flashcards,
                "quizzes": total_quizzes,
                "conversations": total_conversations
            },
            "activity": {
                "quiz_attempts": total_quiz_attempts,
                "avg_attempts_per_quiz": round(total_quiz_attempts / total_quizzes, 2) if total_quizzes > 0 else 0
            }
        }
        
        logger.info(f"System statistics requested by admin {current_user.username}")
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system statistics"
        )


@router.delete("/admin/users/{user_id}", response_model=SuccessResponse)
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Permanently delete a user and all their data (admin only).
    
    WARNING: This action is irreversible and will delete all user data.
    
    - **user_id**: ID of user to delete
    """
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-deletion
        if user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        username = user.username
        
        # Delete user (cascade will handle related data)
        await db.delete(user)
        await db.commit()
        
        logger.warning(f"User {username} (ID: {user_id}) deleted by admin {current_user.username}")
        
        return SuccessResponse(
            success=True,
            message=f"User {username} and all associated data deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
