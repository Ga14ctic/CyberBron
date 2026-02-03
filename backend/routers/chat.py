"""
Chat router - Handle AI conversations and chat history
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone
import logging
from typing import List

from ..database import get_db, User, Conversation
from ..schemas import ChatRequest, ChatResponse, SuccessResponse
from ..utils.auth import get_current_user
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a chat message and get AI response.
    
    - **message**: User message
    - **conversation_id**: Optional conversation ID to continue
    - **use_web_search**: Enable web search for better responses
    """
    try:
        # Get or create conversation
        conversation = None
        if request.conversation_id:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == request.conversation_id,
                    Conversation.user_id == current_user.id
                )
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
        else:
            # Create new conversation
            conversation = Conversation(
                user_id=current_user.id,
                messages=[]
            )
            db.add(conversation)
            await db.flush()
        
        # Get messages history
        messages = conversation.messages or []
        
        # Add user message
        messages.append({
            "role": "user",
            "content": request.message,
            "timestamp": str(datetime.now(timezone.utc))
        })
        
        # TODO: Integrate with Ollama/LLM service for AI response
        # For now, return a placeholder response
        import json
        
        ai_message = f"This is a placeholder response. In production, this would integrate with Ollama to process: {request.message}"
        sources = []
        web_search_used = False
        
        # Add AI response to history
        messages.append({
            "role": "assistant",
            "content": ai_message,
            "timestamp": str(datetime.now(timezone.utc)),
            "sources": sources,
            "web_search_used": web_search_used
        })
        
        # Update conversation
        conversation.messages = messages
        await db.commit()
        await db.refresh(conversation)
        
        logger.info(f"Chat message processed for user {current_user.username}, conversation {conversation.id}")
        
        return ChatResponse(
            message=ai_message,
            conversation_id=conversation.id,
            sources=sources,
            web_search_used=web_search_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@router.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all conversations for the current user.
    """
    try:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == current_user.id)
            .order_by(Conversation.updated_at.desc())
        )
        conversations = result.scalars().all()
        
        # Return conversation summaries
        return [
            {
                "id": conv.id,
                "message_count": len(conv.messages) if conv.messages else 0,
                "last_message": conv.messages[-1]["content"][:100] if conv.messages else "",
                "created_at": conv.created_at,
                "updated_at": conv.updated_at
            }
            for conv in conversations
        ]
        
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list conversations"
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific conversation with full message history.
    """
    try:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {
            "id": conversation.id,
            "messages": conversation.messages or [],
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation"
        )


@router.delete("/conversations/{conversation_id}", response_model=SuccessResponse)
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a conversation.
    """
    try:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        await db.delete(conversation)
        await db.commit()
        
        logger.info(f"Deleted conversation {conversation_id} for user {current_user.username}")
        
        return SuccessResponse(
            success=True,
            message="Conversation deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )
