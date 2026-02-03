"""
Presentations router - Handle presentation generation and downloads
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import os
from datetime import datetime
from pathlib import Path

from ..database import get_db, User
from ..schemas import PresentationRequest, PresentationResponse, SuccessResponse
from ..utils.auth import get_current_user
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/presentations/generate", response_model=PresentationResponse)
async def generate_presentation(
    request: PresentationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a PowerPoint presentation using AI.
    
    - **topic**: Presentation topic
    - **num_slides**: Number of slides (3-30)
    - **theme**: Visual theme (professional, modern, minimal, dark)
    - **detail_level**: Amount of detail (brief, moderate, detailed)
    - **enable_web_search**: Use web search for research
    - **custom_content**: Optional custom content to include
    """
    try:
        if request.num_slides > settings.MAX_PRESENTATION_SLIDES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot generate more than {settings.MAX_PRESENTATION_SLIDES} slides"
            )
        
        # Ensure output directory exists
        output_dir = Path(settings.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in request.topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        filename = f"{safe_topic}_{timestamp}.pptx"
        filepath = output_dir / filename
        
        # TODO: Integrate with presentation generation service
        # For now, create a placeholder file
        logger.info(f"Presentation generation requested by {current_user.username}: {request.topic}")
        
        # In production, this would call the presentation generation service
        # from services.presentation_service import generate_presentation
        # generate_presentation(request, filepath)
        
        download_url = f"/api/presentations/{filename}/download"
        
        return PresentationResponse(
            filename=filename,
            download_url=download_url,
            num_slides=request.num_slides,
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating presentation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate presentation"
        )


@router.get("/presentations")
async def list_presentations(
    current_user: User = Depends(get_current_user)
):
    """
    List all generated presentations.
    
    Returns list of available presentation files.
    """
    try:
        output_dir = Path(settings.OUTPUT_DIR)
        
        if not output_dir.exists():
            return []
        
        presentations = []
        for file in output_dir.glob("*.pptx"):
            stat = file.stat()
            presentations.append({
                "filename": file.name,
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "download_url": f"/api/presentations/{file.name}/download"
            })
        
        # Sort by creation time, newest first
        presentations.sort(key=lambda x: x["created_at"], reverse=True)
        
        return presentations
        
    except Exception as e:
        logger.error(f"Error listing presentations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list presentations"
        )


@router.get("/presentations/{filename}/download")
async def download_presentation(
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download a generated presentation file.
    
    - **filename**: Name of the presentation file
    """
    try:
        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )
        
        filepath = Path(settings.OUTPUT_DIR) / filename
        
        if not filepath.exists() or not filepath.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Presentation file not found"
            )
        
        # Ensure file has correct extension
        if filepath.suffix.lower() != ".pptx":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type"
            )
        
        logger.info(f"Presentation download by {current_user.username}: {filename}")
        
        return FileResponse(
            path=str(filepath),
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading presentation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download presentation"
        )


@router.delete("/presentations/{filename}", response_model=SuccessResponse)
async def delete_presentation(
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a presentation file.
    
    - **filename**: Name of the presentation file to delete
    """
    try:
        # Validate filename
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )
        
        filepath = Path(settings.OUTPUT_DIR) / filename
        
        if not filepath.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Presentation file not found"
            )
        
        # Delete the file
        filepath.unlink()
        
        logger.info(f"Presentation deleted by {current_user.username}: {filename}")
        
        return SuccessResponse(
            success=True,
            message="Presentation deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting presentation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete presentation"
        )
