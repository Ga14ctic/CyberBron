"""
Notes Service for Complete Notes Management
Handles CRUD operations, search, tagging, and organization of notes.
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class NotesService:
    """Service for managing user notes."""
    
    def __init__(self, notes_dir: str = "notes"):
        """
        Initialize the notes service.
        
        Args:
            notes_dir: Directory to store notes files
        """
        self.notes_dir = notes_dir
        self.notes_file = os.path.join(notes_dir, "notes.json")
        
        self._ensure_notes_dir()
        logger.info(f"NotesService initialized with notes_dir={notes_dir}")
    
    def _ensure_notes_dir(self):
        """Ensure notes directory exists."""
        if not os.path.exists(self.notes_dir):
            os.makedirs(self.notes_dir)
            logger.info(f"Created notes directory: {self.notes_dir}")
    
    def _load_notes(self) -> List[Dict]:
        """Load all notes from storage."""
        if os.path.exists(self.notes_file):
            try:
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading notes: {e}")
                return []
        return []
    
    def _save_notes(self, notes: List[Dict]):
        """Save notes to storage."""
        try:
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                json.dump(notes, f, indent=4, ensure_ascii=False)
            logger.debug("Saved notes to storage")
        except Exception as e:
            logger.error(f"Error saving notes: {e}")
            raise
    
    def create_note(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        folder: str = "General",
        source: str = "manual"
    ) -> Dict:
        """
        Create a new note.
        
        Args:
            title: Note title
            content: Note content
            tags: Optional list of tags
            folder: Folder/category for the note
            source: Source of the note (manual, conversation, import)
            
        Returns:
            Created note object
        """
        notes = self._load_notes()
        
        note = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "tags": tags or [],
            "folder": folder,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        notes.append(note)
        self._save_notes(notes)
        
        logger.info(f"Created note: {title} (id={note['id']})")
        return note
    
    def get_note(self, note_id: str) -> Optional[Dict]:
        """Get a specific note by ID."""
        notes = self._load_notes()
        for note in notes:
            if note.get("id") == note_id:
                return note
        return None
    
    def get_all_notes(self) -> List[Dict]:
        """Get all notes."""
        return self._load_notes()
    
    def update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        folder: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Update an existing note.
        
        Args:
            note_id: ID of note to update
            title: Optional new title
            content: Optional new content
            tags: Optional new tags
            folder: Optional new folder
            
        Returns:
            Updated note or None if not found
        """
        notes = self._load_notes()
        
        for note in notes:
            if note.get("id") == note_id:
                if title is not None:
                    note["title"] = title
                if content is not None:
                    note["content"] = content
                if tags is not None:
                    note["tags"] = tags
                if folder is not None:
                    note["folder"] = folder
                
                note["updated_at"] = datetime.now().isoformat()
                self._save_notes(notes)
                
                logger.info(f"Updated note: {note['title']} (id={note_id})")
                return note
        
        logger.warning(f"Note not found: {note_id}")
        return None
    
    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note.
        
        Args:
            note_id: ID of note to delete
            
        Returns:
            True if deleted, False if not found
        """
        notes = self._load_notes()
        original_count = len(notes)
        
        notes = [note for note in notes if note.get("id") != note_id]
        
        if len(notes) < original_count:
            self._save_notes(notes)
            logger.info(f"Deleted note: {note_id}")
            return True
        
        logger.warning(f"Note not found for deletion: {note_id}")
        return False
    
    def search_notes(self, query: str) -> List[Dict]:
        """
        Search notes by title and content.
        
        Args:
            query: Search query
            
        Returns:
            List of matching notes
        """
        notes = self._load_notes()
        query_lower = query.lower()
        
        matching_notes = []
        for note in notes:
            if (query_lower in note.get("title", "").lower() or
                query_lower in note.get("content", "").lower()):
                matching_notes.append(note)
        
        logger.info(f"Search for '{query}' found {len(matching_notes)} notes")
        return matching_notes
    
    def get_notes_by_folder(self, folder: str) -> List[Dict]:
        """Get all notes in a specific folder."""
        notes = self._load_notes()
        return [note for note in notes if note.get("folder") == folder]
    
    def get_notes_by_tag(self, tag: str) -> List[Dict]:
        """Get all notes with a specific tag."""
        notes = self._load_notes()
        return [note for note in notes if tag in note.get("tags", [])]
    
    def get_all_folders(self) -> List[str]:
        """Get list of all unique folders."""
        notes = self._load_notes()
        folders = set(note.get("folder", "General") for note in notes)
        return sorted(list(folders))
    
    def get_all_tags(self) -> List[str]:
        """Get list of all unique tags."""
        notes = self._load_notes()
        tags = set()
        for note in notes:
            tags.update(note.get("tags", []))
        return sorted(list(tags))
    
    def export_note_to_markdown(self, note_id: str, export_dir: str = "exports") -> Optional[str]:
        """
        Export a note to Markdown file.
        
        Args:
            note_id: ID of note to export
            export_dir: Directory to export to
            
        Returns:
            Path to exported file or None if failed
        """
        note = self.get_note(note_id)
        if not note:
            return None
        
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        filename = f"{note['title'].replace(' ', '_')}_{note['id'][:8]}.md"
        filepath = os.path.join(export_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {note['title']}\n\n")
                f.write(f"**Created:** {note['created_at']}\n")
                f.write(f"**Updated:** {note['updated_at']}\n")
                f.write(f"**Folder:** {note['folder']}\n")
                if note.get('tags'):
                    f.write(f"**Tags:** {', '.join(note['tags'])}\n")
                f.write(f"\n---\n\n")
                f.write(note['content'])
            
            logger.info(f"Exported note to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting note: {e}")
            return None
