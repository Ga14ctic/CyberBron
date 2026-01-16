"""
Notes Tab UI Component
Complete notes management interface.
"""
import streamlit as st
from services.notes_service import NotesService
from typing import Optional


def render_notes_tab(notes_service: NotesService):
    """
    Render the notes management tab.
    
    Args:
        notes_service: NotesService instance
    """
    st.header("📝 Notes Manager")
    
    st.markdown("*Create notes, generate flashcards & quizzes with one click*")
    
    # Quick create note (always visible at top)
    with st.expander("➕ Quick Create Note", expanded=False):
        with st.form("quick_note"):
            qn_title = st.text_input("Title*")
            qn_content = st.text_area("Content*", height=100)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                qn_submit = st.form_submit_button("💾 Save", use_container_width=True)
            
            if qn_submit and qn_title and qn_content:
                note = notes_service.create_note(
                    title=qn_title,
                    content=qn_content,
                    folder="General"
                )
                st.success(f"✅ Note saved!")
                st.rerun()
    
    st.divider()
    
    # Sidebar for notes actions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Your Notes")
    
    with col2:
        if st.button("➕ New Note", use_container_width=True):
            st.session_state.notes_creating_new = True
            st.rerun()
    
    # Create new note form
    if st.session_state.get("notes_creating_new", False):
        with st.form("new_note_form"):
            st.subheader("Create New Note")
            
            title = st.text_input("Title*", placeholder="Enter note title...")
            
            col1, col2 = st.columns(2)
            with col1:
                folder = st.text_input("Folder", value="General")
            with col2:
                tags_input = st.text_input("Tags (comma-separated)", placeholder="e.g., network, security")
            
            content = st.text_area("Content*", height=200, placeholder="Write your note content here...")
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                submit = st.form_submit_button("Save", use_container_width=True)
            with col2:
                cancel = st.form_submit_button("Cancel", use_container_width=True)
            
            if submit and title and content:
                tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
                note = notes_service.create_note(
                    title=title,
                    content=content,
                    tags=tags,
                    folder=folder
                )
                st.success(f"✅ Note '{title}' created successfully!")
                st.session_state.notes_creating_new = False
                st.rerun()
            
            if cancel:
                st.session_state.notes_creating_new = False
                st.rerun()
        
        st.divider()
    
    # Search and filter
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        search_query = st.text_input("🔍 Search notes", placeholder="Search by title or content...")
    
    with col2:
        folders = ["All"] + notes_service.get_all_folders()
        selected_folder = st.selectbox("📁 Folder", folders)
    
    with col3:
        tags = ["All"] + notes_service.get_all_tags()
        selected_tag = st.selectbox("🏷️ Tag", tags)
    
    # Get and filter notes
    if search_query:
        notes = notes_service.search_notes(search_query)
    else:
        notes = notes_service.get_all_notes()
    
    # Apply filters
    if selected_folder != "All":
        notes = [n for n in notes if n.get("folder") == selected_folder]
    
    if selected_tag != "All":
        notes = [n for n in notes if selected_tag in n.get("tags", [])]
    
    # Sort by updated date (most recent first)
    notes = sorted(notes, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    # Display notes
    if not notes:
        st.info("No notes found. Create your first note!")
    else:
        st.caption(f"Showing {len(notes)} note(s)")
        
        for note in notes:
            with st.expander(f"📄 {note['title']}", expanded=False):
                # Note metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📁 {note.get('folder', 'General')}")
                with col2:
                    st.caption(f"📅 {note.get('updated_at', '')[:10]}")
                with col3:
                    if note.get('tags'):
                        st.caption(f"🏷️ {', '.join(note['tags'])}")
                
                # Note content
                st.markdown(note['content'])
                
                st.divider()
                
                # Quick Actions Row
                st.markdown("**Quick Actions:**")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button("✏️ Edit", key=f"edit_{note['id']}", use_container_width=True):
                        st.session_state.notes_editing = note['id']
                        st.rerun()
                
                with col2:
                    if st.button("🎴 Flashcards", key=f"flash_{note['id']}", use_container_width=True):
                        st.session_state.generate_flashcards_from_note = note['id']
                        st.session_state.active_tab = "Flashcards"
                        st.success("✅ Navigate to Flashcards tab to generate!")
                
                with col3:
                    if st.button("📊 Quiz", key=f"quiz_{note['id']}", use_container_width=True):
                        st.session_state.generate_quiz_from_note = note['id']
                        st.session_state.active_tab = "Quiz"
                        st.success("✅ Navigate to Quiz tab to generate!")
                
                with col4:
                    if st.button("📥 Export", key=f"export_{note['id']}", use_container_width=True):
                        filepath = notes_service.export_note_to_markdown(note['id'])
                        if filepath:
                            st.success(f"Exported!")
                
                with col5:
                    if st.button("🗑️ Delete", key=f"delete_{note['id']}", use_container_width=True):
                        if notes_service.delete_note(note['id']):
                            st.success("Note deleted!")
                            st.rerun()
                
                # Edit form
                if st.session_state.get("notes_editing") == note['id']:
                    st.divider()
                    with st.form(f"edit_note_form_{note['id']}"):
                        st.subheader("Edit Note")
                        
                        new_title = st.text_input("Title", value=note['title'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            new_folder = st.text_input("Folder", value=note.get('folder', 'General'))
                        with col2:
                            current_tags = ', '.join(note.get('tags', []))
                            new_tags_input = st.text_input("Tags", value=current_tags)
                        
                        new_content = st.text_area("Content", value=note['content'], height=200)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            update = st.form_submit_button("Update", use_container_width=True)
                        with col2:
                            cancel_edit = st.form_submit_button("Cancel", use_container_width=True)
                        
                        if update:
                            new_tags = [tag.strip() for tag in new_tags_input.split(",") if tag.strip()]
                            notes_service.update_note(
                                note['id'],
                                title=new_title,
                                content=new_content,
                                tags=new_tags,
                                folder=new_folder
                            )
                            st.success("Note updated!")
                            st.session_state.notes_editing = None
                            st.rerun()
                        
                        if cancel_edit:
                            st.session_state.notes_editing = None
                            st.rerun()
