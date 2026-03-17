"""
Notes Tab UI Component
A clean, document-editor-style notes interface.
"""
import streamlit as st
from services.notes_service import NotesService


def render_notes_tab(notes_service: NotesService):
    """Render the notes tab as a proper writing environment."""

    # Initialize state
    if "notes_active_id" not in st.session_state:
        st.session_state.notes_active_id = None
    if "notes_editing" not in st.session_state:
        st.session_state.notes_editing = False

    notes = notes_service.get_all_notes()
    notes = sorted(notes, key=lambda x: x.get("updated_at", ""), reverse=True)
    folders = notes_service.get_all_folders()

    # ── Top bar ──────────────────────────────────────────────────
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.header("Notes")
    with top_right:
        if st.button("New Note", use_container_width=True, key="notes_new_btn"):
            note = notes_service.create_note(
                title="Untitled Note",
                content="",
                folder="General"
            )
            st.session_state.notes_active_id = note["id"]
            st.session_state.notes_editing = True
            st.rerun()

    # ── Two-column layout: note list | editor ────────────────────
    list_col, editor_col = st.columns([1, 3])

    # ── Left panel: note list ────────────────────────────────────
    with list_col:
        search_query = st.text_input("Search", placeholder="Filter notes...", key="notes_search", label_visibility="collapsed")
        folder_filter = st.selectbox("Folder", ["All"] + folders, key="notes_folder_filter", label_visibility="collapsed")

        filtered = notes
        if search_query:
            q = search_query.lower()
            filtered = [n for n in filtered if q in n.get("title", "").lower() or q in n.get("content", "").lower()]
        if folder_filter != "All":
            filtered = [n for n in filtered if n.get("folder") == folder_filter]

        if not filtered:
            st.caption("No notes yet.")
        else:
            for note in filtered:
                is_active = st.session_state.notes_active_id == note["id"]
                preview = note.get("content", "")[:80].replace("\n", " ")
                date_str = note.get("updated_at", "")[:10]
                label = f"**{note['title']}**\n{date_str} · {note.get('folder', 'General')}"

                if st.button(
                    label,
                    key=f"note_select_{note['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.notes_active_id = note["id"]
                    st.session_state.notes_editing = False
                    st.rerun()

    # ── Right panel: editor / viewer ─────────────────────────────
    with editor_col:
        active_note = None
        if st.session_state.notes_active_id:
            active_note = notes_service.get_note(st.session_state.notes_active_id)

        if active_note is None:
            st.markdown(
                "<div style='text-align:center; padding: 80px 20px; opacity: 0.5;'>"
                "<h3>Select a note or create a new one</h3>"
                "<p>Your notes appear on the left. Click one to open it here.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        # ── Toolbar ──────────────────────────────────────────────
        tb1, tb2, tb3, tb4, tb5 = st.columns([1, 1, 1, 1, 1])
        with tb1:
            edit_label = "Editing" if st.session_state.notes_editing else "Edit"
            if st.button(edit_label, use_container_width=True, key="notes_edit_toggle",
                         type="primary" if st.session_state.notes_editing else "secondary"):
                st.session_state.notes_editing = not st.session_state.notes_editing
                st.rerun()
        with tb2:
            if st.button("Flashcards", use_container_width=True, key="notes_to_flash"):
                st.session_state.generate_flashcards_from_note = active_note["id"]
                st.toast("Switch to the Flashcards tab to generate!")
        with tb3:
            if st.button("Quiz", use_container_width=True, key="notes_to_quiz"):
                st.session_state.generate_quiz_from_note = active_note["id"]
                st.toast("Switch to the Quiz tab to generate!")
        with tb4:
            if st.button("Export .md", use_container_width=True, key="notes_export"):
                fp = notes_service.export_note_to_markdown(active_note["id"])
                if fp:
                    st.toast(f"Exported to {fp}")
        with tb5:
            if st.button("Delete", use_container_width=True, key="notes_delete"):
                notes_service.delete_note(active_note["id"])
                st.session_state.notes_active_id = None
                st.session_state.notes_editing = False
                st.rerun()

        st.divider()

        # ── Editor mode ─────────────────────────────────────────
        if st.session_state.notes_editing:
            with st.form("notes_editor_form", border=False):
                # Title
                new_title = st.text_input(
                    "Title",
                    value=active_note["title"],
                    key="notes_edit_title",
                    label_visibility="collapsed",
                    placeholder="Note title",
                )

                # Metadata row
                m1, m2 = st.columns(2)
                with m1:
                    new_folder = st.text_input("Folder", value=active_note.get("folder", "General"), key="notes_edit_folder")
                with m2:
                    current_tags = ", ".join(active_note.get("tags", []))
                    new_tags_str = st.text_input("Tags (comma-separated)", value=current_tags, key="notes_edit_tags")

                # Big editor — the main writing area
                new_content = st.text_area(
                    "Content",
                    value=active_note["content"],
                    height=500,
                    key="notes_edit_content",
                    label_visibility="collapsed",
                    placeholder="Start writing... Markdown is supported.",
                )

                # Word count
                word_count = len(new_content.split()) if new_content.strip() else 0
                char_count = len(new_content)

                s1, s2, s3 = st.columns([2, 2, 1])
                with s1:
                    st.caption(f"{word_count} words · {char_count} characters")
                with s3:
                    saved = st.form_submit_button("Save", use_container_width=True, type="primary")

                if saved:
                    new_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]
                    notes_service.update_note(
                        active_note["id"],
                        title=new_title,
                        content=new_content,
                        tags=new_tags,
                        folder=new_folder,
                    )
                    st.toast("Saved!")
                    st.rerun()

        # ── View mode ────────────────────────────────────────────
        else:
            # Title
            st.markdown(f"## {active_note['title']}")

            # Metadata
            meta_parts = [f"**{active_note.get('folder', 'General')}**"]
            meta_parts.append(active_note.get("updated_at", "")[:10])
            if active_note.get("tags"):
                meta_parts.append(" · ".join(active_note["tags"]))
            word_count = len(active_note["content"].split()) if active_note["content"].strip() else 0
            meta_parts.append(f"{word_count} words")
            st.caption(" · ".join(meta_parts))

            st.divider()

            # Rendered markdown content
            if active_note["content"].strip():
                st.markdown(active_note["content"])
            else:
                st.caption("This note is empty. Click Edit to start writing.")
