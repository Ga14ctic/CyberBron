"""
Flashcards Tab UI Component
Flashcard viewer with spaced repetition.
"""
import streamlit as st
from services.flashcard_service import FlashcardService
from generators.flashcard_generator import FlashcardGenerator


def render_flashcards_tab(flashcard_service: FlashcardService, flashcard_generator: FlashcardGenerator, notes_service=None):
    """
    Render the flashcards tab.
    
    Args:
        flashcard_service: FlashcardService instance
        flashcard_generator: FlashcardGenerator instance
        notes_service: Optional NotesService for generating from notes
    """
    st.header("🎴 Flashcards")
    
    # Tab navigation within flashcards
    tab1, tab2, tab3 = st.tabs(["📚 Study", "➕ Create", "📊 Decks"])
    
    with tab1:
        render_study_mode(flashcard_service)
    
    with tab2:
        render_create_flashcard(flashcard_service, flashcard_generator, notes_service)
    
    with tab3:
        render_deck_management(flashcard_service)


def render_study_mode(flashcard_service: FlashcardService):
    """Render flashcard study mode."""
    st.subheader("Study Flashcards")
    
    # Select deck
    decks = flashcard_service.get_all_decks()
    
    if not decks:
        st.info("No flashcards yet! Create some in the 'Create' tab.")
        return
    
    selected_deck = st.selectbox("Select Deck", ["All"] + decks)
    
    # Get flashcards
    if selected_deck == "All":
        cards = flashcard_service.get_all_flashcards()
    else:
        cards = flashcard_service.get_flashcards_by_deck(selected_deck)
    
    if not cards:
        st.info("No flashcards in this deck.")
        return
    
    # Study mode options
    col1, col2 = st.columns(2)
    with col1:
        study_mode = st.radio("Study Mode", ["All Cards", "Due for Review"])
    with col2:
        shuffle = st.checkbox("Shuffle", value=True)
    
    if study_mode == "Due for Review":
        cards = flashcard_service.get_due_flashcards(None if selected_deck == "All" else selected_deck)
    
    if shuffle:
        import random
        cards = list(cards)
        random.shuffle(cards)
    
    if not cards:
        st.success("🎉 No cards due for review! You're all caught up!")
        return
    
    st.caption(f"Studying {len(cards)} card(s)")
    
    # Initialize session state for card index
    if "flashcard_index" not in st.session_state:
        st.session_state.flashcard_index = 0
    if "flashcard_flipped" not in st.session_state:
        st.session_state.flashcard_flipped = False
    
    # Ensure index is valid
    if st.session_state.flashcard_index >= len(cards):
        st.session_state.flashcard_index = 0
    
    current_card = cards[st.session_state.flashcard_index]
    
    # Progress
    progress = (st.session_state.flashcard_index + 1) / len(cards)
    st.progress(progress, text=f"Card {st.session_state.flashcard_index + 1} of {len(cards)}")
    
    # Flashcard display
    card_container = st.container()
    
    with card_container:
        if not st.session_state.flashcard_flipped:
            # Show question
            st.markdown(f"""
            <div style='background-color: #161b22; border: 2px solid #00ff88; border-radius: 10px; padding: 40px; text-align: center; min-height: 200px; display: flex; align-items: center; justify-content: center;'>
                <h2 style='color: #00ff88;'>{current_card['question']}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Flip Card", use_container_width=True, key="flip"):
                    st.session_state.flashcard_flipped = True
                    st.rerun()
        else:
            # Show answer
            st.markdown(f"""
            <div style='background-color: #161b22; border: 2px solid #00d4ff; border-radius: 10px; padding: 40px; min-height: 200px;'>
                <h3 style='color: #00d4ff; text-align: center;'>Question:</h3>
                <p style='color: #c9d1d9; text-align: center; margin-bottom: 30px;'>{current_card['question']}</p>
                <h3 style='color: #00ff88; text-align: center;'>Answer:</h3>
                <p style='color: #c9d1d9; text-align: center;'>{current_card['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### How well did you know this?")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("😓 Hard (1 day)", use_container_width=True, key="hard"):
                    flashcard_service.record_review(current_card['id'], "hard")
                    next_card()
            
            with col2:
                if st.button("🙂 Medium (3 days)", use_container_width=True, key="medium"):
                    flashcard_service.record_review(current_card['id'], "medium")
                    next_card()
            
            with col3:
                if st.button("😄 Easy (7 days)", use_container_width=True, key="easy"):
                    flashcard_service.record_review(current_card['id'], "easy")
                    next_card()


def next_card():
    """Move to next flashcard."""
    st.session_state.flashcard_index += 1
    st.session_state.flashcard_flipped = False
    st.rerun()


def render_create_flashcard(flashcard_service: FlashcardService, flashcard_generator: FlashcardGenerator, notes_service=None):
    """Render flashcard creation interface."""
    st.subheader("Create Flashcards")
    
    # Check if coming from notes tab
    if st.session_state.get("generate_flashcards_from_note") and notes_service:
        note_id = st.session_state.generate_flashcards_from_note
        note = notes_service.get_note(note_id)
        
        if note:
            st.info(f"📝 Generating flashcards from note: **{note['title']}**")
            
            with st.form("generate_from_note"):
                col1, col2 = st.columns(2)
                with col1:
                    num_cards = st.number_input("Number of cards", min_value=1, max_value=20, value=10)
                with col2:
                    deck = st.text_input("Deck", value=note.get('folder', 'General'))
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    generate = st.form_submit_button("🤖 Generate", use_container_width=True)
                with col2:
                    cancel = st.form_submit_button("Cancel", use_container_width=True)
                
                if generate:
                    with st.spinner("Generating flashcards from your note..."):
                        flashcards = flashcard_generator.generate_from_text(
                            note['content'], 
                            num_cards, 
                            note['title']
                        )
                        
                        if flashcards:
                            for card in flashcards:
                                flashcard_service.create_flashcard(
                                    question=card['question'],
                                    answer=card['answer'],
                                    deck=deck,
                                    topic=note['title'],
                                    source="note"
                                )
                            st.success(f"✅ Generated {len(flashcards)} flashcards from '{note['title']}'!")
                            st.session_state.generate_flashcards_from_note = None
                        else:
                            st.error("Failed to generate flashcards. Please try again.")
                
                if cancel:
                    st.session_state.generate_flashcards_from_note = None
                    st.rerun()
            
            st.divider()
    
    create_mode = st.radio("Creation Mode", ["Manual", "AI Generate from Text"])
    
    if create_mode == "Manual":
        with st.form("create_flashcard"):
            question = st.text_area("Question (Front)*", placeholder="Enter the question...")
            answer = st.text_area("Answer (Back)*", placeholder="Enter the answer...")
            
            col1, col2 = st.columns(2)
            with col1:
                deck = st.text_input("Deck", value="General")
            with col2:
                topic = st.text_input("Topic (optional)")
            
            submit = st.form_submit_button("Create Flashcard")
            
            if submit and question and answer:
                flashcard_service.create_flashcard(
                    question=question,
                    answer=answer,
                    deck=deck,
                    topic=topic if topic else None
                )
                st.success("✅ Flashcard created!")
    
    else:  # AI Generate
        st.markdown("Generate flashcards from your notes or text using AI.")
        
        with st.form("generate_flashcards"):
            # Content source selection
            content_source = st.radio(
                "Content Source",
                ["Paste Text", "Select from Notes"],
                horizontal=True
            )
            
            text_input = ""
            if content_source == "Paste Text":
                text_input = st.text_area("Paste text content here", height=200)
            else:
                if notes_service:
                    notes = notes_service.get_all_notes()
                    if notes:
                        note_options = {f"{note['title']} ({note.get('folder', 'General')})": note for note in notes}
                        selected_note_name = st.selectbox("Select Note", list(note_options.keys()))
                        if selected_note_name:
                            selected_note = note_options[selected_note_name]
                            text_input = selected_note['content']
                            st.info(f"📝 Using note: **{selected_note['title']}**")
                    else:
                        st.warning("No notes available. Create notes first or paste text instead.")
                else:
                    st.warning("Notes service not available. Please paste text instead.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_cards = st.number_input("Number of cards", min_value=1, max_value=20, value=10)
            with col2:
                deck = st.text_input("Deck", value="Generated")
            with col3:
                topic = st.text_input("Topic (optional)")
            
            generate = st.form_submit_button("🤖 Generate Flashcards")
            
            if generate and text_input:
                with st.spinner("Generating flashcards..."):
                    flashcards = flashcard_generator.generate_from_text(text_input, num_cards, topic)
                    
                    if flashcards:
                        for card in flashcards:
                            flashcard_service.create_flashcard(
                                question=card['question'],
                                answer=card['answer'],
                                deck=deck,
                                topic=topic if topic else None,
                                source="generated"
                            )
                        st.success(f"✅ Generated {len(flashcards)} flashcards!")
                    else:
                        st.error("Failed to generate flashcards. Please try again.")


def render_deck_management(flashcard_service: FlashcardService):
    """Render deck management interface."""
    st.subheader("Deck Management")
    
    decks = flashcard_service.get_all_decks()
    
    if not decks:
        st.info("No decks yet. Create flashcards to start building decks!")
        return
    
    for deck in decks:
        with st.expander(f"📚 {deck}", expanded=False):
            stats = flashcard_service.get_deck_stats(deck)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cards", stats['total_cards'])
            with col2:
                st.metric("Due for Review", stats['due_cards'])
            with col3:
                st.metric("Reviewed", stats['reviewed_cards'])
            with col4:
                st.metric("Mastered", stats['mastered_cards'])
            
            # Show cards in deck
            cards = flashcard_service.get_flashcards_by_deck(deck)
            
            if st.checkbox(f"Show all cards in {deck}", key=f"show_{deck}"):
                for card in cards:
                    st.markdown(f"**Q:** {card['question']}")
                    st.caption(f"**A:** {card['answer']}")
                    st.divider()
