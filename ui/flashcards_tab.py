"""
Flashcards Tab UI Component
Clean, focused flashcard study and management interface.
"""
import streamlit as st
import random
from services.flashcard_service import FlashcardService
from generators.flashcard_generator import FlashcardGenerator


def render_flashcards_tab(flashcard_service: FlashcardService, flashcard_generator: FlashcardGenerator, notes_service=None):
    """Render the flashcards tab."""
    tab_study, tab_create, tab_decks = st.tabs(["Study", "Create", "Decks"])

    with tab_study:
        _render_study(flashcard_service)
    with tab_create:
        _render_create(flashcard_service, flashcard_generator, notes_service)
    with tab_decks:
        _render_decks(flashcard_service)


# ─── Study mode ──────────────────────────────────────────────────────
def _render_study(flashcard_service: FlashcardService):
    decks = flashcard_service.get_all_decks()

    if not decks:
        st.info("No flashcards yet. Head to the Create tab to make some.")
        return

    # Controls row
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        selected_deck = st.selectbox("Deck", ["All"] + decks, key="fc_study_deck")
    with c2:
        mode = st.selectbox("Show", ["All Cards", "Due for Review"], key="fc_study_mode")
    with c3:
        shuffle = st.checkbox("Shuffle", value=True, key="fc_shuffle")

    # Get cards
    if selected_deck == "All":
        cards = flashcard_service.get_all_flashcards()
    else:
        cards = flashcard_service.get_flashcards_by_deck(selected_deck)

    if mode == "Due for Review":
        cards = flashcard_service.get_due_flashcards(None if selected_deck == "All" else selected_deck)

    if not cards:
        st.success("Nothing due for review - you're all caught up!")
        return

    cards = list(cards)
    if shuffle:
        if "fc_seed" not in st.session_state:
            st.session_state.fc_seed = random.randint(0, 99999)
        rng = random.Random(st.session_state.fc_seed)
        rng.shuffle(cards)

    # Card index
    if "fc_idx" not in st.session_state:
        st.session_state.fc_idx = 0
    if "fc_flipped" not in st.session_state:
        st.session_state.fc_flipped = False
    if st.session_state.fc_idx >= len(cards):
        st.session_state.fc_idx = 0

    idx = st.session_state.fc_idx
    card = cards[idx]

    # Progress bar
    st.progress((idx + 1) / len(cards), text=f"Card {idx + 1} of {len(cards)}")

    # ── Card display ─────────────────────────────────────────────
    if not st.session_state.fc_flipped:
        # Question side
        st.markdown(
            f"""<div style="
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 12px;
                padding: 48px 40px;
                min-height: 220px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            ">
                <p style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; opacity: 0.4; margin-bottom: 24px;">Question</p>
                <p style="font-size: 22px; line-height: 1.5;">{card['question']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button("Show Answer", use_container_width=True, key="fc_flip", type="primary"):
                st.session_state.fc_flipped = True
                st.rerun()

    else:
        # Answer side
        st.markdown(
            f"""<div style="
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 12px;
                padding: 40px;
                min-height: 220px;
            ">
                <p style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; opacity: 0.4; margin-bottom: 16px;">Question</p>
                <p style="font-size: 16px; opacity: 0.7; margin-bottom: 28px;">{card['question']}</p>
                <div style="border-top: 1px solid rgba(255,255,255,0.1); margin-bottom: 24px;"></div>
                <p style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; opacity: 0.4; margin-bottom: 16px;">Answer</p>
                <p style="font-size: 18px; line-height: 1.6;">{card['answer']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.caption("How well did you know this?")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Hard · 1d", use_container_width=True, key="fc_hard"):
                flashcard_service.record_review(card["id"], "hard")
                _next_card(cards)
        with c2:
            if st.button("Medium · 3d", use_container_width=True, key="fc_medium"):
                flashcard_service.record_review(card["id"], "medium")
                _next_card(cards)
        with c3:
            if st.button("Easy · 7d", use_container_width=True, key="fc_easy"):
                flashcard_service.record_review(card["id"], "easy")
                _next_card(cards)
        with c4:
            if st.button("Skip", use_container_width=True, key="fc_skip"):
                _next_card(cards)


def _next_card(cards):
    st.session_state.fc_idx += 1
    st.session_state.fc_flipped = False
    if st.session_state.fc_idx >= len(cards):
        st.session_state.fc_idx = 0
        st.session_state.fc_seed = random.randint(0, 99999)
    st.rerun()


# ─── Create mode ─────────────────────────────────────────────────────
def _render_create(flashcard_service: FlashcardService, flashcard_generator: FlashcardGenerator, notes_service=None):

    # Handle incoming request from notes tab
    if st.session_state.get("generate_flashcards_from_note") and notes_service:
        note_id = st.session_state.generate_flashcards_from_note
        note = notes_service.get_note(note_id)
        if note:
            st.info(f"Generating flashcards from note: **{note['title']}**")
            with st.form("fc_from_note"):
                c1, c2 = st.columns(2)
                with c1:
                    num = st.number_input("Cards to generate", 1, 20, 10, key="fc_note_num")
                with c2:
                    deck = st.text_input("Deck", value=note.get("folder", "General"), key="fc_note_deck")
                go = st.form_submit_button("Generate", type="primary")
                if go:
                    with st.spinner("Generating..."):
                        cards = flashcard_generator.generate_from_text(note["content"], num, note["title"])
                        if cards:
                            for c in cards:
                                flashcard_service.create_flashcard(c["question"], c["answer"], deck=deck, topic=note["title"], source="note")
                            st.success(f"Created {len(cards)} flashcards!")
                            st.session_state.generate_flashcards_from_note = None
                            st.rerun()
                        else:
                            st.error("Generation failed - try again.")
            st.divider()

    create_mode = st.radio("Mode", ["Manual", "AI Generate"], horizontal=True, key="fc_create_mode")

    if create_mode == "Manual":
        with st.form("fc_manual"):
            question = st.text_area("Question (Front)", height=100, placeholder="What is...?")
            answer = st.text_area("Answer (Back)", height=100, placeholder="It is...")
            c1, c2 = st.columns(2)
            with c1:
                deck = st.text_input("Deck", value="General", key="fc_manual_deck")
            with c2:
                topic = st.text_input("Topic (optional)", key="fc_manual_topic")
            submit = st.form_submit_button("Create Card", type="primary")
            if submit and question and answer:
                flashcard_service.create_flashcard(question, answer, deck=deck, topic=topic or None)
                st.success("Card created!")
                st.rerun()

    else:
        with st.form("fc_ai_gen"):
            source = st.radio("Source", ["Paste text", "From a note"], horizontal=True, key="fc_ai_source")

            text_input = ""
            if source == "Paste text":
                text_input = st.text_area("Content", height=250, placeholder="Paste study material here...", key="fc_ai_text")
            else:
                if notes_service:
                    all_notes = notes_service.get_all_notes()
                    if all_notes:
                        opts = {f"{n['title']} ({n.get('folder','General')})": n for n in all_notes}
                        sel = st.selectbox("Note", list(opts.keys()), key="fc_ai_note_sel")
                        if sel:
                            text_input = opts[sel]["content"]
                    else:
                        st.caption("No notes available.")
                else:
                    st.caption("Notes service unavailable.")

            c1, c2 = st.columns(2)
            with c1:
                num = st.number_input("Cards", 1, 20, 10, key="fc_ai_num")
            with c2:
                deck = st.text_input("Deck", value="Generated", key="fc_ai_deck")

            gen = st.form_submit_button("Generate Flashcards", type="primary")
            if gen and text_input:
                with st.spinner("Generating flashcards with Claude Code..."):
                    cards = flashcard_generator.generate_from_text(text_input, num)
                    if cards:
                        for c in cards:
                            flashcard_service.create_flashcard(c["question"], c["answer"], deck=deck, source="generated")
                        st.success(f"Created {len(cards)} flashcards!")
                        st.rerun()
                    else:
                        st.error("Generation failed. Try again or check your content.")


# ─── Deck management ────────────────────────────────────────────────
def _render_decks(flashcard_service: FlashcardService):
    decks = flashcard_service.get_all_decks()

    if not decks:
        st.info("No decks yet. Create some flashcards first.")
        return

    for deck in decks:
        stats = flashcard_service.get_deck_stats(deck)
        cards = flashcard_service.get_flashcards_by_deck(deck)

        with st.expander(f"{deck}  —  {stats['total_cards']} cards", expanded=False):
            # Stats row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", stats["total_cards"])
            c2.metric("Due", stats["due_cards"])
            c3.metric("Reviewed", stats["reviewed_cards"])
            c4.metric("Mastered", stats["mastered_cards"])

            st.divider()

            # Card list
            for card in cards:
                c1, c2, c3 = st.columns([4, 4, 1])
                with c1:
                    st.markdown(f"**Q:** {card['question']}")
                with c2:
                    st.caption(card["answer"])
                with c3:
                    if st.button("X", key=f"fc_del_{card['id']}"):
                        flashcard_service.delete_flashcard(card["id"])
                        st.rerun()
