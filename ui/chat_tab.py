"""
Chat Tab UI Component
Enhanced chat interface with source citations and quick actions.
"""
import streamlit as st
from typing import List, Dict


def render_chat_tab(
    messages: List[Dict],
    rag_chain,
    chat_history_for_chain: List,
    config: Dict,
    on_save_to_notes=None,
    on_generate_flashcards=None,
    on_create_presentation=None
):
    """
    Render the chat tab interface.
    
    Args:
        messages: List of chat messages
        rag_chain: RAG chain for generating responses
        chat_history_for_chain: Formatted chat history for the chain
        config: Configuration dictionary
        on_save_to_notes: Callback for saving to notes
        on_generate_flashcards: Callback for generating flashcards
        on_create_presentation: Callback for creating presentations
    """
    # Display chat history
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add quick action buttons for assistant messages
            if message["role"] == "assistant" and i == len(messages) - 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📝 Save to Notes", key=f"save_note_{i}"):
                        if on_save_to_notes:
                            on_save_to_notes(message["content"])
                
                with col2:
                    if st.button("🎴 Make Flashcards", key=f"flashcard_{i}"):
                        if on_generate_flashcards:
                            # Get last few messages for context
                            recent = messages[max(0, i-3):i+1]
                            on_generate_flashcards(recent)
                
                with col3:
                    if st.button("🎯 Create Slides", key=f"slides_{i}"):
                        if on_create_presentation:
                            # Get conversation topic
                            recent = messages[max(0, i-3):i+1]
                            on_create_presentation(recent)
    
    return None


def show_source_indicator(source_type: str):
    """
    Show a visual indicator for the source of information.
    
    Args:
        source_type: Type of source (docs, model, web)
    """
    if source_type == "docs":
        st.caption("📚 From your documents")
    elif source_type == "model":
        st.caption("🧠 From AI knowledge")
    elif source_type == "web":
        st.caption("🌐 From web search")


def format_search_results_display(search_results: List[Dict]):
    """
    Display search results in an expandable section.
    
    Args:
        search_results: List of search result dictionaries
    """
    if not search_results:
        return
    
    with st.expander("🌐 Web Search Results", expanded=False):
        for i, result in enumerate(search_results, 1):
            st.markdown(f"**{i}. {result['title']}**")
            st.caption(result['snippet'])
            st.markdown(f"[🔗 {result['link']}]({result['link']})")
            st.divider()
