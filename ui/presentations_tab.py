"""
Presentations Tab UI Component
Presentation generation interface integrated with SlideBron-style logic.
"""
import streamlit as st
import os
from services.presentation_service import PresentationService
from generators.pptx_generator import PPTXGenerator


def render_presentations_tab(presentation_service: PresentationService, llm, search_service=None):
    """
    Render the presentations tab.
    
    Args:
        presentation_service: PresentationService instance
        llm: Language model for content generation
        search_service: Optional SearchService for web research
    """
    st.header("🎯 Presentation Generator")
    
    st.markdown("""
    Generate professional PowerPoint presentations on cybersecurity topics.
    Powered by AI with optional web research.
    """)
    
    with st.form("create_presentation"):
        st.subheader("Configure Your Presentation")
        
        topic = st.text_input(
            "Presentation Topic*",
            placeholder="e.g., Introduction to Network Security"
        )
        
        content_source = st.radio(
            "Content Source",
            ["Generate from Topic", "Use Custom Content"]
        )
        
        custom_content = None
        if content_source == "Use Custom Content":
            custom_content = st.text_area(
                "Paste your content here",
                height=200,
                placeholder="Paste notes, outlines, or key points..."
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_slides = st.number_input(
                "Number of Slides",
                min_value=3,
                max_value=20,
                value=7
            )
        
        with col2:
            theme = st.selectbox(
                "Visual Theme",
                presentation_service.get_available_themes()
            )
        
        with col3:
            detail_level = st.selectbox(
                "Detail Level",
                ["Brief", "Moderate", "Detailed"]
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_search = st.checkbox(
                "🌐 Enable Web Research",
                value=True,
                help="Search the web for additional information"
            )
        
        with col2:
            enable_images = st.checkbox(
                "🖼️ Suggest Images",
                value=False,
                help="Include image suggestions (not actual images)"
            )
        
        generate = st.form_submit_button("🎨 Generate Presentation", use_container_width=True)
        
        if generate and topic:
            with st.spinner("🎨 Generating your presentation... This may take a moment..."):
                try:
                    # Generate presentation content
                    slides_content = generate_presentation_content(
                        llm=llm,
                        topic=topic,
                        num_slides=num_slides,
                        custom_content=custom_content,
                        detail_level=detail_level,
                        enable_search=enable_search,
                        search_service=search_service
                    )
                    
                    if slides_content:
                        # Create PPTX file
                        pptx_gen = PPTXGenerator(theme=theme)
                        
                        # Ensure output directory exists
                        os.makedirs(presentation_service.output_dir, exist_ok=True)
                        
                        # Generate filename
                        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_topic = safe_topic.replace(' ', '_')
                        filename = f"{safe_topic}.pptx"
                        output_path = os.path.join(presentation_service.output_dir, filename)
                        
                        # Create presentation
                        pptx_gen.create_presentation(
                            title=topic,
                            slides_content=slides_content,
                            output_path=output_path
                        )
                        
                        st.success(f"✅ Presentation created successfully!")
                        st.info(f"📁 Saved to: `{output_path}`")
                        
                        # Display slide preview
                        with st.expander("📋 Slide Preview", expanded=True):
                            st.markdown(f"### {topic}")
                            for i, slide in enumerate(slides_content, 1):
                                st.markdown(f"#### Slide {i+1}: {slide['title']}")
                                if isinstance(slide['content'], str):
                                    st.write(slide['content'])
                                elif isinstance(slide['content'], list):
                                    for point in slide['content']:
                                        st.markdown(f"- {point}")
                                st.divider()
                        
                        # Download button
                        if os.path.exists(output_path):
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Presentation",
                                    data=f,
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                    use_container_width=True
                                )
                    else:
                        st.error("Failed to generate presentation content. Please try again.")
                
                except Exception as e:
                    st.error(f"Error generating presentation: {e}")
                    import logging
                    logging.exception("Presentation generation error")
    
    # Show recent presentations
    st.divider()
    st.subheader("📚 Recent Presentations")
    
    if os.path.exists(presentation_service.output_dir):
        files = [f for f in os.listdir(presentation_service.output_dir) if f.endswith('.pptx')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(presentation_service.output_dir, x)), reverse=True)
        
        if files:
            for filename in files[:5]:  # Show last 5
                filepath = os.path.join(presentation_service.output_dir, filename)
                file_size = os.path.getsize(filepath) / 1024  # KB
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📄 {filename}")
                with col2:
                    st.caption(f"{file_size:.1f} KB")
                with col3:
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="⬇️",
                            data=f,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key=f"download_{filename}"
                        )
        else:
            st.info("No presentations yet. Generate your first one above!")
    else:
        st.info("No presentations yet. Generate your first one above!")


def generate_presentation_content(
    llm,
    topic: str,
    num_slides: int,
    custom_content: str = None,
    detail_level: str = "Moderate",
    enable_search: bool = False,
    search_service = None
) -> list:
    """
    Generate presentation content using LLM.
    
    Args:
        llm: Language model instance
        topic: Presentation topic
        num_slides: Number of slides to generate
        custom_content: Optional custom content to base slides on
        detail_level: Level of detail (Brief, Moderate, Detailed)
        enable_search: Whether to use web search
        search_service: SearchService instance
        
    Returns:
        List of slide dictionaries with 'title' and 'content'
    """
    # Perform web search if enabled
    search_context = ""
    if enable_search and search_service:
        search_results = search_service.search_cybersecurity(topic)
        if search_results:
            search_context = "\n\nWeb Research Context:\n"
            for result in search_results[:3]:  # Use top 3 results
                search_context += f"- {result['title']}: {result['snippet']}\n"
    
    # Construct prompt
    detail_instructions = {
        "Brief": "Write 1-2 concise paragraphs (3-4 sentences each) explaining the key concepts.",
        "Moderate": "Write 2-3 well-developed paragraphs (4-5 sentences each) with clear explanations and examples.",
        "Detailed": "Write 3-4 comprehensive paragraphs (5-6 sentences each) with in-depth analysis, examples, and thorough coverage."
    }
    
    content_instruction = ""
    if custom_content:
        content_instruction = f"\n\nBase the presentation on this content:\n{custom_content}\n"
    
    prompt = f"""Create a professional presentation outline for: "{topic}"

Generate exactly {num_slides} content slides (excluding title and closing slides).
{detail_instructions[detail_level]}

IMPORTANT: Do NOT use bullet points. Write full paragraphs that go into depth on each topic.
Each slide should have well-written, flowing paragraphs that thoroughly explain the concept.
The content should be suitable for T-Level cybersecurity students.
{content_instruction}
{search_context}

Format your response as a JSON array with this structure:
[
  {{
    "title": "Slide Title",
    "content": "Full paragraph text explaining the topic in depth. This should be multiple sentences that flow naturally and provide comprehensive information about the subject matter."
  }}
]

Make sure each slide has a clear title and detailed paragraph content (not bullet points).
Focus on clarity, depth, accuracy, and educational value.
"""
    
    try:
        response = llm.invoke(prompt)
        
        # Parse JSON response
        import json
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            slides = json.loads(json_str)
            
            # Validate slides
            valid_slides = []
            for slide in slides[:num_slides]:  # Limit to requested number
                if isinstance(slide, dict) and 'title' in slide and 'content' in slide:
                    # Content can be either a string (paragraph) or list (for backward compatibility)
                    if isinstance(slide['content'], (str, list)):
                        # Convert list to paragraph if needed
                        if isinstance(slide['content'], list):
                            slide['content'] = ' '.join(slide['content'])
                        valid_slides.append(slide)
            
            return valid_slides
        else:
            # Fallback: parse text manually
            return parse_presentation_from_text(response, num_slides)
    
    except Exception as e:
        import logging
        logging.error(f"Error generating presentation content: {e}")
        return []


def parse_presentation_from_text(text: str, num_slides: int) -> list:
    """Fallback parser for presentation content from text."""
    slides = []
    lines = text.split('\n')
    
    current_slide = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a slide title (starts with # or is all caps or contains "Slide")
        if (line.startswith('#') or 
            line.isupper() or 
            'slide' in line.lower() and ':' in line):
            
            if current_slide and current_slide['content']:
                slides.append(current_slide)
            
            title = line.replace('#', '').replace('Slide', '').replace(':', '').strip()
            current_slide = {'title': title, 'content': []}
        
        # Check if it's a bullet point
        elif line.startswith(('-', '•', '*', '+')):
            if current_slide:
                point = line.lstrip('-•*+ ').strip()
                if point:
                    current_slide['content'].append(point)
    
    # Add last slide
    if current_slide and current_slide['content']:
        slides.append(current_slide)
    
    return slides[:num_slides]
