"""
Presentations Tab UI Component
Generates professional, content-rich PowerPoint presentations using Claude Code.
"""
import streamlit as st
import os
import json
import logging
from services.presentation_service import PresentationService
from generators.pptx_generator import PPTXGenerator

logger = logging.getLogger(__name__)


def render_presentations_tab(presentation_service: PresentationService, llm, search_service=None):
    """Render the presentations tab."""
    st.header("Presentation Generator")

    st.markdown("Generate professional PowerPoint presentations with rich, detailed content.")

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
                ["professional", "modern", "minimal", "dark", "cyber"]
            )

        with col3:
            detail_level = st.selectbox(
                "Detail Level",
                ["Brief", "Moderate", "Detailed"]
            )

        enable_search = st.checkbox(
            "Enable Web Research",
            value=True,
            help="Search the web for additional information"
        )

        generate = st.form_submit_button("Generate Presentation", use_container_width=True)

        if generate and topic:
            with st.spinner("Generating your presentation with Claude Code..."):
                try:
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
                        pptx_gen = PPTXGenerator(theme=theme)
                        os.makedirs(presentation_service.output_dir, exist_ok=True)

                        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_topic = safe_topic.replace(' ', '_')
                        filename = f"{safe_topic}.pptx"
                        output_path = os.path.join(presentation_service.output_dir, filename)

                        pptx_gen.create_presentation(
                            title=topic,
                            slides_content=slides_content,
                            output_path=output_path
                        )

                        st.success(f"Presentation created with {len(slides_content)} content slides!")
                        st.info(f"Saved to: `{output_path}`")

                        with st.expander("Slide Preview", expanded=True):
                            st.markdown(f"### {topic}")
                            for i, slide in enumerate(slides_content, 1):
                                st.markdown(f"**Slide {i}: {slide['title']}**")
                                if isinstance(slide['content'], str):
                                    st.write(slide['content'][:500] + ("..." if len(slide['content']) > 500 else ""))
                                elif isinstance(slide['content'], list):
                                    for point in slide['content'][:5]:
                                        st.markdown(f"- {point}")
                                st.divider()

                        if os.path.exists(output_path):
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="Download Presentation",
                                    data=f,
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                    use_container_width=True
                                )
                    else:
                        st.error("Failed to generate presentation content. Please try again.")

                except Exception as e:
                    st.error(f"Error generating presentation: {e}")
                    logger.exception("Presentation generation error")

    # Show recent presentations
    st.divider()
    st.subheader("Recent Presentations")

    if os.path.exists(presentation_service.output_dir):
        files = [f for f in os.listdir(presentation_service.output_dir) if f.endswith('.pptx')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(presentation_service.output_dir, x)), reverse=True)

        if files:
            for filename in files[:5]:
                filepath = os.path.join(presentation_service.output_dir, filename)
                file_size = os.path.getsize(filepath) / 1024

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"{filename}")
                with col2:
                    st.caption(f"{file_size:.1f} KB")
                with col3:
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="Download",
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
    search_service=None
) -> list:
    """Generate rich, detailed presentation content using Claude Code."""

    # Perform web search if enabled
    search_context = ""
    if enable_search and search_service:
        search_results = search_service.search_cybersecurity(topic)
        if search_results:
            search_context = "\n\nAdditional research context from the web:\n"
            for result in search_results[:3]:
                search_context += f"- {result['title']}: {result['snippet']}\n"

    # Detail level instructions - the key fix for thin content
    detail_instructions = {
        "Brief": """Each slide must have 2-3 substantial paragraphs (3-4 sentences each).
Cover the key concepts with clear explanations. Include at least one concrete example or real-world scenario per slide.""",

        "Moderate": """Each slide must have 3-4 substantial paragraphs (4-6 sentences each).
Provide thorough explanations with examples, real-world applications, and technical details.
Include specific tools, frameworks, standards, or case studies where relevant.
Each slide should contain roughly 150-250 words of content.""",

        "Detailed": """Each slide must have 4-5 rich, comprehensive paragraphs (5-7 sentences each).
Go deep: explain the concept, why it matters, how it works technically, real-world examples,
common pitfalls, best practices, and connections to other topics.
Include specific tools, CVEs, standards (NIST, ISO 27001, OWASP), case studies, and statistics where relevant.
Each slide should contain roughly 250-400 words of content. Do NOT be brief. Be thorough.""",
    }

    content_instruction = ""
    if custom_content:
        content_instruction = f"\n\nBase the presentation on this source material:\n{custom_content}\n"

    prompt = f"""You are an expert presentation content writer for cybersecurity education.

Create detailed, educational slide content for a presentation titled: "{topic}"

Generate exactly {num_slides} content slides (the title slide and closing slide are added automatically).

CONTENT REQUIREMENTS:
{detail_instructions[detail_level]}

CRITICAL RULES:
- Write in FULL PARAGRAPHS with complete sentences. Never use bullet points or dashes.
- Each paragraph should flow naturally and explain concepts in depth.
- Include real examples, specific technologies, tools, or standards.
- Content should be suitable for T-Level cybersecurity students.
- Do NOT write thin, vague, or generic content. Be specific and educational.
- Each slide should teach something concrete, not just mention topics.
{content_instruction}
{search_context}

FORMAT: Respond with ONLY a JSON array. No markdown, no code fences, no explanation outside the JSON.

[
  {{
    "title": "Concise Slide Title",
    "content": "First paragraph explaining the core concept in depth with specific details and examples. This should be multiple complete sentences that thoroughly cover the topic.\\n\\nSecond paragraph diving deeper into technical aspects, real-world applications, or related considerations. Include specific tools, standards, or case studies.\\n\\nThird paragraph covering best practices, common challenges, or how this connects to the broader cybersecurity landscape."
  }}
]

Remember: Each slide's content field must contain multiple paragraphs separated by \\n\\n. Write substantive, educational content — not summaries or outlines."""

    try:
        response = llm.invoke(prompt)

        # Parse JSON response
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            slides = json.loads(json_str)

            valid_slides = []
            for slide in slides[:num_slides]:
                if isinstance(slide, dict) and 'title' in slide and 'content' in slide:
                    if isinstance(slide['content'], list):
                        slide['content'] = '\n\n'.join(str(item) for item in slide['content'])
                    if isinstance(slide['content'], str):
                        valid_slides.append(slide)

            return valid_slides
        else:
            logger.warning("Could not find JSON array in response, attempting fallback parse")
            return parse_presentation_from_text(response, num_slides)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        # Try to clean up common JSON issues
        try:
            json_str = response[start_idx:end_idx]
            # Fix common issues: trailing commas, unescaped newlines
            json_str = json_str.replace(",\n]", "\n]").replace(",\r\n]", "\r\n]")
            slides = json.loads(json_str)
            return [s for s in slides[:num_slides] if isinstance(s, dict) and 'title' in s and 'content' in s]
        except Exception:
            return parse_presentation_from_text(response, num_slides)
    except Exception as e:
        logger.error(f"Error generating presentation content: {e}")
        return []


def parse_presentation_from_text(text: str, num_slides: int) -> list:
    """Fallback parser for presentation content from text."""
    slides = []
    current_slide = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current_slide and current_slide.get('_paragraphs'):
                current_slide['_paragraphs'].append('')
            continue

        if (line.startswith('#') or
            (line.isupper() and len(line) > 5) or
            ('slide' in line.lower() and ':' in line)):

            if current_slide:
                current_slide['content'] = '\n\n'.join(
                    p for p in current_slide.pop('_paragraphs', []) if p
                )
                if current_slide['content']:
                    slides.append(current_slide)

            title = line.replace('#', '').strip()
            for prefix in ['Slide ', 'SLIDE ']:
                if title.upper().startswith(prefix.upper()):
                    title = title[len(prefix):].lstrip('0123456789').lstrip(':').lstrip(' -')
            current_slide = {'title': title, '_paragraphs': []}

        elif current_slide is not None:
            point = line.lstrip('-*+ ').strip()
            if point:
                if current_slide['_paragraphs'] and current_slide['_paragraphs'][-1]:
                    current_slide['_paragraphs'][-1] += ' ' + point
                else:
                    current_slide['_paragraphs'].append(point)

    if current_slide:
        current_slide['content'] = '\n\n'.join(
            p for p in current_slide.pop('_paragraphs', []) if p
        )
        if current_slide['content']:
            slides.append(current_slide)

    return slides[:num_slides]
