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
                max_value=50,
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
    Generate presentation content using LLM with enhanced detail and specifics.
    
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
    # Import enhanced content module for T-Level integration
    try:
        from enhanced_content import PearsonTLevelIntegration
        
        # Get technical specifics for the topic
        specifics = PearsonTLevelIntegration.get_technical_specifics(topic)
        curriculum_context = PearsonTLevelIntegration.get_topic_context(topic)
        
        specifics_text = ""
        if specifics:
            specifics_text = "\n\nTechnical Specifics Database (USE THIS INFORMATION):\n"
            for category, items in specifics.items():
                specifics_text += f"\n{category.upper()}:\n"
                if isinstance(items, list):
                    for item in items[:5]:  # Top 5 items per category
                        specifics_text += f"- {item}\n"
        
        curriculum_text = ""
        if curriculum_context:
            curriculum_text = f"\n\nT-Level Curriculum: {curriculum_context.get('unit', '')} - {curriculum_context.get('title', '')}"
    except ImportError:
        specifics_text = ""
        curriculum_text = ""
    
    # Perform web search if enabled
    search_context = ""
    if enable_search and search_service:
        try:
            search_results = search_service.search_cybersecurity(topic)
            if search_results:
                search_context = "\n\nWeb Research Context:\n"
                for result in search_results[:10]:  # Use top 10 results for comprehensive research
                    search_context += f"- {result['title']}: {result['snippet']}\n"
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
    
    # Construct prompt with enhanced detail
    detail_instructions = {
        "Brief": """Provide 8-10 substantial points per slide, each containing:
- Specific technical details with names and specifications
- Real-world examples with concrete data
- Industry standards and frameworks by name
- Statistics and quantifiable metrics
- Multiple paragraphs of explanation for complex topics
Focus on delivering comprehensive information in a concise format.""",
        
        "Moderate": """Provide 12-16 comprehensive points per slide with multiple paragraphs covering:
- Detailed technical specifications, protocols, and algorithms
- Extensive real-world examples with case studies
- Industry standards, compliance frameworks, and methodologies
- Statistical analysis with trends and comparative data
- Practical applications with implementation details
- Best practices and common pitfalls with specific scenarios
- Historical context where relevant
Each point should be 2-3 sentences minimum with rich technical content.""",
        
        "Detailed": """Provide 20-30 in-depth points per slide, structured as multiple detailed paragraphs covering:
- Complete technical specifications with protocol details, algorithm breakdowns, and architectural patterns
- Multiple real-world case studies with incident analysis, timeline details, and impact assessment
- Comprehensive coverage of industry standards (ISO, NIST, OWASP), compliance frameworks (GDPR, HIPAA, PCI-DSS), and methodologies
- Extensive statistical analysis including trends over time, comparative analysis, market data, and research findings
- Detailed practical implementations with code examples, configuration guides, and deployment strategies
- Thorough best practices documentation with pros/cons analysis, trade-offs, and decision criteria
- Common pitfalls with specific failure scenarios, remediation strategies, and lessons learned
- Historical evolution of the technology/concept with key milestones and future trends
- Academic and research context where applicable
- Relationship to other concepts and technologies with comparison tables
Each point should be substantial (3-5 sentences) and information-dense. For complex topics like hash functions or encryption algorithms:
  * Include algorithm specifics (SHA-256: 256-bit output, 64 rounds, Merkle-Damgård construction)
  * Usage contexts (SHA-256 in Bitcoin mining, SSL/TLS certificates, digital signatures)
  * Comparative analysis (SHA-256 vs SHA-3 vs Blake2: performance benchmarks, security margins, adoption rates)
  * Vulnerability history (SHA-1 collision attacks in 2017, migration timelines, deprecated usage)
  * Implementation details (hash rates, hardware acceleration, library recommendations)
  * Academic background (Designed by NSA in 2001, published in FIPS 180-2)
  * Future outlook (Quantum resistance, post-quantum alternatives)"""
    }
    
    content_instruction = ""
    if custom_content:
        content_instruction = f"\n\nBase the presentation on this content:\n{custom_content}\n"
    
    prompt = f"""Create a highly detailed, professional, and information-rich presentation for: "{topic}"

Generate exactly {num_slides} content slides (excluding title and closing slides).
{curriculum_text}

CRITICAL REQUIREMENTS FOR MAXIMUM DETAIL AND SPECIFICITY:

1. TECHNICAL DEPTH:
   - Include specific technical details: protocol names, algorithm specifications, architectural patterns
   - Provide concrete examples with real names, version numbers, and specifications
   - Reference industry standards, frameworks, and methodologies by exact name (e.g., ISO 27001, NIST CSF, OWASP Top 10)
   - Add extensive statistics, percentages, and quantifiable data with sources and dates
   - Include actual tools, technologies, and products with version information

2. COMPREHENSIVE COVERAGE:
   - For algorithms (e.g., hash functions): Include design details, bit operations, round counts, security margins
   - For vulnerabilities: Provide CVE numbers, discovery dates, affected systems, impact analysis
   - For technologies: Cover history, current usage, market adoption, future trends
   - Include comparative analysis with pros/cons tables when discussing alternatives
   - Add implementation details, configuration examples, and practical deployment guidance

3. RELATIONAL AND CONTEXTUAL:
   - Explain relationships between concepts (how X relates to Y, when to use A vs B)
   - Provide historical context and evolution timeline
   - Include academic background and research foundations
   - Discuss real-world applications across different industries
   - Connect theory to practice with specific use cases

4. REAL-WORLD GROUNDING:
   - Reference actual security incidents with dates, organizations, and impact details
   - Include case studies with specific scenarios and outcomes
   - Mention tools used in industry (open-source and commercial)
   - Provide statistics from recent reports (OWASP, Verizon DBIR, etc.)
   - Include current trends and threat landscape updates

5. STUDENT-FOCUSED DETAIL:
   - Make content suitable for T-Level Cybersecurity with job-relevant depth
   - Include practical skills and knowledge needed for the industry
   - Provide exam-relevant details and certification-aligned content
   - Add best practices that professionals actually use
   - Include common pitfalls and how to avoid them

{detail_instructions[detail_level]}

For each slide, structure content as an array of substantial, information-dense points.
Each point should include multiple sentences with:
- Technical specifications and precise terminology
- Real examples with names, numbers, dates, and sources
- Quantifiable data with context and comparisons
- Practical implications and applications
- Relational connections to other concepts
{content_instruction}
{specifics_text}
{search_context}

Format your response as a JSON array with this structure:
[
  {{
    "title": "Specific, Technical Slide Title That Clearly States the Topic",
    "content": [
      "First comprehensive point: Start with a clear statement, then provide specific technical details with exact names and specifications. Include real-world examples with concrete data points. Add statistics with sources. Explain the practical significance and relationships to other concepts. Minimum 3-4 sentences.",
      "Second comprehensive point: Follow the same detailed pattern with different aspects. Include comparisons, alternatives, or historical context. Reference specific tools, standards, or frameworks. Provide implementation details or usage contexts. Maintain information density throughout.",
      "Continue with additional points following the comprehensive pattern, ensuring each point delivers substantial technical and practical value...",
      "For detailed level: Include extensive coverage with multiple paragraphs per point, academic references, vulnerability details with CVE numbers, market statistics, performance benchmarks, migration strategies, future trends, and quantum computing implications where relevant."
    ]
  }}
]

EXAMPLE OF EXCELLENT CONTENT (highly specific and detailed):

Title: "SHA-2 Family: SHA-256 and SHA-512 Deep Dive"
Content:
[
  "SHA-256 Architecture: Designed by NSA and published by NIST in 2001 (FIPS 180-2), SHA-256 uses a Merkle-Damgård construction with 64 rounds of processing. It produces a 256-bit (32-byte) hash digest from input data processed in 512-bit blocks. The algorithm employs 8 working variables (a-h), 64 round constants derived from cube roots of first 64 primes, and bitwise operations (AND, OR, XOR, NOT, ROTR, SHR). Computational complexity: O(n) where n is message length. Modern CPUs can hash at 200-400 MB/s, while specialized ASICs (Bitcoin miners) achieve 100+ TH/s.",
  
  "Real-World Usage and Adoption: SHA-256 is the backbone of Bitcoin blockchain (mining difficulty target requires hash with specific number of leading zeros), SSL/TLS certificates (replacing deprecated SHA-1), digital signatures in PKI, HMAC-SHA256 in JWT tokens, and Git commit IDs. Google announced full SHA-1 deprecation in 2017 following collision attacks. As of 2024, 95% of SSL certificates use SHA-256. Performance: OpenSSL benchmarks show 450 MB/s on modern x86_64 processors vs 180 MB/s for SHA-512 on 32-bit systems.",
  
  "Comparative Analysis - SHA-256 vs Alternatives: SHA-256 offers 128-bit security level (2^128 operations for collision). Comparison: SHA-3 (Keccak) uses sponge construction, more resistant to length extension attacks but 20-30% slower; BLAKE2 is faster (twice SHA-256 speed) but less standardized; SHA-1 deprecated due to 2017 collision attacks by Google/CWI (SHAttered). Migration timeline: Major CAs stopped issuing SHA-1 certs in 2016, browsers blocked SHA-1 by 2017. Trade-offs: SHA-256 balance of security (no known vulnerabilities after 20+ years), speed, and widespread support.",
  
  "Security Analysis and Vulnerability History: SHA-256 has no known practical attacks as of 2024. Best theoretical attack: 2^252 operations (Keccak team, 2009) vs brute force 2^256. Length extension vulnerability: Mitigated using HMAC or SHA-512/256 truncated variant. Quantum computing threat: Grover's algorithm reduces security to 2^128 operations, requiring 256-bit output minimum. NIST Post-Quantum Cryptography project (2016-present) evaluating quantum-resistant alternatives. Current recommendation: SHA-256 secure until at least 2030, SHA-384/512 for long-term security (30+ years).",
  
  "Implementation Best Practices: Use established libraries: OpenSSL (C/C++), hashlib (Python), crypto (Node.js), Java.security.MessageDigest. Avoid: Rolling your own crypto, using MD5/SHA-1 for security, applying single hash to passwords (use Argon2/bcrypt/PBKDF2 instead). Common pitfalls: Not validating input length (DoS via large inputs), comparing hashes with == instead of constant-time comparison (timing attacks), forgetting to include salt in password hashing. Production example: Django uses PBKDF2-SHA256 with 150,000 iterations by default. Hardware acceleration: AES-NI instructions provide 2-3x speedup for SHA-256 on Intel/AMD CPUs.",
  
  "Industry Standards and Compliance: NIST FIPS 180-4 (2015) is current specification. Required by: HIPAA for medical records (when using HMAC), PCI-DSS 3.2+ for payment data, FISMA for US federal systems. Compliance requirements typically mandate SHA-256 minimum for new deployments. EU GDPR Article 32 recommends 'state of the art' encryption (interpreted as SHA-256+). Common Criteria EAL4+ certification available for hardware implementations. ISO/IEC 10118-3:2018 international standard. NIST deprecation timeline: No plans to deprecate SHA-256; SHA-1 fully deprecated 2030, MD5 already prohibited for security use.",
  
  "Future Trends and Post-Quantum Landscape: Quantum computers with 4000+ logical qubits could break SHA-256 via Grover's algorithm (estimated 2035-2040 timeframe per IBM roadmap). NIST's post-quantum candidates: SPHINCS+ (stateless hash-based signatures) selected 2022. Hybrid approaches emerging: SHA-256 + lattice-based schemes for transition period. Research focus: Homomorphic hashing for privacy-preserving computation, hardware acceleration for IoT devices, zero-knowledge proofs using hash functions. Academic interest: Collision resistance proofs, quantum algorithm analysis, side-channel attack mitigation. Industry migration plans: Google Project Zero tracking quantum threats, major cloud providers adding quantum-safe options by 2025."
]

EXAMPLE OF POOR CONTENT (too generic, avoid this):
"SHA-256 is a cryptographic hash function. It is used for security purposes in many applications. It is considered secure and widely used. Organizations use it to protect data. It has some advantages over older algorithms."

Generate slides with the level of detail shown in the EXCELLENT example. Every slide must be information-dense, technically precise, and rich with specific details, examples, and data.
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
                    # Content should be a list of bullet points for detailed information
                    # If it's a string, keep it as is for backward compatibility
                    if isinstance(slide['content'], (str, list)):
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
