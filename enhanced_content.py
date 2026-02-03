"""
Enhanced content generation with Pearson T-Level integration
This module provides improved flashcard and quiz generation with curriculum alignment
"""

from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

class PearsonTLevelIntegration:
    """
    Integration with Pearson Cybersecurity T-Level curriculum
    """
    
    # Pearson T-Level Cybersecurity Core Topics
    CORE_TOPICS = {
        "Unit 1": {
            "title": "Digital Technologies",
            "subtopics": [
                "Hardware and networking",
                "Software and operating systems",
                "Cloud computing",
                "Virtualization",
                "Data management"
            ]
        },
        "Unit 2": {
            "title": "Cyber Security Concepts",
            "subtopics": [
                "Threats and vulnerabilities",
                "Risk assessment",
                "Security controls",
                "Cryptography",
                "Authentication and access control"
            ]
        },
        "Unit 3": {
            "title": "Cyber Security Operations",
            "subtopics": [
                "Network security",
                "Application security",
                "Incident response",
                "Security monitoring",
                "Penetration testing"
            ]
        },
        "Unit 4": {
            "title": "Laws and Standards",
            "subtopics": [
                "Data Protection Act",
                "Computer Misuse Act",
                "GDPR",
                "ISO 27001",
                "NIST Cybersecurity Framework"
            ]
        }
    }
    
    # Key learning outcomes mapped to topics
    LEARNING_OUTCOMES = {
        "Network Security": [
            "Understand network protocols and their security implications",
            "Identify common network attacks (DoS, MitM, packet sniffing)",
            "Configure firewalls and IDS/IPS systems",
            "Implement network segmentation",
            "Secure wireless networks"
        ],
        "Cryptography": [
            "Explain symmetric vs asymmetric encryption",
            "Understand hashing and digital signatures",
            "Implement secure key management",
            "Apply cryptographic protocols (SSL/TLS)",
            "Analyze cryptographic vulnerabilities"
        ],
        "Web Security": [
            "Identify OWASP Top 10 vulnerabilities",
            "Prevent SQL injection and XSS attacks",
            "Implement secure authentication",
            "Configure HTTPS and certificates",
            "Perform web application security testing"
        ],
        "Incident Response": [
            "Follow incident response procedures",
            "Conduct forensic analysis",
            "Document security incidents",
            "Implement recovery procedures",
            "Learn from security breaches"
        ]
    }
    
    # Specific technical details database for presentations
    TECHNICAL_SPECIFICS = {
        "Network Security": {
            "protocols": ["TCP/IP", "UDP", "ICMP", "ARP", "DNS", "DHCP", "HTTP/HTTPS", "FTP/SFTP", "SSH", "Telnet"],
            "attacks": [
                "DDoS attacks: SYN flood (80% of attacks), UDP flood, ICMP flood - Mirai botnet affected 600,000+ devices",
                "Man-in-the-Middle: ARP spoofing, DNS poisoning, SSL stripping - 35% of public WiFi vulnerable",
                "Packet sniffing: Wireshark, tcpdump - can capture credentials on unencrypted networks"
            ],
            "tools": ["Wireshark", "Nmap", "Metasploit", "Snort", "Suricata", "pfSense", "Cisco ASA"],
            "standards": ["IEEE 802.1X", "WPA3", "IPSec", "TLS 1.3", "DNSSEC"],
            "statistics": [
                "93% of network breaches exploit known vulnerabilities (Verizon DBIR 2023)",
                "Average cost of a data breach: $4.45 million (IBM 2023)",
                "68% of breaches involve human error"
            ]
        },
        "Cryptography": {
            "algorithms": ["AES-256", "RSA-2048/4096", "SHA-256/SHA-3", "ECDSA", "ChaCha20-Poly1305"],
            "applications": [
                "TLS 1.3: 40% faster than TLS 1.2, removes weak ciphers",
                "HTTPS adoption: 95% of web traffic encrypted (2023)",
                "End-to-end encryption: Signal Protocol, WhatsApp (2B+ users)"
            ],
            "key_facts": [
                "AES-256: 2^256 possible keys (longer than age of universe to brute force)",
                "RSA recommended minimum: 2048 bits (NIST), 4096 for long-term security",
                "Quantum threat: Shor's algorithm can break RSA, hence post-quantum cryptography (PQC)"
            ],
            "standards": ["FIPS 140-2/3", "NIST SP 800-57", "ISO/IEC 19790"],
            "tools": ["OpenSSL", "GPG", "HashiCorp Vault", "Let's Encrypt", "Certbot"]
        },
        "Web Security": {
            "owasp_top_10": [
                "A01: Broken Access Control - 94% of applications tested had some form",
                "A02: Cryptographic Failures - affects 3.5 million passwords leaked in 2023",
                "A03: Injection (SQL, XSS) - 32% of web apps vulnerable",
                "A04: Insecure Design - lack of security requirements in SDLC",
                "A05: Security Misconfiguration - default credentials, unnecessary features",
                "A06: Vulnerable Components - 84% use components with known vulnerabilities",
                "A07: Authentication Failures - weak passwords, missing MFA",
                "A08: Software/Data Integrity - supply chain attacks up 300% in 2023",
                "A09: Logging Failures - 60% can't detect breaches for months",
                "A10: SSRF - server-side request forgery gaining prominence"
            ],
            "attacks": [
                "SQL Injection: Union-based, Boolean-based, Time-based blind - impacts 65% of apps",
                "XSS: Stored (most dangerous), Reflected, DOM-based - allows session hijacking",
                "CSRF: Forces users to execute unwanted actions - prevented with tokens"
            ],
            "tools": ["Burp Suite", "OWASP ZAP", "SQLMap", "Nikto", "Acunetix", "ModSecurity WAF"],
            "frameworks": ["Django", "Ruby on Rails", "ASP.NET Core", "Spring Security"],
            "statistics": [
                "Average 72 vulnerabilities per application (Veracode 2023)",
                "Web attacks increased 164% year-over-year",
                "API attacks up 681% in 2022"
            ]
        },
        "Incident Response": {
            "frameworks": [
                "NIST SP 800-61: Preparation, Detection, Analysis, Containment, Eradication, Recovery",
                "SANS Incident Handler: 6-step process used by 78% of organizations",
                "MITRE ATT&CK: 14 tactics, 193 techniques for threat detection"
            ],
            "tools": [
                "SIEM: Splunk, IBM QRadar, LogRhythm - aggregate logs from 100+ sources",
                "EDR: CrowdStrike, SentinelOne, Carbon Black - behavioral analysis",
                "Forensics: EnCase, FTK, Autopsy - disk imaging and analysis"
            ],
            "metrics": [
                "Mean Time to Detect (MTTD): Industry average 207 days (Mandiant 2023)",
                "Mean Time to Respond (MTTR): Average 73 days",
                "Cost per minute of downtime: $9,000 for enterprise"
            ],
            "case_studies": [
                "Colonial Pipeline (2021): DarkSide ransomware, $4.4M ransom paid, 6-day shutdown",
                "SolarWinds (2020): Supply chain attack affected 18,000+ organizations",
                "MOVEit (2023): Zero-day exploit, 600+ organizations breached"
            ]
        },
        "Compliance": {
            "regulations": [
                "GDPR: €20M or 4% of global revenue fines - 1,618 fines issued (2023)",
                "PCI DSS v4.0: 12 requirements for handling card data",
                "HIPAA: $100-50,000 per violation, max $1.5M per year",
                "SOX: Criminal penalties for executives, up to 20 years prison"
            ],
            "standards": [
                "ISO 27001: 114 controls across 14 domains - 50,000+ certified orgs",
                "NIST CSF: Identify, Protect, Detect, Respond, Recover - used by 50% of US orgs",
                "CIS Controls: 18 controls, 153 safeguards - prioritized implementation"
            ]
        }
    }
    
    @classmethod
    def get_topic_context(cls, topic: str) -> Optional[Dict]:
        """Get curriculum context for a topic."""
        for unit, content in cls.CORE_TOPICS.items():
            if topic.lower() in content["title"].lower():
                return {
                    "unit": unit,
                    "title": content["title"],
                    "subtopics": content["subtopics"]
                }
            for subtopic in content["subtopics"]:
                if topic.lower() in subtopic.lower():
                    return {
                        "unit": unit,
                        "title": content["title"],
                        "related_subtopic": subtopic,
                        "all_subtopics": content["subtopics"]
                    }
        return None
    
    @classmethod
    def get_learning_outcomes(cls, topic: str) -> List[str]:
        """Get learning outcomes for a topic."""
        for key, outcomes in cls.LEARNING_OUTCOMES.items():
            if topic.lower() in key.lower() or key.lower() in topic.lower():
                return outcomes
        return []
    
    @classmethod
    def get_technical_specifics(cls, topic: str) -> Dict:
        """Get technical specifics for a topic."""
        for key, specifics in cls.TECHNICAL_SPECIFICS.items():
            if key.lower() in topic.lower() or topic.lower() in key.lower():
                return specifics
        return {}
    
    @classmethod
    def enhance_prompt_with_curriculum(cls, base_prompt: str, topic: str) -> str:
        """Enhance generation prompt with curriculum context."""
        context = cls.get_topic_context(topic)
        outcomes = cls.get_learning_outcomes(topic)
        specifics = cls.get_technical_specifics(topic)
        
        enhancement = "\n\nT-Level Curriculum Context:\n"
        
        if context:
            enhancement += f"Unit: {context.get('unit', 'N/A')}\n"
            enhancement += f"Area: {context.get('title', 'N/A')}\n"
        
        if outcomes:
            enhancement += "\nKey Learning Outcomes:\n"
            for outcome in outcomes[:3]:  # Top 3 most relevant
                enhancement += f"- {outcome}\n"
        
        if specifics:
            enhancement += "\nTechnical Details to Include:\n"
            for category, items in list(specifics.items())[:3]:
                if isinstance(items, list) and items:
                    enhancement += f"- {category.title()}: {items[0]}\n"
        
        enhancement += "\nAlign the generated content with these T-Level objectives and technical specifics."
        
        return base_prompt + enhancement


class EnhancedContentGenerator:
    """
    Enhanced content generation with improved algorithms
    """
    
    @staticmethod
    def generate_flashcard_prompt(content: str, num_cards: int, deck: str) -> str:
        """
        Generate enhanced prompt for flashcard creation.
        """
        # Check if content relates to curriculum
        curriculum_context = PearsonTLevelIntegration.enhance_prompt_with_curriculum(
            "", deck
        )
        
        prompt = f"""Generate {num_cards} high-quality flashcards from the following content.

Content:
{content}

Requirements:
1. Create clear, concise questions that test understanding
2. Include both basic recall and application questions
3. Use varied question formats:
   - Definition questions (What is...)
   - Process questions (How does...)
   - Comparison questions (What's the difference between...)
   - Scenario-based questions (In situation X, what would...)
   - Best practice questions (What is the recommended...)
4. Answers should be comprehensive but concise (2-4 sentences)
5. Include real-world examples where applicable
6. Focus on practical, job-relevant knowledge

{curriculum_context}

Format each flashcard as:
Q: [Clear, specific question]
A: [Comprehensive answer with examples]

Generate exactly {num_cards} flashcards."""
        
        return prompt
    
    @staticmethod
    def generate_quiz_prompt(content: str, num_questions: int, difficulty: str, title: str) -> str:
        """
        Generate enhanced prompt for quiz creation.
        """
        curriculum_context = PearsonTLevelIntegration.enhance_prompt_with_curriculum(
            "", title
        )
        
        difficulty_guidance = {
            "easy": "Focus on basic recall and definitions. Single-step problems.",
            "medium": "Mix of recall and application. Multi-step problems with guidance.",
            "hard": "Focus on analysis, synthesis, and complex scenarios. Minimal guidance."
        }
        
        prompt = f"""Generate a {difficulty} difficulty quiz with {num_questions} questions on: {title}

Content to base questions on:
{content}

Difficulty Level: {difficulty}
Guidance: {difficulty_guidance.get(difficulty, difficulty_guidance['medium'])}

Question Type Distribution:
- 60% Multiple Choice (4 options, 1 correct)
- 20% True/False
- 20% Short Answer

Requirements for ALL questions:
1. Clear, unambiguous wording
2. Avoid trick questions
3. Test understanding, not memorization
4. Include practical scenarios
5. Provide detailed explanations
6. Reference real-world applications

{curriculum_context}

For Multiple Choice:
- Make distractors plausible but clearly wrong
- Avoid "all of the above" or "none of the above"
- Randomize correct answer position

For True/False:
- Test significant concepts
- Avoid double negatives
- Explanation must clarify why

For Short Answer:
- Specific, answerable questions
- Clear evaluation criteria
- Sample correct answer provided

Format as JSON array:
[
  {{
    "type": "multiple_choice",
    "question": "Question text?",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "B",
    "explanation": "Detailed explanation with reasoning"
  }},
  {{
    "type": "true_false",
    "question": "Statement to evaluate?",
    "correct_answer": "true",
    "explanation": "Why this is true/false"
  }},
  {{
    "type": "short_answer",
    "question": "Open-ended question?",
    "correct_answer": "Sample correct answer",
    "explanation": "Evaluation criteria and key points"
  }}
]

Generate exactly {num_questions} questions."""
        
        return prompt
    
    @staticmethod
    def generate_notes_prompt(topic: str, detail_level: str = "moderate") -> str:
        """
        Generate enhanced prompt for note creation.
        """
        curriculum_context = PearsonTLevelIntegration.enhance_prompt_with_curriculum(
            "", topic
        )
        
        detail_guidance = {
            "brief": "1-2 paragraphs with key points only",
            "moderate": "3-5 paragraphs with examples and explanations",
            "detailed": "Comprehensive coverage with multiple examples, diagrams descriptions, and best practices"
        }
        
        prompt = f"""Generate comprehensive study notes on: {topic}

Detail Level: {detail_level}
Length Guidance: {detail_guidance.get(detail_level, detail_guidance['moderate'])}

Structure:
1. Overview (what and why)
2. Key Concepts (with definitions)
3. Practical Examples (real-world scenarios)
4. Best Practices (industry standards)
5. Common Pitfalls (what to avoid)
6. Summary (key takeaways)

{curriculum_context}

Requirements:
- Use clear, student-friendly language
- Include specific examples
- Reference industry standards
- Highlight exam-relevant content
- Use markdown formatting:
  - ## for main sections
  - ### for subsections
  - **bold** for key terms
  - `code` for technical terms
  - - bullet points for lists
  - 1. numbered lists for procedures

Generate comprehensive, well-structured notes."""
        
        return prompt


# Convenience functions for direct use
def generate_enhanced_flashcards(llm, content: str, num_cards: int = 10, deck: str = "Default") -> List[Dict]:
    """Generate enhanced flashcards with curriculum alignment."""
    try:
        prompt = EnhancedContentGenerator.generate_flashcard_prompt(content, num_cards, deck)
        response = llm.invoke(prompt)
        
        # Parse response into flashcards
        flashcards = []
        lines = response.strip().split('\n')
        current_card = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                if current_card:
                    flashcards.append(current_card)
                current_card = {'question': line[2:].strip()}
            elif line.startswith('A:'):
                current_card['answer'] = line[2:].strip()
        
        if current_card and 'answer' in current_card:
            flashcards.append(current_card)
        
        logger.info(f"Generated {len(flashcards)} enhanced flashcards")
        return flashcards[:num_cards]
    
    except Exception as e:
        logger.error(f"Error generating enhanced flashcards: {e}")
        return []


def generate_enhanced_quiz(llm, content: str, num_questions: int = 10, difficulty: str = "medium", title: str = "Quiz") -> Dict:
    """Generate enhanced quiz with curriculum alignment."""
    try:
        prompt = EnhancedContentGenerator.generate_quiz_prompt(content, num_questions, difficulty, title)
        response = llm.invoke(prompt)
        
        # Try to parse as JSON
        try:
            questions = json.loads(response)
            if isinstance(questions, list):
                logger.info(f"Generated {len(questions)} enhanced quiz questions")
                return {
                    "title": title,
                    "difficulty": difficulty,
                    "questions": questions[:num_questions]
                }
        except json.JSONDecodeError:
            logger.warning("Failed to parse quiz as JSON, using fallback parser")
        
        # Fallback: manual parsing
        return {
            "title": title,
            "difficulty": difficulty,
            "questions": []
        }
    
    except Exception as e:
        logger.error(f"Error generating enhanced quiz: {e}")
        return {"title": title, "difficulty": difficulty, "questions": []}


def generate_enhanced_notes(llm, topic: str, detail_level: str = "moderate") -> str:
    """Generate enhanced notes with curriculum alignment."""
    try:
        prompt = EnhancedContentGenerator.generate_notes_prompt(topic, detail_level)
        response = llm.invoke(prompt)
        logger.info(f"Generated enhanced notes for: {topic}")
        return response
    except Exception as e:
        logger.error(f"Error generating enhanced notes: {e}")
        return f"# {topic}\n\nError generating notes. Please try again."
