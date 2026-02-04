"""
Flashcard Generator using AI
Generates flashcards from text content using LLM.
"""
import logging
from typing import List, Dict
import json

logger = logging.getLogger(__name__)


class FlashcardGenerator:
    """AI-powered flashcard generator."""
    
    def __init__(self, llm):
        """
        Initialize the flashcard generator.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        logger.info("FlashcardGenerator initialized")
    
    def generate_from_text(self, text: str, num_cards: int = 10, topic: str = None) -> List[Dict]:
        """
        Generate flashcards from text content.
        
        Args:
            text: Source text to generate flashcards from
            num_cards: Number of flashcards to generate
            topic: Optional topic for the flashcards
            
        Returns:
            List of flashcard dictionaries with 'question' and 'answer'
        """
        prompt = f"""Generate {num_cards} high-quality, diverse flashcards from the following text.

FLASHCARD QUALITY REQUIREMENTS:
1. CREATE VARIED QUESTION TYPES:
   - Conceptual understanding questions (What is...?, Explain...)
   - Application-based questions (How would you...?, When should...)
   - Comparative questions (Compare X and Y..., What's the difference...)
   - Multi-step reasoning questions (If X happens, what should you do? Why?)
   - Real-life scenario questions (In a situation where..., How does this apply...)
   - Relational questions (How does X relate to Y?, What connects...)

2. FOCUS ON RELATIONAL UNDERSTANDING:
   - Connect concepts to each other (How does encryption relate to authentication?)
   - Link theory to practice (When would you use SHA-256 vs SHA-512 in production?)
   - Relate to real-world applications (What companies/tools use this?)
   - Show cause and effect relationships
   - Demonstrate concept hierarchies and dependencies

3. INCLUDE RICH CONTEXT:
   - Add specific examples, not generic statements
   - Include relevant statistics, dates, or quantifiable data
   - Reference real tools, frameworks, standards, or technologies
   - Mention practical implications and use cases
   - Provide comprehensive answers with multiple aspects

4. ENSURE VARIETY IN STRUCTURE:
   - Mix simple definition cards with complex scenario cards
   - Include cards that require analysis, not just recall
   - Create cards that test different cognitive levels (remember, understand, apply, analyze)
   - Balance factual cards with conceptual cards
   - Add cards covering edge cases, limitations, or common mistakes

5. MAKE ANSWERS COMPREHENSIVE:
   - Answers should be detailed but clear (2-5 sentences typically)
   - Include the "why" and "how", not just "what"
   - Mention related concepts or alternatives
   - Add practical tips or common pitfalls where relevant
   - Use specific technical terminology correctly

EXAMPLE OF EXCELLENT FLASHCARDS:

Q: "Compare SHA-256 and SHA-3 in terms of design, performance, and security. When would you choose one over the other?"
A: "SHA-256 uses Merkle-Damgård construction (64 rounds) and produces 256-bit hash at ~400 MB/s on modern CPUs. SHA-3 (Keccak) uses sponge construction with different internal structure, runs 20-30% slower but more resistant to length extension attacks. Choose SHA-256 for: widespread compatibility, established security (20+ years), hardware acceleration support. Choose SHA-3 for: new applications requiring resistance to length extension, future-proofing against cryptanalysis advances, NIST standardization requirements (FIPS 202)."

Q: "In a scenario where you're implementing user authentication, why would you use Argon2 or bcrypt instead of simply hashing passwords with SHA-256?"
A: "SHA-256 is designed for speed (~400 MB/s), making it vulnerable to brute-force attacks where attackers can test billions of password combinations per second. Argon2/bcrypt are intentionally slow (10-100ms per hash) and memory-hard, dramatically increasing the cost of brute-force attacks. They also include automatic salt generation. For example, an attacker with GPU cluster could test 100 billion SHA-256 hashes/second but only 100,000 Argon2 hashes/second - a million-fold difference. OWASP and NIST recommend these password-hashing algorithms specifically."

Q: "What is the relationship between encryption, authentication, and integrity in cybersecurity, and why are all three necessary?"
A: "These are complementary security properties, not interchangeable: Encryption (confidentiality) prevents unauthorized reading; Authentication verifies identity of sender/receiver; Integrity ensures data hasn't been modified. All three are needed because: encryption alone doesn't prevent man-in-the-middle attacks (authentication needed), authentication doesn't prevent data tampering (integrity needed), integrity doesn't hide content (encryption needed). Real example: HTTPS uses TLS which provides all three - RSA/ECDH for authentication, AES for encryption, and HMAC-SHA256 for integrity."

Text to generate flashcards from:
{text}

Generate exactly {num_cards} flashcards following the quality requirements above.
Format your response as a JSON array:
[
  {{"question": "Detailed, specific question that tests understanding...", "answer": "Comprehensive answer with context and examples..."}},
  {{"question": "Different type of question...", "answer": "Detailed answer..."}}
]
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            # Look for JSON array in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                flashcards = json.loads(json_str)
                logger.info(f"Generated {len(flashcards)} flashcards from text")
                return flashcards
            else:
                logger.warning("Could not parse flashcards from LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            return []
    
    def generate_from_conversation(self, messages: List[Dict], num_cards: int = 10) -> List[Dict]:
        """
        Generate flashcards from a conversation.
        
        Args:
            messages: List of conversation messages
            num_cards: Number of flashcards to generate
            
        Returns:
            List of flashcard dictionaries
        """
        # Extract text from conversation
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        ])
        
        return self.generate_from_text(conversation_text, num_cards)
