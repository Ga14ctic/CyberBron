"""
Web Search Service using DuckDuckGo
Provides real-time web search capabilities for CyberBron.
"""
import logging
from typing import List, Dict, Optional
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class SearchService:
    """Service for performing web searches using DuckDuckGo."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the search service.
        
        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        logger.info(f"SearchService initialized with max_results={max_results}")
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Perform a web search using DuckDuckGo.
        
        Args:
            query: Search query string
            max_results: Optional override for max results
            
        Returns:
            List of search results with title, link, and snippet
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
        
        results_limit = max_results if max_results is not None else self.max_results
        
        try:
            logger.info(f"Searching for: {query}")
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=results_limit):
                    results.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("body", "")
                    })
                
                logger.info(f"Found {len(results)} results for query: {query}")
                return results
                
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []
    
    def search_cybersecurity(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a cybersecurity-focused search.
        Adds cybersecurity context to the query.
        
        Args:
            query: Base search query
            
        Returns:
            List of search results
        """
        # Add cybersecurity context
        enhanced_query = f"{query} cybersecurity"
        return self.search(enhanced_query)
    
    def search_cve(self, cve_id: str) -> List[Dict[str, str]]:
        """
        Search for information about a specific CVE.
        
        Args:
            cve_id: CVE identifier (e.g., "CVE-2024-1234")
            
        Returns:
            List of search results about the CVE
        """
        query = f"{cve_id} vulnerability details"
        return self.search(query)
    
    def should_trigger_search(self, query: str, keywords: List[str]) -> bool:
        """
        Determine if a query should trigger web search based on keywords.
        
        Args:
            query: User query
            keywords: List of keywords that trigger search
            
        Returns:
            True if search should be triggered
        """
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in keywords)
    
    def format_search_results(self, results: List[Dict[str, str]]) -> str:
        """
        Format search results into a readable string for LLM context.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string of search results
        """
        if not results:
            return "No search results found."
        
        formatted = "🌐 Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result['title']}**\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   Source: {result['link']}\n\n"
        
        return formatted
