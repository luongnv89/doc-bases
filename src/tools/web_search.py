"""
Web search tool for fallback retrieval in Corrective RAG.
Uses DuckDuckGo (free, no API key required).
"""

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger()

# Try to import DuckDuckGo search
try:
    from langchain_community.tools import DuckDuckGoSearchResults

    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for additional context when document retrieval is insufficient.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Concatenated search results as text
    """
    if not DUCKDUCKGO_AVAILABLE:
        return "Web search unavailable. Install: pip install duckduckgo-search"

    try:
        search = DuckDuckGoSearchResults(num_results=max_results)
        results = search.run(query)
        logger.info(f"Web search for '{query}': {len(results)} chars returned")
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search error: {e}"


def web_search_to_documents(query: str, max_results: int = 3) -> list[Document]:
    """
    Perform web search and return results as LangChain Documents.

    Useful for integrating web results into RAG pipeline.
    """
    if not DUCKDUCKGO_AVAILABLE:
        return []

    try:
        search = DuckDuckGoSearchResults(num_results=max_results)
        results = search.invoke(query)

        documents = []
        if isinstance(results, list):
            for result in results:
                doc = Document(
                    page_content=result.get("snippet", str(result)),
                    metadata={"source": result.get("link", "web_search"), "title": result.get("title", ""), "content_type": "web_search"},
                )
                documents.append(doc)
        else:
            # Handle string results
            documents.append(Document(page_content=str(results), metadata={"source": "web_search", "content_type": "web_search"}))

        return documents
    except Exception as e:
        logger.error(f"Web search to documents failed: {e}")
        return []


def is_web_search_available() -> bool:
    """Check if web search is available."""
    return DUCKDUCKGO_AVAILABLE
