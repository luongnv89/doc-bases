"""
Semantic chunking using embeddings for natural text boundaries.
Replaces naive character-count splitting for better retrieval quality.
"""
import os
from typing import List
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger()

# Conditional import - langchain_experimental may not be installed
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False


class SemanticDocumentSplitter:
    """
    Split documents at semantic boundaries using embedding similarity.

    Preserves topic coherence within chunks better than character-based splitting.
    """

    def __init__(
        self,
        embeddings=None,
        breakpoint_type: str = "percentile",
        breakpoint_threshold: float = 95
    ):
        """
        Initialize semantic splitter.

        Args:
            embeddings: Embedding model (uses default if None)
            breakpoint_type: How to detect split points
                           (percentile, standard_deviation, interquartile)
            breakpoint_threshold: Threshold for creating new chunks
        """
        if not SEMANTIC_CHUNKER_AVAILABLE:
            raise ImportError(
                "SemanticChunker not available. Install with: pip install langchain-experimental"
            )

        if embeddings is None:
            from src.models.embeddings import get_embedding_model
            embeddings = get_embedding_model()

        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_threshold
        )
        logger.info(f"SemanticDocumentSplitter initialized with {breakpoint_type} strategy")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents at semantic boundaries.

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents with preserved metadata
        """
        try:
            chunks = self.splitter.split_documents(documents)
            logger.info(f"Semantic splitting: {len(documents)} docs -> {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Semantic splitting failed: {e}")
            raise


def get_chunking_strategy() -> str:
    """Get configured chunking strategy from environment."""
    return os.getenv("CHUNKING_STRATEGY", "recursive").lower()


def is_semantic_chunker_available() -> bool:
    """Check if semantic chunker is available."""
    return SEMANTIC_CHUNKER_AVAILABLE
