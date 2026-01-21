"""
BM25 index management for hybrid search.

This module provides BM25 (Best Matching 25) keyword-based retrieval
to complement dense vector search in hybrid retrieval.
"""

import os
import pickle
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger

logger = get_logger()

# Default knowledge base directory
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledges")


def tokenize(text: str) -> list[str]:
    """
    Simple tokenization for BM25.

    Converts text to lowercase and splits on whitespace and punctuation.

    Args:
        text: Input text to tokenize.

    Returns:
        List of tokens.
    """
    # Simple tokenization: lowercase, split on non-alphanumeric
    import re

    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


class BM25Index:
    """
    Manages BM25 index for a knowledge base.

    The index is persisted to disk alongside the vector store for
    efficient hybrid search without rebuilding on each query.

    Attributes:
        kb_name: Name of the knowledge base.
        index_path: Path to the persisted index file.
        bm25: The BM25Okapi instance.
        documents: List of indexed documents.
    """

    def __init__(self, kb_name: str):
        """
        Initialize BM25Index for a knowledge base.

        Args:
            kb_name: Name of the knowledge base.
        """
        self.kb_name = kb_name
        self.index_path = Path(KNOWLEDGE_BASE_DIR) / kb_name / "bm25_index.pkl"
        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []
        self._corpus: list[list[str]] = []

    def build_index(self, documents: list[Document]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of LangChain Document objects to index.
        """
        if not documents:
            logger.warning(f"No documents provided for BM25 index '{self.kb_name}'")
            return

        self.documents = documents
        self._corpus = [tokenize(doc.page_content) for doc in documents]

        # Filter out empty documents
        valid_indices = [i for i, tokens in enumerate(self._corpus) if tokens]
        if len(valid_indices) < len(documents):
            logger.warning(f"Filtered {len(documents) - len(valid_indices)} empty documents from BM25 index")
            self._corpus = [self._corpus[i] for i in valid_indices]
            self.documents = [self.documents[i] for i in valid_indices]

        if not self._corpus:
            logger.warning(f"All documents were empty for BM25 index '{self.kb_name}'")
            return

        self.bm25 = BM25Okapi(self._corpus)
        logger.info(f"Built BM25 index with {len(self.documents)} documents")

    def load_index(self) -> bool:
        """
        Load existing BM25 index from disk.

        Returns:
            True if index was loaded successfully, False otherwise.
        """
        if not self.index_path.exists():
            logger.debug(f"BM25 index not found at {self.index_path}")
            return False

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)

            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self._corpus = data["corpus"]

            logger.info(f"Loaded BM25 index for '{self.kb_name}' with {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def save_index(self) -> None:
        """
        Persist BM25 index to disk.

        Raises:
            ValueError: If no index has been built.
        """
        if self.bm25 is None:
            raise ValueError("No BM25 index to save. Call build_index first.")

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "corpus": self._corpus,
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index to {self.index_path}")

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """
        Search BM25 index and return documents with scores.

        Args:
            query: Search query string.
            k: Maximum number of documents to return.

        Returns:
            List of (Document, score) tuples sorted by descending score.
        """
        if self.bm25 is None:
            logger.warning("BM25 index not initialized, returning empty results")
            return []

        if not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning("Query tokenized to empty, returning empty results")
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score (descending)
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        results = [(self.documents[idx], score) for idx, score in scored_indices if score > 0]  # Filter zero-score documents

        return results

    def delete_index(self) -> bool:
        """
        Delete the persisted BM25 index file.

        Returns:
            True if deleted successfully, False otherwise.
        """
        if self.index_path.exists():
            try:
                self.index_path.unlink()
                logger.info(f"Deleted BM25 index at {self.index_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete BM25 index: {e}")
                return False
        return True

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded and ready for search."""
        return self.bm25 is not None and len(self.documents) > 0
