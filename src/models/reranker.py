"""
Reranker models for improving retrieval relevance.

This module provides a factory function for creating reranker instances
that can score and reorder documents based on query relevance.

Supported providers:
- cross-encoder: Local sentence-transformers CrossEncoder (free, requires GPU for speed)
- cohere: Cohere Rerank API (paid, cloud-based)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger()


class Reranker(ABC):
    """Abstract base class for reranking models."""

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: Documents to rerank.
            top_k: Maximum number of documents to return.

        Returns:
            Reranked documents sorted by descending relevance.
        """
        pass


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder based reranker using sentence-transformers.

    Cross-encoders jointly encode query-document pairs and produce
    a relevance score, providing more accurate ranking than bi-encoders
    at the cost of higher latency.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Fast and good quality for general English text
    - ~80MB model size
    - Works well on CPU, faster on GPU
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize CrossEncoder reranker.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                Options:
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (default, balanced)
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, lower quality)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better quality)
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers is required for CrossEncoderReranker. " "Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded CrossEncoder reranker: {model_name}")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """
        Rerank documents using cross-encoder scoring.

        Args:
            query: Search query.
            documents: Documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Top-k documents sorted by cross-encoder score.
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score (descending) and return top_k
        scored_docs = sorted(zip(documents, scores, strict=False), key=lambda x: x[1], reverse=True)

        return [doc for doc, _score in scored_docs[:top_k]]


class CohereReranker(Reranker):
    """
    Cohere Rerank API-based reranker.

    Uses Cohere's cloud-based reranking service for high-quality
    relevance scoring. Requires API key and has usage costs.

    Default model: rerank-english-v3.0
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key. If None, reads from COHERE_API_KEY env var.
            model: Cohere rerank model name.
                Options:
                - rerank-english-v3.0 (default)
                - rerank-multilingual-v3.0 (for non-English)
                - rerank-english-v2.0 (legacy)
        """
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere is required for CohereReranker. " "Install with: pip install cohere")

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY environment variable " "or pass api_key parameter.")

        self.model = model
        self.client = cohere.Client(self.api_key)
        logger.info(f"Initialized Cohere reranker: {model}")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """
        Rerank documents using Cohere Rerank API.

        Args:
            query: Search query.
            documents: Documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Top-k documents sorted by Cohere relevance score.
        """
        if not documents:
            return []

        # Extract document texts
        doc_texts = [doc.page_content for doc in documents]

        # Call Cohere Rerank API
        response = self.client.rerank(
            query=query,
            documents=doc_texts,
            model=self.model,
            top_n=top_k,
        )

        # Reorder documents based on response
        reranked = [documents[result.index] for result in response.results]

        return reranked


class PassthroughReranker(Reranker):
    """
    No-op reranker that returns documents unchanged.

    Useful for testing or when reranking is disabled.
    """

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Return documents unchanged (truncated to top_k)."""
        return documents[:top_k]


def get_reranker(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Reranker | None:
    """
    Factory function for reranker models.

    Reads configuration from environment variables if parameters not provided.

    Args:
        provider: Reranker provider ("cross-encoder", "cohere", or None).
            If None, reads from RERANKER_PROVIDER env var.
        model: Model name. If None, reads from RERANKER_MODEL env var.
        api_key: API key for cloud providers.

    Returns:
        Configured Reranker instance, or None if no provider specified.

    Environment Variables:
        RERANKER_PROVIDER: Provider name (cross-encoder, cohere)
        RERANKER_MODEL: Model name
        COHERE_API_KEY: API key for Cohere

    Examples:
        # From environment variables
        reranker = get_reranker()

        # Explicit configuration
        reranker = get_reranker(
            provider="cross-encoder",
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    """
    # Read from environment if not provided
    provider = provider or os.getenv("RERANKER_PROVIDER", "").strip()

    if not provider:
        logger.debug("No reranker provider configured")
        return None

    provider = provider.lower()

    if provider == "cross-encoder":
        model = model or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            return CrossEncoderReranker(model_name=model)
        except ImportError as e:
            logger.warning(f"Cannot create CrossEncoder reranker: {e}")
            return None

    elif provider == "cohere":
        model = model or os.getenv("RERANKER_MODEL", "rerank-english-v3.0")
        try:
            return CohereReranker(api_key=api_key, model=model)
        except ImportError as e:
            logger.warning(f"Cannot create Cohere reranker: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Cannot create Cohere reranker: {e}")
            return None

    elif provider == "none" or provider == "passthrough":
        return PassthroughReranker()

    else:
        logger.warning(f"Unknown reranker provider: {provider}")
        return None
