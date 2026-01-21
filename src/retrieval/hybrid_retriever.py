"""
Hybrid Retriever combining BM25 and dense vector search.

This module provides a unified retriever that combines keyword-based (BM25)
and semantic (dense vector) retrieval with Reciprocal Rank Fusion,
optionally followed by cross-encoder reranking.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from src.retrieval.bm25_index import BM25Index
from src.retrieval.rrf_fusion import reciprocal_rank_fusion
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from langchain_chroma import Chroma

    from src.models.reranker import Reranker

logger = get_logger()


class HybridRetrieverWrapper(BaseRetriever):
    """
    LangChain-compatible retriever wrapper for HybridRetriever.

    This wrapper implements the BaseRetriever interface so it can be used
    as a drop-in replacement in LangChain chains and agents.
    """

    hybrid_retriever: Any = Field(description="The HybridRetriever instance")
    search_kwargs: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Sync retrieval."""
        k = self.search_kwargs.get("k")
        return self.hybrid_retriever.invoke(query, k=k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Async retrieval."""
        k = self.search_kwargs.get("k")
        return await self.hybrid_retriever.ainvoke(query, k=k)


class HybridRetriever:
    """
    Unified retriever combining BM25 + Dense search with optional reranking.

    Supports three retrieval modes:
    - dense: Traditional vector similarity search only (default)
    - hybrid: Combines BM25 and dense search using RRF fusion

    Optionally applies cross-encoder reranking for improved relevance.

    Attributes:
        vectorstore: ChromaDB vector store for dense retrieval.
        kb_name: Knowledge base name for BM25 index location.
        retrieval_mode: Either "dense" or "hybrid".
        reranker: Optional reranker for final scoring.
        retrieval_k: Number of candidates to retrieve from each method.
        final_k: Number of documents to return after fusion/reranking.
        rrf_constant: RRF fusion parameter (default 60).
    """

    def __init__(
        self,
        vectorstore: Chroma,
        kb_name: str,
        retrieval_mode: str = "dense",
        reranker: Reranker | None = None,
        retrieval_k: int = 10,
        final_k: int = 5,
        rrf_constant: int = 60,
    ):
        """
        Initialize HybridRetriever.

        Args:
            vectorstore: ChromaDB vector store.
            kb_name: Name of the knowledge base.
            retrieval_mode: "dense" or "hybrid".
            reranker: Optional reranker instance.
            retrieval_k: Candidates to retrieve from each method.
            final_k: Final documents to return.
            rrf_constant: RRF constant (default 60).
        """
        self.vectorstore = vectorstore
        self.kb_name = kb_name
        self.retrieval_mode = retrieval_mode
        self.reranker = reranker
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        self.rrf_constant = rrf_constant

        # Initialize BM25 index for hybrid mode
        self._bm25_index: BM25Index | None = None
        if retrieval_mode == "hybrid":
            self._bm25_index = BM25Index(kb_name)
            if not self._bm25_index.load_index():
                logger.warning(f"BM25 index not found for '{kb_name}'. " "Hybrid search will fall back to dense-only until index is built.")

        logger.info(f"Initialized HybridRetriever: mode={retrieval_mode}, " f"k={retrieval_k}, final_k={final_k}, reranker={reranker is not None}")

    def as_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        """
        Return a LangChain-compatible retriever.

        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 5}).

        Returns:
            BaseRetriever-compatible wrapper.
        """
        return HybridRetrieverWrapper(
            hybrid_retriever=self,
            search_kwargs=search_kwargs or {},
        )

    def invoke(self, query: str, k: int | None = None) -> list[Document]:
        """
        Synchronous retrieval.

        Args:
            query: Search query string.
            k: Optional override for final_k.

        Returns:
            List of retrieved documents.
        """
        final_k = k if k is not None else self.final_k

        if self.retrieval_mode == "hybrid" and self._bm25_index and self._bm25_index.is_loaded:
            results = self._hybrid_search(query, final_k)
        else:
            results = self._dense_search(query, final_k)

        # Apply reranking if configured
        if self.reranker and results:
            results = self._rerank(query, results, final_k)

        return results

    async def ainvoke(self, query: str, k: int | None = None) -> list[Document]:
        """
        Asynchronous retrieval.

        Args:
            query: Search query string.
            k: Optional override for final_k.

        Returns:
            List of retrieved documents.
        """
        # Run sync retrieval in thread pool for true async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, query, k)

    def _dense_search(self, query: str, k: int) -> list[Document]:
        """
        Perform dense vector search.

        Args:
            query: Search query.
            k: Number of documents to retrieve.

        Returns:
            List of documents.
        """
        # Use similarity_search_with_score for consistency with hybrid
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # ChromaDB returns (doc, distance), lower is better
        # Convert to documents only
        return [doc for doc, _score in results]

    def _bm25_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query.
            k: Number of documents to retrieve.

        Returns:
            List of (document, score) tuples.
        """
        if self._bm25_index is None or not self._bm25_index.is_loaded:
            return []
        return self._bm25_index.search(query, k=k)

    def _hybrid_search(self, query: str, final_k: int) -> list[Document]:
        """
        Perform hybrid search combining BM25 and dense retrieval.

        Args:
            query: Search query.
            final_k: Number of documents to return after fusion.

        Returns:
            List of fused documents.
        """
        # Get candidates from both retrievers
        dense_results = self.vectorstore.similarity_search_with_score(query, k=self.retrieval_k)
        # Convert ChromaDB results: (doc, distance) -> (doc, similarity)
        # ChromaDB uses L2 distance by default, so we convert to similarity
        dense_results_converted = [(doc, 1.0 / (1.0 + distance)) for doc, distance in dense_results]

        bm25_results = self._bm25_search(query, k=self.retrieval_k)

        # Fuse results using RRF
        if bm25_results:
            fused = reciprocal_rank_fusion(
                [dense_results_converted, bm25_results],
                k=self.rrf_constant,
                top_n=final_k if not self.reranker else self.retrieval_k,
            )
            documents = [doc for doc, _score in fused]
        else:
            # Fall back to dense-only if BM25 returned nothing
            documents = [doc for doc, _score in dense_results_converted[:final_k]]

        logger.debug(f"Hybrid search: dense={len(dense_results)}, bm25={len(bm25_results)}, " f"fused={len(documents)}")

        return documents

    def _rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """
        Rerank documents using the configured reranker.

        Args:
            query: Original query.
            documents: Documents to rerank.
            top_k: Number of documents to return after reranking.

        Returns:
            Reranked documents.
        """
        if not self.reranker:
            return documents[:top_k]

        try:
            reranked = self.reranker.rerank(query, documents, top_k=top_k)
            logger.debug(f"Reranked {len(documents)} docs to {len(reranked)}")
            return reranked
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original order.")
            return documents[:top_k]

    def build_bm25_index(self, documents: list[Document]) -> None:
        """
        Build and save BM25 index from documents.

        Call this when creating or updating the knowledge base.

        Args:
            documents: Documents to index.
        """
        if self._bm25_index is None:
            self._bm25_index = BM25Index(self.kb_name)

        self._bm25_index.build_index(documents)
        self._bm25_index.save_index()
        logger.info(f"Built BM25 index for '{self.kb_name}'")


def create_hybrid_retriever(
    vectorstore: Chroma,
    kb_name: str,
    retrieval_mode: str | None = None,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    """
    Factory function to create a HybridRetriever with configuration from environment.

    Args:
        vectorstore: ChromaDB vector store.
        kb_name: Knowledge base name.
        retrieval_mode: Override for RETRIEVAL_MODE env var.
        reranker: Optional reranker instance.

    Returns:
        Configured HybridRetriever instance.
    """
    # Read configuration from environment
    mode = retrieval_mode or os.getenv("RETRIEVAL_MODE", "dense")
    retrieval_k = int(os.getenv("RETRIEVAL_K", "10"))
    final_k = int(os.getenv("RETRIEVAL_FINAL_K", "5"))
    rrf_constant = int(os.getenv("RRF_CONSTANT", "60"))

    # Create reranker if configured and not provided
    if reranker is None:
        reranker_provider = os.getenv("RERANKER_PROVIDER", "").strip()
        if reranker_provider:
            try:
                from src.models.reranker import get_reranker

                reranker = get_reranker()
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                reranker = None

    return HybridRetriever(
        vectorstore=vectorstore,
        kb_name=kb_name,
        retrieval_mode=mode,
        reranker=reranker,
        retrieval_k=retrieval_k,
        final_k=final_k,
        rrf_constant=rrf_constant,
    )
