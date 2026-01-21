"""Tests for Hybrid Retriever."""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverWrapper, create_hybrid_retriever


class TestHybridRetriever:
    """Tests for the HybridRetriever class."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create a mock vectorstore."""
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="Dense result 1"), 0.1),
            (Document(page_content="Dense result 2"), 0.2),
            (Document(page_content="Dense result 3"), 0.3),
        ]
        return vectorstore

    @pytest.fixture
    def mock_bm25_index(self):
        """Create a mock BM25 index."""
        with patch("src.retrieval.hybrid_retriever.BM25Index") as MockBM25:
            index = MagicMock()
            index.is_loaded = True
            index.load_index.return_value = True
            index.search.return_value = [
                (Document(page_content="BM25 result 1"), 5.0),
                (Document(page_content="BM25 result 2"), 4.0),
            ]
            MockBM25.return_value = index
            yield MockBM25, index

    def test_init_dense_mode(self, mock_vectorstore):
        """Should initialize in dense mode."""
        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
        )

        assert retriever.retrieval_mode == "dense"
        assert retriever._bm25_index is None

    def test_init_hybrid_mode(self, mock_vectorstore, mock_bm25_index):
        """Should initialize BM25 index in hybrid mode."""
        MockBM25, index = mock_bm25_index

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="hybrid",
        )

        assert retriever.retrieval_mode == "hybrid"
        assert retriever._bm25_index is not None
        MockBM25.assert_called_once_with("test_kb")
        index.load_index.assert_called_once()

    def test_invoke_dense_mode(self, mock_vectorstore):
        """Should use dense search in dense mode."""
        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
            final_k=3,
        )

        docs = retriever.invoke("test query")

        assert len(docs) == 3
        mock_vectorstore.similarity_search_with_score.assert_called_once()

    def test_invoke_hybrid_mode(self, mock_vectorstore, mock_bm25_index):
        """Should combine BM25 and dense search in hybrid mode."""
        MockBM25, index = mock_bm25_index

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="hybrid",
            retrieval_k=5,
            final_k=3,
        )

        docs = retriever.invoke("test query")

        # Both retrievers should be called
        mock_vectorstore.similarity_search_with_score.assert_called()
        index.search.assert_called()

        # Should return fused results
        assert len(docs) <= 3

    def test_invoke_with_k_override(self, mock_vectorstore):
        """Should respect k override in invoke."""
        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
            final_k=5,
        )

        retriever.invoke("test query", k=2)

        # Should pass k=2 to vectorstore despite final_k=5
        mock_vectorstore.similarity_search_with_score.assert_called_with("test query", k=2)

    @pytest.mark.asyncio
    async def test_ainvoke(self, mock_vectorstore):
        """Should support async invocation."""
        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
        )

        docs = await retriever.ainvoke("test query")

        assert len(docs) > 0

    def test_as_retriever(self, mock_vectorstore):
        """Should return LangChain-compatible retriever."""
        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
        )

        lc_retriever = retriever.as_retriever(search_kwargs={"k": 3})

        assert isinstance(lc_retriever, HybridRetrieverWrapper)

    def test_with_reranker(self, mock_vectorstore):
        """Should apply reranker when provided."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            Document(page_content="Reranked result 1"),
            Document(page_content="Reranked result 2"),
        ]

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
            reranker=mock_reranker,
            final_k=2,
        )

        docs = retriever.invoke("test query")

        mock_reranker.rerank.assert_called_once()
        assert len(docs) == 2

    def test_reranker_error_fallback(self, mock_vectorstore):
        """Should fallback to original order if reranker fails."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = Exception("Rerank failed")

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
            reranker=mock_reranker,
            final_k=2,
        )

        # Should not raise, should return original results
        docs = retriever.invoke("test query")
        assert len(docs) > 0

    def test_hybrid_fallback_when_bm25_empty(self, mock_vectorstore, mock_bm25_index):
        """Should fallback to dense-only when BM25 returns nothing."""
        MockBM25, index = mock_bm25_index
        index.search.return_value = []  # Empty BM25 results

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="hybrid",
        )

        docs = retriever.invoke("test query")

        # Should still return dense results
        assert len(docs) > 0

    def test_build_bm25_index(self, mock_vectorstore, mock_bm25_index):
        """Should build and save BM25 index."""
        MockBM25, index = mock_bm25_index

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
        )

        docs = [Document(page_content="Test doc")]
        retriever.build_bm25_index(docs)

        index.build_index.assert_called_once_with(docs)
        index.save_index.assert_called_once()


class TestHybridRetrieverWrapper:
    """Tests for the HybridRetrieverWrapper class."""

    @pytest.fixture
    def mock_hybrid_retriever(self):
        """Create a mock hybrid retriever."""
        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="Result 1"),
            Document(page_content="Result 2"),
        ]
        return retriever

    def test_get_relevant_documents(self, mock_hybrid_retriever):
        """Should call hybrid retriever invoke."""
        wrapper = HybridRetrieverWrapper(
            hybrid_retriever=mock_hybrid_retriever,
            search_kwargs={"k": 3},
        )

        docs = wrapper._get_relevant_documents("test query")

        mock_hybrid_retriever.invoke.assert_called_once_with("test query", k=3)
        assert len(docs) == 2


class TestCreateHybridRetriever:
    """Tests for the create_hybrid_retriever factory function."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create a mock vectorstore."""
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_score.return_value = []
        return vectorstore

    def test_default_config(self, mock_vectorstore):
        """Should use default configuration."""
        with patch.dict(os.environ, {}, clear=True):
            retriever = create_hybrid_retriever(
                vectorstore=mock_vectorstore,
                kb_name="test_kb",
            )

        assert retriever.retrieval_mode == "dense"
        assert retriever.retrieval_k == 10
        assert retriever.final_k == 5
        assert retriever.rrf_constant == 60

    def test_env_config(self, mock_vectorstore):
        """Should read configuration from environment."""
        env_vars = {
            "RETRIEVAL_MODE": "hybrid",
            "RETRIEVAL_K": "15",
            "RETRIEVAL_FINAL_K": "8",
            "RRF_CONSTANT": "100",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch("src.retrieval.hybrid_retriever.BM25Index") as MockBM25:
                mock_index = MagicMock()
                mock_index.load_index.return_value = False
                MockBM25.return_value = mock_index

                retriever = create_hybrid_retriever(
                    vectorstore=mock_vectorstore,
                    kb_name="test_kb",
                )

        assert retriever.retrieval_mode == "hybrid"
        assert retriever.retrieval_k == 15
        assert retriever.final_k == 8
        assert retriever.rrf_constant == 100

    def test_override_retrieval_mode(self, mock_vectorstore):
        """Should allow overriding retrieval mode."""
        with patch.dict(os.environ, {"RETRIEVAL_MODE": "hybrid"}, clear=True):
            retriever = create_hybrid_retriever(
                vectorstore=mock_vectorstore,
                kb_name="test_kb",
                retrieval_mode="dense",  # Override
            )

        assert retriever.retrieval_mode == "dense"

    def test_creates_reranker_from_env(self, mock_vectorstore):
        """Should create reranker when configured in environment."""
        env_vars = {
            "RERANKER_PROVIDER": "cross-encoder",
            "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Patch at the source module where get_reranker is defined
            with patch("src.models.reranker.get_reranker") as mock_get_reranker:
                mock_reranker = MagicMock()
                mock_get_reranker.return_value = mock_reranker

                retriever = create_hybrid_retriever(
                    vectorstore=mock_vectorstore,
                    kb_name="test_kb",
                )

        mock_get_reranker.assert_called_once()
        assert retriever.reranker == mock_reranker
