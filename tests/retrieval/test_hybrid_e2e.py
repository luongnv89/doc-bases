"""
End-to-end tests for Hybrid Search with Reranking.

These tests verify the complete flow from document ingestion to retrieval
with hybrid search enabled.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestHybridSearchE2E:
    """End-to-end tests for hybrid search functionality."""

    @pytest.fixture
    def temp_kb_dir(self):
        """Create a temporary knowledge base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledges"
            kb_dir.mkdir()

            # Patch the KNOWLEDGE_BASE_DIR in all relevant modules
            with patch.dict(os.environ, {"KNOWLEDGE_BASE_DIR": str(kb_dir)}):
                import src.retrieval.bm25_index as bm25_module
                import src.utils.rag_utils as rag_utils_module

                original_bm25_dir = bm25_module.KNOWLEDGE_BASE_DIR
                original_rag_dir = rag_utils_module.KNOWLEDGE_BASE_DIR

                bm25_module.KNOWLEDGE_BASE_DIR = str(kb_dir)
                rag_utils_module.KNOWLEDGE_BASE_DIR = str(kb_dir)

                try:
                    yield kb_dir
                finally:
                    bm25_module.KNOWLEDGE_BASE_DIR = original_bm25_dir
                    rag_utils_module.KNOWLEDGE_BASE_DIR = original_rag_dir

    @pytest.fixture
    def sample_documents(self):
        """Create realistic sample documents for testing."""
        return [
            Document(
                page_content="Python is a high-level programming language known for its simplicity and readability. "
                "It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                metadata={"source": "python_intro.md"},
            ),
            Document(
                page_content="JavaScript is a dynamic programming language primarily used for web development. "
                "It runs in browsers and enables interactive web pages. Node.js allows JavaScript to run on servers.",
                metadata={"source": "javascript_intro.md"},
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
                "Python is widely used in machine learning with libraries like TensorFlow, PyTorch, and scikit-learn.",
                metadata={"source": "ml_overview.md"},
            ),
            Document(
                page_content="React is a JavaScript library for building user interfaces. "
                "It uses a component-based architecture and virtual DOM for efficient rendering.",
                metadata={"source": "react_intro.md"},
            ),
            Document(
                page_content="Data science combines statistics, programming, and domain expertise to extract insights from data. "
                "Python and R are popular languages in data science for data analysis and visualization.",
                metadata={"source": "data_science.md"},
            ),
            Document(
                page_content="API stands for Application Programming Interface. REST APIs use HTTP methods like GET, POST, PUT, DELETE. "
                "Python frameworks like Flask and FastAPI make it easy to build REST APIs.",
                metadata={"source": "api_basics.md"},
            ),
            Document(
                page_content="Docker containers package applications with their dependencies for consistent deployment. "
                "Kubernetes orchestrates containers at scale. Both are essential for modern DevOps practices.",
                metadata={"source": "containers.md"},
            ),
            Document(
                page_content="Git is a distributed version control system for tracking changes in source code. "
                "GitHub and GitLab are popular platforms for hosting Git repositories and collaboration.",
                metadata={"source": "git_basics.md"},
            ),
        ]

    def test_bm25_index_creation(self, temp_kb_dir, sample_documents):
        """Test that BM25 index is created when RETRIEVAL_MODE=hybrid."""
        from src.retrieval.bm25_index import BM25Index

        kb_name = "test_hybrid_kb"

        # Create BM25 index
        bm25_index = BM25Index(kb_name)
        bm25_index.build_index(sample_documents)
        bm25_index.save_index()

        # Verify index file exists
        assert bm25_index.index_path.exists()

        # Verify index can be loaded
        bm25_index2 = BM25Index(kb_name)
        loaded = bm25_index2.load_index()
        assert loaded is True
        assert len(bm25_index2.documents) == len(sample_documents)

    def test_bm25_search_finds_keyword_matches(self, temp_kb_dir, sample_documents):
        """Test that BM25 search finds documents with exact keyword matches."""
        from src.retrieval.bm25_index import BM25Index

        kb_name = "test_bm25_search"

        bm25_index = BM25Index(kb_name)
        bm25_index.build_index(sample_documents)

        # Search for specific technical terms
        results = bm25_index.search("Docker containers Kubernetes", k=3)

        # Should find the containers document
        assert len(results) > 0
        contents = [doc.page_content for doc, _ in results]
        assert any("Docker" in c or "Kubernetes" in c for c in contents)

    def test_rrf_fusion_combines_results(self, temp_kb_dir, sample_documents):
        """Test that RRF fusion properly combines BM25 and dense results."""
        from src.retrieval.rrf_fusion import reciprocal_rank_fusion

        # Simulate BM25 results (keyword match for "Python")
        bm25_results = [
            (sample_documents[0], 5.0),  # Python intro - high BM25 score
            (sample_documents[2], 4.5),  # ML (mentions Python)
            (sample_documents[4], 4.0),  # Data science (mentions Python)
        ]

        # Simulate dense results (semantic match for "programming language")
        dense_results = [
            (sample_documents[1], 0.9),  # JavaScript - semantically similar
            (sample_documents[0], 0.85),  # Python intro
            (sample_documents[3], 0.8),  # React
        ]

        # Fuse results
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60, top_n=5)

        # Python intro should be boosted (appears in both lists)
        assert len(fused) > 0
        top_content = fused[0][0].page_content
        assert "Python" in top_content

    def test_hybrid_retriever_dense_mode(self, temp_kb_dir, sample_documents):
        """Test HybridRetriever in dense-only mode."""

        from src.retrieval.hybrid_retriever import HybridRetriever

        # Create mock vectorstore
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.1),
            (sample_documents[1], 0.2),
        ]

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
            final_k=2,
        )

        results = retriever.invoke("Python programming")

        assert len(results) == 2
        mock_vectorstore.similarity_search_with_score.assert_called_once()

    def test_hybrid_retriever_hybrid_mode(self, temp_kb_dir, sample_documents):
        """Test HybridRetriever in hybrid mode with BM25 + dense search."""

        from src.retrieval.bm25_index import BM25Index
        from src.retrieval.hybrid_retriever import HybridRetriever

        kb_name = "test_hybrid_mode"

        # Build BM25 index first
        bm25_index = BM25Index(kb_name)
        bm25_index.build_index(sample_documents)
        bm25_index.save_index()

        # Create mock vectorstore
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (sample_documents[1], 0.1),  # JavaScript
            (sample_documents[3], 0.2),  # React
        ]

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name=kb_name,
            retrieval_mode="hybrid",
            retrieval_k=5,
            final_k=3,
        )

        # Search should combine BM25 and dense results
        results = retriever.invoke("Python machine learning")

        assert len(results) <= 3
        # Both retrievers should be called
        mock_vectorstore.similarity_search_with_score.assert_called()

    def test_hybrid_retriever_with_reranker(self, temp_kb_dir, sample_documents):
        """Test HybridRetriever with a mock reranker."""

        from src.models.reranker import PassthroughReranker
        from src.retrieval.hybrid_retriever import HybridRetriever

        # Create mock vectorstore
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.1),
            (sample_documents[1], 0.2),
            (sample_documents[2], 0.3),
        ]

        # Use passthrough reranker (just truncates to top_k)
        reranker = PassthroughReranker()

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
            reranker=reranker,
            final_k=2,
        )

        results = retriever.invoke("test query")

        # Reranker should limit to final_k
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_hybrid_retriever_async(self, temp_kb_dir, sample_documents):
        """Test async retrieval with HybridRetriever."""

        from src.retrieval.hybrid_retriever import HybridRetriever

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.1),
        ]

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
        )

        results = await retriever.ainvoke("async test query")

        assert len(results) > 0

    def test_create_hybrid_retriever_factory(self, temp_kb_dir):
        """Test the create_hybrid_retriever factory function."""

        from src.retrieval.hybrid_retriever import create_hybrid_retriever

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = []

        # Test with default config
        with patch.dict(os.environ, {"RETRIEVAL_MODE": "dense"}, clear=False):
            retriever = create_hybrid_retriever(
                vectorstore=mock_vectorstore,
                kb_name="test_kb",
            )

            assert retriever.retrieval_mode == "dense"
            assert retriever.retrieval_k == 10
            assert retriever.final_k == 5

    def test_langchain_retriever_interface(self, temp_kb_dir, sample_documents):
        """Test that HybridRetriever works with LangChain retriever interface."""

        from src.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverWrapper

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.1),
        ]

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            kb_name="test_kb",
            retrieval_mode="dense",
        )

        # Get LangChain-compatible retriever
        lc_retriever = retriever.as_retriever(search_kwargs={"k": 3})

        assert isinstance(lc_retriever, HybridRetrieverWrapper)

        # Test retrieval through the wrapper
        results = lc_retriever._get_relevant_documents("test query")
        assert isinstance(results, list)


class TestHybridSearchWithRealEmbeddings:
    """
    Tests that use real embeddings (requires EMB_PROVIDER to be configured).

    These tests are more realistic but require an embedding provider.
    """

    @pytest.fixture
    def temp_kb_dir(self):
        """Create a temporary knowledge base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "knowledges"
            kb_dir.mkdir()

            import src.retrieval.bm25_index as bm25_module
            import src.utils.rag_utils as rag_utils_module

            original_bm25_dir = bm25_module.KNOWLEDGE_BASE_DIR
            original_rag_dir = rag_utils_module.KNOWLEDGE_BASE_DIR

            bm25_module.KNOWLEDGE_BASE_DIR = str(kb_dir)
            rag_utils_module.KNOWLEDGE_BASE_DIR = str(kb_dir)

            try:
                yield kb_dir
            finally:
                bm25_module.KNOWLEDGE_BASE_DIR = original_bm25_dir
                rag_utils_module.KNOWLEDGE_BASE_DIR = original_rag_dir

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            Document(page_content="Python is a programming language"),
            Document(page_content="JavaScript is used for web development"),
            Document(page_content="Machine learning uses Python"),
            Document(page_content="React is a JavaScript framework"),
        ]

    @pytest.mark.skipif(not os.getenv("EMB_PROVIDER"), reason="Requires EMB_PROVIDER to be configured")
    def test_full_vector_store_with_hybrid_search(self, temp_kb_dir, sample_documents):
        """Test creating a real vector store with hybrid search enabled."""
        from src.retrieval.bm25_index import BM25Index
        from src.utils.rag_utils import create_vector_store

        kb_name = "test_real_hybrid"

        with patch.dict(os.environ, {"RETRIEVAL_MODE": "hybrid"}):
            # Create vector store (should also create BM25 index)
            vectorstore = create_vector_store(sample_documents, kb_name)

            assert vectorstore is not None

            # Verify BM25 index was created
            bm25_index = BM25Index(kb_name)
            loaded = bm25_index.load_index()
            assert loaded is True
            assert len(bm25_index.documents) == len(sample_documents)


class TestConfigurationIntegration:
    """Tests for configuration integration."""

    def test_config_manager_has_retrieval_settings(self):
        """Test that ConfigManager includes retrieval settings."""
        from src.cli.config_manager import ConfigManager

        config = ConfigManager.DEFAULT_CONFIG

        assert "retrieval" in config
        assert config["retrieval"]["mode"] == "dense"
        assert config["retrieval"]["k"] == 10
        assert config["retrieval"]["final_k"] == 5
        assert config["retrieval"]["rrf_constant"] == 60

        assert "reranker" in config
        assert config["reranker"]["provider"] is None
        assert "ms-marco" in config["reranker"]["model"]

    def test_env_var_mapping(self):
        """Test that environment variable mappings are correct."""
        from src.cli.config_manager import ConfigManager

        mapping = ConfigManager.ENV_VAR_MAPPING

        assert mapping["retrieval.mode"] == "RETRIEVAL_MODE"
        assert mapping["retrieval.k"] == "RETRIEVAL_K"
        assert mapping["retrieval.final_k"] == "RETRIEVAL_FINAL_K"
        assert mapping["retrieval.rrf_constant"] == "RRF_CONSTANT"
        assert mapping["reranker.provider"] == "RERANKER_PROVIDER"
        assert mapping["reranker.model"] == "RERANKER_MODEL"

    def test_config_validation_accepts_valid_retrieval_mode(self):
        """Test that config validation accepts valid retrieval modes."""
        from src.cli.config_manager import ConfigManager

        with patch.dict(os.environ, {"RETRIEVAL_MODE": "hybrid"}, clear=False):
            manager = ConfigManager()
            manager.set("retrieval.mode", "hybrid")
            errors = manager.validate()

            # Should not have retrieval mode error
            retrieval_errors = [e for e in errors if "retrieval.mode" in e]
            assert len(retrieval_errors) == 0

    def test_config_validation_rejects_invalid_retrieval_mode(self):
        """Test that config validation rejects invalid retrieval modes."""
        from src.cli.config_manager import ConfigManager

        manager = ConfigManager()
        manager.set("retrieval.mode", "invalid_mode")
        errors = manager.validate()

        # Should have retrieval mode error
        retrieval_errors = [e for e in errors if "retrieval.mode" in e]
        assert len(retrieval_errors) == 1
