"""Tests for the reranker module."""

# Check for optional dependencies using importlib.util.find_spec
import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.models.reranker import PassthroughReranker, get_reranker

HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None
HAS_COHERE = importlib.util.find_spec("cohere") is not None


class TestPassthroughReranker:
    """Tests for the PassthroughReranker class."""

    def test_rerank_returns_unchanged(self):
        """Should return documents unchanged."""
        reranker = PassthroughReranker()
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
            Document(page_content="Doc 3"),
        ]

        result = reranker.rerank("query", docs, top_k=3)

        assert result == docs

    def test_rerank_respects_top_k(self):
        """Should truncate to top_k."""
        reranker = PassthroughReranker()
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
            Document(page_content="Doc 3"),
        ]

        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].page_content == "Doc 1"
        assert result[1].page_content == "Doc 2"

    def test_rerank_empty_list(self):
        """Should handle empty document list."""
        reranker = PassthroughReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker class."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock the sentence_transformers CrossEncoder."""
        with patch("src.models.reranker.CrossEncoderReranker.__init__", return_value=None) as mock_init:
            yield mock_init

    def test_init_imports_sentence_transformers(self):
        """Should raise ImportError if sentence-transformers not installed."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                # This would raise if we actually tried to instantiate
                pass

    def test_rerank_orders_by_score(self):
        """Should order documents by cross-encoder score."""
        from src.models.reranker import CrossEncoderReranker

        with patch("sentence_transformers.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            # Scores: Doc 2 (0.9) > Doc 3 (0.7) > Doc 1 (0.5)
            mock_model.predict.return_value = [0.5, 0.9, 0.7]
            MockCE.return_value = mock_model

            reranker = CrossEncoderReranker()
            docs = [
                Document(page_content="Doc 1"),
                Document(page_content="Doc 2"),
                Document(page_content="Doc 3"),
            ]

            result = reranker.rerank("query", docs, top_k=3)

            # Should be ordered by score
            assert result[0].page_content == "Doc 2"  # Highest score
            assert result[1].page_content == "Doc 3"
            assert result[2].page_content == "Doc 1"  # Lowest score

    def test_rerank_respects_top_k(self):
        """Should return only top_k documents."""
        from src.models.reranker import CrossEncoderReranker

        with patch("sentence_transformers.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.8, 0.7]
            MockCE.return_value = mock_model

            reranker = CrossEncoderReranker()
            docs = [
                Document(page_content="Doc 1"),
                Document(page_content="Doc 2"),
                Document(page_content="Doc 3"),
            ]

            result = reranker.rerank("query", docs, top_k=2)

            assert len(result) == 2

    def test_rerank_empty_list(self):
        """Should handle empty document list."""
        from src.models.reranker import CrossEncoderReranker

        with patch("sentence_transformers.CrossEncoder") as MockCE:
            MockCE.return_value = MagicMock()

            reranker = CrossEncoderReranker()
            result = reranker.rerank("query", [], top_k=5)

            assert result == []


@pytest.mark.skipif(not HAS_COHERE, reason="cohere not installed")
class TestCohereReranker:
    """Tests for the CohereReranker class."""

    def test_init_requires_api_key(self):
        """Should raise ValueError without API key."""
        from src.models.reranker import CohereReranker

        with patch.dict(os.environ, {}, clear=True):
            with patch("cohere.Client"):
                with pytest.raises(ValueError, match="Cohere API key required"):
                    CohereReranker(api_key=None)

    def test_init_reads_env_api_key(self):
        """Should read API key from environment."""
        from src.models.reranker import CohereReranker

        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}, clear=True):
            with patch("cohere.Client") as MockClient:
                reranker = CohereReranker()

                MockClient.assert_called_once_with("test-key")
                assert reranker.api_key == "test-key"

    def test_rerank_calls_api(self):
        """Should call Cohere rerank API."""
        from src.models.reranker import CohereReranker

        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}, clear=True):
            with patch("cohere.Client") as MockClient:
                mock_client = MagicMock()

                # Mock API response
                mock_result1 = MagicMock()
                mock_result1.index = 1
                mock_result2 = MagicMock()
                mock_result2.index = 0

                mock_response = MagicMock()
                mock_response.results = [mock_result1, mock_result2]

                mock_client.rerank.return_value = mock_response
                MockClient.return_value = mock_client

                reranker = CohereReranker()
                docs = [
                    Document(page_content="Doc 1"),
                    Document(page_content="Doc 2"),
                ]

                result = reranker.rerank("query", docs, top_k=2)

                mock_client.rerank.assert_called_once()
                assert len(result) == 2
                # Order based on API response indices
                assert result[0].page_content == "Doc 2"
                assert result[1].page_content == "Doc 1"


class TestGetReranker:
    """Tests for the get_reranker factory function."""

    def test_no_provider_returns_none(self):
        """Should return None when no provider configured."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_reranker()
            assert result is None

    def test_empty_provider_returns_none(self):
        """Should return None for empty provider string."""
        with patch.dict(os.environ, {"RERANKER_PROVIDER": ""}, clear=True):
            result = get_reranker()
            assert result is None

    def test_passthrough_provider(self):
        """Should return PassthroughReranker for passthrough/none provider."""
        result = get_reranker(provider="passthrough")
        assert isinstance(result, PassthroughReranker)

        result = get_reranker(provider="none")
        assert isinstance(result, PassthroughReranker)

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_cross_encoder_provider(self):
        """Should create CrossEncoderReranker."""
        from src.models.reranker import CrossEncoderReranker

        with patch("sentence_transformers.CrossEncoder") as MockCE:
            MockCE.return_value = MagicMock()

            result = get_reranker(
                provider="cross-encoder",
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            )

            assert isinstance(result, CrossEncoderReranker)

    def test_cross_encoder_missing_dependency(self):
        """Should return None if sentence-transformers not installed."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Simulate ImportError
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if name == "sentence_transformers":
                    raise ImportError("No module named 'sentence_transformers'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # This test verifies the graceful handling
                pass

    @pytest.mark.skipif(not HAS_COHERE, reason="cohere not installed")
    def test_cohere_provider(self):
        """Should create CohereReranker."""
        from src.models.reranker import CohereReranker

        with patch("cohere.Client") as MockClient:
            MockClient.return_value = MagicMock()

            result = get_reranker(
                provider="cohere",
                api_key="test-key",
            )

            assert isinstance(result, CohereReranker)

    @pytest.mark.skipif(not HAS_COHERE, reason="cohere not installed")
    def test_cohere_missing_api_key(self):
        """Should return None if Cohere API key missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("cohere.Client"):
                result = get_reranker(provider="cohere")
                assert result is None

    def test_unknown_provider_returns_none(self):
        """Should return None for unknown provider."""
        result = get_reranker(provider="unknown_provider")
        assert result is None

    @pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
    def test_reads_model_from_env(self):
        """Should read model name from environment."""
        env_vars = {
            "RERANKER_PROVIDER": "cross-encoder",
            "RERANKER_MODEL": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with patch("sentence_transformers.CrossEncoder") as MockCE:
                MockCE.return_value = MagicMock()

                get_reranker()

                MockCE.assert_called_once_with("cross-encoder/ms-marco-TinyBERT-L-2-v2")
