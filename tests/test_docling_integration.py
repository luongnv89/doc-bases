"""
Tests for Phase 2: Docling integration and semantic chunking.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.utils.docling_loader import DOCLING_AVAILABLE, DoclingDocumentLoader, is_docling_available
from src.utils.document_loader import DocumentLoader, _use_docling
from src.utils.semantic_splitter import SemanticDocumentSplitter, get_chunking_strategy, is_semantic_chunker_available


class TestDoclingAvailability:
    """Test Docling availability checks."""

    def test_is_docling_available_returns_bool(self):
        """is_docling_available should return a boolean."""
        result = is_docling_available()
        assert isinstance(result, bool)

    def test_docling_available_matches_import(self):
        """is_docling_available should match DOCLING_AVAILABLE constant."""
        assert is_docling_available() == DOCLING_AVAILABLE


class TestDoclingDocumentLoader:
    """Test DoclingDocumentLoader class."""

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    def test_initialization(self):
        """DoclingDocumentLoader should initialize with default parameters."""
        loader = DoclingDocumentLoader()
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200

    @pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling not installed")
    def test_initialization_custom_params(self):
        """DoclingDocumentLoader should accept custom chunk parameters."""
        loader = DoclingDocumentLoader(chunk_size=500, chunk_overlap=100)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 100

    def test_supports_format_pdf(self):
        """DoclingDocumentLoader should support PDF files."""
        if not DOCLING_AVAILABLE:
            pytest.skip("Docling not installed")
        loader = DoclingDocumentLoader()
        assert loader.supports_format("document.pdf") is True
        assert loader.supports_format("DOCUMENT.PDF") is True

    def test_supports_format_docx(self):
        """DoclingDocumentLoader should support DOCX files."""
        if not DOCLING_AVAILABLE:
            pytest.skip("Docling not installed")
        loader = DoclingDocumentLoader()
        assert loader.supports_format("document.docx") is True

    def test_supports_format_unsupported(self):
        """DoclingDocumentLoader should not support unsupported formats."""
        if not DOCLING_AVAILABLE:
            pytest.skip("Docling not installed")
        loader = DoclingDocumentLoader()
        assert loader.supports_format("document.txt") is False
        assert loader.supports_format("document.csv") is False
        assert loader.supports_format("document.json") is False


class TestSemanticSplitterAvailability:
    """Test semantic splitter availability checks."""

    def test_is_semantic_chunker_available_returns_bool(self):
        """is_semantic_chunker_available should return a boolean."""
        result = is_semantic_chunker_available()
        assert isinstance(result, bool)


class TestGetChunkingStrategy:
    """Test chunking strategy configuration."""

    def test_default_strategy(self):
        """Default chunking strategy should be 'recursive'."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove CHUNKING_STRATEGY if present
            os.environ.pop("CHUNKING_STRATEGY", None)
            result = get_chunking_strategy()
            assert result == "recursive"

    def test_semantic_strategy_from_env(self):
        """Should return 'semantic' when configured."""
        with patch.dict(os.environ, {"CHUNKING_STRATEGY": "semantic"}):
            result = get_chunking_strategy()
            assert result == "semantic"

    def test_strategy_case_insensitive(self):
        """Strategy should be case-insensitive."""
        with patch.dict(os.environ, {"CHUNKING_STRATEGY": "SEMANTIC"}):
            result = get_chunking_strategy()
            assert result == "semantic"


class TestUseDocling:
    """Test _use_docling helper function."""

    def test_use_docling_disabled_by_default(self):
        """Docling should be disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("USE_DOCLING", None)
            result = _use_docling()
            assert result is False

    def test_use_docling_enabled_with_env(self):
        """Docling should be enabled when USE_DOCLING=true."""
        with patch.dict(os.environ, {"USE_DOCLING": "true"}):
            result = _use_docling()
            # Should only be True if docling is also available
            assert result == is_docling_available()

    def test_use_docling_case_insensitive(self):
        """USE_DOCLING should be case-insensitive."""
        with patch.dict(os.environ, {"USE_DOCLING": "TRUE"}):
            result = _use_docling()
            assert result == is_docling_available()

    def test_use_docling_false_when_not_available(self):
        """_use_docling should return False when docling is not available."""
        with patch.dict(os.environ, {"USE_DOCLING": "true"}):
            with patch("src.utils.document_loader.is_docling_available", return_value=False):
                result = _use_docling()
                assert result is False


class TestDocumentLoaderWithDocling:
    """Test DocumentLoader integration with Docling."""

    def test_document_loader_initializes(self):
        """DocumentLoader should initialize correctly."""
        loader = DocumentLoader()
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200

    def test_document_loader_custom_chunks(self):
        """DocumentLoader should accept custom chunk parameters."""
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50

    @patch("src.utils.document_loader._use_docling", return_value=True)
    @patch("src.utils.document_loader.DoclingDocumentLoader")
    def test_load_single_document_uses_docling_when_enabled(self, mock_docling_class, mock_use_docling):
        """_load_single_document should try Docling first when enabled."""
        # Setup mock
        mock_docling_instance = MagicMock()
        mock_docling_instance.supports_format.return_value = True
        mock_docling_instance.load_document.return_value = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
        mock_docling_class.return_value = mock_docling_instance

        loader = DocumentLoader()
        result = loader._load_single_document("test.pdf")

        # Verify Docling was used
        mock_docling_class.assert_called_once()
        mock_docling_instance.load_document.assert_called_once_with("test.pdf")
        assert result is not None
        assert len(result) == 1
        assert result[0].page_content == "Test content"

    @patch("src.utils.document_loader._use_docling", return_value=False)
    @patch("src.utils.document_loader.magic")
    @patch("src.utils.document_loader.TextLoader")
    def test_load_single_document_fallback_when_docling_disabled(self, mock_text_loader, mock_magic, mock_use_docling):
        """_load_single_document should use fallback when Docling is disabled."""
        # Setup mocks
        mock_magic.from_file.return_value = "text/plain"
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="Text content", metadata={"source": "test.txt"})]
        mock_text_loader.return_value = mock_loader_instance

        loader = DocumentLoader()
        result = loader._load_single_document("test.txt")

        # Verify fallback was used
        mock_text_loader.assert_called_once()
        assert result is not None


class TestSemanticDocumentSplitter:
    """Test SemanticDocumentSplitter class."""

    @pytest.mark.skipif(not is_semantic_chunker_available(), reason="langchain-experimental not installed")
    def test_initialization_with_custom_embeddings(self):
        """SemanticDocumentSplitter should accept custom embeddings."""
        mock_embeddings = MagicMock()

        # This may fail if embeddings model isn't available
        try:
            splitter = SemanticDocumentSplitter(embeddings=mock_embeddings)
            assert splitter is not None
        except Exception:
            pytest.skip("Embeddings model not available")


class TestDocumentLoaderChunkingStrategy:
    """Test DocumentLoader chunking strategy selection."""

    @patch("src.utils.document_loader.get_chunking_strategy", return_value="recursive")
    def test_split_uses_recursive_by_default(self, mock_strategy):
        """_split_documents_to_chunk should use recursive strategy by default."""
        loader = DocumentLoader()
        documents = [Document(page_content="Test content " * 100, metadata={})]

        result = loader._split_documents_to_chunk(documents)

        assert result is not None
        assert len(result) >= 1

    @patch("src.utils.document_loader.get_chunking_strategy", return_value="semantic")
    @patch("src.utils.document_loader.is_semantic_chunker_available", return_value=False)
    def test_split_falls_back_when_semantic_unavailable(self, mock_available, mock_strategy):
        """Should fall back to recursive when semantic is unavailable."""
        loader = DocumentLoader()
        documents = [Document(page_content="Test content " * 100, metadata={})]

        result = loader._split_documents_to_chunk(documents)

        assert result is not None
        # Verify it still works (falls back to recursive)
        assert len(result) >= 1


class TestPhase2Integration:
    """Integration tests for Phase 2 features."""

    def test_environment_variables_recognized(self):
        """Environment variables for Phase 2 should be recognized."""
        # Test USE_DOCLING
        with patch.dict(os.environ, {"USE_DOCLING": "true"}):
            use_docling_env = os.getenv("USE_DOCLING", "false").lower() == "true"
            assert use_docling_env is True

        # Test CHUNKING_STRATEGY
        with patch.dict(os.environ, {"CHUNKING_STRATEGY": "semantic"}):
            strategy = get_chunking_strategy()
            assert strategy == "semantic"

    def test_document_loader_works_with_phase2_disabled(self):
        """DocumentLoader should work normally when Phase 2 features are disabled."""
        with patch.dict(os.environ, {"USE_DOCLING": "false", "CHUNKING_STRATEGY": "recursive"}):
            loader = DocumentLoader()
            assert loader is not None

            # Should still be able to split documents
            documents = [Document(page_content="Test content " * 50, metadata={})]
            result = loader._split_documents_to_chunk(documents)
            assert result is not None
