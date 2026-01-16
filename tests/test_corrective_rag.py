"""
Tests for Phase 3: Corrective RAG implementation.
"""

import os
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.evaluation.rag_evaluator import HallucinationCheck, RAGEvaluator, RelevanceScore
from src.graphs.corrective_rag import CorrectiveRAGGraph, CRAGState
from src.tools.web_search import DUCKDUCKGO_AVAILABLE, is_web_search_available, web_search_to_documents
from src.utils.rag_utils import get_rag_mode


class TestRelevanceScore:
    """Test RelevanceScore model."""

    def test_relevance_score_creation(self):
        """RelevanceScore should be created with valid data."""
        score = RelevanceScore(score=0.8, reasoning="Document is relevant")
        assert score.score == 0.8
        assert score.reasoning == "Document is relevant"

    def test_relevance_score_validation(self):
        """RelevanceScore should accept scores between 0 and 1."""
        score = RelevanceScore(score=0.0, reasoning="Not relevant")
        assert score.score == 0.0

        score = RelevanceScore(score=1.0, reasoning="Highly relevant")
        assert score.score == 1.0


class TestHallucinationCheck:
    """Test HallucinationCheck model."""

    def test_hallucination_check_grounded(self):
        """HallucinationCheck should indicate grounded answer."""
        check = HallucinationCheck(is_grounded=True, unsupported_claims=[])
        assert check.is_grounded is True
        assert check.unsupported_claims == []

    def test_hallucination_check_not_grounded(self):
        """HallucinationCheck should indicate ungrounded answer with claims."""
        check = HallucinationCheck(is_grounded=False, unsupported_claims=["Claim 1", "Claim 2"])
        assert check.is_grounded is False
        assert len(check.unsupported_claims) == 2


class TestRAGEvaluator:
    """Test RAGEvaluator class."""

    def test_evaluator_initialization(self):
        """RAGEvaluator should initialize with an LLM."""
        mock_llm = MagicMock()
        evaluator = RAGEvaluator(mock_llm)
        assert evaluator.llm == mock_llm
        assert evaluator.relevance_parser is not None
        assert evaluator.hallucination_parser is not None

    def test_evaluator_has_prompts(self):
        """RAGEvaluator should have relevance and hallucination prompts."""
        mock_llm = MagicMock()
        evaluator = RAGEvaluator(mock_llm)

        # Verify prompts are configured
        assert evaluator.relevance_prompt is not None
        assert evaluator.hallucination_prompt is not None

        # Verify parsers are configured
        assert evaluator.relevance_parser is not None
        assert evaluator.hallucination_parser is not None


class TestWebSearch:
    """Test web search functionality."""

    def test_is_web_search_available_returns_bool(self):
        """is_web_search_available should return boolean."""
        result = is_web_search_available()
        assert isinstance(result, bool)

    def test_web_search_availability_matches_import(self):
        """Availability should match import status."""
        assert is_web_search_available() == DUCKDUCKGO_AVAILABLE

    @patch("src.tools.web_search.DUCKDUCKGO_AVAILABLE", False)
    def test_web_search_unavailable_message(self):
        """web_search should return message when unavailable."""
        # Need to reimport to get the patched value
        from src.tools.web_search import web_search as ws

        result = ws.invoke("test query")
        assert "unavailable" in result.lower() or "install" in result.lower()

    @patch("src.tools.web_search.DUCKDUCKGO_AVAILABLE", False)
    def test_web_search_to_documents_empty_when_unavailable(self):
        """web_search_to_documents should return empty list when unavailable."""
        result = web_search_to_documents("test query")
        assert result == []


class TestCorrectiveRAGGraph:
    """Test CorrectiveRAGGraph class."""

    def test_crag_state_structure(self):
        """CRAGState should have correct structure."""
        state: CRAGState = {
            "messages": [],
            "question": "test question",
            "documents": [],
            "relevant_docs": [],
            "web_search_needed": False,
            "web_results": [],
            "generation": "",
            "is_grounded": True,
        }
        assert "question" in state
        assert "documents" in state
        assert "web_search_needed" in state

    def test_corrective_rag_initialization(self):
        """CorrectiveRAGGraph should initialize with vectorstore and llm."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = CorrectiveRAGGraph(vectorstore=mock_vectorstore, llm=mock_llm, relevance_threshold=0.6, min_relevant_docs=2)

        assert graph.vectorstore == mock_vectorstore
        assert graph.llm == mock_llm
        assert graph.relevance_threshold == 0.6
        assert graph.min_relevant_docs == 2

    def test_decide_retrieval_quality_needs_web(self):
        """decide_retrieval_quality should return 'web_search' when needed."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = CorrectiveRAGGraph(mock_vectorstore, mock_llm)

        state: CRAGState = {
            "messages": [],
            "question": "test",
            "documents": [],
            "relevant_docs": [],
            "web_search_needed": True,
            "web_results": [],
            "generation": "",
            "is_grounded": True,
        }

        result = graph.decide_retrieval_quality(state)
        assert result == "web_search"

    def test_decide_retrieval_quality_generate(self):
        """decide_retrieval_quality should return 'generate' when docs sufficient."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = CorrectiveRAGGraph(mock_vectorstore, mock_llm)

        state: CRAGState = {
            "messages": [],
            "question": "test",
            "documents": [],
            "relevant_docs": [Document(page_content="relevant doc")],
            "web_search_needed": False,
            "web_results": [],
            "generation": "",
            "is_grounded": True,
        }

        result = graph.decide_retrieval_quality(state)
        assert result == "generate"


class TestGetRagMode:
    """Test get_rag_mode function."""

    def test_default_mode(self):
        """Default RAG mode should be 'basic'."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RAG_MODE", None)
            result = get_rag_mode()
            assert result == "basic"

    def test_corrective_mode(self):
        """Should return 'corrective' when configured."""
        with patch.dict(os.environ, {"RAG_MODE": "corrective"}):
            result = get_rag_mode()
            assert result == "corrective"

    def test_adaptive_mode(self):
        """Should return 'adaptive' when configured."""
        with patch.dict(os.environ, {"RAG_MODE": "adaptive"}):
            result = get_rag_mode()
            assert result == "adaptive"

    def test_mode_case_insensitive(self):
        """RAG mode should be case-insensitive."""
        with patch.dict(os.environ, {"RAG_MODE": "CORRECTIVE"}):
            result = get_rag_mode()
            assert result == "corrective"
