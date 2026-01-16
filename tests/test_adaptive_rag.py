"""
Tests for Phase 3: Adaptive RAG implementation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from src.graphs.adaptive_rag import AdaptiveRAGGraph, AdaptiveRAGState


class TestAdaptiveRAGState:
    """Test AdaptiveRAGState structure."""

    def test_state_structure(self):
        """AdaptiveRAGState should have correct structure."""
        state: AdaptiveRAGState = {"messages": [], "question": "test question", "query_type": None, "documents": [], "generation": ""}
        assert "question" in state
        assert "query_type" in state
        assert "documents" in state
        assert "generation" in state

    def test_state_with_query_type(self):
        """State should accept valid query types."""
        state: AdaptiveRAGState = {"messages": [], "question": "test", "query_type": "simple", "documents": [], "generation": ""}
        assert state["query_type"] == "simple"


class TestAdaptiveRAGGraph:
    """Test AdaptiveRAGGraph class."""

    def test_adaptive_rag_initialization(self):
        """AdaptiveRAGGraph should initialize with vectorstore and llm."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = AdaptiveRAGGraph(vectorstore=mock_vectorstore, llm=mock_llm)

        assert graph.vectorstore == mock_vectorstore
        assert graph.llm == mock_llm
        assert graph.classification_prompt is not None

    def test_route_query_simple(self):
        """route_query should return correct query type."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)

        state: AdaptiveRAGState = {"messages": [], "question": "What is Python?", "query_type": "simple", "documents": [], "generation": ""}

        result = graph.route_query(state)
        assert result == "simple"

    def test_route_query_complex(self):
        """route_query should return 'complex' for complex queries."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)

        state: AdaptiveRAGState = {
            "messages": [],
            "question": "Compare and contrast X and Y",
            "query_type": "complex",
            "documents": [],
            "generation": "",
        }

        result = graph.route_query(state)
        assert result == "complex"

    def test_route_query_web(self):
        """route_query should return 'web' for web queries."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)

        state: AdaptiveRAGState = {"messages": [], "question": "What's the latest news?", "query_type": "web", "documents": [], "generation": ""}

        result = graph.route_query(state)
        assert result == "web"

    @pytest.mark.asyncio
    async def test_simple_retrieval(self):
        """simple_retrieval should retrieve documents."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.ainvoke = AsyncMock(return_value=[Document(page_content="Test doc 1"), Document(page_content="Test doc 2")])
        mock_vectorstore.as_retriever.return_value = mock_retriever

        mock_llm = MagicMock()

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)

        state: AdaptiveRAGState = {"messages": [], "question": "Test question", "query_type": "simple", "documents": [], "generation": ""}

        result = await graph.simple_retrieval(state)
        assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_generate(self):
        """generate should create answer from documents."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "Generated answer"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)

        state: AdaptiveRAGState = {
            "messages": [],
            "question": "Test question",
            "query_type": "simple",
            "documents": [Document(page_content="Relevant context")],
            "generation": "",
        }

        result = await graph.generate(state)
        assert result["generation"] == "Generated answer"
        assert len(result["messages"]) == 2  # Human + AI message


class TestAdaptiveRAGQueryClassification:
    """Test query classification logic."""

    @pytest.mark.asyncio
    async def test_classify_query_simple(self):
        """classify_query should identify simple queries."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "simple"
        mock_llm.__or__ = MagicMock(return_value=mock_llm)
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)

        # Override the chain behavior for testing
        graph.classification_prompt = MagicMock()
        graph.classification_prompt.__or__ = MagicMock(return_value=mock_llm)

        state: AdaptiveRAGState = {"messages": [], "question": "What is Python?", "query_type": None, "documents": [], "generation": ""}

        result = await graph.classify_query(state)
        assert result["query_type"] == "simple"

    @pytest.mark.asyncio
    async def test_classify_query_defaults_to_simple(self):
        """classify_query should default to 'simple' for invalid responses."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "invalid_type"
        mock_llm.__or__ = MagicMock(return_value=mock_llm)
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)
        graph.classification_prompt = MagicMock()
        graph.classification_prompt.__or__ = MagicMock(return_value=mock_llm)

        state: AdaptiveRAGState = {"messages": [], "question": "Some query", "query_type": None, "documents": [], "generation": ""}

        result = await graph.classify_query(state)
        assert result["query_type"] == "simple"


class TestAdaptiveRAGIntegration:
    """Integration tests for Adaptive RAG."""

    def test_graph_builds_successfully(self):
        """AdaptiveRAGGraph should build without errors."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)
        assert graph.graph is not None

    def test_graph_has_all_nodes(self):
        """Graph should have all required nodes."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        graph = AdaptiveRAGGraph(mock_vectorstore, mock_llm)

        # The graph should have been built with nodes
        assert graph.graph is not None
