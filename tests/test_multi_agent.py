"""
Tests for Phase 4: Multi-Agent Orchestration.
"""
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document

from src.agents.retriever_agent import RetrieverAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.critic_agent import CriticAgent, Critique
from src.agents.supervisor import MultiAgentSupervisor, SupervisorState
from src.utils.rag_utils import get_rag_mode


class TestRetrieverAgent:
    """Test RetrieverAgent class."""

    def test_retriever_initialization(self):
        """RetrieverAgent should initialize with vectorstore and llm."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        agent = RetrieverAgent(mock_vectorstore, mock_llm)

        assert agent.vectorstore == mock_vectorstore
        assert agent.llm == mock_llm

    def test_get_documents(self):
        """get_documents should return documents from vectorstore."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2")
        ]
        mock_vectorstore.as_retriever.return_value = mock_retriever

        mock_llm = MagicMock()

        agent = RetrieverAgent(mock_vectorstore, mock_llm)
        docs = agent.get_documents("test query", k=2)

        assert len(docs) == 2
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 2})


class TestSummarizerAgent:
    """Test SummarizerAgent class."""

    def test_summarizer_initialization(self):
        """SummarizerAgent should initialize with llm."""
        mock_llm = MagicMock()
        agent = SummarizerAgent(mock_llm)

        assert agent.llm == mock_llm
        assert agent.summary_prompt is not None

    @pytest.mark.asyncio
    async def test_summarize(self):
        """summarize should generate summary from documents."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a summary"

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.__or__ = MagicMock(return_value=mock_chain)

        agent = SummarizerAgent(mock_llm)

        # Override the chain
        agent.summary_prompt = MagicMock()
        agent.summary_prompt.__or__ = MagicMock(return_value=mock_chain)

        documents = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2")
        ]

        result = await agent.summarize(documents, "test question")
        assert result == "This is a summary"

    def test_summarize_sync(self):
        """summarize_sync should work synchronously."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Synchronous summary"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_llm.__or__ = MagicMock(return_value=mock_chain)

        agent = SummarizerAgent(mock_llm)
        agent.summary_prompt = MagicMock()
        agent.summary_prompt.__or__ = MagicMock(return_value=mock_chain)

        documents = [Document(page_content="Content")]
        result = agent.summarize_sync(documents, "test")

        assert result == "Synchronous summary"


class TestCritiqueModel:
    """Test Critique Pydantic model."""

    def test_critique_creation(self):
        """Critique should be created with valid data."""
        critique = Critique(
            accuracy_score=0.9,
            completeness_score=0.8,
            clarity_score=0.85,
            issues=["Minor issue"],
            suggestions=["Suggestion 1"],
            needs_revision=False
        )

        assert critique.accuracy_score == 0.9
        assert critique.completeness_score == 0.8
        assert critique.clarity_score == 0.85
        assert len(critique.issues) == 1
        assert critique.needs_revision is False

    def test_critique_default_values(self):
        """Critique should have sensible defaults."""
        critique = Critique(
            accuracy_score=0.5,
            completeness_score=0.5,
            clarity_score=0.5,
            needs_revision=True
        )

        assert critique.issues == []
        assert critique.suggestions == []


class TestCriticAgent:
    """Test CriticAgent class."""

    def test_critic_initialization(self):
        """CriticAgent should initialize with llm."""
        mock_llm = MagicMock()
        agent = CriticAgent(mock_llm)

        assert agent.llm == mock_llm
        assert agent.parser is not None
        assert agent.critique_prompt is not None

    def test_get_overall_score(self):
        """get_overall_score should calculate weighted score."""
        mock_llm = MagicMock()
        agent = CriticAgent(mock_llm)

        critique = Critique(
            accuracy_score=1.0,
            completeness_score=1.0,
            clarity_score=1.0,
            needs_revision=False
        )

        score = agent.get_overall_score(critique)
        assert score == 1.0

    def test_get_overall_score_weighted(self):
        """Overall score should be weighted correctly."""
        mock_llm = MagicMock()
        agent = CriticAgent(mock_llm)

        critique = Critique(
            accuracy_score=0.8,  # 0.5 weight
            completeness_score=0.6,  # 0.3 weight
            clarity_score=0.4,  # 0.2 weight
            needs_revision=True
        )

        # 0.8*0.5 + 0.6*0.3 + 0.4*0.2 = 0.4 + 0.18 + 0.08 = 0.66
        score = agent.get_overall_score(critique)
        assert score == 0.66


class TestSupervisorState:
    """Test SupervisorState structure."""

    def test_state_structure(self):
        """SupervisorState should have correct structure."""
        state: SupervisorState = {
            "messages": [],
            "question": "test question",
            "next_agent": None,
            "documents": [],
            "summary": "",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": 3
        }

        assert "question" in state
        assert "next_agent" in state
        assert "documents" in state
        assert "summary" in state
        assert "answer" in state
        assert "critique" in state
        assert "iteration" in state
        assert "max_iterations" in state


class TestMultiAgentSupervisor:
    """Test MultiAgentSupervisor class."""

    def test_supervisor_initialization(self):
        """MultiAgentSupervisor should initialize correctly."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(
            mock_vectorstore,
            mock_llm,
            max_iterations=5
        )

        assert supervisor.vectorstore == mock_vectorstore
        assert supervisor.llm == mock_llm
        assert supervisor.max_iterations == 5
        assert supervisor.summarizer is not None
        assert supervisor.critic is not None

    @pytest.mark.asyncio
    async def test_supervisor_routing_to_retriever(self):
        """Supervisor should route to retriever when no documents."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test",
            "next_agent": None,
            "documents": [],
            "summary": "",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": 3
        }

        result = await supervisor.supervisor_routing(state)
        assert result["next_agent"] == "retriever"

    @pytest.mark.asyncio
    async def test_supervisor_routing_to_summarizer(self):
        """Supervisor should route to summarizer when documents exist."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test",
            "next_agent": None,
            "documents": [Document(page_content="doc")],
            "summary": "",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": 3
        }

        result = await supervisor.supervisor_routing(state)
        assert result["next_agent"] == "summarizer"

    @pytest.mark.asyncio
    async def test_supervisor_routing_to_generator(self):
        """Supervisor should route to generator when summary exists."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test",
            "next_agent": None,
            "documents": [Document(page_content="doc")],
            "summary": "Summary text",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": 3
        }

        result = await supervisor.supervisor_routing(state)
        assert result["next_agent"] == "generator"

    @pytest.mark.asyncio
    async def test_supervisor_routing_to_critic(self):
        """Supervisor should route to critic when answer exists."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test",
            "next_agent": None,
            "documents": [Document(page_content="doc")],
            "summary": "Summary",
            "answer": "Answer text",
            "critique": {},
            "iteration": 1,
            "max_iterations": 3
        }

        result = await supervisor.supervisor_routing(state)
        assert result["next_agent"] == "critic"

    @pytest.mark.asyncio
    async def test_supervisor_routing_to_end(self):
        """Supervisor should route to END when complete."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test",
            "next_agent": None,
            "documents": [Document(page_content="doc")],
            "summary": "Summary",
            "answer": "Answer",
            "critique": {"needs_revision": False},
            "iteration": 2,
            "max_iterations": 3
        }

        result = await supervisor.supervisor_routing(state)
        assert result["next_agent"] == "END"

    def test_route_to_agent(self):
        """route_to_agent should return the next_agent value."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test",
            "next_agent": "summarizer",
            "documents": [],
            "summary": "",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": 3
        }

        result = supervisor.route_to_agent(state)
        assert result == "summarizer"

    @pytest.mark.asyncio
    async def test_call_retriever(self):
        """call_retriever should fetch documents."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.ainvoke = AsyncMock(return_value=[
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2")
        ])
        mock_vectorstore.as_retriever.return_value = mock_retriever

        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test question",
            "next_agent": None,
            "documents": [],
            "summary": "",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": 3
        }

        result = await supervisor.call_retriever(state)
        assert len(result["documents"]) == 2
        assert result["iteration"] == 1

    def test_get_workflow_summary(self):
        """get_workflow_summary should return execution summary."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        state: SupervisorState = {
            "messages": [],
            "question": "test",
            "next_agent": None,
            "documents": [Document(page_content="d1"), Document(page_content="d2")],
            "summary": "Summary",
            "answer": "Answer",
            "critique": {"overall_score": 0.85},
            "iteration": 2,
            "max_iterations": 3
        }

        summary = supervisor.get_workflow_summary(state)

        assert summary["question"] == "test"
        assert summary["documents_retrieved"] == 2
        assert summary["iterations"] == 2
        assert summary["final_score"] == 0.85


class TestMultiAgentRagMode:
    """Test multi_agent RAG mode configuration."""

    def test_multi_agent_mode_from_env(self):
        """Should return 'multi_agent' when configured."""
        with patch.dict(os.environ, {"RAG_MODE": "multi_agent"}):
            result = get_rag_mode()
            assert result == "multi_agent"

    def test_multi_agent_mode_case_insensitive(self):
        """RAG mode should be case-insensitive."""
        with patch.dict(os.environ, {"RAG_MODE": "MULTI_AGENT"}):
            result = get_rag_mode()
            assert result == "multi_agent"


class TestMultiAgentIntegration:
    """Integration tests for multi-agent system."""

    def test_supervisor_builds_graph(self):
        """Supervisor should build workflow graph."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)
        assert supervisor.graph is not None

    def test_supervisor_has_all_components(self):
        """Supervisor should have all required components."""
        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()

        supervisor = MultiAgentSupervisor(mock_vectorstore, mock_llm)

        assert hasattr(supervisor, 'summarizer')
        assert hasattr(supervisor, 'critic')
        assert hasattr(supervisor, 'routing_prompt')
        assert hasattr(supervisor, 'graph')
