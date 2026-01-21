"""End-to-end tests for all RAG modes.

These tests verify that each RAG mode (basic, corrective, adaptive, multi_agent)
can successfully load a knowledge base and answer queries.
"""

import os
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Test configuration
TEST_KB_NAME = "rag-e2e-test"
TEST_DOCS_DIR = Path("test_rag_e2e")
KNOWLEDGES_DIR = Path("knowledges")


@pytest.fixture(scope="module", autouse=True)
def setup_test_kb():
    """Create a test knowledge base if it doesn't exist."""
    kb_path = KNOWLEDGES_DIR / TEST_KB_NAME

    if not kb_path.exists():
        # Create test documents directory
        TEST_DOCS_DIR.mkdir(exist_ok=True)

        # Create sample document
        sample_doc = TEST_DOCS_DIR / "sample_doc.md"
        sample_doc.write_text(
            """# DocBases Test Document

## Overview

DocBases is a RAG (Retrieval-Augmented Generation) system designed for document-based question answering.

## RAG Modes

DocBases supports four RAG modes:

1. **Basic RAG**: Simple retrieve-and-generate pipeline.
2. **Corrective RAG**: Includes relevance checking and automatic refinement.
3. **Adaptive RAG**: Routes queries to different processing paths.
4. **Multi-Agent RAG**: Uses specialized agents for complex reasoning.

## Test Facts

- The capital of France is Paris.
- Python was created by Guido van Rossum.
- Water boils at 100 degrees Celsius at sea level.
- The Earth is approximately 4.5 billion years old.
"""
        )

        # Create KB
        from src.utils.document_loader import DocumentLoader
        from src.utils.rag_utils import create_vector_store

        doc_loader = DocumentLoader()
        docs = doc_loader.load_documents_from_directory(str(TEST_DOCS_DIR))
        create_vector_store(docs, TEST_KB_NAME)

    yield

    # Cleanup is optional - leave KB for inspection if needed


class TestBasicRAG:
    """Test Basic RAG mode."""

    def test_basic_rag_loads_successfully(self):
        """Test that Basic RAG mode loads without errors."""
        # Set RAG_MODE for this test
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "basic"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None, "Basic RAG agent should load successfully"
        finally:
            # Restore original RAG_MODE
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]

    def test_basic_rag_answers_query(self):
        """Test that Basic RAG mode can answer a query."""
        # Set RAG_MODE for this test
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "basic"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None

            # Generate thread ID
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Execute query
            input_data = {"messages": [HumanMessage(content="What RAG modes does DocBases support?")]}
            result = agent.invoke(input_data, config)

            # Verify result
            assert result is not None
            assert "messages" in result
            assert len(result["messages"]) > 0

            # Get answer from last message
            last_msg = result["messages"][-1]
            answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

            # Verify answer contains expected content
            assert len(answer) > 10, "Answer should be non-trivial"
            print(f"\nBasic RAG Answer: {answer[:500]}...")
        finally:
            # Restore original RAG_MODE
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]


class TestCorrectiveRAG:
    """Test Corrective RAG mode."""

    def test_corrective_rag_loads_successfully(self):
        """Test that Corrective RAG mode loads without errors."""
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "corrective"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None, "Corrective RAG agent should load successfully"
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]

    @pytest.mark.asyncio
    async def test_corrective_rag_answers_query(self):
        """Test that Corrective RAG mode can answer a query."""
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "corrective"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None

            # Generate thread ID
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Execute query (CorrectiveRAGGraph uses invoke method directly)
            query = "What is the capital of France according to the document?"
            answer = await agent.invoke(query, config=config)

            # Verify answer
            assert answer is not None
            assert len(answer) > 0, "Answer should be non-empty"
            print(f"\nCorrective RAG Answer: {answer[:500]}...")
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]


class TestAdaptiveRAG:
    """Test Adaptive RAG mode."""

    def test_adaptive_rag_loads_successfully(self):
        """Test that Adaptive RAG mode loads without errors."""
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "adaptive"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None, "Adaptive RAG agent should load successfully"
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]

    @pytest.mark.asyncio
    async def test_adaptive_rag_answers_query(self):
        """Test that Adaptive RAG mode can answer a query."""
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "adaptive"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None

            # Generate thread ID
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Execute query (AdaptiveRAGGraph uses invoke method directly)
            query = "Who created Python?"
            answer = await agent.invoke(query, config=config)

            # Verify answer
            assert answer is not None
            assert len(answer) > 0, "Answer should be non-empty"
            print(f"\nAdaptive RAG Answer: {answer[:500]}...")
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]


class TestMultiAgentRAG:
    """Test Multi-Agent RAG mode."""

    def test_multi_agent_rag_loads_successfully(self):
        """Test that Multi-Agent RAG mode loads without errors."""
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "multi_agent"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None, "Multi-Agent RAG agent should load successfully"
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout for multi-agent
    async def test_multi_agent_rag_answers_query(self):
        """Test that Multi-Agent RAG mode can answer a query.

        Note: Multi-agent RAG involves multiple sequential LLM calls
        (retriever -> summarizer -> generator -> critic), so this test
        may take longer than other RAG modes.
        """
        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "multi_agent"

        try:
            from src.utils.rag_utils import load_rag_chain

            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None

            # Generate thread ID
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Execute query (MultiAgentSupervisor uses invoke method directly)
            query = "At what temperature does water boil?"
            answer = await agent.invoke(query, config=config)

            # Verify answer
            assert answer is not None
            assert len(answer) > 0, "Answer should be non-empty"
            print(f"\nMulti-Agent RAG Answer: {answer[:500]}...")
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]


class TestRAGModeSelection:
    """Test that RAG mode selection works correctly."""

    def test_default_mode_is_basic(self):
        """Test that default RAG mode is 'basic'."""
        from src.utils.rag_utils import get_rag_mode

        # Save and clear RAG_MODE
        original_mode = os.environ.get("RAG_MODE")
        if "RAG_MODE" in os.environ:
            del os.environ["RAG_MODE"]

        try:
            mode = get_rag_mode()
            assert mode == "basic", f"Default mode should be 'basic', got '{mode}'"
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode

    def test_mode_case_insensitive(self):
        """Test that RAG mode is case insensitive."""
        from src.utils.rag_utils import get_rag_mode

        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "CORRECTIVE"

        try:
            mode = get_rag_mode()
            assert mode == "corrective", f"Mode should be 'corrective' (lowercase), got '{mode}'"
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]

    def test_multi_agent_mode(self):
        """Test that multi_agent mode is recognized."""
        from src.utils.rag_utils import get_rag_mode

        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "multi_agent"

        try:
            mode = get_rag_mode()
            assert mode == "multi_agent"
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_kb_returns_none(self):
        """Test that loading a non-existent KB returns None."""
        from src.utils.rag_utils import load_rag_chain

        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "basic"

        try:
            agent = load_rag_chain("nonexistent-kb-12345")
            assert agent is None, "Should return None for non-existent KB"
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]

    def test_empty_query_handling(self):
        """Test that empty queries are handled gracefully."""
        from src.utils.rag_utils import load_rag_chain

        original_mode = os.environ.get("RAG_MODE")
        os.environ["RAG_MODE"] = "basic"

        try:
            agent = load_rag_chain(TEST_KB_NAME)
            assert agent is not None

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Empty query should still work (agent handles it)
            input_data = {"messages": [HumanMessage(content="")]}
            result = agent.invoke(input_data, config)

            assert result is not None
        finally:
            if original_mode is not None:
                os.environ["RAG_MODE"] = original_mode
            elif "RAG_MODE" in os.environ:
                del os.environ["RAG_MODE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
