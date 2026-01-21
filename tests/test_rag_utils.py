# tests/test_rag_utils.py
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.utils.rag_utils import delete_knowledge_base, list_knowledge_bases, load_rag_chain, setup_rag


# Fixtures
@pytest.fixture
def test_documents():
    """Fixture for test documents."""
    return [
        Document(page_content="This is a test document.", metadata={"source": "test"}),
        Document(page_content="Another test document.", metadata={"source": "test"}),
    ]


@pytest.fixture
def test_knowledge_base_name():
    """Fixture for the test knowledge base name."""
    return "test_knowledge_base"


@pytest.fixture
def mock_temp_dir():
    """Fixture to create and clean up a unique temporary directory for each test."""
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    os.chmod(temp_dir, 0o777)  # Ensure write permissions
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_llm():
    """Fixture to mock LLM model."""
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


@pytest.fixture
def mock_embeddings():
    """Fixture to mock embeddings model."""
    mock = MagicMock()
    mock.embed_documents = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    mock.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    return mock


# Tests
def test_setup_rag(test_documents, test_knowledge_base_name, mock_temp_dir, mock_llm, mock_embeddings):
    """Test setting up the RAG system."""
    with (
        patch("src.utils.rag_utils.KNOWLEDGE_BASE_DIR", mock_temp_dir),
        patch("src.utils.rag_utils.get_embedding_model", return_value=mock_embeddings),
        patch("src.utils.rag_utils.get_rag_mode", return_value="basic"),
        patch("src.utils.rag_utils.Chroma") as mock_chroma,
        patch("src.utils.rag_utils.create_react_agent") as mock_agent,
    ):
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_agent.return_value = MagicMock()

        qa_chain = setup_rag(test_documents, test_knowledge_base_name, llm=mock_llm)

        assert qa_chain is not None
        mock_chroma.assert_called_once()
        mock_agent.assert_called_once()


def test_list_knowledge_bases(test_documents, test_knowledge_base_name, mock_temp_dir, mock_llm, mock_embeddings):
    """Test listing available knowledge bases."""
    # Create a fake knowledge base directory
    kb_dir = os.path.join(mock_temp_dir, test_knowledge_base_name)
    os.makedirs(kb_dir, exist_ok=True)

    with patch("src.utils.rag_utils.KNOWLEDGE_BASE_DIR", mock_temp_dir):
        knowledge_bases = list_knowledge_bases()
        assert test_knowledge_base_name in knowledge_bases


def test_delete_knowledge_base(test_documents, test_knowledge_base_name, mock_temp_dir, mock_llm, mock_embeddings):
    """Test deleting a knowledge base."""
    # Create a fake knowledge base directory
    kb_dir = os.path.join(mock_temp_dir, test_knowledge_base_name)
    os.makedirs(kb_dir, exist_ok=True)

    with patch("src.utils.rag_utils.KNOWLEDGE_BASE_DIR", mock_temp_dir):
        delete_knowledge_base(test_knowledge_base_name)
        assert not os.path.exists(kb_dir)


def test_load_rag_chain(test_documents, test_knowledge_base_name, mock_temp_dir, mock_llm, mock_embeddings):
    """Test loading a RAG chain from disk."""
    # Create a fake knowledge base directory
    kb_dir = os.path.join(mock_temp_dir, test_knowledge_base_name)
    os.makedirs(kb_dir, exist_ok=True)

    with (
        patch("src.utils.rag_utils.KNOWLEDGE_BASE_DIR", mock_temp_dir),
        patch("src.utils.rag_utils.get_embedding_model", return_value=mock_embeddings),
        patch("src.utils.rag_utils.get_rag_mode", return_value="basic"),
        patch("src.utils.rag_utils.Chroma") as mock_chroma,
        patch("src.utils.rag_utils.create_react_agent") as mock_agent,
    ):
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        mock_agent.return_value = MagicMock()

        qa_chain = load_rag_chain(test_knowledge_base_name, llm=mock_llm)

        assert qa_chain is not None
        mock_chroma.assert_called_once()
        mock_agent.assert_called_once()


def test_load_rag_chain_not_found(mock_temp_dir):
    """Test loading a RAG chain that doesn't exist."""
    with patch("src.utils.rag_utils.KNOWLEDGE_BASE_DIR", mock_temp_dir):
        qa_chain = load_rag_chain("nonexistent_kb")
        assert qa_chain is None


def test_interactive_cli(test_knowledge_base_name, mock_temp_dir, mock_llm, mock_embeddings, monkeypatch):
    """Test the interactive CLI."""
    from src.utils.rag_utils import interactive_cli

    # Create a fake knowledge base directory
    kb_dir = os.path.join(mock_temp_dir, test_knowledge_base_name)
    os.makedirs(kb_dir, exist_ok=True)

    # Mock user input to exit immediately
    monkeypatch.setattr("builtins.input", lambda _: "exit")

    with (
        patch("src.utils.rag_utils.KNOWLEDGE_BASE_DIR", mock_temp_dir),
        patch("src.utils.rag_utils.get_embedding_model", return_value=mock_embeddings),
        patch("src.utils.rag_utils.get_llm_model", return_value=mock_llm),
        patch("src.utils.rag_utils.get_rag_mode", return_value="basic"),
        patch("src.utils.rag_utils.Chroma") as mock_chroma,
        patch("src.utils.rag_utils.create_react_agent") as mock_agent,
        patch("src.utils.rag_utils.console") as mock_console,
    ):
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore

        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance

        # Mock console.input to return "exit"
        mock_console.input.return_value = "exit"

        interactive_cli()

        # Verify the agent was created
        mock_agent.assert_called_once()


def test_rag_query(test_documents, test_knowledge_base_name, mock_temp_dir, mock_llm, mock_embeddings):
    """Test querying the RAG system."""
    with (
        patch("src.utils.rag_utils.KNOWLEDGE_BASE_DIR", mock_temp_dir),
        patch("src.utils.rag_utils.get_embedding_model", return_value=mock_embeddings),
        patch("src.utils.rag_utils.get_rag_mode", return_value="basic"),
        patch("src.utils.rag_utils.Chroma") as mock_chroma,
        patch("src.utils.rag_utils.create_react_agent") as mock_agent,
    ):
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore

        # Mock the agent to return a proper response
        mock_agent_instance = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "This is about test documents."
        mock_agent_instance.invoke.return_value = {"messages": [mock_message]}
        mock_agent.return_value = mock_agent_instance

        qa_chain = setup_rag(test_documents, test_knowledge_base_name, llm=mock_llm)

        # Invoke the agent with the proper message format
        result = qa_chain.invoke({"messages": [{"role": "user", "content": "What is this document about?"}]})

        assert "messages" in result
        assert len(result["messages"]) > 0
        assert "test document" in result["messages"][-1].content.lower()
