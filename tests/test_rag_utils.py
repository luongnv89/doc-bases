# tests/test_rag_utils.py
import os
import shutil
import tempfile

import pytest
from langchain.schema import Document

from src.models.llm import get_llm_model
from src.utils.rag_utils import delete_knowledge_base, interactive_cli, list_knowledge_bases, load_rag_chain, setup_rag


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
def llm():
    """Fixture for the LLM model."""
    return get_llm_model()


@pytest.fixture
def mock_temp_dir():
    """Fixture to create and clean up a unique temporary directory for each test."""
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    os.chmod(temp_dir, 0o777)  # Ensure write permissions
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


# Tests
def test_setup_rag(test_documents, test_knowledge_base_name, llm, mock_temp_dir):
    """Test setting up the RAG system."""
    os.environ["KNOWLEDGE_BASE_DIR"] = mock_temp_dir
    qa_chain = setup_rag(test_documents, test_knowledge_base_name, llm=llm)
    assert qa_chain is not None


def test_list_knowledge_bases(test_documents, test_knowledge_base_name, llm, mock_temp_dir):
    """Test listing available knowledge bases."""
    os.environ["KNOWLEDGE_BASE_DIR"] = mock_temp_dir
    setup_rag(test_documents, test_knowledge_base_name, llm=llm)
    knowledge_bases = list_knowledge_bases()
    assert test_knowledge_base_name in knowledge_bases


def test_delete_knowledge_base(test_documents, test_knowledge_base_name, llm, mock_temp_dir):
    """Test deleting a knowledge base."""
    os.environ["KNOWLEDGE_BASE_DIR"] = mock_temp_dir
    setup_rag(test_documents, test_knowledge_base_name, llm=llm)
    delete_knowledge_base(test_knowledge_base_name)
    assert not os.path.exists(os.path.join(mock_temp_dir, test_knowledge_base_name))


def test_load_rag_chain(test_documents, test_knowledge_base_name, llm, mock_temp_dir):
    """Test loading a RAG chain from disk."""
    os.environ["KNOWLEDGE_BASE_DIR"] = mock_temp_dir
    setup_rag(test_documents, test_knowledge_base_name, llm=llm)
    qa_chain = load_rag_chain(test_knowledge_base_name, llm=llm)
    assert qa_chain is not None


def test_interactive_cli(test_documents, test_knowledge_base_name, llm, mock_temp_dir, monkeypatch):
    """Test the interactive CLI."""
    os.environ["KNOWLEDGE_BASE_DIR"] = mock_temp_dir
    setup_rag(test_documents, test_knowledge_base_name, llm=llm)
    monkeypatch.setattr("builtins.input", lambda _: "exit")
    interactive_cli()


def test_rag_query(test_documents, test_knowledge_base_name, llm, mock_temp_dir):
    """Test querying the RAG system."""
    os.environ["KNOWLEDGE_BASE_DIR"] = mock_temp_dir
    qa_chain = setup_rag(test_documents, test_knowledge_base_name, llm=llm)
    result = qa_chain.invoke({"query": "What is this document about?"})
    assert "test document" in result["result"]
