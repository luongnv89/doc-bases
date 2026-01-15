# tests/test_document_loader.py
import os
import shutil
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from src.utils.document_loader import DocumentLoader

# Constants
TEST_TEMP_DIR = "temps"
TEST_REPO_URL = "https://github.com/example/repo.git"
TEST_FILE_URL = "https://example.com/file.txt"
TEST_WEBSITE_URL = "https://example.com"
TEST_FILE_CONTENT = "This is a test file."
TEST_WEBSITE_CONTENT = "<html><body><p>This is a test website.</p></body></html>"


# Fixtures
@pytest.fixture
def document_loader():
    """Fixture for DocumentLoader instance."""
    return DocumentLoader()


@pytest.fixture
def mock_temp_dir():
    """Fixture to create and clean up the temporary directory."""
    os.makedirs(TEST_TEMP_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_TEMP_DIR):
        shutil.rmtree(TEST_TEMP_DIR)


@pytest.fixture
def mock_file():
    """Fixture to create a test file."""
    file_path = os.path.join(TEST_TEMP_DIR, "test_file.txt")
    with open(file_path, "w") as f:
        f.write(TEST_FILE_CONTENT)
    return file_path


@pytest.fixture
def mock_repo_dir():
    """Fixture to create a mock repository directory."""
    repo_dir = os.path.join(TEST_TEMP_DIR, "repo")
    os.makedirs(repo_dir, exist_ok=True)
    file_path = os.path.join(repo_dir, "test_file.txt")
    with open(file_path, "w") as f:
        f.write(TEST_FILE_CONTENT)
    return repo_dir


# Tests
def test_set_chunks(document_loader):
    """Test setting chunk size and overlap."""
    document_loader.set_chunks(500, 100)
    assert document_loader.chunk_size == 500
    assert document_loader.chunk_overlap == 100


def test_get_repo_name_from_url(document_loader):
    """Test extracting repository name from URL."""
    repo_name = document_loader._get_repo_name_from_url(TEST_REPO_URL)
    assert repo_name == "repo"


def test_clone_repo(document_loader, mock_temp_dir):
    """Test cloning a repository."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = document_loader._clone_repo(TEST_REPO_URL)
        assert result is True


def test_load_single_document(document_loader, mock_file):
    """Test loading a single document."""
    documents = document_loader._load_single_document(mock_file)
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], Document)


def test_load_text_folder(document_loader, mock_repo_dir):
    """Test loading documents from a folder."""
    documents = document_loader._load_text_folder(mock_repo_dir)
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], Document)


def test_split_documents_to_chunk(document_loader, mock_file):
    """Test splitting documents into chunks."""
    documents = document_loader._load_single_document(mock_file)
    chunked_documents = document_loader._split_documents_to_chunk(documents)
    assert isinstance(chunked_documents, list)
    assert len(chunked_documents) > 0
    assert isinstance(chunked_documents[0], Document)


def test_clone_and_parse_repo(document_loader, mock_temp_dir):
    """Test cloning and parsing a repository."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        documents = document_loader._clone_and_parse_repo(TEST_REPO_URL)
        assert isinstance(documents, list)
        assert len(documents) > 0
        assert isinstance(documents[0], Document)


def test_download_file(document_loader, mock_temp_dir):
    """Test downloading a file."""
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            iter_content=lambda: [b"This is a test file."],
            headers={"content-length": "20"},
        )
        result = document_loader._download_file(TEST_FILE_URL, "test_file.txt")
        assert result is True


def test_scrape_website(document_loader):
    """Test scraping a website."""
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=TEST_WEBSITE_CONTENT.encode("utf-8"),
        )
        document = document_loader._scrape_website(TEST_WEBSITE_URL)
        assert isinstance(document, Document)
        assert "This is a test website." in document.page_content


def test_load_documents_from_url(document_loader, mock_temp_dir):
    """Test loading documents from a URL."""
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            iter_content=lambda: [b"This is a test file."],
            headers={"content-length": "20"},
        )
        documents = document_loader.load_documents_from_url(TEST_FILE_URL)
        assert isinstance(documents, list)
        assert len(documents) > 0
        assert isinstance(documents[0], Document)


def test_load_documents_from_directory(document_loader, mock_repo_dir):
    """Test loading documents from a directory."""
    documents = document_loader.load_documents_from_directory(mock_repo_dir)
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], Document)


def test_load_documents_from_file(document_loader, mock_file):
    """Test loading documents from a file."""
    documents = document_loader.load_documents_from_file(mock_file)
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], Document)


def test_load_documents_from_repo(document_loader, mock_temp_dir):
    """Test loading documents from a repository."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        documents = document_loader.load_documents_from_repo(TEST_REPO_URL)
        assert isinstance(documents, list)
        assert len(documents) > 0
        assert isinstance(documents[0], Document)


def test_load_documents_from_website(document_loader):
    """Test loading documents from a website."""
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            content=TEST_WEBSITE_CONTENT.encode("utf-8"),
        )
        documents = document_loader.load_documents_from_website(TEST_WEBSITE_URL)
        assert isinstance(documents, list)
        assert len(documents) > 0
        assert isinstance(documents[0], Document)
