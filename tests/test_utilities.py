# tests/test_utils.py
import os
import pytest
from unittest.mock import patch, MagicMock
from src.utils.logger import get_logger
from src.utils.utilities import get_version_from_git, generate_knowledge_base_name

# Constants
TEST_GIT_HASH = "abc1234"
TEST_VERSION_FILE_CONTENT = "1.2.3"
TEST_FALLBACK_VERSION = "0.1.0 (fallback)"

# Fixtures
@pytest.fixture
def mock_git():
    """Fixture to mock Git subprocess calls."""
    with patch("subprocess.check_output") as mock_check_output:
        yield mock_check_output

@pytest.fixture
def mock_version_file():
    """Fixture to mock the VERSION file."""
    with patch("builtins.open") as mock_open:
        yield mock_open

@pytest.fixture
def logger():
    """Fixture to return the logger instance."""
    return get_logger()

# Tests
def test_get_version_from_git_success(mock_git, logger):
    """Test get_version_from_git with successful Git commit retrieval."""
    mock_git.return_value = TEST_GIT_HASH
    version = get_version_from_git()
    assert version == f"Version: {TEST_GIT_HASH}"

def test_get_version_from_git_failure_fallback_version_file(mock_git, mock_version_file, logger):
    """Test get_version_from_git with Git failure and fallback to VERSION file."""
    mock_git.side_effect = Exception("Git error")
    mock_version_file.return_value.__enter__.return_value.read.return_value = TEST_VERSION_FILE_CONTENT
    version = get_version_from_git()
    assert version == f"Version: {TEST_VERSION_FILE_CONTENT}"

def test_get_version_from_git_failure_fallback_hardcoded(mock_git, mock_version_file, logger):
    """Test get_version_from_git with Git failure and fallback to hardcoded version."""
    mock_git.side_effect = Exception("Git error")
    mock_version_file.side_effect = FileNotFoundError
    version = get_version_from_git()
    assert version == f"Version: {TEST_FALLBACK_VERSION}"

def test_generate_knowledge_base_name_repo():
    """Test generate_knowledge_base_name for repository source type."""
    repo_url = "https://github.com/example/repo.git"
    result = generate_knowledge_base_name(1, repo_url)
    assert result == "repo"

def test_generate_knowledge_base_name_local_file():
    """Test generate_knowledge_base_name for local file source type."""
    file_path = "/path/to/file.txt"
    result = generate_knowledge_base_name(2, file_path)
    assert result == "file"

def test_generate_knowledge_base_name_local_folder():
    """Test generate_knowledge_base_name for local folder source type."""
    folder_path = "/path/to/folder"
    result = generate_knowledge_base_name(3, folder_path)
    assert result == "folder"

def test_generate_knowledge_base_name_website_url():
    """Test generate_knowledge_base_name for website URL source type."""
    website_url = "https://example.com/path/to/page"
    result = generate_knowledge_base_name(4, website_url)
    assert result == "example.com_path_to_page"

def test_generate_knowledge_base_name_download_file_url():
    """Test generate_knowledge_base_name for download file URL source type."""
    file_url = "https://example.com/path/to/file.txt"
    result = generate_knowledge_base_name(5, file_url)
    assert result == "file.txt"

def test_generate_knowledge_base_name_default():
    """Test generate_knowledge_base_name for an unknown source type."""
    input_str = "some_input"
    result = generate_knowledge_base_name(99, input_str)
    assert result == "default_knowledge_base"