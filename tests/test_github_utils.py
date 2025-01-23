# tests/test_github_utils.py
import pytest
from src.utils.github_utils import clone_and_parse_repo


def test_clone_and_parse_repo():
    # Test with a valid repository URL
    repo_url = "https://github.com/luongnv89/inbash"
    docs = clone_and_parse_repo(repo_url)
    assert docs is not None
