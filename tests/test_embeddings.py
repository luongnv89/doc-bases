# tests/test_embeddings.py
import os
from unittest.mock import patch

import pytest

from src.models.embeddings import get_embedding_model

# Constants
TEST_PROVIDER = "openai"
TEST_MODEL = "text-embedding-3-small"
TEST_API_KEY = "test_api_key"
TEST_API_BASE = "http://test-api-base.com"


# Fixtures
@pytest.fixture
def mock_env_vars():
    """Fixture to mock environment variables."""
    env_vars = {
        "EMB_PROVIDER": TEST_PROVIDER,
        "EMB_MODEL": TEST_MODEL,
        "OPENAI_API_KEY": TEST_API_KEY,
        "GOOGLE_API_KEY": TEST_API_KEY,
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def mock_embeddings():
    """Fixture to mock embedding instances."""
    with (
        patch("src.models.embeddings.OpenAIEmbeddings") as mock_openai,
        patch("src.models.embeddings.OllamaEmbeddings") as mock_ollama,
        patch("src.models.embeddings.GoogleGenerativeAIEmbeddings") as mock_google,
    ):
        yield {
            "openai": mock_openai,
            "ollama": mock_ollama,
            "google": mock_google,
        }


# Tests
def test_get_embedding_model_default(mock_env_vars, mock_embeddings):
    """Test get_embedding_model with default behavior (using environment variables)."""
    # Clear EMB_API_BASE to test with base_url='' (empty string from env)
    with patch.dict(os.environ, {"EMB_API_BASE": ""}, clear=False):
        get_embedding_model()
        # base_url will be empty string when EMB_API_BASE is set to ""
        mock_embeddings["openai"].assert_called_once_with(openai_api_key=TEST_API_KEY, base_url="")


def test_get_embedding_model_openai(mock_embeddings):
    """Test get_embedding_model for OpenAI provider."""
    get_embedding_model(
        provider="openai",
        model=TEST_MODEL,
        api_key=TEST_API_KEY,
        api_base=TEST_API_BASE,
    )
    mock_embeddings["openai"].assert_called_once_with(openai_api_key=TEST_API_KEY, base_url=TEST_API_BASE)


def test_get_embedding_model_ollama(mock_embeddings):
    """Test get_embedding_model for Ollama provider."""
    get_embedding_model(provider="ollama", model=TEST_MODEL)
    mock_embeddings["ollama"].assert_called_once_with(model=TEST_MODEL)


def test_get_embedding_model_google(mock_embeddings):
    """Test get_embedding_model for Google provider."""
    get_embedding_model(provider="google", model=TEST_MODEL, api_key=TEST_API_KEY)
    mock_embeddings["google"].assert_called_once_with(model=f"models/{TEST_MODEL}", google_api_key=TEST_API_KEY)


def test_get_embedding_model_custom_provider(mock_embeddings):
    """Test get_embedding_model for a custom provider."""
    custom_provider = "custom_provider"
    with patch.dict(os.environ, {f"{custom_provider.upper()}_API_KEY": TEST_API_KEY}):
        get_embedding_model(provider=custom_provider, model=TEST_MODEL, api_base=TEST_API_BASE)
    mock_embeddings["openai"].assert_called_once_with(openai_api_key=TEST_API_KEY, base_url=TEST_API_BASE)


def test_get_embedding_model_missing_provider():
    """Test get_embedding_model with missing provider."""
    with patch.dict(os.environ, {"EMB_PROVIDER": ""}, clear=False):
        with pytest.raises(ValueError, match="EMB_PROVIDER not found in environment variables."):
            get_embedding_model(provider=None)


def test_get_embedding_model_missing_model(mock_env_vars):
    """Test get_embedding_model with missing model."""
    with patch.dict(os.environ, {"EMB_MODEL": ""}):
        with pytest.raises(ValueError, match="EMB_MODEL not found in environment variables."):
            get_embedding_model()


def test_get_embedding_model_missing_api_key(mock_env_vars):
    """Test get_embedding_model with missing API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        with pytest.raises(
            ValueError,
            match="OPENAI_API_KEY not found in environment variables or user input.",
        ):
            get_embedding_model(provider="openai")


def test_get_embedding_model_unsupported_provider():
    """Test get_embedding_model with an unsupported provider."""
    with pytest.raises(ValueError, match="Provider 'unsupported_provider' not supported and no API key provided"):
        get_embedding_model(provider="unsupported_provider")
