# tests/test_embeddings.py
import os
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings

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
        patch("langchain_openai.OpenAIEmbeddings") as mock_openai,
        patch("langchain_ollama.OllamaEmbeddings") as mock_ollama,
        patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as mock_google,
    ):
        yield {
            "openai": mock_openai,
            "ollama": mock_ollama,
            "google": mock_google,
        }


# Tests
def test_get_embedding_model_default(mock_env_vars, mock_embeddings):
    """Test get_embedding_model with default behavior (using environment variables)."""
    embeddings = get_embedding_model()
    assert isinstance(embeddings, Embeddings)
    mock_embeddings["openai"].assert_called_once_with(openai_api_key=TEST_API_KEY, base_url=None)


def test_get_embedding_model_openai(mock_embeddings):
    """Test get_embedding_model for OpenAI provider."""
    embeddings = get_embedding_model(
        provider="openai",
        model=TEST_MODEL,
        api_key=TEST_API_KEY,
        api_base=TEST_API_BASE,
    )
    assert isinstance(embeddings, Embeddings)
    mock_embeddings["openai"].assert_called_once_with(openai_api_key=TEST_API_KEY, base_url=TEST_API_BASE)


def test_get_embedding_model_ollama(mock_embeddings):
    """Test get_embedding_model for Ollama provider."""
    embeddings = get_embedding_model(provider="ollama", model=TEST_MODEL)
    assert isinstance(embeddings, Embeddings)
    mock_embeddings["ollama"].assert_called_once_with(model=TEST_MODEL)


def test_get_embedding_model_google(mock_embeddings):
    """Test get_embedding_model for Google provider."""
    embeddings = get_embedding_model(provider="google", model=TEST_MODEL, api_key=TEST_API_KEY)
    assert isinstance(embeddings, Embeddings)
    mock_embeddings["google"].assert_called_once_with(model=f"models/{TEST_MODEL}", google_api_key=TEST_API_KEY)


def test_get_embedding_model_custom_provider(mock_embeddings):
    """Test get_embedding_model for a custom provider."""
    custom_provider = "custom_provider"
    with patch.dict(os.environ, {f"{custom_provider.upper()}_API_KEY": TEST_API_KEY}):
        embeddings = get_embedding_model(provider=custom_provider, model=TEST_MODEL, api_base=TEST_API_BASE)
    assert isinstance(embeddings, Embeddings)
    mock_embeddings["openai"].assert_called_once_with(openai_api_key=TEST_API_KEY, base_url=TEST_API_BASE)


def test_get_embedding_model_missing_provider():
    """Test get_embedding_model with missing provider."""
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
    with pytest.raises(ValueError, match="Provider 'unsupported_provider' not supported and no api_key"):
        get_embedding_model(provider="unsupported_provider")
