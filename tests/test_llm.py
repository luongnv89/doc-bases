# tests/test_llm.py
import os
from unittest.mock import patch

import pytest

from src.models.llm import get_llm_model

# Constants
TEST_PROVIDER = "google"
TEST_MODEL = "gemini-1.5-flash"
TEST_API_KEY = "test_api_key"
TEST_API_BASE = "http://test-api-base.com"


# Fixtures
@pytest.fixture
def mock_env_vars():
    """Fixture to mock environment variables."""
    env_vars = {
        "LLM_PROVIDER": TEST_PROVIDER,
        "LLM_MODEL": TEST_MODEL,
        "GOOGLE_API_KEY": TEST_API_KEY,
        "OPENAI_API_KEY": TEST_API_KEY,
        "GROQ_API_KEY": TEST_API_KEY,
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def mock_llm():
    """Fixture to mock LLM instances."""
    with (
        patch("src.models.llm.ChatGoogleGenerativeAI") as mock_google,
        patch("src.models.llm.ChatOpenAI") as mock_openai,
        patch("src.models.llm.ChatGroq") as mock_groq,
        patch("src.models.llm.ChatOllama") as mock_ollama,
    ):
        yield {
            "google": mock_google,
            "openai": mock_openai,
            "grok": mock_groq,
            "ollama": mock_ollama,
        }


# Tests
def test_get_llm_model_default(mock_env_vars, mock_llm):
    """Test get_llm_model with default behavior (using environment variables)."""
    get_llm_model()
    mock_llm["google"].assert_called_once_with(model=TEST_MODEL, google_api_key=TEST_API_KEY)


def test_get_llm_model_google(mock_llm):
    """Test get_llm_model for Google provider."""
    get_llm_model(provider="google", model=TEST_MODEL, api_key=TEST_API_KEY)
    mock_llm["google"].assert_called_once_with(model=TEST_MODEL, google_api_key=TEST_API_KEY)


def test_get_llm_model_openai(mock_llm):
    """Test get_llm_model for OpenAI provider."""
    get_llm_model(
        provider="openai",
        model=TEST_MODEL,
        api_key=TEST_API_KEY,
        api_base=TEST_API_BASE,
    )
    mock_llm["openai"].assert_called_once_with(model=TEST_MODEL, openai_api_key=TEST_API_KEY, base_url=TEST_API_BASE)


def test_get_llm_model_groq(mock_llm):
    """Test get_llm_model for Groq provider."""
    get_llm_model(provider="grok", model=TEST_MODEL, api_key=TEST_API_KEY)
    mock_llm["grok"].assert_called_once_with(api_key=TEST_API_KEY, model=TEST_MODEL)


def test_get_llm_model_ollama(mock_llm):
    """Test get_llm_model for Ollama provider."""
    get_llm_model(provider="ollama", model=TEST_MODEL, api_base=TEST_API_BASE)
    mock_llm["ollama"].assert_called_once_with(model=TEST_MODEL, base_url=TEST_API_BASE)


def test_get_llm_model_custom_provider(mock_llm):
    """Test get_llm_model for a custom provider."""
    custom_provider = "custom_provider"
    with patch.dict(os.environ, {f"{custom_provider.upper()}_API_KEY": TEST_API_KEY}):
        get_llm_model(provider=custom_provider, model=TEST_MODEL, api_base=TEST_API_BASE)
    mock_llm["openai"].assert_called_once_with(model=TEST_MODEL, openai_api_key=TEST_API_KEY, base_url=TEST_API_BASE)


def test_get_llm_model_missing_provider():
    """Test get_llm_model with missing provider."""
    with patch.dict(os.environ, {"LLM_PROVIDER": ""}, clear=False):
        with pytest.raises(ValueError, match="LLM_PROVIDER not found in environment variables."):
            get_llm_model(provider=None)


def test_get_llm_model_missing_model(mock_env_vars):
    """Test get_llm_model with missing model."""
    with patch.dict(os.environ, {"LLM_MODEL": ""}):
        with pytest.raises(ValueError, match="LLM_MODEL not found in environment variables."):
            get_llm_model()


def test_get_llm_model_missing_api_key(mock_env_vars):
    """Test get_llm_model with missing API key."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
        with pytest.raises(
            ValueError,
            match="GOOGLE_API_KEY not found in environment variables or user input.",
        ):
            get_llm_model(provider="google")


def test_get_llm_model_unsupported_provider():
    """Test get_llm_model with an unsupported provider."""
    with pytest.raises(ValueError, match="Provider 'unsupported_provider' not supported and no API key provided"):
        get_llm_model(provider="unsupported_provider")
