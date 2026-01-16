"""
This module provides functions for getting embedding models.
"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from rich.console import Console

from src.utils.logger import custom_theme, get_logger  # Import custom_theme

# Load environment variables from .env file
load_dotenv()
logger = get_logger()

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)


def get_embedding_model(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> Embeddings:
    """
    Gets the embedding model based on the provided provider and model name.

    Args:
        provider (str, optional): The provider of the embedding model.
            Defaults to None (get from .env).
        model (str, optional): The name of the embedding model.
            Defaults to None (get from .env).
        api_key (str, optional): The API key for the provider.
            Defaults to None.
        api_base (str, optional): The API base URL for the provider.
            Defaults to None, read from env if present.

    Returns:
        Embeddings: The configured embedding model.
    """
    if not provider:
        provider = os.getenv("EMB_PROVIDER")
        if not provider:
            console.print("[error]EMB_PROVIDER not found in environment variables.[/error]")
            raise ValueError("EMB_PROVIDER not found in environment variables.")
    if not model:
        model = os.getenv("EMB_MODEL")
        if not model:
            console.print("[error]EMB_MODEL not found in environment variables.[/error]")
            raise ValueError("EMB_MODEL not found in environment variables.")
    if not api_base:
        api_base = os.getenv("EMB_API_BASE")

    console.print(f"[info]Getting Embedding Model: Provider={provider}, Model={model}[/info]")
    if api_key:
        console.print("[info]User provided API key[/info]")
    if api_base:
        console.print(f"[info]Using custom API base: {api_base}[/info]")

    try:
        if provider == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                console.print("[info]Getting OPENAI_API_KEY from env[/info]")
            if not api_key:
                console.print("[error]OPENAI_API_KEY not found in environment variables or user input.[/error]")
                raise ValueError("OPENAI_API_KEY not found in environment variables or user input.")
            console.print("[success]Using OpenAI Embeddings Model[/success]")
            return OpenAIEmbeddings(openai_api_key=api_key, base_url=api_base)

        elif provider == "ollama":
            console.print("[success]Using Ollama Embeddings Model[/success]")
            return OllamaEmbeddings(model=model)

        elif provider == "google":
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
                console.print("[info]Getting GOOGLE_API_KEY from env[/info]")
            if not api_key:
                console.print("[error]GOOGLE_API_KEY not found in environment variables or user input.[/error]")
                raise ValueError("GOOGLE_API_KEY not found in environment variables or user input.")
            # Ensure the model name is in the correct format
            if not model.startswith("models/"):
                model = f"models/{model}"
            console.print(f"[success]Using Google Embeddings Model: {model}[/success]")
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

        else:
            if not api_key:
                api_key_env_key = f"{provider.upper()}_API_KEY"
                api_key = os.getenv(api_key_env_key)
                console.print(f"[info]Getting {api_key_env_key} from env[/info]")
            if api_key:
                console.print(f"[success]Using Custom Embedding Model provider {provider}[/success]")
                return OpenAIEmbeddings(openai_api_key=api_key, base_url=api_base)
            else:
                console.print(f"[error]Provider '{provider}' not supported and no API key provided.[/error]")
                raise ValueError(f"Provider '{provider}' not supported and no API key provided.")

    except Exception as e:
        console.print(f"[error]Error getting embedding model: {e}[/error]")
        raise
