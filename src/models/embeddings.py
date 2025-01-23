# src/models/embeddings.py
"""
This module provides functions for getting embedding models.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()
logger = get_logger()


def get_embedding_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
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
            raise ValueError(
                "EMB_PROVIDER not found in environment variables."
            )
    if not model:
        model = os.getenv("EMB_MODEL")
        if not model:
            raise ValueError(
                "EMB_MODEL not found in environment variables."
            )
    if not api_base:
        api_base = os.getenv("EMB_API_BASE")
    logger.info(f"Getting Embedding Model: Provider={provider}, Model={model}")
    if api_key:
        logger.info("User provided api key")
    if api_base:
        logger.info(f"Using custom API base: {api_base}")

    try:
        if provider == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                logger.info("Getting OPENAI_API_KEY from env")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables or user input."
                )
            logger.info("Using OpenAI Embeddings Model")
            return OpenAIEmbeddings(openai_api_key=api_key,
                                    base_url=api_base)
        elif provider == "ollama":
            logger.info("Using Ollama Embeddings Model")
            return OllamaEmbeddings(model=model)
        elif provider == "google":
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
                logger.info("Getting GOOGLE_API_KEY from env")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found in environment variables or user input."
                )
            # Ensure the model name is in the correct format
            if not model.startswith("models/"):
                model = f"models/{model}"
            logger.info(f"Using Google Embeddings Model: {model}")
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        else:
            if not api_key:
                api_key_env_key = f"{provider.upper()}_API_KEY"
                api_key = os.getenv(api_key_env_key)
                logger.info(f"Getting {api_key_env_key} from env")
            if api_key:
                logger.info(f"Using Custom Embedding Model provider {provider}")
                return OpenAIEmbeddings(openai_api_key=api_key,
                                        base_url=api_base)
            else:
                raise ValueError(f"Provider '{provider}' not supported and no api_key")
    except Exception as e:
        logger.error(f"Error getting embedding model: {e}")
        raise