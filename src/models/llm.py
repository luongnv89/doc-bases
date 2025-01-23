# src/models/llm.py
"""
This module provides functions for getting LLM models.
"""
import os
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_core.language_models import LLM
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()
logger = get_logger()

def get_llm_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> LLM:
    """Gets the LLM model based on the provided provider and model name.

    Args:
        provider (str, optional): The provider of the LLM model.
            Defaults to None (get from .env).
        model (str, optional): The name of the LLM model.
             Defaults to None (get from .env).
        api_key (str, optional): The API key for the provider.
            Defaults to None.
        api_base (str, optional): The API base URL for the provider.
            Defaults to None, read from env if present.

    Returns:
        LLM: The configured LLM model.
    """
    if not provider:
        provider = os.getenv("LLM_PROVIDER")
        if not provider:
          raise ValueError(
                "LLM_PROVIDER not found in environment variables."
            )
    if not model:
        model = os.getenv("LLM_MODEL")
        if not model:
            raise ValueError(
                "LLM_MODEL not found in environment variables."
            )
    if not api_base:
        api_base = os.getenv("LLM_API_BASE")

    logger.info(f"Getting LLM Model: Provider={provider}, Model={model}")
    if api_key:
        logger.info("User provided api key")
    if api_base:
        logger.info(f"Using custom API base: {api_base}")

    try:
        if provider == "google":
          if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
            logger.info("Getting GOOGLE_API_KEY from env")
          if not api_key:
              raise ValueError(
                  "GOOGLE_API_KEY not found in environment variables or user input."
              )
          logger.info("Using Google GenAI Model")
          return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

        elif provider == "openai":
          if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            logger.info("Getting OPENAI_API_KEY from env")
          if not api_key:
              raise ValueError(
                  "OPENAI_API_KEY not found in environment variables or user input."
              )
          logger.info("Using OpenAI Model")
          return ChatOpenAI(model=model, openai_api_key=api_key,
                            base_url=api_base)

        elif provider == "grok":
            if not api_key:
              api_key = os.getenv("GROQ_API_KEY")
              logger.info("Getting GROQ_API_KEY from env")
            if not api_key:
                  raise ValueError(
                      "GROQ_API_KEY not found in environment variables or user input."
                  )
            logger.info("Using Groq Model")
            return ChatGroq(api_key=api_key, model=model)
        elif provider == "ollama":
            logger.info("Using Ollama Model")
            return OllamaLLM(model=model, base_url=api_base)
        else:
          if not api_key:
            api_key_env_key = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_env_key)
            logger.info(f"Getting {api_key_env_key} from env")
          if api_key:
             logger.info(f"Using Custom Model provider {provider}")
             return ChatOpenAI(model=model, openai_api_key=api_key,
                               base_url = api_base)
          else:
              raise ValueError(f"Provider '{provider}' not supported and no api_key")
    except Exception as e:
        logger.error(f"Error getting LLM model: {e}")
        raise