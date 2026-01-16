"""
This module provides functions for getting LLM models.
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from rich.console import Console

from src.utils.logger import custom_theme, get_logger  # Import custom_theme

# Load environment variables from .env file
load_dotenv()
logger = get_logger()

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)


def get_llm_model(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> BaseChatModel:
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
        BaseChatModel: The configured chat model with tool calling support.
    """
    if not provider:
        provider = os.getenv("LLM_PROVIDER")
        if not provider:
            console.print("[error]LLM_PROVIDER not found in environment variables.[/error]")
            raise ValueError("LLM_PROVIDER not found in environment variables.")
    if not model:
        model = os.getenv("LLM_MODEL")
        if not model:
            console.print("[error]LLM_MODEL not found in environment variables.[/error]")
            raise ValueError("LLM_MODEL not found in environment variables.")
    if not api_base:
        api_base = os.getenv("LLM_API_BASE")

    console.print(f"[info]Getting LLM Model: Provider={provider}, Model={model}[/info]")
    if api_key:
        console.print("[info]User provided API key[/info]")
    if api_base:
        console.print(f"[info]Using custom API base: {api_base}[/info]")

    try:
        if provider == "google":
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
                console.print("[info]Getting GOOGLE_API_KEY from env[/info]")
            if not api_key:
                console.print("[error]GOOGLE_API_KEY not found in environment variables or user input.[/error]")
                raise ValueError("GOOGLE_API_KEY not found in environment variables or user input.")
            console.print("[success]Using Google GenAI Model[/success]")
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

        elif provider == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                console.print("[info]Getting OPENAI_API_KEY from env[/info]")
            if not api_key:
                console.print("[error]OPENAI_API_KEY not found in environment variables or user input.[/error]")
                raise ValueError("OPENAI_API_KEY not found in environment variables or user input.")
            console.print("[success]Using OpenAI Model[/success]")
            return ChatOpenAI(model=model, openai_api_key=api_key, base_url=api_base)

        elif provider == "grok":
            if not api_key:
                api_key = os.getenv("GROQ_API_KEY")
                console.print("[info]Getting GROQ_API_KEY from env[/info]")
            if not api_key:
                console.print("[error]GROQ_API_KEY not found in environment variables or user input.[/error]")
                raise ValueError("GROQ_API_KEY not found in environment variables or user input.")
            console.print("[success]Using Groq Model[/success]")
            return ChatGroq(api_key=api_key, model=model)

        elif provider == "ollama":
            console.print("[success]Using Ollama Model[/success]")
            return ChatOllama(model=model, base_url=api_base)

        else:
            if not api_key:
                api_key_env_key = f"{provider.upper()}_API_KEY"
                api_key = os.getenv(api_key_env_key)
                console.print(f"[info]Getting {api_key_env_key} from env[/info]")
            if api_key:
                console.print(f"[success]Using Custom Model provider {provider}[/success]")
                return ChatOpenAI(model=model, openai_api_key=api_key, base_url=api_base)
            else:
                console.print(f"[error]Provider '{provider}' not supported and no API key provided.[/error]")
                raise ValueError(f"Provider '{provider}' not supported and no API key provided.")

    except Exception as e:
        console.print(f"[error]Error getting LLM model: {e}[/error]")
        raise
