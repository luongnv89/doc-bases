# src/models/llm.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Registry for LLM models
LLM_REGISTRY = {
    "google": {
        "default": lambda: ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
        ),
    },
    "openai": {
        "default": lambda: ChatOpenAI(
            model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")
        ),
    },
    "ollama": {
        "default": None,  # Will be set dynamically based on developer input
    },
}


def get_llm_model(llm_model_name: str = "llama3.2"):
    """Returns the appropriate LLM model based on detected API keys or developer input.

    Returns:
        BaseLLM: The configured LLM model.
    """
    if os.getenv("GOOGLE_API_KEY"):
        return LLM_REGISTRY["google"]["default"]()
    elif os.getenv("OPENAI_API_KEY"):
        return LLM_REGISTRY["openai"]["default"]()
    else:
        # Fallback to Ollama
        if LLM_REGISTRY["ollama"]["default"] is None:
            LLM_REGISTRY["ollama"]["default"] = lambda: ChatOllama(model=llm_model_name)
        return LLM_REGISTRY["ollama"]["default"]()
