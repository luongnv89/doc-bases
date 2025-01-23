# src/models/embeddings.py
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# Registry for embedding models
EMBEDDING_REGISTRY = {
    "openai": {
        "default": lambda: OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    },
    "ollama": {
        "default": None,  # Will be set dynamically based on developer input
    },
}


def get_embedding_model(embedding_model_name: str = "nomic-embed-text"):
    """Returns the appropriate embedding model based on detected API keys or developer input.

    Returns:
        Embeddings: The configured embedding model.
    """
    if os.getenv("OPENAI_API_KEY"):
        return EMBEDDING_REGISTRY["openai"]["default"]()
    else:
        # Fallback to Ollama
        if EMBEDDING_REGISTRY["ollama"]["default"] is None:
            EMBEDDING_REGISTRY["ollama"]["default"] = lambda: OllamaEmbeddings(
                model=embedding_model_name
            )
        return EMBEDDING_REGISTRY["ollama"]["default"]()
