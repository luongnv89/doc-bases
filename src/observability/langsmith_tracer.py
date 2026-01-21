"""
LangSmith integration for distributed tracing and debugging.

LangSmith provides:
- Request tracing for all LangChain/LangGraph operations
- Performance monitoring and latency analysis
- Debug logging for prompt/response pairs
- Cost tracking for API calls
"""

import os

from src.utils.logger import get_logger

logger = get_logger()


def setup_langsmith_tracing(api_key: str | None = None, project: str | None = None, enabled: bool | None = None) -> bool:
    """
    Configure LangSmith tracing for the application.

    Sets up environment variables required by LangChain to enable
    automatic tracing of all LLM calls, retrievals, and agent actions.

    Args:
        api_key: LangSmith API key. Falls back to LANGSMITH_API_KEY env var.
        project: Project name in LangSmith. Falls back to LANGSMITH_PROJECT env var.
        enabled: Whether to enable tracing. Falls back to LANGSMITH_TRACING env var.

    Returns:
        True if tracing was successfully enabled, False otherwise.
    """
    # Determine if tracing should be enabled
    if enabled is None:
        enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    if not enabled:
        logger.info("LangSmith tracing is disabled")
        return False

    # Get API key
    api_key = api_key or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning("LANGSMITH_API_KEY not set - tracing disabled")
        return False

    # Get project name
    project_name: str = project or os.getenv("LANGSMITH_PROJECT", "doc-bases") or "doc-bases"

    # Set environment variables for LangChain
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name

    logger.info(f"LangSmith tracing enabled for project: {project_name}")
    return True


def disable_langsmith_tracing() -> None:
    """
    Disable LangSmith tracing.

    Removes the tracing environment variable to stop trace collection.
    """
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    logger.info("LangSmith tracing disabled")


def is_tracing_enabled() -> bool:
    """
    Check if LangSmith tracing is currently enabled.

    Returns:
        True if tracing is enabled, False otherwise.
    """
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


def get_tracing_status() -> dict:
    """
    Get current tracing configuration status.

    Returns:
        Dictionary with tracing configuration details.
    """
    return {
        "enabled": is_tracing_enabled(),
        "project": os.getenv("LANGCHAIN_PROJECT", "not set"),
        "api_key_configured": bool(os.getenv("LANGCHAIN_API_KEY")),
    }
