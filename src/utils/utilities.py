import os
import subprocess

from src.utils.logger import get_logger

# Setup logging
logger = get_logger()


def get_version_from_git():
    """Retrieves the version number from the latest Git commit."""
    try:
        # Get the latest commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return f"Version: {commit_hash}"
    except Exception as e:
        logger.error(f"Failed to retrieve Git version: {e}")
        # Fallback to a hardcoded version or read from a VERSION file
        try:
            with open("VERSION") as f:
                return f"Version: {f.read().strip()}"
        except FileNotFoundError:
            return "Version: 0.1.0 (fallback)"


def generate_knowledge_base_name(source_type: int, input_str: str) -> str:
    """Generates a knowledge base name from user input."""
    if source_type == 1:  # Repo
        repo_name = input_str.split("/")[-1].replace(".git", "")
        return repo_name
    elif source_type == 2:  # Local File
        return os.path.splitext(os.path.basename(input_str))[0]
    elif source_type == 3:  # Local Folder
        return os.path.basename(input_str)
    elif source_type == 4:  # Website url
        return input_str.replace("https://", "").replace("http://", "").replace("/", "_")
    elif source_type == 5:  # Download file url
        return os.path.basename(input_str)
    else:
        return "default_knowledge_base"
