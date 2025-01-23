# src/utils/github_utils.py
import os
import subprocess
from github import Github
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# src/utils/github_utils.py
import os
import shutil
from github import Github
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def clone_and_parse_repo(repo_url: str):
    """Clones a GitHub repository, processes its files, and splits the text into chunks for LangChain.

    Args:
        repo_url (str): The URL of the GitHub repository.

    Returns:
        list: A list of LangChain Documents, each containing a chunk of text from the repository.
              Returns None if there's an error during cloning or processing.
    """
    try:
        # Assuming you have a Github token for authentication
        g = Github()  # If you need authentication, use Github('your_token')

        # Get the repository info
        repo = g.get_repo(repo_url.split("/")[-2] + "/" + repo_url.split("/")[-1])
        print(repo)

        # Determine the default branch
        default_branch = repo.default_branch
        print(f"Default branch detected: {default_branch}")

        # Determine a local directory to clone the repository into
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        local_path = os.path.join(os.getcwd(), "temp_" + repo_name)

        # This will clone the repo to the specified local directory
        loader = GitLoader(
            repo_path=local_path,  # Local directory to clone to
            clone_url=repo_url,  # URL to clone from
            branch=default_branch,  # Use the default branch
        )
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        split_documents = text_splitter.split_documents(documents)

        # Clean up: Remove the temporary repository directory
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
            print(f"Removed temporary repository directory: {local_path}")

        return split_documents
    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up in case of an error
        if "local_path" in locals() and os.path.exists(local_path):
            shutil.rmtree(local_path)
            print(f"Removed temporary repository directory due to error: {local_path}")
        return None


def get_version_from_git():
    """Retrieves the version number from the latest Git commit."""
    try:
        # Get the latest commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        return f"Version: {commit_hash}"
    except Exception as e:
        print(f"Failed to retrieve Git version: {e}")
        # Fallback to a hardcoded version or read from a VERSION file
        try:
            with open("VERSION", "r") as f:
                return f"Version: {f.read().strip()}"
        except FileNotFoundError:
            return "Version: 0.1.0 (fallback)"
