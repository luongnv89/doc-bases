"""
This module provides a DocumentLoader class for processing text files,
preparing them for embedding by splitting them into small chunks.
It includes functionalities for loading text-based documents, splitting documents
into chunks, cloning repositories, downloading files from URLs, and
scraping content from websites.
"""

import os
import subprocess
import shutil
import re
import magic
import requests
import fnmatch
from rich.console import Console
from typing import List, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.schema import Document
from src.utils.logger import get_logger, custom_theme  # Import custom_theme

# Setup logging
logger = get_logger()

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)


class DocumentLoader:
    """
    A class for loading documents from various sources and processing them into chunks.
    """

    TEMP_DIR = "temps"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the DocumentLoader with the given chunk size and overlap.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create the temp directory if it doesn't exist
        if not os.path.exists(self.TEMP_DIR):
            os.makedirs(self.TEMP_DIR)
            logger.info(f"Created temporary directory: {self.TEMP_DIR}")
        logger.info(
            f"DocumentLoader initialized with chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}"
        )

    def set_chunks(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Set the chunk size and overlap.

        Args:
            chunk_size: The new chunk size.
            chunk_overlap: The new chunk overlap.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Chunk size and overlap set to: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """Extracts the repository name from a GitHub URL."""
        if not repo_url:
            return ""
        match = re.search(r"/([^/]+?)(\.git)?$", repo_url)
        repo_name = match.group(1) if match else ""
        logger.debug(f"Extracted repo name: {repo_name} from URL: {repo_url}")
        return repo_name

    def _clone_repo(
        self, repo_url: str, local_path: Optional[str] = None, overwrite: bool = True
    ) -> bool:
        """
        Clones a Git repository to a local path within the 'temps' directory.

        If no local path is provided, the repository will be cloned into
        a directory with the same name as the repository in the 'temps' directory.

        Args:
            repo_url: The URL of the Git repository.
            local_path: The local directory to clone the repository to within 'temps'.
                If None, defaults to the repository name.
            overwrite: If True, overwrite the local path if it exists, default is True

        Returns:
            True if the cloning was successful, False otherwise.
        """
        repo_name = self._get_repo_name_from_url(repo_url)
        if not repo_name:
            console.print("[error]Could not determine repository name from URL.[/error]")
            return False
        if local_path is None:
            local_path = os.path.join(self.TEMP_DIR, repo_name)

        if os.path.exists(local_path) and overwrite:
            console.print(f"[warning]Overwriting existing path: {local_path}[/warning]")
            shutil.rmtree(local_path)

        try:
            console.print(f"[info]Cloning repository from {repo_url} to {local_path}...[/info]")
            subprocess.run(
                ["git", "clone", repo_url, local_path],
                check=True,
                capture_output=True,
                text=True,
            )
            console.print(f"[success]Repository cloned successfully to {local_path}[/success]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[error]Error cloning repository: {e.stderr}[/error]")
            return False
        except Exception as e:
            console.print(f"[error]An unexpected error occurred during cloning: {e}[/error]")
            return False

    def _load_single_document(self, file_path: str) -> Optional[List[Document]]:
        """
        Loads a single document of a supported type.

        Args:
            file_path: Path to the file.

        Returns:
            A list of LangChain documents or None if loading fails.
        """
        try:
            mime_type = magic.from_file(file_path, mime=True)
            logger.info(f"File MIME Type: {mime_type}")

            if mime_type.startswith("text/"):
                console.print(f"[info]Loading Text file: {file_path}[/info]")
                return TextLoader(file_path).load()
            elif mime_type == "application/pdf":
                console.print(f"[info]Loading PDF file: {file_path}[/info]")
                return PyPDFLoader(file_path).load()
            elif (
                mime_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                console.print(f"[info]Loading Word document: {file_path}[/info]")
                return UnstructuredWordDocumentLoader(file_path).load()
            elif (
                mime_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                console.print(f"[info]Loading Excel file: {file_path}[/info]")
                return UnstructuredExcelLoader(file_path).load()
            elif (
                mime_type
                == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                or mime_type == "application/vnd.ms-powerpoint"
            ):
                console.print(f"[info]Loading PowerPoint file: {file_path}[/info]")
                return UnstructuredPowerPointLoader(file_path).load()
            else:
                console.print(f"[warning]Unsupported file type: {mime_type} for {file_path}[/warning]")
                return None
        except Exception as e:
            console.print(f"[error]Error loading document {file_path}: {e}[/error]")
            return None

    def _load_text_folder(self, folder_path: str) -> Optional[List[Document]]:
        """
        Loads supported text files from a folder, including subfolders, to LangChain documents.

        This function supports loading text files (.txt, .md, .py, .json, .csv),
        PDF files (.pdf), Word documents (.docx), and Excel files (.xlsx).

        Args:
            folder_path: The path to the folder containing the files.

        Returns:
            A list of LangChain Documents, or None if there is an error or no files are found.
        """
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            if not os.path.isdir(folder_path):
                raise NotADirectoryError(f"Not a directory: {folder_path}")

            # Load .gitignore patterns if the file exists
            gitignore_path = os.path.join(folder_path, ".gitignore")
            gitignore_patterns = []
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r") as f:
                    gitignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]

            documents = []
            console.print(f"[info]Loading documents from {folder_path}...[/info]")
            for root, dirs, files in os.walk(folder_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]

                for file in tqdm(files, desc="Processing Files"):
                    file_path = os.path.join(root, file)

                    # Skip hidden files
                    if file.startswith("."):
                        continue

                    # Skip files that match .gitignore patterns
                    relative_path = os.path.relpath(file_path, folder_path)
                    if any(fnmatch.fnmatch(relative_path, pattern) for pattern in gitignore_patterns):
                        logger.debug(f"Skipping file due to .gitignore: {file_path}")
                        continue

                    loaded_docs = self._load_single_document(file_path)
                    if loaded_docs:
                        documents.extend(loaded_docs)

            if not documents:
                console.print("[warning]No valid files found in the folder[/warning]")
                return None

            return documents

        except Exception as e:
            console.print(f"[error]Error while loading text folder: {e}[/error]")
            return None

    def _split_documents_to_chunk(
        self, documents: List[Document]
    ) -> Optional[List[Document]]:
        """
        Splits a list of documents into smaller chunks using LangChain's text splitter.

        Args:
            documents: A list of LangChain documents to be split.

        Returns:
            A list of LangChain documents, each containing a smaller chunk of text,
            or None if there's an error during splitting.
        """
        try:
            console.print("[info]Splitting documents into chunks...[/info]")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            split_documents = text_splitter.split_documents(documents)
            console.print(f"[success]Successfully split documents into {len(split_documents)} chunks.[/success]")
            return split_documents
        except Exception as e:
            console.print(f"[error]Error splitting documents: {e}[/error]")
            return None

    def _load_text_folder_to_chunk(self, folder_path: str) -> Optional[List[Document]]:
        """
        Combines loading and chunking processes for text files in a folder.

        This function first loads the text files using load_text_folder function,
        then splits the loaded documents into smaller chunks using the
        split_documents_to_chunk function.

        Args:
            folder_path: The path to the folder containing the files.

        Returns:
            A list of LangChain documents, each containing a smaller chunk of text,
            or None if there's an error during the loading or splitting processes.
        """
        try:
            documents = self._load_text_folder(folder_path)
            if not documents:
                console.print("[warning]No documents loaded from the folder[/warning]")
                return None
            chunked_documents = self._split_documents_to_chunk(documents)
            if not chunked_documents:
                console.print("[error]Failed to split documents into chunks[/error]")
                return None
            return chunked_documents
        except Exception as e:
            console.print(f"[error]Error in load_text_folder_to_chunk: {e}[/error]")
            return None

    def _clone_and_parse_repo(
        self, repo_url: str, overwrite: bool = False
    ) -> Optional[List[Document]]:
        """
        Clones a GitHub repository, processes its files, and splits the text into chunks.

        This function first clones a repository to a local path, then loads
        the text files, splits them into chunks, and finally cleans up the
        cloned repo.

        Args:
            repo_url: The URL of the GitHub repository.
            overwrite: If True, overwrite the local path if it exists

        Returns:
            A list of LangChain documents, each containing a smaller chunk of text,
            or None if there's an error during the cloning, loading, or splitting process.
        """
        repo_name = self._get_repo_name_from_url(repo_url)
        if not repo_name:
            console.print("[error]Could not determine repository name from URL.[/error]")
            return None

        local_path = os.path.join(self.TEMP_DIR, repo_name)
        try:
            if not self._clone_repo(repo_url, local_path, overwrite):
                console.print("[error]Failed to clone the repository.[/error]")
                return None
            chunked_documents = self._load_text_folder_to_chunk(local_path)
            if not chunked_documents:
                console.print("[error]Failed to load and chunk documents.[/error]")
                return None
            return chunked_documents

        except Exception as e:
            console.print(f"[error]An error occurred during repo processing: {e}[/error]")
            return None
        finally:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
                console.print(f"[info]Cleaned up cloned repo at {local_path}[/info]")

    def _download_file(self, url: str, local_path: str, overwrite: bool = True) -> bool:
        """
        Downloads a file from a URL to a local path within the 'temps' directory.

        Args:
            url: The URL of the file to download.
            local_path: The local path to save the downloaded file to within 'temps'.
            overwrite: If True, overwrite the local path if it exists, default is True

        Returns:
            True if the download was successful, False otherwise.
        """
        local_path = os.path.join(self.TEMP_DIR, local_path)

        if os.path.exists(local_path) and overwrite:
            console.print(f"[warning]Overwriting existing path: {local_path}[/warning]")
            os.remove(local_path)

        try:
            console.print(f"[info]Downloading file from {url} to {local_path}...[/info]")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(
                total=total_size, unit="iB", unit_scale=True, desc="Downloading"
            )

            with open(local_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    file.write(chunk)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                console.print("[warning]Download incomplete[/warning]")
                return False

            console.print(f"[success]Downloaded file successfully to {local_path}[/success]")
            return True
        except requests.exceptions.RequestException as e:
            console.print(f"[error]Error downloading file: {e}[/error]")
            return False
        except Exception as e:
            console.print(f"[error]An unexpected error occurred during downloading: {e}[/error]")
            return False

    def _scrape_website(self, url: str) -> Optional[Document]:
        """
        Scrapes the content of a webpage and returns it as a Langchain Document.

        Args:
            url: The URL of the webpage to scrape.

        Returns:
            A Langchain Document containing the text content of the webpage, or
            None if there's an error during the scraping process.
        """
        try:
            console.print(f"[info]Scraping content from {url}...[/info]")
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            text_content = soup.get_text(separator=" ", strip=True)
            console.print("[success]Successfully scraped content from the web page[/success]")
            return Document(page_content=text_content, metadata={"source": url})
        except requests.exceptions.RequestException as e:
            console.print(f"[error]Error during the web scraping: {e}[/error]")
            return None
        except Exception as e:
            console.print(f"[error]An unexpected error occurred during scraping: {e}[/error]")
            return None

    def load_documents_from_url(
        self, file_url: str, overwrite: bool = False
    ) -> Optional[List[Document]]:
        """
        Downloads a file from a URL and returns chunked documents.

        Args:
            file_url: The URL of the file to download.
            overwrite: If True, overwrite the local file if it exists

        Returns:
            A list of LangChain documents, or None if there was an error during
            downloading or processing the document.
        """
        try:
            local_file_name = os.path.basename(file_url)
            if not self._download_file(file_url, local_file_name, overwrite):
                console.print(f"[error]Failed to download file from url: {file_url}[/error]")
                return None
            local_path = os.path.join(self.TEMP_DIR, local_file_name)
            return self._load_single_document(local_path)
        except Exception as e:
            console.print(f"[error]An error occurred during downloading or processing documents: {e}[/error]")
            return None
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    def load_documents_from_directory(
        self, folder_path: str
    ) -> Optional[List[Document]]:
        """
        Loads documents from a directory and returns chunked documents.

        Args:
            folder_path: The path to the folder containing the files.

        Returns:
            A list of LangChain documents, or None if there was an error during
            loading or processing the documents.
        """
        try:
            return self._load_text_folder_to_chunk(folder_path)
        except Exception as e:
            console.print(f"[error]An error occurred during loading or processing documents: {e}[/error]")
            return None

    def load_documents_from_file(self, file_path: str) -> Optional[List[Document]]:
        """
        Loads documents from a single file and returns chunked documents.

        Args:
            file_path: Path to a single file.

        Returns:
            A list of LangChain documents, or None if there was an error during
            loading or processing the document.
        """
        try:
            documents = self._load_single_document(file_path)
            if not documents:
                console.print("[warning]No documents loaded from the file[/warning]")
                return None
            return self._split_documents_to_chunk(documents)
        except Exception as e:
            console.print(f"[error]An error occurred during loading or processing document: {e}[/error]")
            return None

    def load_documents_from_repo(
        self, repo_url: str, overwrite: bool = False
    ) -> Optional[List[Document]]:
        """
        Clones a repository from a URL and returns chunked documents.

        Args:
            repo_url: The URL of the Git repository.
            overwrite: If True, overwrite the local file if it exists

        Returns:
            A list of LangChain documents, or None if there was an error during
            cloning or processing the documents.
        """
        try:
            return self._clone_and_parse_repo(repo_url, overwrite)
        except Exception as e:
            console.print(f"[error]An error occurred during loading or processing the repo: {e}[/error]")
            return None

    def load_documents_from_website(self, url: str) -> Optional[List[Document]]:
        """
        Scrapes content from a website and returns chunked documents.

        Args:
            url: The url of the website to be scraped.

        Returns:
             A list of LangChain documents, or None if there was an error during
            scraping or processing the document
        """
        try:
            scraped_document = self._scrape_website(url)
            if not scraped_document:
                console.print(f"[error]Failed to scrape website from {url}[/error]")
                return None
            return self._split_documents_to_chunk([scraped_document])
        except Exception as e:
            console.print(f"[error]An error occurred during scraping or processing the website: {e}[/error]")
            return None