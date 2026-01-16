"""
Docling-based document loader for enhanced PDF/document processing.
Provides superior table extraction, layout preservation, and metadata.
"""
import os
from typing import List, Optional
from langchain_core.documents import Document
from rich.console import Console

# Conditional import - Docling may not be installed
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from src.utils.logger import get_logger, custom_theme

logger = get_logger()
console = Console(theme=custom_theme)


class DoclingDocumentLoader:
    """
    Enhanced document loader using IBM's Docling library.

    Supports: PDF, DOCX, PPTX, XLSX, HTML, images
    Features: Table extraction, formula parsing, layout preservation
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling not installed. Install with: pip install docling"
            )
        self.converter = DocumentConverter()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info("DoclingDocumentLoader initialized")

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load and convert document using Docling.

        Args:
            file_path: Path to document file

        Returns:
            List of LangChain Document objects
        """
        console.print(f"[info]Loading with Docling: {file_path}[/info]")

        try:
            result = self.converter.convert(file_path)
            doc = result.document

            # Extract text with metadata
            documents = []

            # Process main text content
            text_content = doc.export_to_markdown()

            # Create document with rich metadata
            metadata = {
                "source": file_path,
                "loader": "docling",
                "num_pages": len(doc.pages) if hasattr(doc, 'pages') else 1,
            }

            documents.append(Document(
                page_content=text_content,
                metadata=metadata
            ))

            # Extract tables as separate documents (for better retrieval)
            if hasattr(doc, 'tables'):
                for i, table in enumerate(doc.tables):
                    table_doc = Document(
                        page_content=table.export_to_markdown(),
                        metadata={
                            **metadata,
                            "content_type": "table",
                            "table_index": i
                        }
                    )
                    documents.append(table_doc)

            console.print(f"[success]Docling extracted {len(documents)} documents[/success]")
            return documents

        except Exception as e:
            console.print(f"[error]Docling failed: {e}[/error]")
            logger.exception(f"Docling error for {file_path}")
            raise

    def supports_format(self, file_path: str) -> bool:
        """Check if Docling supports this file format."""
        supported_extensions = {'.pdf', '.docx', '.pptx', '.xlsx', '.html', '.md'}
        _, ext = os.path.splitext(file_path.lower())
        return ext in supported_extensions


def is_docling_available() -> bool:
    """Check if Docling is installed and available."""
    return DOCLING_AVAILABLE
