# DocBases Modernization Plan

## Executive Summary

**Project:** DocBases RAG System Modernization
**Objective:** Transform from basic ReAct agent to production-grade agentic RAG system
**Timeline:** 12-16 weeks (5 phases)
**Approach:** Pragmatic Balanced with Fresh Start (no backward compatibility constraint)

### Current State vs Target State

| Component | Current State | Target State |
|-----------|--------------|--------------|
| **Orchestration** | LangChain 0.3.15 + LangGraph 0.4.8 | LangChain 1.2.x + LangGraph 1.0 |
| **Vector Store** | ChromaDB 0.5.23 | ChromaDB 1.4.x with hybrid search |
| **Document Parsing** | Unstructured loaders | Docling (tables, formulas, layouts) |
| **RAG Pattern** | Basic ReAct | Corrective + Adaptive + Multi-Agent |
| **Memory** | In-memory MemorySaver | SQLite persistent checkpointer |
| **Observability** | None | LangSmith tracing + metrics |
| **Python** | 3.9+ | 3.10+ (required by LangChain 1.x) |

### Key Design Decisions

1. **Fresh Start:** No backward compatibility with existing knowledge bases (simplifies migration)
2. **Feature Flags:** All new capabilities are opt-in via environment variables
3. **Phased Delivery:** Each phase delivers independent value
4. **Standard Timeline:** Sequential phases for thorough testing

---

## Table of Contents

1. [Phase 1: Foundation & Core Dependencies](#phase-1-foundation--core-dependencies-weeks-1-2)
2. [Phase 2: Document Processing with Docling](#phase-2-document-processing-with-docling-weeks-3-5)
3. [Phase 3: Advanced RAG Patterns](#phase-3-advanced-rag-patterns-weeks-6-9)
4. [Phase 4: Multi-Agent Orchestration](#phase-4-multi-agent-orchestration-weeks-10-13)
5. [Phase 5: Persistent Memory & Observability](#phase-5-persistent-memory--observability-weeks-14-16)
6. [File Reference](#file-reference)
7. [Environment Configuration](#environment-configuration)
8. [Testing Strategy](#testing-strategy)
9. [Risk Mitigation](#risk-mitigation)
10. [Resources](#resources)

---

## Phase 1: Foundation & Core Dependencies (Weeks 1-2)

### Objectives

- Upgrade Python to 3.10+
- Update all LangChain ecosystem packages to latest stable
- Upgrade ChromaDB with hybrid search support
- Ensure CLI continues working throughout

### Prerequisites

- Python 3.10 or higher installed
- Virtual environment management (venv/conda)
- Git for version control

### Tasks

#### 1.1 Create Feature Branch

```bash
git checkout -b feature/modernize-rag-v2
```

#### 1.2 Python & Project Configuration

**File:** `pyproject.toml`

Update the following sections:

```toml
[project]
name = "doc-bases"
version = "2.0.0"
requires-python = ">=3.10"

[tool.black]
target-version = ['py310', 'py311', 'py312']

[tool.mypy]
python_version = "3.10"

[tool.ruff]
target-version = "py310"
```

#### 1.3 Dependencies Update

**File:** `requirements.txt`

Replace with updated versions:

```txt
# =============================================================================
# DocBases v2.0 Dependencies
# =============================================================================

# -----------------------------------------------------------------------------
# Core Framework - UPGRADED
# -----------------------------------------------------------------------------
langchain>=1.2.0,<2.0.0
langchain-core>=1.0.0,<2.0.0
langchain-community>=1.0.0,<2.0.0
langchain-text-splitters>=1.0.0,<2.0.0
langgraph>=1.0.0,<2.0.0

# -----------------------------------------------------------------------------
# Vector Store - UPGRADED
# -----------------------------------------------------------------------------
chromadb>=1.4.0,<2.0.0
langchain-chroma>=1.0.0,<2.0.0

# -----------------------------------------------------------------------------
# LLM Providers - UPDATED FOR COMPATIBILITY
# -----------------------------------------------------------------------------
langchain-openai>=1.0.0,<2.0.0
langchain-google-genai>=2.0.0
langchain-groq>=1.0.0
langchain-ollama>=1.0.0

# -----------------------------------------------------------------------------
# Document Processing (Phase 2)
# -----------------------------------------------------------------------------
docling>=2.0.0
langchain-experimental>=0.3.0
unstructured>=0.16.0

# -----------------------------------------------------------------------------
# Web Search (Phase 3)
# -----------------------------------------------------------------------------
duckduckgo-search>=6.0.0

# -----------------------------------------------------------------------------
# Observability (Phase 5)
# -----------------------------------------------------------------------------
langsmith>=0.3.0

# -----------------------------------------------------------------------------
# Existing Utilities - KEEP
# -----------------------------------------------------------------------------
rich>=13.9.4
python-dotenv>=1.0.1
tqdm>=4.67.0
python-magic>=0.4.27
beautifulsoup4>=4.12.0
requests>=2.32.0
pydantic>=2.0.0

# -----------------------------------------------------------------------------
# Development Dependencies
# -----------------------------------------------------------------------------
pytest>=7.0.0
pytest-asyncio>=0.23.0
black>=24.0.0
ruff>=0.1.0
mypy>=1.0.0
```

#### 1.4 Import Updates

**File:** `src/utils/rag_utils.py`

Update imports at the top of the file:

```python
# Line 15: Update Document import
# OLD: from langchain.schema import Document
# NEW:
from langchain_core.documents import Document

# Lines 17-19: Verify LangGraph imports (should be compatible)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
```

**File:** `src/utils/document_loader.py`

Update imports:

```python
# Line 19: Update Document import
# OLD: from langchain.schema import Document
# NEW:
from langchain_core.documents import Document

# Line 20: Update text splitter import
# OLD: from langchain.text_splitter import RecursiveCharacterTextSplitter
# NEW:
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

**File:** `src/models/llm.py`

Update imports:

```python
# Line 9: Update base class import
# OLD: from langchain_core.language_models import LLM
# NEW:
from langchain_core.language_models.chat_models import BaseChatModel
```

#### 1.5 Installation & Testing

```bash
# Create new virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run existing tests
pytest tests/ -v

# Manual testing
python -m src.main
# Test: Create a knowledge base from a local file
# Test: Query the knowledge base
# Test: Delete the knowledge base
```

### Checklist

- [x] Create feature branch `feature/modernize-rag-v2`
- [x] Update `pyproject.toml` Python version to 3.10+
- [x] Update `requirements.txt` with new versions
- [x] Fix import statements in `src/utils/rag_utils.py`
- [x] Fix import statements in `src/utils/document_loader.py`
- [x] Fix import statements in `src/models/llm.py`
- [x] Fix import statements in `src/models/embeddings.py`
- [x] Create Python 3.10+ virtual environment
- [x] Run `pip install -r requirements.txt`
- [x] Run existing tests: `pytest tests/`
- [x] Manual test: Create knowledge base
- [x] Manual test: Query knowledge base
- [x] Manual test: Delete knowledge base
- [x] Document any breaking changes encountered
- [x] Commit changes with message: `feat: upgrade to LangChain 1.2.x, LangGraph 1.0, ChromaDB 1.4.x`

**Status: ✅ COMPLETED**

### Success Criteria

- All existing tests pass
- Can create new knowledge base from repo/file/folder
- Can query existing knowledge base via interactive CLI
- No deprecation warnings for updated imports

### Rollback Plan

```bash
# If issues occur, revert to previous state
git checkout main
# Re-create venv with old requirements
```

---

## Phase 2: Document Processing with Docling (Weeks 3-5)

### Objectives

- Integrate Docling for advanced document parsing
- Add semantic chunking capabilities
- Maintain fallback to existing loaders
- Feature flag for gradual rollout

### New Files to Create

#### 2.1 Docling Document Loader

**File:** `src/utils/docling_loader.py`

```python
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
```

#### 2.2 Semantic Chunking Strategy

**File:** `src/utils/semantic_splitter.py`

```python
"""
Semantic chunking using embeddings for natural text boundaries.
Replaces naive character-count splitting for better retrieval quality.
"""
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.models.embeddings import get_embedding_model
from src.utils.logger import get_logger

logger = get_logger()


class SemanticDocumentSplitter:
    """
    Split documents at semantic boundaries using embedding similarity.

    Preserves topic coherence within chunks better than character-based splitting.
    """

    def __init__(
        self,
        embeddings=None,
        breakpoint_type: str = "percentile",
        breakpoint_threshold: float = 95
    ):
        """
        Initialize semantic splitter.

        Args:
            embeddings: Embedding model (uses default if None)
            breakpoint_type: How to detect split points
                           (percentile, standard_deviation, interquartile)
            breakpoint_threshold: Threshold for creating new chunks
        """
        if embeddings is None:
            embeddings = get_embedding_model()

        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_threshold
        )
        logger.info(f"SemanticDocumentSplitter initialized with {breakpoint_type} strategy")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents at semantic boundaries.

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents with preserved metadata
        """
        try:
            chunks = self.splitter.split_documents(documents)
            logger.info(f"Semantic splitting: {len(documents)} docs → {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Semantic splitting failed: {e}")
            raise


def get_chunking_strategy() -> str:
    """Get configured chunking strategy from environment."""
    return os.getenv("CHUNKING_STRATEGY", "recursive").lower()
```

### Files to Modify

#### 2.3 Update Document Loader

**File:** `src/utils/document_loader.py`

Add the following imports at the top:

```python
import os
from src.utils.docling_loader import DoclingDocumentLoader, is_docling_available
from src.utils.semantic_splitter import SemanticDocumentSplitter, get_chunking_strategy
```

Add helper function:

```python
def _use_docling() -> bool:
    """Check if Docling should be used for document loading."""
    use_docling = os.getenv("USE_DOCLING", "false").lower() == "true"
    return use_docling and is_docling_available()
```

Modify `_load_single_document` method (around line 125):

```python
def _load_single_document(self, file_path: str) -> Optional[List[Document]]:
    """Load single document with Docling or fallback loaders."""

    # Try Docling first if enabled
    if _use_docling():
        try:
            docling_loader = DoclingDocumentLoader(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            if docling_loader.supports_format(file_path):
                console.print("[info]Using Docling parser[/info]")
                return docling_loader.load_document(file_path)
        except Exception as e:
            console.print(f"[warning]Docling failed, falling back: {e}[/warning]")

    # Existing MIME-type based loading logic continues below...
    try:
        mime_type = magic.from_file(file_path, mime=True)
        # ... rest of existing code ...
```

Modify `_split_documents_to_chunk` method (around line 223):

```python
def _split_documents_to_chunk(self, documents: List[Document]) -> Optional[List[Document]]:
    """Split documents using configured chunking strategy."""

    try:
        strategy = get_chunking_strategy()

        if strategy == "semantic":
            console.print("[info]Using semantic chunking strategy[/info]")
            splitter = SemanticDocumentSplitter()
            return splitter.split_documents(documents)
        else:
            # Default: recursive character splitting (existing logic)
            console.print("[info]Using recursive chunking strategy[/info]")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_documents = text_splitter.split_documents(documents)
            console.print(f"[success]Split into {len(split_documents)} chunks[/success]")
            return split_documents

    except Exception as e:
        console.print(f"[error]Error splitting documents: {e}[/error]")
        return None
```

### Checklist

- [x] Create `src/utils/docling_loader.py`
- [x] Create `src/utils/semantic_splitter.py`
- [x] Update imports in `src/utils/document_loader.py`
- [x] Add `_use_docling()` helper function
- [x] Modify `_load_single_document()` method
- [x] Modify `_split_documents_to_chunk()` method
- [x] Add environment variables to `.env`
- [x] Install Docling: `pip install docling>=2.0.0`
- [x] Install experimental: `pip install langchain-experimental>=0.3.0`
- [ ] Test Docling with sample PDF containing tables
- [ ] Test semantic chunking vs recursive chunking
- [x] Create tests: `tests/test_docling_integration.py`
- [ ] Benchmark chunk quality improvement
- [ ] Commit: `feat: add Docling document parsing and semantic chunking`

**Status: IN PROGRESS - Code implementation complete, manual testing pending**

### Success Criteria

- Docling successfully extracts tables from PDFs
- Semantic chunking produces more coherent chunks
- Fallback to existing loaders works when Docling fails
- Feature flags allow toggling between strategies

---

## Phase 3: Advanced RAG Patterns (Weeks 6-9)

### Objectives

- Implement Corrective RAG (CRAG) with self-grading
- Implement Adaptive RAG with query routing
- Add web search fallback for out-of-domain queries
- Integrate as selectable modes in CLI

### Directory Structure

Create the following directories:

```bash
mkdir -p src/evaluation
mkdir -p src/tools
mkdir -p src/graphs

touch src/evaluation/__init__.py
touch src/tools/__init__.py
touch src/graphs/__init__.py
```

### New Files to Create

#### 3.1 RAG Evaluation Framework

**File:** `src/evaluation/__init__.py`

```python
"""RAG evaluation components."""
from src.evaluation.rag_evaluator import RAGEvaluator, RelevanceScore, HallucinationCheck

__all__ = ["RAGEvaluator", "RelevanceScore", "HallucinationCheck"]
```

**File:** `src/evaluation/rag_evaluator.py`

```python
"""
RAG quality evaluation using LLM-based grading.
Provides relevance scoring and hallucination detection.
"""
from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.utils.logger import get_logger

logger = get_logger()


class RelevanceScore(BaseModel):
    """Structured output for relevance grading."""
    score: float = Field(description="Relevance score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation for the score")


class HallucinationCheck(BaseModel):
    """Structured output for hallucination detection."""
    is_grounded: bool = Field(description="Whether answer is grounded in documents")
    unsupported_claims: List[str] = Field(
        default=[],
        description="Claims not supported by documents"
    )


class RAGEvaluator:
    """
    Evaluate RAG retrieval and generation quality.

    Used by Corrective RAG to decide if retrieval is sufficient.
    """

    def __init__(self, llm):
        self.llm = llm
        self.relevance_parser = PydanticOutputParser(pydantic_object=RelevanceScore)
        self.hallucination_parser = PydanticOutputParser(pydantic_object=HallucinationCheck)

        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance grader. Evaluate if a document is relevant to answering a question.

Score from 0.0 (completely irrelevant) to 1.0 (highly relevant).

{format_instructions}"""),
            ("human", """Question: {question}

Document: {document}

Grade the relevance:""")
        ])

        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker. Determine if an answer is fully grounded in the provided documents.

{format_instructions}"""),
            ("human", """Documents:
{documents}

Answer to check:
{answer}

Is this answer grounded in the documents?""")
        ])

    async def grade_relevance(self, question: str, document: str) -> RelevanceScore:
        """Grade relevance of a document to a question."""
        chain = self.relevance_prompt | self.llm | self.relevance_parser
        result = await chain.ainvoke({
            "question": question,
            "document": document,
            "format_instructions": self.relevance_parser.get_format_instructions()
        })
        logger.debug(f"Relevance score: {result.score}")
        return result

    async def check_hallucination(self, documents: str, answer: str) -> HallucinationCheck:
        """Check if answer contains hallucinations."""
        chain = self.hallucination_prompt | self.llm | self.hallucination_parser
        result = await chain.ainvoke({
            "documents": documents,
            "answer": answer,
            "format_instructions": self.hallucination_parser.get_format_instructions()
        })
        logger.debug(f"Grounded: {result.is_grounded}")
        return result

    async def grade_documents_batch(
        self,
        question: str,
        documents: List[str],
        threshold: float = 0.5
    ) -> Tuple[List[str], List[str]]:
        """
        Grade multiple documents and split into relevant/irrelevant.

        Returns:
            Tuple of (relevant_docs, irrelevant_docs)
        """
        relevant = []
        irrelevant = []

        for doc in documents:
            score = await self.grade_relevance(question, doc)
            if score.score >= threshold:
                relevant.append(doc)
            else:
                irrelevant.append(doc)

        logger.info(f"Graded {len(documents)} docs: {len(relevant)} relevant")
        return relevant, irrelevant
```

#### 3.2 Web Search Tool

**File:** `src/tools/__init__.py`

```python
"""Tool components for RAG system."""
from src.tools.web_search import web_search, web_search_to_documents

__all__ = ["web_search", "web_search_to_documents"]
```

**File:** `src/tools/web_search.py`

```python
"""
Web search tool for fallback retrieval in Corrective RAG.
Uses DuckDuckGo (free, no API key required).
"""
import os
from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger()

# Try to import DuckDuckGo search
try:
    from langchain_community.tools import DuckDuckGoSearchResults
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for additional context when document retrieval is insufficient.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Concatenated search results as text
    """
    if not DUCKDUCKGO_AVAILABLE:
        return "Web search unavailable. Install: pip install duckduckgo-search"

    try:
        search = DuckDuckGoSearchResults(max_results=max_results)
        results = search.run(query)
        logger.info(f"Web search for '{query}': {len(results)} chars returned")
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search error: {e}"


def web_search_to_documents(query: str, max_results: int = 3) -> List[Document]:
    """
    Perform web search and return results as LangChain Documents.

    Useful for integrating web results into RAG pipeline.
    """
    if not DUCKDUCKGO_AVAILABLE:
        return []

    try:
        search = DuckDuckGoSearchResults(max_results=max_results)
        results = search.invoke(query)

        documents = []
        if isinstance(results, list):
            for result in results:
                doc = Document(
                    page_content=result.get("snippet", str(result)),
                    metadata={
                        "source": result.get("link", "web_search"),
                        "title": result.get("title", ""),
                        "content_type": "web_search"
                    }
                )
                documents.append(doc)
        else:
            # Handle string results
            documents.append(Document(
                page_content=str(results),
                metadata={"source": "web_search", "content_type": "web_search"}
            ))

        return documents
    except Exception as e:
        logger.error(f"Web search to documents failed: {e}")
        return []
```

#### 3.3 Corrective RAG Graph

**File:** `src/graphs/__init__.py`

```python
"""LangGraph workflow definitions."""
from src.graphs.corrective_rag import CorrectiveRAGGraph
from src.graphs.adaptive_rag import AdaptiveRAGGraph

__all__ = ["CorrectiveRAGGraph", "AdaptiveRAGGraph"]
```

**File:** `src/graphs/corrective_rag.py`

```python
"""
Corrective RAG (CRAG) implementation using LangGraph.

Workflow:
1. Retrieve documents from vectorstore
2. Grade relevance of each document
3. If sufficient relevant docs → Generate answer
4. If insufficient → Web search → Generate answer
5. Check for hallucinations
6. Return validated answer
"""
import os
from typing import TypedDict, List, Literal, Annotated
from operator import add

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.evaluation.rag_evaluator import RAGEvaluator
from src.tools.web_search import web_search_to_documents
from src.utils.logger import get_logger

logger = get_logger()


class CRAGState(TypedDict):
    """State for Corrective RAG workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    documents: List[Document]
    relevant_docs: List[Document]
    web_search_needed: bool
    web_results: List[Document]
    generation: str
    is_grounded: bool


class CorrectiveRAGGraph:
    """
    Self-corrective RAG with relevance grading and web search fallback.
    """

    def __init__(
        self,
        vectorstore,
        llm,
        checkpointer=None,
        relevance_threshold: float = 0.5,
        min_relevant_docs: int = 1
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.checkpointer = checkpointer
        self.evaluator = RAGEvaluator(llm)
        self.relevance_threshold = relevance_threshold
        self.min_relevant_docs = min_relevant_docs

        self.graph = self._build_graph()
        logger.info("CorrectiveRAGGraph initialized")

    async def retrieve(self, state: CRAGState) -> CRAGState:
        """Retrieve documents from vectorstore."""
        logger.info(f"Retrieving documents for: {state['question']}")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = await retriever.ainvoke(state["question"])

        state["documents"] = docs
        logger.info(f"Retrieved {len(docs)} documents")
        return state

    async def grade_documents(self, state: CRAGState) -> CRAGState:
        """Grade relevance of retrieved documents."""
        logger.info("Grading document relevance")

        relevant_docs = []
        for doc in state["documents"]:
            score = await self.evaluator.grade_relevance(
                state["question"],
                doc.page_content
            )
            if score.score >= self.relevance_threshold:
                relevant_docs.append(doc)

        state["relevant_docs"] = relevant_docs
        state["web_search_needed"] = len(relevant_docs) < self.min_relevant_docs

        logger.info(f"Relevant: {len(relevant_docs)}, Web search: {state['web_search_needed']}")
        return state

    def decide_retrieval_quality(self, state: CRAGState) -> Literal["generate", "web_search"]:
        """Route based on document relevance."""
        if state["web_search_needed"]:
            return "web_search"
        return "generate"

    async def web_search(self, state: CRAGState) -> CRAGState:
        """Fallback to web search for additional context."""
        logger.info(f"Web search for: {state['question']}")

        web_docs = web_search_to_documents(state["question"], max_results=3)
        state["web_results"] = web_docs
        state["relevant_docs"].extend(web_docs)

        logger.info(f"Added {len(web_docs)} web results")
        return state

    async def generate(self, state: CRAGState) -> CRAGState:
        """Generate answer from relevant documents."""
        logger.info("Generating answer")

        context = "\n\n".join([doc.page_content for doc in state["relevant_docs"]])

        prompt = f"""Answer the question based on the following context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {state['question']}

Answer:"""

        response = await self.llm.ainvoke(prompt)
        state["generation"] = response.content if hasattr(response, 'content') else str(response)

        state["messages"].append(HumanMessage(content=state["question"]))
        state["messages"].append(AIMessage(content=state["generation"]))

        return state

    async def check_hallucination(self, state: CRAGState) -> CRAGState:
        """Check if answer is grounded in documents."""
        logger.info("Checking for hallucinations")

        context = "\n\n".join([doc.page_content for doc in state["relevant_docs"]])
        check = await self.evaluator.check_hallucination(context, state["generation"])

        state["is_grounded"] = check.is_grounded

        if not check.is_grounded:
            logger.warning(f"Hallucination detected: {check.unsupported_claims}")

        return state

    def _build_graph(self) -> StateGraph:
        """Build the CRAG workflow graph."""
        workflow = StateGraph(CRAGState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)
        workflow.add_node("check_hallucination", self.check_hallucination)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_retrieval_quality,
            {"generate": "generate", "web_search": "web_search"}
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", "check_hallucination")
        workflow.add_edge("check_hallucination", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def invoke(self, question: str, config: dict = None) -> str:
        """Run the CRAG workflow."""
        initial_state = {
            "messages": [],
            "question": question,
            "documents": [],
            "relevant_docs": [],
            "web_search_needed": False,
            "web_results": [],
            "generation": "",
            "is_grounded": True
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result["generation"]
```

#### 3.4 Adaptive RAG Graph

**File:** `src/graphs/adaptive_rag.py`

```python
"""
Adaptive RAG with query routing and strategy selection.

Routes queries to optimal retrieval strategy:
- Simple: Direct vectorstore retrieval (fast)
- Complex: Multi-step retrieval with reranking
- Web: External web search for out-of-domain queries
"""
from typing import TypedDict, List, Literal, Annotated
from operator import add

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.tools.web_search import web_search_to_documents
from src.utils.logger import get_logger

logger = get_logger()


class AdaptiveRAGState(TypedDict):
    """State for Adaptive RAG workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    query_type: Literal["simple", "complex", "web"] | None
    documents: List[Document]
    generation: str


class AdaptiveRAGGraph:
    """
    Adaptive RAG with intelligent query routing.
    """

    def __init__(self, vectorstore, llm, checkpointer=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.checkpointer = checkpointer

        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify the query type:

- simple: Direct factual question, single document lookup
- complex: Requires reasoning or multiple documents
- web: Current events or external information

Respond with ONLY: simple, complex, or web"""),
            ("human", "{question}")
        ])

        self.graph = self._build_graph()
        logger.info("AdaptiveRAGGraph initialized")

    async def classify_query(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Classify query complexity/type."""
        logger.info(f"Classifying: {state['question']}")

        chain = self.classification_prompt | self.llm
        response = await chain.ainvoke({"question": state["question"]})

        classification = response.content.strip().lower() if hasattr(response, 'content') else "simple"

        if classification not in ["simple", "complex", "web"]:
            classification = "simple"

        state["query_type"] = classification
        logger.info(f"Classified as: {classification}")
        return state

    def route_query(self, state: AdaptiveRAGState) -> str:
        """Route to appropriate retrieval strategy."""
        return state["query_type"]

    async def simple_retrieval(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Fast direct retrieval for simple queries."""
        logger.info("Simple retrieval")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = await retriever.ainvoke(state["question"])

        state["documents"] = docs
        return state

    async def complex_retrieval(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Multi-step retrieval for complex queries."""
        logger.info("Complex retrieval")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        initial_docs = await retriever.ainvoke(state["question"])

        # Generate sub-queries
        subquery_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate 2 related queries. One per line."),
            ("human", "{question}")
        ])

        chain = subquery_prompt | self.llm
        response = await chain.ainvoke({"question": state["question"]})

        subqueries = response.content.strip().split("\n") if hasattr(response, 'content') else []

        all_docs = list(initial_docs)
        for sq in subqueries[:2]:
            if sq.strip():
                sub_docs = await retriever.ainvoke(sq.strip())
                all_docs.extend(sub_docs)

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        state["documents"] = unique_docs[:8]
        return state

    async def web_retrieval(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Web search for external information."""
        logger.info("Web retrieval")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        local_docs = await retriever.ainvoke(state["question"])

        web_docs = web_search_to_documents(state["question"], max_results=3)

        state["documents"] = local_docs + web_docs
        return state

    async def generate(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Generate answer."""
        logger.info("Generating")

        context = "\n\n".join([doc.page_content for doc in state["documents"]])

        prompt = f"""Context:
{context}

Question: {state['question']}

Answer:"""

        response = await self.llm.ainvoke(prompt)
        state["generation"] = response.content if hasattr(response, 'content') else str(response)

        state["messages"].append(HumanMessage(content=state["question"]))
        state["messages"].append(AIMessage(content=state["generation"]))

        return state

    def _build_graph(self) -> StateGraph:
        """Build workflow graph."""
        workflow = StateGraph(AdaptiveRAGState)

        workflow.add_node("classify", self.classify_query)
        workflow.add_node("simple_retrieval", self.simple_retrieval)
        workflow.add_node("complex_retrieval", self.complex_retrieval)
        workflow.add_node("web_retrieval", self.web_retrieval)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("classify")

        workflow.add_conditional_edges(
            "classify",
            self.route_query,
            {
                "simple": "simple_retrieval",
                "complex": "complex_retrieval",
                "web": "web_retrieval"
            }
        )

        workflow.add_edge("simple_retrieval", "generate")
        workflow.add_edge("complex_retrieval", "generate")
        workflow.add_edge("web_retrieval", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def invoke(self, question: str, config: dict = None) -> str:
        """Run workflow."""
        initial_state = {
            "messages": [],
            "question": question,
            "query_type": None,
            "documents": [],
            "generation": ""
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result["generation"]
```

### Files to Modify

#### 3.5 Update RAG Utils

**File:** `src/utils/rag_utils.py`

Add imports at top:

```python
import os
import asyncio
from src.graphs.corrective_rag import CorrectiveRAGGraph
from src.graphs.adaptive_rag import AdaptiveRAGGraph
```

Add helper function:

```python
def get_rag_mode() -> str:
    """Get configured RAG mode from environment."""
    return os.getenv("RAG_MODE", "basic").lower()
```

Modify `setup_rag` function:

```python
def setup_rag(documents: List[Document], knowledge_base_name: str, llm=None):
    """Setup RAG with selected mode."""
    logger.info(f"Setting up RAG for: '{knowledge_base_name}'")

    persist_directory = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)
    embeddings = get_embedding_model()

    console.print(f"[header]Creating vector store for '{knowledge_base_name}'...[/header]")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    with tqdm(total=len(documents), desc="Embedding data") as pbar:
        for doc in documents:
            vectorstore.add_documents([doc])
            pbar.update(1)

    console.print(f"[success]Vector store created.[/success]")

    if not llm:
        llm = get_llm_model()

    memory = MemorySaver()
    rag_mode = get_rag_mode()

    if rag_mode == "corrective":
        logger.info("Setting up Corrective RAG mode")
        console.print("[info]RAG Mode: Corrective[/info]")
        agent = CorrectiveRAGGraph(vectorstore, llm, checkpointer=memory)
    elif rag_mode == "adaptive":
        logger.info("Setting up Adaptive RAG mode")
        console.print("[info]RAG Mode: Adaptive[/info]")
        agent = AdaptiveRAGGraph(vectorstore, llm, checkpointer=memory)
    else:
        logger.info("Setting up Basic RAG mode")
        console.print("[info]RAG Mode: Basic[/info]")
        retrieve_tool = get_retriever_tool(vectorstore)
        agent = create_react_agent(llm, [retrieve_tool], checkpointer=memory)

    return agent
```

Modify `load_rag_chain` function similarly.

Modify `interactive_cli` function to handle async:

```python
def interactive_cli() -> None:
    """Interactive CLI with async agent support."""
    # ... existing setup code ...

    rag_mode = get_rag_mode()
    console.print(f"[info]RAG Mode: {rag_mode}[/info]")

    while True:
        query = console.input("[info]You: [/info]")
        if query.lower() == "exit":
            break

        # Validation...

        with Live(Spinner("dots"), refresh_per_second=20):
            time.sleep(0.5)
            if rag_mode in ["corrective", "adaptive"]:
                # Async agents
                answer = asyncio.run(agent.invoke(query, config=config))
            else:
                # Basic agent
                messages.append({"role": "user", "content": query})
                result = agent.invoke({"messages": messages}, config=config)
                answer = result["messages"][-1].content

        # Display answer...
```

### Checklist

- [x] Create `src/evaluation/` directory and `__init__.py`
- [x] Create `src/evaluation/rag_evaluator.py`
- [x] Create `src/tools/` directory and `__init__.py`
- [x] Create `src/tools/web_search.py`
- [x] Create `src/graphs/` directory and `__init__.py`
- [x] Create `src/graphs/corrective_rag.py`
- [x] Create `src/graphs/adaptive_rag.py`
- [x] Update `src/utils/rag_utils.py` with mode selection
- [x] Install: `pip install duckduckgo-search`
- [x] Add `RAG_MODE` to `.env`
- [ ] Test Corrective RAG with poor retrieval scenario
- [ ] Test Adaptive RAG query routing
- [ ] Benchmark: basic vs corrective vs adaptive
- [x] Create tests: `tests/test_corrective_rag.py`
- [x] Create tests: `tests/test_adaptive_rag.py`
- [ ] Commit: `feat: add Corrective RAG and Adaptive RAG patterns`

**Status: IN PROGRESS - Code implementation complete, manual testing pending**

### Success Criteria

- Corrective RAG triggers web search when documents are irrelevant
- Adaptive RAG correctly classifies query types (>90% accuracy)
- Hallucination detection catches unsupported claims
- No significant latency regression for simple queries (<2x)

---

## Phase 4: Multi-Agent Orchestration (Weeks 10-13)

### Objectives

- Implement specialized agents (Retriever, Summarizer, Critic)
- Create supervisor for agent orchestration
- Enable iterative refinement through agent collaboration

### Directory Structure

```bash
mkdir -p src/agents
touch src/agents/__init__.py
```

### New Files to Create

#### 4.1 Agent Package Init

**File:** `src/agents/__init__.py`

```python
"""Multi-agent system components."""
from src.agents.retriever_agent import RetrieverAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.critic_agent import CriticAgent
from src.agents.supervisor import MultiAgentSupervisor

__all__ = [
    "RetrieverAgent",
    "SummarizerAgent",
    "CriticAgent",
    "MultiAgentSupervisor"
]
```

#### 4.2 Retriever Agent

**File:** `src/agents/retriever_agent.py`

```python
"""
Specialized retriever agent with query decomposition.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.utils.logger import get_logger

logger = get_logger()


class RetrieverAgent:
    """Agent specialized in document retrieval."""

    def __init__(self, vectorstore, llm, checkpointer=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.checkpointer = checkpointer
        self.agent = self._create_agent()

    def _create_agent(self):
        """Create ReAct agent with retrieval tools."""
        vectorstore = self.vectorstore
        llm = self.llm

        @tool
        def retrieve_documents(query: str, k: int = 4) -> str:
            """Retrieve relevant documents from knowledge base."""
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
            return "\n\n".join([d.page_content for d in docs])

        @tool
        def decompose_query(complex_query: str) -> str:
            """Break complex query into simpler sub-queries."""
            prompt = f"""Break this question into 2-3 simpler queries:
Question: {complex_query}

Return as numbered list."""
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

        tools = [retrieve_documents, decompose_query]
        return create_react_agent(self.llm, tools, checkpointer=self.checkpointer)

    async def retrieve(self, question: str, config: dict = None) -> dict:
        """Execute retrieval workflow."""
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Retrieve for: {question}"}]},
            config=config
        )
        return result
```

#### 4.3 Summarizer Agent

**File:** `src/agents/summarizer_agent.py`

```python
"""
Summarizer agent for condensing documents.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.utils.logger import get_logger

logger = get_logger()


class SummarizerAgent:
    """Agent specialized in summarization."""

    def __init__(self, llm):
        self.llm = llm

        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Create concise summaries focused on answering questions."""),
            ("human", """Summarize for this question:

Question: {question}

Documents:
{documents}

Summary:""")
        ])

    async def summarize(self, documents: List[Document], question: str) -> str:
        """Generate query-focused summary."""
        doc_text = "\n\n---\n\n".join([d.page_content for d in documents])

        chain = self.summary_prompt | self.llm
        response = await chain.ainvoke({
            "question": question,
            "documents": doc_text
        })

        return response.content if hasattr(response, 'content') else str(response)

    async def map_reduce_summarize(self, documents: List[Document], question: str) -> str:
        """Map-reduce for large document sets."""
        # Map: Summarize each
        summaries = []
        for doc in documents:
            summary = await self.summarize([doc], question)
            summaries.append(summary)

        # Reduce: Combine
        combined = Document(page_content="\n\n".join(summaries))
        final = await self.summarize([combined], question)

        return final
```

#### 4.4 Critic Agent

**File:** `src/agents/critic_agent.py`

```python
"""
Critic agent for answer validation.
"""
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.utils.logger import get_logger

logger = get_logger()


class Critique(BaseModel):
    """Structured critique."""
    accuracy_score: float = Field(description="0-1 factual accuracy")
    completeness_score: float = Field(description="0-1 completeness")
    clarity_score: float = Field(description="0-1 clarity")
    issues: List[str] = Field(default=[], description="Issues found")
    suggestions: List[str] = Field(default=[], description="Improvements")
    needs_revision: bool = Field(description="Needs revision?")


class CriticAgent:
    """Agent for answer validation."""

    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=Critique)

        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate answers for accuracy, completeness, clarity.

{format_instructions}"""),
            ("human", """Question: {question}

Documents:
{documents}

Answer:
{answer}

Critique:""")
        ])

    async def critique(self, question: str, answer: str, documents: str) -> Critique:
        """Critique an answer."""
        chain = self.critique_prompt | self.llm | self.parser

        result = await chain.ainvoke({
            "question": question,
            "documents": documents,
            "answer": answer,
            "format_instructions": self.parser.get_format_instructions()
        })

        logger.info(f"Scores: acc={result.accuracy_score}, comp={result.completeness_score}")
        return result

    async def suggest_improvements(self, answer: str, critique: Critique) -> str:
        """Improve answer based on critique."""
        if not critique.needs_revision:
            return answer

        prompt = f"""Improve this answer:

Original: {answer}

Issues:
{chr(10).join(['- ' + i for i in critique.issues])}

Suggestions:
{chr(10).join(['- ' + s for s in critique.suggestions])}

Improved:"""

        response = await self.llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
```

#### 4.5 Multi-Agent Supervisor

**File:** `src/agents/supervisor.py`

```python
"""
Supervisor for multi-agent orchestration.
"""
from typing import TypedDict, Literal, Annotated, List
from operator import add

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.agents.summarizer_agent import SummarizerAgent
from src.agents.critic_agent import CriticAgent
from src.utils.logger import get_logger

logger = get_logger()


class SupervisorState(TypedDict):
    """State for supervisor."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    next_agent: Literal["retriever", "summarizer", "generator", "critic", "END"] | None
    documents: List[Document]
    summary: str
    answer: str
    critique: dict
    iteration: int
    max_iterations: int


class MultiAgentSupervisor:
    """
    Orchestrates specialized agents.

    Flow:
    1. Retriever fetches documents
    2. Summarizer condenses
    3. Generator creates answer
    4. Critic validates
    5. Iterate if needed
    """

    def __init__(self, vectorstore, llm, checkpointer=None, max_iterations: int = 3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.checkpointer = checkpointer
        self.max_iterations = max_iterations

        self.summarizer = SummarizerAgent(llm)
        self.critic = CriticAgent(llm)

        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """Decide next agent:
- retriever: Need documents
- summarizer: Need to condense
- generator: Need answer
- critic: Need validation
- END: Task complete

Reply with ONE word."""),
            ("human", """State:
- Question: {question}
- Has docs: {has_docs}
- Has summary: {has_summary}
- Has answer: {has_answer}
- Iteration: {iteration}/{max_iter}
- Critique: {critique}

Next:""")
        ])

        self.graph = self._build_graph()
        logger.info("MultiAgentSupervisor initialized")

    async def supervisor_routing(self, state: SupervisorState) -> SupervisorState:
        """Decide next agent."""
        chain = self.routing_prompt | self.llm

        response = await chain.ainvoke({
            "question": state["question"],
            "has_docs": len(state["documents"]) > 0,
            "has_summary": bool(state["summary"]),
            "has_answer": bool(state["answer"]),
            "iteration": state["iteration"],
            "max_iter": state["max_iterations"],
            "critique": state["critique"].get("summary", "None")
        })

        next_agent = response.content.strip().lower() if hasattr(response, 'content') else "END"

        if next_agent not in ["retriever", "summarizer", "generator", "critic", "END"]:
            next_agent = "END"

        state["next_agent"] = next_agent
        logger.info(f"Routing to: {next_agent}")
        return state

    def route_to_agent(self, state: SupervisorState) -> str:
        """Return next agent."""
        return state["next_agent"]

    async def call_retriever(self, state: SupervisorState) -> SupervisorState:
        """Retrieve documents."""
        logger.info("Retriever agent")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = await retriever.ainvoke(state["question"])

        state["documents"] = docs
        state["iteration"] += 1
        return state

    async def call_summarizer(self, state: SupervisorState) -> SupervisorState:
        """Summarize documents."""
        logger.info("Summarizer agent")

        summary = await self.summarizer.summarize(state["documents"], state["question"])
        state["summary"] = summary
        return state

    async def generate_answer(self, state: SupervisorState) -> SupervisorState:
        """Generate answer."""
        logger.info("Generator")

        context = state["summary"] or "\n\n".join([d.page_content for d in state["documents"]])

        prompt = f"""Context:
{context}

Question: {state['question']}

Answer:"""

        response = await self.llm.ainvoke(prompt)
        state["answer"] = response.content if hasattr(response, 'content') else str(response)
        return state

    async def call_critic(self, state: SupervisorState) -> SupervisorState:
        """Validate answer."""
        logger.info("Critic agent")

        docs_text = "\n\n".join([d.page_content for d in state["documents"]])
        critique = await self.critic.critique(state["question"], state["answer"], docs_text)

        state["critique"] = {
            "accuracy": critique.accuracy_score,
            "completeness": critique.completeness_score,
            "needs_revision": critique.needs_revision,
            "summary": f"Acc: {critique.accuracy_score:.2f}"
        }

        if critique.needs_revision and state["iteration"] < state["max_iterations"]:
            improved = await self.critic.suggest_improvements(state["answer"], critique)
            state["answer"] = improved

        return state

    async def finalize(self, state: SupervisorState) -> SupervisorState:
        """Finalize."""
        state["messages"].append(HumanMessage(content=state["question"]))
        state["messages"].append(AIMessage(content=state["answer"]))
        return state

    def _build_graph(self) -> StateGraph:
        """Build graph."""
        workflow = StateGraph(SupervisorState)

        workflow.add_node("supervisor", self.supervisor_routing)
        workflow.add_node("retriever", self.call_retriever)
        workflow.add_node("summarizer", self.call_summarizer)
        workflow.add_node("generator", self.generate_answer)
        workflow.add_node("critic", self.call_critic)
        workflow.add_node("finalize", self.finalize)

        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            self.route_to_agent,
            {
                "retriever": "retriever",
                "summarizer": "summarizer",
                "generator": "generator",
                "critic": "critic",
                "END": "finalize"
            }
        )

        workflow.add_edge("retriever", "supervisor")
        workflow.add_edge("summarizer", "supervisor")
        workflow.add_edge("generator", "supervisor")
        workflow.add_edge("critic", "supervisor")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def invoke(self, question: str, config: dict = None) -> str:
        """Run workflow."""
        initial_state = {
            "messages": [],
            "question": question,
            "next_agent": None,
            "documents": [],
            "summary": "",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": self.max_iterations
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result["answer"]
```

### Files to Modify

Update `src/utils/rag_utils.py` to include multi-agent mode:

```python
from src.agents.supervisor import MultiAgentSupervisor

# In setup_rag and load_rag_chain:
if rag_mode == "multi_agent":
    logger.info("Setting up Multi-Agent RAG mode")
    console.print("[info]RAG Mode: Multi-Agent[/info]")
    agent = MultiAgentSupervisor(vectorstore, llm, checkpointer=memory)
```

### Checklist

- [x] Create `src/agents/` directory
- [x] Create `src/agents/__init__.py`
- [x] Create `src/agents/retriever_agent.py`
- [x] Create `src/agents/summarizer_agent.py`
- [x] Create `src/agents/critic_agent.py`
- [x] Create `src/agents/supervisor.py`
- [x] Update `src/utils/rag_utils.py` with multi-agent mode
- [x] Test supervisor routing
- [x] Test iterative refinement
- [ ] Benchmark multi-agent vs single-agent
- [x] Create tests: `tests/test_multi_agent.py`
- [ ] Commit: `feat: add multi-agent orchestration system`

**Status: IN PROGRESS - Code implementation complete, manual testing pending**

### Success Criteria

- Supervisor correctly routes to agents
- Critic improves answer quality
- Multi-agent produces higher quality answers
- Max iterations prevents infinite loops

---

## Phase 5: Persistent Memory & Observability (Weeks 14-16)

### Objectives

- Replace MemorySaver with SQLite persistent checkpointer
- Add LangSmith tracing
- Create metrics dashboard

### Directory Structure

```bash
mkdir -p src/checkpointing
mkdir -p src/observability

touch src/checkpointing/__init__.py
touch src/observability/__init__.py
```

### New Files to Create

#### 5.1 SQLite Checkpointer

**File:** `src/checkpointing/__init__.py`

```python
"""Checkpointing components."""
from src.checkpointing.sqlite_saver import PersistentCheckpointer, get_checkpointer

__all__ = ["PersistentCheckpointer", "get_checkpointer"]
```

**File:** `src/checkpointing/sqlite_saver.py`

```python
"""
SQLite-based persistent checkpointer.
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from src.utils.logger import get_logger, custom_theme

logger = get_logger()
console = Console(theme=custom_theme)


class PersistentCheckpointer:
    """SQLite-backed checkpointer."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("CHECKPOINT_DB_PATH", "knowledges/checkpoints.db")

        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")

        self.saver = SqliteSaver(self.conn)
        logger.info(f"PersistentCheckpointer at {db_path}")

    def get_saver(self) -> SqliteSaver:
        """Get SqliteSaver."""
        return self.saver

    def list_sessions(self, limit: int = 10) -> list:
        """List recent sessions."""
        try:
            cursor = self.conn.execute("""
                SELECT DISTINCT thread_id, MAX(created_at) as last_active
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY last_active DESC
                LIMIT ?
            """, (limit,))
            return cursor.fetchall()
        except Exception:
            return []

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete old sessions."""
        cutoff = datetime.now() - timedelta(days=days)
        try:
            cursor = self.conn.execute("""
                DELETE FROM checkpoints WHERE created_at < ?
            """, (cutoff.isoformat(),))
            deleted = cursor.rowcount
            self.conn.commit()
            logger.info(f"Cleaned {deleted} entries")
            return deleted
        except Exception:
            return 0

    def delete_session(self, thread_id: str) -> bool:
        """Delete session."""
        try:
            cursor = self.conn.execute("""
                DELETE FROM checkpoints WHERE thread_id = ?
            """, (thread_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False

    def close(self):
        """Close connection."""
        self.conn.close()


def get_checkpointer(use_persistent: bool = None):
    """Get checkpointer based on config."""
    if use_persistent is None:
        use_persistent = os.getenv("USE_PERSISTENT_MEMORY", "true").lower() == "true"

    if use_persistent:
        try:
            checkpointer = PersistentCheckpointer()
            return checkpointer.get_saver()
        except Exception as e:
            logger.warning(f"SQLite failed, using memory: {e}")
            return MemorySaver()
    else:
        return MemorySaver()
```

#### 5.2 LangSmith Integration

**File:** `src/observability/__init__.py`

```python
"""Observability components."""
from src.observability.langsmith_tracer import setup_langsmith_tracing
from src.observability.metrics import MetricsTracker, get_metrics_tracker

__all__ = ["setup_langsmith_tracing", "MetricsTracker", "get_metrics_tracker"]
```

**File:** `src/observability/langsmith_tracer.py`

```python
"""
LangSmith integration.
"""
import os
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger()


def setup_langsmith_tracing(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    enabled: Optional[bool] = None
) -> bool:
    """Configure LangSmith tracing."""
    if enabled is None:
        enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    if not enabled:
        logger.info("LangSmith disabled")
        return False

    api_key = api_key or os.getenv("LANGSMITH_API_KEY")
    project = project or os.getenv("LANGSMITH_PROJECT", "doc-bases")

    if not api_key:
        logger.warning("LANGSMITH_API_KEY not set")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    logger.info(f"LangSmith enabled: {project}")
    return True
```

#### 5.3 Metrics Tracking

**File:** `src/observability/metrics.py`

```python
"""
Metrics tracking.
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table

from src.utils.logger import get_logger, custom_theme

logger = get_logger()
console = Console(theme=custom_theme)


class MetricsTracker:
    """Track RAG metrics."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("METRICS_DB_PATH", "knowledges/metrics.db")

        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
        logger.info(f"MetricsTracker at {db_path}")

    def _init_tables(self):
        """Create tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                retrieval_count INTEGER DEFAULT 0,
                rag_mode TEXT NOT NULL,
                session_id TEXT,
                success BOOLEAN DEFAULT TRUE,
                error TEXT
            )
        """)
        self.conn.commit()

    def log_query(
        self,
        query: str,
        latency_ms: int,
        retrieval_count: int = 0,
        rag_mode: str = "basic",
        session_id: str = None,
        success: bool = True,
        error: str = None
    ):
        """Log query metric."""
        self.conn.execute("""
            INSERT INTO query_metrics
            (timestamp, query, latency_ms, retrieval_count, rag_mode, session_id, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            query,
            latency_ms,
            retrieval_count,
            rag_mode,
            session_id,
            success,
            error
        ))
        self.conn.commit()

    def get_stats(self, days: int = 7) -> Dict:
        """Get stats."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        total = self.conn.execute(
            "SELECT COUNT(*) FROM query_metrics WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0]

        success = self.conn.execute(
            "SELECT COUNT(*) FROM query_metrics WHERE timestamp > ? AND success = TRUE", (cutoff,)
        ).fetchone()[0]

        avg_latency = self.conn.execute(
            "SELECT AVG(latency_ms) FROM query_metrics WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0] or 0

        by_mode = self.conn.execute("""
            SELECT rag_mode, COUNT(*) FROM query_metrics
            WHERE timestamp > ? GROUP BY rag_mode
        """, (cutoff,)).fetchall()

        return {
            "total_queries": total,
            "success_rate": (success / total * 100) if total > 0 else 0,
            "avg_latency_ms": round(avg_latency, 2),
            "queries_by_mode": dict(by_mode),
            "period_days": days
        }

    def display_dashboard(self, days: int = 7):
        """Display dashboard."""
        stats = self.get_stats(days)

        table = Table(title=f"[header]Metrics (Last {days} days)[/header]")
        table.add_column("Metric", style="info")
        table.add_column("Value", style="success")

        table.add_row("Total Queries", str(stats["total_queries"]))
        table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
        table.add_row("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")

        for mode, count in stats["queries_by_mode"].items():
            table.add_row(f"Mode: {mode}", str(count))

        console.print(table)

    def close(self):
        """Close connection."""
        self.conn.close()


_metrics: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """Get metrics tracker."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsTracker()
    return _metrics
```

### Files to Modify

#### Update `src/utils/rag_utils.py`

Add imports:

```python
from src.checkpointing.sqlite_saver import get_checkpointer
from src.observability.langsmith_tracer import setup_langsmith_tracing
from src.observability.metrics import get_metrics_tracker
import time

# At module level
setup_langsmith_tracing()
```

Replace `MemorySaver()` with `get_checkpointer()`:

```python
# OLD: memory = MemorySaver()
# NEW:
memory = get_checkpointer()
```

Add metrics to `interactive_cli`:

```python
def interactive_cli() -> None:
    metrics = get_metrics_tracker()
    # ...

    while True:
        query = console.input("[info]You: [/info]")
        if query.lower() == "exit":
            break

        start_time = time.time()
        success = True
        error = None

        try:
            # ... query logic ...
        except Exception as e:
            success = False
            error = str(e)

        latency_ms = int((time.time() - start_time) * 1000)

        metrics.log_query(
            query=query,
            latency_ms=latency_ms,
            rag_mode=get_rag_mode(),
            session_id=session_id,
            success=success,
            error=error
        )
```

#### Update `src/main.py`

Add metrics dashboard option:

```python
from src.observability.metrics import get_metrics_tracker


def cli_helper():
    # ... existing options ...
    table.add_row("7", "View Metrics: Performance dashboard")
    table.add_row("8", "Exit")


def main():
    # ...
    while True:
        # ...
        elif action == 7:
            metrics = get_metrics_tracker()
            metrics.display_dashboard(days=7)
        elif action == 8:
            console.print("[success]Exiting...[/success]")
            break
```

### Checklist

- [x] Create `src/checkpointing/` directory
- [x] Create `src/checkpointing/__init__.py`
- [x] Create `src/checkpointing/sqlite_saver.py`
- [x] Create `src/observability/` directory
- [x] Create `src/observability/__init__.py`
- [x] Create `src/observability/langsmith_tracer.py`
- [x] Create `src/observability/metrics.py`
- [x] Update `src/utils/rag_utils.py` with checkpointer
- [x] Update `src/main.py` with metrics dashboard
- [x] Install: `pip install langsmith langgraph-checkpoint-sqlite`
- [x] Create tests: `tests/test_sqlite_checkpointer.py`
- [ ] Test session persistence across restarts (manual test pending)
- [ ] Test LangSmith trace capture (requires API key)
- [ ] Test metrics dashboard (manual test pending)
- [ ] Commit: `feat: add persistent memory and observability`

**Status: IN PROGRESS - Code implementation complete, manual testing pending**

### Success Criteria

- Conversations persist after restart
- Session resumption works
- LangSmith captures traces
- Metrics dashboard shows insights
- No performance degradation

---

## File Reference

### New Files Summary (27 total)

| Phase | File | Purpose |
|-------|------|---------|
| 2 | `src/utils/docling_loader.py` | Docling parser |
| 2 | `src/utils/semantic_splitter.py` | Semantic chunking |
| 3 | `src/evaluation/__init__.py` | Package init |
| 3 | `src/evaluation/rag_evaluator.py` | Relevance grading |
| 3 | `src/tools/__init__.py` | Package init |
| 3 | `src/tools/web_search.py` | Web search |
| 3 | `src/graphs/__init__.py` | Package init |
| 3 | `src/graphs/corrective_rag.py` | CRAG |
| 3 | `src/graphs/adaptive_rag.py` | Adaptive RAG |
| 4 | `src/agents/__init__.py` | Package init |
| 4 | `src/agents/retriever_agent.py` | Retriever |
| 4 | `src/agents/summarizer_agent.py` | Summarizer |
| 4 | `src/agents/critic_agent.py` | Critic |
| 4 | `src/agents/supervisor.py` | Supervisor |
| 5 | `src/checkpointing/__init__.py` | Package init |
| 5 | `src/checkpointing/sqlite_saver.py` | Persistent memory |
| 5 | `src/observability/__init__.py` | Package init |
| 5 | `src/observability/langsmith_tracer.py` | Tracing |
| 5 | `src/observability/metrics.py` | Metrics |

### Modified Files Summary (8 total)

| Phase | File | Changes |
|-------|------|---------|
| 1 | `pyproject.toml` | Python 3.10+ |
| 1 | `requirements.txt` | Dependencies |
| 1-5 | `src/utils/rag_utils.py` | Modes, checkpointer |
| 2 | `src/utils/document_loader.py` | Docling, chunking |
| 1 | `src/models/llm.py` | Imports |
| 1 | `src/models/embeddings.py` | Imports |
| 5 | `src/main.py` | Metrics |
| All | `.env` | Configuration |

---

## Environment Configuration

### Complete `.env` Template

```bash
# =============================================================================
# DocBases v2.0 Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_API_BASE=http://localhost:11434

# For OpenAI:
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4
# OPENAI_API_KEY=sk-...

# For Google:
# LLM_PROVIDER=google
# LLM_MODEL=gemini-pro
# GOOGLE_API_KEY=...

# -----------------------------------------------------------------------------
# Embedding Configuration
# -----------------------------------------------------------------------------
EMB_PROVIDER=ollama
EMB_MODEL=nomic-embed-text
EMB_API_BASE=http://localhost:11434

# -----------------------------------------------------------------------------
# Document Processing (Phase 2)
# -----------------------------------------------------------------------------
USE_DOCLING=true
CHUNKING_STRATEGY=semantic  # Options: recursive, semantic

# -----------------------------------------------------------------------------
# RAG Mode (Phase 3-4)
# -----------------------------------------------------------------------------
RAG_MODE=adaptive  # Options: basic, corrective, adaptive, multi_agent

# -----------------------------------------------------------------------------
# Persistent Memory (Phase 5)
# -----------------------------------------------------------------------------
USE_PERSISTENT_MEMORY=true
CHECKPOINT_DB_PATH=knowledges/checkpoints.db
METRICS_DB_PATH=knowledges/metrics.db

# -----------------------------------------------------------------------------
# Observability (Phase 5)
# -----------------------------------------------------------------------------
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=doc-bases
```

---

## Testing Strategy

### Test Files to Create

```
tests/
├── test_migration_phase1.py      # Phase 1 dependency tests
├── test_docling_integration.py   # Phase 2 docling tests
├── test_semantic_splitter.py     # Phase 2 chunking tests
├── test_corrective_rag.py        # Phase 3 CRAG tests
├── test_adaptive_rag.py          # Phase 3 adaptive tests
├── test_multi_agent.py           # Phase 4 agent tests
├── test_sqlite_checkpointer.py   # Phase 5 persistence tests
└── fixtures/
    └── test_documents.py         # Shared test data
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific phase
pytest tests/test_corrective_rag.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Dependency conflicts | Pin exact versions, test in isolated venv |
| Breaking changes | Feature flags, gradual rollout |
| Performance regression | Benchmark at each phase |
| Data loss | Fresh start, no migration needed |

### Rollback Commands

```bash
# Phase rollback
git checkout HEAD~1

# Full rollback
git checkout main
pip install -r requirements.txt.backup
```

---

## Resources

### Documentation

- [LangChain 1.x Docs](https://python.langchain.com/docs/)
- [LangGraph 1.0 Docs](https://langchain-ai.github.io/langgraph/)
- [Docling Docs](https://docling-project.github.io/docling/)
- [ChromaDB 1.4 Docs](https://docs.trychroma.com/)
- [LangSmith Docs](https://docs.smith.langchain.com/)

### References

- [LangChain Changelog](https://changelog.langchain.com/)
- [LangGraph 1.0 Release](https://blog.langchain.com/langchain-langgraph-1dot0/)
- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)
- [RAG Architectures Guide](https://humanloop.com/blog/rag-architectures)

---

## Contact

For questions about this plan, contact the original author or refer to the project repository.

---

*Last updated: January 2026*
*Plan version: 2.0*
