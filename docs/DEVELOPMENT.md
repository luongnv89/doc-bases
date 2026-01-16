# Development Guide

Setup, testing, and contribution guidelines for DocBases.

## Quick Setup with Ollama

**New to development?** Follow the **[Ollama Setup Guide](OLLAMA_SETUP.md)** first for local development (5 minutes, no API keys needed).

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment tool (venv, conda, etc.)
- **Ollama** (for local development) or cloud API key (OpenAI/Google/Groq)

### Step 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/luongnv89/doc-bases.git
cd doc-bases

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
```

### Step 2: Configure Environment

Edit `.env` with your settings:

```env
# Required: LLM provider
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_API_BASE=http://localhost:11434

# Required: Embedding provider
EMB_PROVIDER=ollama
EMB_MODEL=nomic-embed-text
EMB_API_BASE=http://localhost:11434

# Optional: RAG mode (default: basic)
RAG_MODE=adaptive

# Optional: Document processing
USE_DOCLING=false
CHUNKING_STRATEGY=recursive
```

### Step 3: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "from src.utils.rag_utils import setup_rag; print('OK')"
```

## Running the Application

### Development Mode

```bash
# With hot-reload (requires watchdog)
python -m src.main

# Or direct execution
python src/main.py
```

### With Logging

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python src/main.py
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_corrective_rag.py -v

# Single test function
pytest tests/test_corrective_rag.py::test_crag_initialization -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

```
tests/
├── test_corrective_rag.py       # CRAG workflow tests
├── test_adaptive_rag.py          # Adaptive RAG tests
├── test_multi_agent.py           # Multi-agent tests
├── test_document_loader.py       # Document loading
├── test_embeddings.py            # Embedding models
├── test_llm.py                   # LLM interfaces
├── test_rag_utils.py             # RAG utilities
├── test_sqlite_checkpointer.py   # Persistence
├── test_docling_integration.py   # Docling parser
└── fixtures/                     # Shared test data
```

### Writing Tests

Example test structure:

```python
import pytest
from src.graphs.corrective_rag import CorrectiveRAGGraph
from langchain_core.documents import Document

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="RAG is a technique for improving LLM responses.",
            metadata={"source": "test.txt"}
        )
    ]

@pytest.mark.asyncio
async def test_crag_initialization(sample_documents, mock_vectorstore, mock_llm):
    """Test CRAG initialization."""
    crag = CorrectiveRAGGraph(mock_vectorstore, mock_llm)
    assert crag is not None
    assert crag.relevance_threshold == 0.5

@pytest.mark.asyncio
async def test_crag_workflow(mock_vectorstore, mock_llm):
    """Test complete CRAG workflow."""
    crag = CorrectiveRAGGraph(mock_vectorstore, mock_llm)
    answer = await crag.invoke("What is RAG?")
    assert answer is not None
    assert len(answer) > 0
```

### Fixtures for Testing

```python
# conftest.py - shared fixtures
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    mock = AsyncMock()
    mock.ainvoke.return_value.content = "Test response"
    return mock

@pytest.fixture
def mock_vectorstore():
    """Create mock vectorstore."""
    mock = Mock()
    mock.as_retriever.return_value.ainvoke.return_value = [
        Document(page_content="Test content", metadata={"source": "test"})
    ]
    return mock
```

## Code Style & Quality

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Check formatting
black --check src/ tests/

# Configure in pyproject.toml
[tool.black]
line-length = 100
target-version = ['py310', 'py311']
```

### Linting

```bash
# Lint with Ruff
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

### Type Checking

```bash
# Type check with mypy
mypy src/

# Configure in pyproject.toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Configuration** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

## Development Workflow

### 1. Feature Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Push to remote
git push -u origin feature/your-feature-name
```

### 2. Commit Message Format

Follow conventional commits:

```
feat: Add new RAG mode
fix: Correct relevance grading
docs: Update architecture guide
test: Add tests for web search
refactor: Simplify embedding interface
```

### 3. Pull Request

```bash
# Push commits
git push

# Create PR on GitHub
# Include:
# - Description of changes
# - Motivation and context
# - Testing performed
# - Screenshots (if UI changes)
```

### 4. Code Review Checklist

Before submitting:
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Code formatted (`black src/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation updated
- [ ] Commits are clean and descriptive

## Debugging

### Debug Logging

```python
from src.utils.logger import get_logger

logger = get_logger()
logger.debug("Variable value:", extra={"var": variable})
```

### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use pytest
pytest tests/test_file.py -v -s --pdb
```

### LangSmith Debugging

Enable tracing to see detailed execution:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls_...
LANGSMITH_PROJECT=doc-bases-dev
```

Then view traces at [smith.langchain.com](https://smith.langchain.com).

## Performance Profiling

### Timing Analysis

```python
import time
from src.utils.logger import get_logger

logger = get_logger()

start = time.time()
# Operation to profile
result = vectorstore.similarity_search(query, k=5)
duration = time.time() - start

logger.info(f"Search took {duration:.3f}s")
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile specific function
python -m memory_profiler src/main.py
```

## Common Development Tasks

### Adding a New RAG Mode

1. **Create graph class**:
```python
# src/graphs/my_rag.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyRAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

class MyRAGGraph:
    def __init__(self, vectorstore, llm, checkpointer=None):
        # Setup
        pass

    async def invoke(self, question: str) -> str:
        # Implementation
        pass
```

2. **Register in rag_utils.py**:
```python
from src.graphs.my_rag import MyRAGGraph

def setup_rag(documents, knowledge_base_name, llm=None):
    # ...
    if rag_mode == "my_mode":
        agent = MyRAGGraph(vectorstore, llm, checkpointer=memory)
```

3. **Add to .env.example**:
```env
RAG_MODE=my_mode          # New option
```

4. **Add tests**:
```python
# tests/test_my_rag.py
@pytest.mark.asyncio
async def test_my_rag():
    graph = MyRAGGraph(mock_vectorstore, mock_llm)
    answer = await graph.invoke("Test question")
    assert answer
```

### Adding a New Agent

1. **Create agent class**:
```python
# src/agents/my_agent.py
from langchain_core.tools import tool

class MyAgent:
    def __init__(self, llm):
        self.llm = llm

    @tool
    def my_tool(self, input: str) -> str:
        """Tool description."""
        return result
```

2. **Export in __init__.py**:
```python
# src/agents/__init__.py
from src.agents.my_agent import MyAgent
__all__ = ["MyAgent"]
```

3. **Integrate with supervisor** (if needed):
```python
# src/agents/supervisor.py
from src.agents.my_agent import MyAgent

class MultiAgentSupervisor:
    def __init__(self, ...):
        self.my_agent = MyAgent(llm)
```

### Adding Documentation

1. Create markdown file in `docs/`
2. Link from main README.md
3. Use Mermaid for diagrams
4. Keep examples concise and runnable

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Verify virtual environment is activated
which python  # Should point to venv

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**LLM Connection Failed**
```env
# Check LLM_API_BASE
LLM_API_BASE=http://localhost:11434

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

**ChromaDB Lock**
```bash
# Delete lock files if persistent errors
rm -rf knowledges/.chroma_lock

# Use in-memory store temporarily
USE_PERSISTENT_MEMORY=false
```

**Test Failures**
```bash
# Run with verbose output
pytest tests/test_file.py -vv -s

# Run single test
pytest tests/test_file.py::test_function -vv
```

## Documentation Standards

### Module Documentation

```python
"""
Module description.

This module handles document loading from various sources
including repositories, files, and websites.

Usage:
    >>> loader = DocumentLoader()
    >>> docs = loader.load_from_file("/path/to/file")
"""
```

### Function Documentation

```python
def setup_rag(
    documents: List[Document],
    knowledge_base_name: str,
    llm: Optional[BaseChatModel] = None
) -> Any:
    """
    Setup RAG system with selected mode.

    Args:
        documents: List of documents to add to knowledge base
        knowledge_base_name: Name for the knowledge base
        llm: LLM model (uses default if None)

    Returns:
        Configured RAG agent for the selected mode

    Raises:
        ValueError: If knowledge base name is invalid
        DocumentLoaderError: If document processing fails

    Examples:
        >>> from src.utils.document_loader import DocumentLoader
        >>> loader = DocumentLoader()
        >>> docs = loader.load_from_file("doc.pdf")
        >>> agent = setup_rag(docs, "my-kb")
    """
```

## Resources

### Learning Materials
- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [RAG Architectures](https://arxiv.org/abs/2501.09136)

### Tools
- [Black Code Formatter](https://black.readthedocs.io/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [mypy Type Checker](https://www.mypy-lang.org/)
- [pytest Testing](https://docs.pytest.org/)

---

**Last Updated**: January 2026
**Development Version**: 2.0
