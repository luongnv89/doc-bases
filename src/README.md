# Source Code Guide

Detailed documentation of all source components and modules.

## Directory Structure

```
src/
├── agents/              # Multi-agent system
│   ├── __init__.py
│   ├── retriever_agent.py      # Document retrieval agent
│   ├── summarizer_agent.py     # Summarization agent
│   ├── critic_agent.py         # Quality validation agent
│   └── supervisor.py           # Agent orchestration
├── checkpointing/       # Persistent memory
│   ├── __init__.py
│   └── sqlite_saver.py         # SQLite checkpointer
├── evaluation/          # RAG quality evaluation
│   ├── __init__.py
│   └── rag_evaluator.py        # Relevance grading & hallucination detection
├── graphs/              # LangGraph RAG workflows
│   ├── __init__.py
│   ├── corrective_rag.py       # CRAG implementation
│   └── adaptive_rag.py         # Adaptive RAG implementation
├── models/              # LLM and Embedding interfaces
│   ├── __init__.py
│   ├── llm.py                  # LLM provider abstraction
│   └── embeddings.py           # Embedding model abstraction
├── observability/       # Monitoring and tracing
│   ├── __init__.py
│   ├── langsmith_tracer.py     # LangSmith integration
│   └── metrics.py              # Metrics tracking & dashboard
├── tools/               # Specialized tools
│   ├── __init__.py
│   └── web_search.py           # Web search tool
├── utils/               # Utilities and helpers
│   ├── __init__.py
│   ├── document_loader.py      # Document loading from various sources
│   ├── docling_loader.py       # Advanced PDF/document parsing
│   ├── rag_utils.py            # RAG setup and utilities
│   ├── semantic_splitter.py    # Semantic text chunking
│   ├── logger.py               # Logging configuration
│   └── utilities.py            # General utilities
├── main.py              # CLI entry point
└── __init__.py
```

## Module Details

### agents/

Multi-agent system for sophisticated RAG workflows.

#### RetrieverAgent (`retriever_agent.py`)

Specializes in document retrieval with query decomposition.

**Capabilities**:
- Vector similarity search
- Query decomposition (breaking complex questions into sub-queries)
- Document ranking and filtering

**Tools**:
- `retrieve_documents(query, k)`: Vector search with configurable document count
- `decompose_query(complex_query)`: Generate related sub-queries

**Usage**:
```python
from src.agents.retriever_agent import RetrieverAgent

agent = RetrieverAgent(vectorstore, llm)
result = await agent.retrieve("What is RAG?")
```

#### SummarizerAgent (`summarizer_agent.py`)

Specializes in generating query-focused summaries.

**Capabilities**:
- Query-focused summarization
- Map-reduce for large document sets
- Iterative refinement

**Usage**:
```python
from src.agents.summarizer_agent import SummarizerAgent

agent = SummarizerAgent(llm)
summary = await agent.summarize(documents, "What is RAG?")
```

#### CriticAgent (`critic_agent.py`)

Validates answer quality and suggests improvements.

**Outputs**:
```python
Critique(
    accuracy_score: float,      # 0.0-1.0
    completeness_score: float,  # 0.0-1.0
    clarity_score: float,       # 0.0-1.0
    issues: List[str],          # Problems found
    suggestions: List[str],     # Improvements
    needs_revision: bool        # Should revise?
)
```

**Usage**:
```python
from src.agents.critic_agent import CriticAgent

agent = CriticAgent(llm)
critique = await agent.critique(question, answer, documents)
```

#### MultiAgentSupervisor (`supervisor.py`)

Orchestrates retriever, summarizer, and critic agents.

**Workflow**:
1. Supervisor decides next agent
2. Agent executes task
3. Repeats until task complete or max iterations reached

**Configuration**:
- `max_iterations`: Default 3 (prevents infinite loops)

**Usage**:
```python
from src.agents.supervisor import MultiAgentSupervisor

supervisor = MultiAgentSupervisor(vectorstore, llm, max_iterations=3)
answer = await supervisor.invoke("Your question here")
```

### checkpointing/

Persistent memory backend for session continuity.

#### PersistentCheckpointer (`sqlite_saver.py`)

SQLite-based session persistence.

**Features**:
- Thread-safe with WAL mode
- Automatic cleanup of old sessions
- Session listing and retrieval

**Configuration**:
```env
USE_PERSISTENT_MEMORY=true
CHECKPOINT_DB_PATH=knowledges/checkpoints.db
```

**Usage**:
```python
from src.checkpointing.sqlite_saver import get_checkpointer

checkpointer = get_checkpointer()
# Used automatically by RAG graphs via config
```

### evaluation/

RAG quality evaluation components.

#### RAGEvaluator (`rag_evaluator.py`)

Grades retrieval and generation quality.

**Methods**:

1. **Relevance Grading**:
```python
score = await evaluator.grade_relevance(question, document)
# Returns: RelevanceScore(score: 0.0-1.0, reasoning: str)
```

2. **Hallucination Detection**:
```python
check = await evaluator.check_hallucination(documents, answer)
# Returns: HallucinationCheck(is_grounded: bool, unsupported_claims: List[str])
```

3. **Batch Grading**:
```python
relevant, irrelevant = await evaluator.grade_documents_batch(
    question, documents, threshold=0.5
)
```

**Configuration**:
```python
threshold: float = 0.5  # Relevance cutoff
```

### graphs/

LangGraph workflow implementations for different RAG patterns.

#### CorrectiveRAGGraph (`corrective_rag.py`)

Self-correcting RAG with validation.

**Workflow**:
```
Query → Retrieve → Grade → [If insufficient] Web Search → Generate → Check Hallucination
```

**State**:
```python
class CRAGState(TypedDict):
    messages: List[BaseMessage]
    question: str
    documents: List[Document]
    relevant_docs: List[Document]
    web_search_needed: bool
    web_results: List[Document]
    generation: str
    is_grounded: bool
```

**Configuration**:
```python
relevance_threshold: float = 0.5      # Relevance cutoff
min_relevant_docs: int = 1             # Min docs before fallback
```

**Usage**:
```python
from src.graphs.corrective_rag import CorrectiveRAGGraph

crag = CorrectiveRAGGraph(vectorstore, llm)
answer = await crag.invoke("Your question")
```

#### AdaptiveRAGGraph (`adaptive_rag.py`)

Query-routing RAG with strategy selection.

**Query Classification**:
- **simple**: Factual questions, fast retrieval (K=3)
- **complex**: Multi-step reasoning with sub-queries (K=6)
- **web**: External information, includes web search

**Usage**:
```python
from src.graphs.adaptive_rag import AdaptiveRAGGraph

arag = AdaptiveRAGGraph(vectorstore, llm)
answer = await arag.invoke("Your question")
```

### models/

LLM and embedding model abstractions.

#### llm.py

LLM provider interface.

**Function**:
```python
def get_llm_model() -> BaseChatModel:
    """Get configured LLM based on environment variables."""

def get_available_models() -> List[str]:
    """List available models for current provider."""
```

**Supported Providers**:
- Ollama (local)
- OpenAI (GPT-3.5, GPT-4)
- Google (Gemini)
- Groq (fast inference)

**Configuration**:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_API_BASE=http://localhost:11434
LLM_TEMPERATURE=0.7
```

#### embeddings.py

Embedding model interface.

**Function**:
```python
def get_embedding_model() -> Embeddings:
    """Get configured embedding model."""

def get_embedding_dimension() -> int:
    """Get embedding vector dimension."""
```

**Supported Providers**:
- Ollama (local, 384-dim)
- OpenAI (ada-002, 1536-dim)
- Google (Embedding Gecko, 768-dim)

**Configuration**:
```env
EMB_PROVIDER=ollama
EMB_MODEL=nomic-embed-text
EMB_API_BASE=http://localhost:11434
```

### observability/

Monitoring and tracing components.

#### langsmith_tracer.py

LangSmith distributed tracing integration.

**Function**:
```python
def setup_langsmith_tracing(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    enabled: Optional[bool] = None
) -> bool:
    """Configure LangSmith tracing."""
```

**Traces Captured**:
- LLM calls with input/output
- Tool execution
- Agent decisions
- Retrieval operations

**Configuration**:
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls_...
LANGSMITH_PROJECT=doc-bases
```

**Access**: [smith.langchain.com](https://smith.langchain.com)

#### metrics.py

Performance metrics collection and dashboard.

**Tracked Metrics**:
- Query latency (ms)
- Retrieval document count
- Success/failure rate
- RAG mode distribution
- Error messages

**Database**: SQLite at `knowledges/metrics.db`

**Usage**:
```python
from src.observability.metrics import get_metrics_tracker

metrics = get_metrics_tracker()
metrics.log_query(
    query="Your question",
    latency_ms=1240,
    rag_mode="adaptive",
    success=True
)

# View stats
stats = metrics.get_stats(days=7)
metrics.display_dashboard()
```

### tools/

Specialized tool implementations.

#### web_search.py

Web search using DuckDuckGo (no API key required).

**Function**:
```python
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search web for additional context."""

def web_search_to_documents(
    query: str,
    max_results: int = 3
) -> List[Document]:
    """Perform web search, return as LangChain Documents."""
```

**Usage**:
- Corrective RAG fallback when retrieval insufficient
- Adaptive RAG for web routing
- Multi-agent for additional context

**Configuration**:
```python
max_results: int = 3    # Results per query
```

### utils/

Utility modules for document processing and RAG setup.

#### document_loader.py

Multi-source document loading pipeline.

**Sources**:
1. **GitHub Repository**
   ```python
   docs = loader.load_from_repository(
       repo_url="https://github.com/user/repo",
       repo_name="repo-name",
       file_extensions=[".md", ".txt"]
   )
   ```

2. **Local File**
   ```python
   docs = loader.load_from_file("/path/to/file.pdf")
   ```

3. **Local Folder**
   ```python
   docs = loader.load_from_folder(
       folder_path="/path/to/docs",
       file_extensions=[".pdf", ".md"]
   )
   ```

4. **Website**
   ```python
   docs = loader.load_from_website("https://example.com")
   ```

5. **Download URL**
   ```python
   docs = loader.load_from_download("https://example.com/doc.pdf")
   ```

**Processing Pipeline**:
1. Load from source
2. Parse (Docling or fallback)
3. Split into chunks
4. Generate embeddings
5. Store in ChromaDB

#### docling_loader.py

Advanced document parsing using Docling library.

**Supported Formats**:
- PDF (with table extraction)
- DOCX (Word documents)
- PPTX (Presentations)
- XLSX (Spreadsheets)
- HTML
- Markdown
- Images

**Features**:
- Table structure preservation
- Formula recognition
- Layout analysis
- Multi-page support

**Configuration**:
```env
USE_DOCLING=true
```

**Usage**:
```python
from src.utils.docling_loader import DoclingDocumentLoader

docling = DoclingDocumentLoader()
docs = docling.load_document("/path/to/file.pdf")
```

#### semantic_splitter.py

Semantic text chunking using embedding similarity.

**Strategy**: Embedding-based boundaries instead of character counts

**Advantages**:
- Preserves semantic coherence
- Reduces fragmentation
- Better retrieval relevance

**Configuration**:
```env
CHUNKING_STRATEGY=semantic
```

**Breakpoint Types**:
- `percentile` (default, 95th percentile)
- `standard_deviation`
- `interquartile`

**Usage**:
```python
from src.utils.semantic_splitter import SemanticDocumentSplitter

splitter = SemanticDocumentSplitter()
chunks = splitter.split_documents(documents)
```

#### rag_utils.py

Core RAG setup and utilities.

**Main Functions**:
```python
def setup_rag(documents, knowledge_base_name, llm=None):
    """Setup RAG with selected mode."""

def load_rag_chain(knowledge_base_name, llm=None):
    """Load existing RAG."""

def query_rag(agent, query, session_id=None):
    """Execute single query."""

def delete_knowledge_base(knowledge_base_name):
    """Delete knowledge base."""

def list_knowledge_bases():
    """List all knowledge bases."""

def interactive_cli():
    """Start interactive CLI."""
```

**RAG Mode Routing**:
- `basic`: Simple retrieval + generation
- `corrective`: CRAG with validation
- `adaptive`: Query routing
- `multi_agent`: Supervisor orchestration

#### logger.py

Centralized logging configuration.

**Usage**:
```python
from src.utils.logger import get_logger

logger = get_logger()
logger.info("Starting knowledge base setup")
logger.warning("Low relevance score detected")
logger.error("Failed to retrieve documents")
logger.debug("Detailed execution trace")
```

**Levels**:
- `DEBUG`: Detailed execution traces
- `INFO`: Normal operations
- `WARNING`: Potentially problematic situations
- `ERROR`: Error conditions

#### utilities.py

General utility functions.

**Examples**:
- String manipulation
- Type checking
- Data validation

### main.py

CLI entry point and menu system.

**Features**:
1. Setup RAG from documents
2. Interactive CLI for querying
3. Knowledge base management
4. Metrics dashboard
5. Log toggling

**Usage**:
```bash
python src/main.py
```

## Data Flow

### Document Ingestion

```
Source (Repo/File/Web)
    ↓
DocumentLoader.load_*()
    ↓
[Parse: Docling/Fallback]
    ↓
[Split: Semantic/Recursive]
    ↓
get_embedding_model()
    ↓
ChromaDB.add_documents()
```

### Query Processing

```
Query Input
    ↓
RAG Mode Selection (RAG_MODE env var)
    ↓
[Basic/CRAG/Adaptive/MultiAgent]
    ↓
Vectorstore.similarity_search()
    ↓
[Evaluation: Grading/Hallucination Check]
    ↓
LLM.generate()
    ↓
Log Metrics
    ↓
Answer Output
```

## Best Practices

### Adding New Components

1. **Create module** in appropriate subdirectory
2. **Add to __init__.py** for clean imports
3. **Document thoroughly** with docstrings
4. **Write tests** in tests/ directory
5. **Update README.md** with description

### Dependency Injection

```python
# ✓ Good: Dependencies injected
def __init__(self, vectorstore, llm, checkpointer=None):
    self.vectorstore = vectorstore
    self.llm = llm

# ✗ Avoid: Hard-coded dependencies
def __init__(self):
    self.vectorstore = Chroma(...)
    self.llm = get_llm_model()
```

### Error Handling

```python
from src.utils.logger import get_logger

logger = get_logger()

try:
    result = operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Handle gracefully
    return default_value
```

### Type Hints

```python
from typing import List, Optional, Dict
from langchain_core.documents import Document

def process_documents(
    documents: List[Document],
    config: Optional[Dict[str, str]] = None
) -> List[Document]:
    """Process documents."""
```

---

**Last Updated**: January 2026
**Source Version**: 2.0
