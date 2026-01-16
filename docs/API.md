# API Reference

Complete reference for DocBases components, configuration, and interfaces.

## Environment Variables

### LLM Configuration

```env
# Provider selection
LLM_PROVIDER=ollama              # Options: ollama, openai, google, groq
LLM_MODEL=llama3.1:8b            # Model name/ID

# Ollama
LLM_API_BASE=http://localhost:11434

# OpenAI
OPENAI_API_KEY=sk-...

# Google
GOOGLE_API_KEY=...

# Groq
GROQ_API_KEY=...
```

**Default**: Ollama llama3.1:8b (requires local Ollama server)

### Embedding Configuration

```env
# Provider selection
EMB_PROVIDER=ollama              # Options: ollama, openai, google
EMB_MODEL=nomic-embed-text       # Model name/ID

# Ollama
EMB_API_BASE=http://localhost:11434

# OpenAI
OPENAI_API_KEY=sk-...

# Google
GOOGLE_API_KEY=...
```

**Default**: Ollama nomic-embed-text

### Document Processing

```env
# Docling parser
USE_DOCLING=false                # Enable advanced document parsing

# Chunking strategy
CHUNKING_STRATEGY=recursive       # Options: recursive, semantic

# Chunk sizes
CHUNK_SIZE=1000                   # Characters per chunk
CHUNK_OVERLAP=200                 # Overlap between chunks
```

### RAG Mode

```env
# Active RAG implementation
RAG_MODE=basic                    # Options: basic, corrective, adaptive, multi_agent
```

**Modes**:
- `basic`: Fast, simple retrieval + generation
- `corrective`: Relevance grading + web search fallback
- `adaptive`: Query routing (simple/complex/web)
- `multi_agent`: Supervisor with specialized agents

### Persistence & Memory

```env
# Persistent memory backend
USE_PERSISTENT_MEMORY=true
CHECKPOINT_DB_PATH=knowledges/checkpoints.db

# Storage configuration
KNOWLEDGE_BASE_DIR=knowledges    # Directory for vector stores
LOGS_DIR=logs                    # Directory for application logs
TEMPS_DIR=temps                  # Directory for temporary files
```

### Observability

```env
# LangSmith tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=ls_...
LANGSMITH_PROJECT=doc-bases

# Metrics tracking
METRICS_DB_PATH=knowledges/metrics.db
LOG_LEVEL=INFO                   # Options: DEBUG, INFO, WARNING, ERROR
```

## Core Classes

### Document Loader (`src/utils/document_loader.py`)

```python
class DocumentLoader:
    """Load documents from various sources."""

    def load_from_repository(
        self,
        repo_url: str,
        repo_name: str,
        file_extensions: List[str] = None
    ) -> List[Document]:
        """Load documents from GitHub repository."""

    def load_from_file(
        self,
        file_path: str
    ) -> List[Document]:
        """Load single local file."""

    def load_from_folder(
        self,
        folder_path: str,
        file_extensions: List[str] = None
    ) -> List[Document]:
        """Load all files from folder recursively."""

    def load_from_website(
        self,
        website_url: str
    ) -> List[Document]:
        """Scrape and load website content."""

    def load_from_download(
        self,
        download_url: str
    ) -> List[Document]:
        """Download and load file."""
```

**Example Usage**:
```python
from src.utils.document_loader import DocumentLoader

loader = DocumentLoader()

# From GitHub repository
docs = loader.load_from_repository(
    repo_url="https://github.com/user/repo",
    repo_name="my-repo",
    file_extensions=[".md", ".txt"]
)

# From local folder
docs = loader.load_from_folder(
    folder_path="/path/to/docs",
    file_extensions=[".pdf"]
)
```

### RAG Utilities (`src/utils/rag_utils.py`)

```python
def setup_rag(
    documents: List[Document],
    knowledge_base_name: str,
    llm=None
) -> Any:
    """Setup RAG system with selected mode."""
    # Returns appropriate agent (ReAct, CRAG, Adaptive, or Supervisor)

def load_rag_chain(knowledge_base_name: str, llm=None) -> Any:
    """Load existing RAG for a knowledge base."""

def interactive_cli() -> None:
    """Start interactive CLI session."""

def query_rag(agent, query: str, session_id: str = None) -> str:
    """Execute single query against RAG."""

def delete_knowledge_base(knowledge_base_name: str) -> bool:
    """Delete knowledge base and storage."""

def list_knowledge_bases() -> List[str]:
    """List all available knowledge bases."""
```

### LLM Interface (`src/models/llm.py`)

```python
def get_llm_model() -> BaseChatModel:
    """Get configured LLM model."""
    # Returns langchain ChatModel instance
    # Configured via LLM_PROVIDER, LLM_MODEL, etc.

def get_available_models() -> List[str]:
    """List available models for current provider."""
```

### Embeddings Interface (`src/models/embeddings.py`)

```python
def get_embedding_model() -> Embeddings:
    """Get configured embedding model."""
    # Returns langchain Embeddings instance
    # Configured via EMB_PROVIDER, EMB_MODEL, etc.

def get_embedding_dimension() -> int:
    """Get embedding vector dimension."""
```

## RAG Graph Classes

### Corrective RAG (`src/graphs/corrective_rag.py`)

```python
class CorrectiveRAGGraph:
    """Self-corrective RAG with quality validation."""

    def __init__(
        self,
        vectorstore: Chroma,
        llm: BaseChatModel,
        checkpointer=None,
        relevance_threshold: float = 0.5,
        min_relevant_docs: int = 1
    ):
        """Initialize Corrective RAG."""

    async def invoke(
        self,
        question: str,
        config: dict = None
    ) -> str:
        """Execute CRAG workflow and return answer."""

# Workflow state
class CRAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    documents: List[Document]
    relevant_docs: List[Document]
    web_search_needed: bool
    web_results: List[Document]
    generation: str
    is_grounded: bool
```

**Usage**:
```python
from src.graphs.corrective_rag import CorrectiveRAGGraph

crag = CorrectiveRAGGraph(vectorstore, llm)
answer = await crag.invoke("What is RAG?")
```

### Adaptive RAG (`src/graphs/adaptive_rag.py`)

```python
class AdaptiveRAGGraph:
    """Query-routing RAG with strategy selection."""

    def __init__(
        self,
        vectorstore: Chroma,
        llm: BaseChatModel,
        checkpointer=None
    ):
        """Initialize Adaptive RAG."""

    async def invoke(
        self,
        question: str,
        config: dict = None
    ) -> str:
        """Execute Adaptive RAG workflow."""

# Query types
class QueryType(Literal):
    simple: str      # Direct factual lookup
    complex: str     # Multi-step reasoning
    web: str         # External information
```

**Configuration**:
```python
# Environment controls routing
# Simple: K=3 retrieval
# Complex: K=6 + sub-queries
# Web: local + web search
```

## Agent Classes

### Multi-Agent Supervisor (`src/agents/supervisor.py`)

```python
class MultiAgentSupervisor:
    """Orchestrates specialized agents."""

    def __init__(
        self,
        vectorstore: Chroma,
        llm: BaseChatModel,
        checkpointer=None,
        max_iterations: int = 3
    ):
        """Initialize supervisor."""

    async def invoke(
        self,
        question: str,
        config: dict = None
    ) -> str:
        """Execute multi-agent workflow."""

class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    next_agent: Literal["retriever", "summarizer", "generator", "critic", "END"]
    documents: List[Document]
    summary: str
    answer: str
    critique: dict
    iteration: int
    max_iterations: int
```

### Retriever Agent (`src/agents/retriever_agent.py`)

```python
class RetrieverAgent:
    """Specialized in document retrieval."""

    async def retrieve(
        self,
        question: str,
        config: dict = None
    ) -> dict:
        """Execute retrieval workflow."""
```

**Tools**:
- `retrieve_documents(query, k)`: Vector search
- `decompose_query(complex_query)`: Sub-query generation

### Summarizer Agent (`src/agents/summarizer_agent.py`)

```python
class SummarizerAgent:
    """Specialized in summarization."""

    async def summarize(
        self,
        documents: List[Document],
        question: str
    ) -> str:
        """Generate query-focused summary."""

    async def map_reduce_summarize(
        self,
        documents: List[Document],
        question: str
    ) -> str:
        """Map-reduce for large document sets."""
```

### Critic Agent (`src/agents/critic_agent.py`)

```python
class CriticAgent:
    """Specialized in answer validation."""

    async def critique(
        self,
        question: str,
        answer: str,
        documents: str
    ) -> Critique:
        """Evaluate answer quality."""

    async def suggest_improvements(
        self,
        answer: str,
        critique: Critique
    ) -> str:
        """Generate improved version."""

class Critique(BaseModel):
    accuracy_score: float        # 0.0-1.0
    completeness_score: float    # 0.0-1.0
    clarity_score: float         # 0.0-1.0
    issues: List[str]            # Problems found
    suggestions: List[str]       # Improvements
    needs_revision: bool         # Revise?
```

## Evaluation Classes

### RAG Evaluator (`src/evaluation/rag_evaluator.py`)

```python
class RAGEvaluator:
    """Evaluate RAG quality."""

    async def grade_relevance(
        self,
        question: str,
        document: str
    ) -> RelevanceScore:
        """Grade document relevance (0.0-1.0)."""

    async def check_hallucination(
        self,
        documents: str,
        answer: str
    ) -> HallucinationCheck:
        """Check if answer is grounded."""

    async def grade_documents_batch(
        self,
        question: str,
        documents: List[str],
        threshold: float = 0.5
    ) -> Tuple[List[str], List[str]]:
        """Grade multiple documents, return (relevant, irrelevant)."""

class RelevanceScore(BaseModel):
    score: float           # 0.0-1.0
    reasoning: str         # Why this score

class HallucinationCheck(BaseModel):
    is_grounded: bool           # Grounded in docs?
    unsupported_claims: List[str]  # Unsupported statements
```

## Tool Classes

### Web Search (`src/tools/web_search.py`)

```python
@tool
def web_search(
    query: str,
    max_results: int = 3
) -> str:
    """Search web with DuckDuckGo."""

def web_search_to_documents(
    query: str,
    max_results: int = 3
) -> List[Document]:
    """Perform web search, return as Documents."""
```

## Persistence Classes

### SQLite Checkpointer (`src/checkpointing/sqlite_saver.py`)

```python
class PersistentCheckpointer:
    """SQLite-backed session persistence."""

    def __init__(self, db_path: str = None):
        """Initialize checkpointer."""

    def get_saver(self) -> SqliteSaver:
        """Get LangGraph saver."""

    def list_sessions(self, limit: int = 10) -> list:
        """List recent sessions."""

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete sessions older than N days."""

    def delete_session(self, thread_id: str) -> bool:
        """Delete specific session."""

    def close(self):
        """Close database connection."""

def get_checkpointer(use_persistent: bool = None):
    """Get checkpointer (persistent or memory)."""
```

## Observability Classes

### Metrics Tracker (`src/observability/metrics.py`)

```python
class MetricsTracker:
    """Track RAG performance metrics."""

    def __init__(self, db_path: str = None):
        """Initialize metrics database."""

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
        """Log single query execution."""

    def get_stats(self, days: int = 7) -> Dict:
        """Get aggregated metrics."""

    def display_dashboard(self, days: int = 7):
        """Display metrics dashboard."""

def get_metrics_tracker() -> MetricsTracker:
    """Get singleton metrics tracker."""
```

**Returns**:
```python
{
    "total_queries": 1245,
    "success_rate": 98.5,      # Percentage
    "avg_latency_ms": 1240,
    "queries_by_mode": {
        "basic": 200,
        "corrective": 412,
        "adaptive": 623
    },
    "period_days": 7
}
```

### LangSmith Tracer (`src/observability/langsmith_tracer.py`)

```python
def setup_langsmith_tracing(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    enabled: Optional[bool] = None
) -> bool:
    """Configure LangSmith tracing."""
```

## Common Patterns

### Creating a Custom RAG Mode

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class CustomRAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    documents: List[Document]
    generation: str

class CustomRAGGraph:
    def __init__(self, vectorstore, llm, checkpointer=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.graph = self._build_graph()

    async def retrieve(self, state: CustomRAGState) -> CustomRAGState:
        # Your retrieval logic
        return state

    async def generate(self, state: CustomRAGState) -> CustomRAGState:
        # Your generation logic
        return state

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CustomRAGState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile(checkpointer=self.checkpointer)

    async def invoke(self, question: str, config=None) -> str:
        initial_state = {
            "messages": [],
            "question": question,
            "documents": [],
            "generation": ""
        }
        result = await self.graph.ainvoke(initial_state, config=config)
        return result["generation"]
```

### Logging

```python
from src.utils.logger import get_logger

logger = get_logger()

logger.info("Processing document")
logger.warning("Low confidence score")
logger.error("Failed to retrieve documents")
logger.debug("Detailed operation trace")
```

### Creating a Custom Tool

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str, param: int = 5) -> str:
    """
    Description of what tool does.

    Args:
        query: Input query string
        param: Optional parameter

    Returns:
        Result as string
    """
    # Implementation
    return result
```

## Error Handling

### Common Exceptions

```python
# Document loading errors
DocumentLoaderError          # File not found, unsupported format
RepositoryError              # Git clone failed

# RAG errors
RetrievalError               # Vector search failed
GenerationError              # LLM call failed
EvaluationError              # Grading failed

# Configuration errors
ConfigurationError           # Invalid environment variables
ModelNotFoundError           # LLM/embedding model not available
```

### Error Handling Pattern

```python
from src.utils.logger import get_logger

logger = get_logger()

try:
    documents = loader.load_from_file(file_path)
except DocumentLoaderError as e:
    logger.error(f"Failed to load document: {e}")
    # Fallback strategy
    documents = []
```

---

**Last Updated**: January 2026
**API Version**: 2.0
