# DocBases

**Intelligent Answers from Any Document**

An advanced document querying system that combines agentic workflows, RAG patterns (Corrective, Adaptive, & Multi-Agent), and semantic intelligence to provide accurate, context-aware answers from your knowledge bases.

### One-Line Usage

```bash
docb query interactive
```

Query your documents with natural language in an interactive, conversational interface. No configuration needed - just ask!

## Quick Start

> 🚀 **New to DocBases?** Start with **[INSTALLATION.md](INSTALLATION.md)** - Get running locally in 5 minutes with no API keys needed!

### Installation (5 Minutes)

```bash
# Clone repository
git clone https://github.com/luongnv89/doc-bases.git
cd doc-bases

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Install as editable package (enables 'docb' CLI)
pip install -e .
```

### Prerequisites

DocBases is configured by default to use **Ollama** (local, free, no API keys). To use it:

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Run Ollama**: `ollama serve` (in separate terminal)
3. **Pull models**:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

See **[INSTALLATION.md](INSTALLATION.md)** for quick setup, or **[docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)** for detailed steps and cloud provider alternatives (OpenAI, Google, Groq).

### Basic Usage

#### CLI Commands (Recommended)

```bash
# See all available commands
docb --help

# Configure settings
docb config list
docb config set rag.mode corrective

# Check health
docb health check

# Manage knowledge bases
docb kb list
docb kb add file /path/to/document.pdf --name "My KB"
docb kb info "My KB"

# Query knowledge base
docb query interactive                           # Interactive mode
docb query single --query "Your question?"       # Single query
docb query batch queries.txt --output results.json  # Batch queries

# View version
docb version
```

#### Legacy Menu Mode (Optional)

```bash
# Start with menu interface (for backwards compatibility)
python src/main.py --legacy
```

Then in the legacy menu:
1. **Setup RAG** - Load documents from GitHub repos, local files, websites, or download URLs
2. **Interactive CLI** - Query your knowledge bases with natural language
3. **Manage Knowledge Bases** - List, delete, or switch between bases

## Key Features

### Core RAG Capabilities
- **4 Multi-RAG Modes**:
  - **Basic**: Fast ReAct agent for simple queries (200-800ms)
  - **Corrective (CRAG)**: Self-validating with hallucination detection (600-1600ms)
  - **Adaptive**: Intelligent query routing (simple/complex/web) (500-1700ms)
  - **Multi-Agent**: Specialized agents with iterative refinement (2-6s)

### Document Processing
- **Advanced PDF Parsing**: Docling integration with table extraction, formula recognition, layout analysis
- **Multiple Source Types**: GitHub repos, local files, folders, websites, download URLs
- **Semantic Chunking**: Embedding-based chunking preserves topic coherence vs. character-based splitting
- **Automatic Format Detection**: Auto-detects MIME types (PDF, DOCX, PPTX, XLSX, HTML, Markdown, Images)

### Knowledge Base Management
- **File Change Detection**: Automatically detects added/modified/deleted source files with metadata tracking
- **Multiple Knowledge Bases**: Independent vector stores and sessions per knowledge base
- **Metadata Persistence**: Tracks file hashes, modification times, and source information

### Agent System
- **Specialized Agents**: Retriever, Summarizer, Critic, and Supervisor orchestrator
- **Intelligent Routing**: Query classification and optimal strategy selection
- **Iterative Refinement**: Automatic answer improvement based on critic feedback (up to 3 iterations)
- **Query Decomposition**: Complex queries broken into sub-queries for comprehensive answers

### Retrieval & Generation
- **Flexible LLM Support**: Ollama (local), OpenAI, Google GenAI, Groq, custom providers
- **Multiple Embeddings**: Ollama, OpenAI, Google GenAI embedding models
- **Web Search Integration**: DuckDuckGo fallback for out-of-domain queries (Corrective & Adaptive RAG)
- **Relevance Grading**: LLM-based document relevance scoring and filtering

### Persistence & Observability
- **Persistent Memory**: SQLite-backed conversation sessions (resumable across restarts)
- **Metrics Tracking**: Query latency, retrieval stats, success rates, mode distribution
- **LangSmith Integration**: Optional distributed tracing for debugging and optimization
- **Quality Evaluation**: Hallucination detection, relevance grading, answer validation

## Documentation

- **[🚀 Ollama Setup Guide](docs/OLLAMA_SETUP.md)** - Get started locally in 5 minutes (START HERE!)
- **[CLI Usage Guide](docs/CLI_USAGE.md)** - Complete CLI command reference and examples
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design, RAG patterns, agent workflows
- **[API Reference](docs/API.md)** - Component interfaces, configuration, environment variables
- **[Development Guide](docs/DEVELOPMENT.md)** - Setup, testing, contributing guidelines
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment, scaling considerations
- **[Component Details](src/README.md)** - Module documentation and specific components

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│            Interactive CLI Interface                 │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼──┐   ┌────▼──┐   ┌────▼──┐
   │ Basic │   │Correct.│   │Adaptive│   ┌──────────┐
   │  RAG  │   │  RAG   │   │  RAG   │───│Multi-Agt │
   └────┬──┘   └────┬──┘   └────┬──┘   └──────────┘
        │           │            │
        └───────────┼────────────┘
                    │
        ┌───────────▼────────────┐
        │   Document Retrieval    │
        │   & Embedding Layer     │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │    ChromaDB Vector      │
        │        Store           │
        └────────────────────────┘
```

For detailed architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Configuration

Key environment variables (see `.env.example` for complete list):

```env
# LLM Provider
LLM_PROVIDER=ollama          # Options: ollama, openai, google, groq
LLM_MODEL=llama3.1:8b

# RAG Mode
RAG_MODE=adaptive            # Options: basic, corrective, adaptive, multi_agent

# Document Processing
USE_DOCLING=true
CHUNKING_STRATEGY=semantic   # Options: recursive, semantic

# Persistence
USE_PERSISTENT_MEMORY=true
CHECKPOINT_DB_PATH=knowledges/checkpoints.db

# Observability
LANGSMITH_TRACING=false
```

See [docs/API.md](docs/API.md) for all configuration options.

## Use Cases

### 1. Research & Knowledge Base Queries
Load documentation, research papers, or knowledge bases and ask complex questions with multi-step reasoning.

### 2. Document Analytics
Analyze large document collections with semantic chunking and specialized agents for summarization.

### 3. Real-time Information
Enable Adaptive RAG's web search for current events and external information not in your documents.

### 4. Quality Assurance
Use Corrective RAG's hallucination detection and critic agent to ensure answer accuracy.

## Project Structure

```
doc-bases/
├── src/
│   ├── agents/              # Multi-agent system (Retriever, Summarizer, Critic, Supervisor)
│   ├── graphs/              # LangGraph RAG workflows (Corrective, Adaptive)
│   ├── evaluation/          # RAG quality evaluation (relevance grading, hallucination check)
│   ├── tools/               # Specialized tools (web search)
│   ├── checkpointing/       # Persistent memory (SQLite)
│   ├── observability/       # Monitoring (LangSmith, metrics)
│   ├── models/              # LLM and Embedding interfaces
│   ├── utils/               # Document loading, RAG utilities, logging
│   └── main.py              # CLI entry point
├── docs/                    # Comprehensive documentation
├── tests/                   # Test suite
├── MODERNIZATION_PLAN.md    # Implementation roadmap (5 phases)
└── requirements.txt         # Python dependencies
```

See [src/README.md](src/README.md) for detailed component breakdown.

## Performance

Typical metrics (varies by model & hardware):

| Operation | Latency | Notes |
|-----------|---------|-------|
| Document Ingestion | ~100ms/doc | Includes parsing, embedding, storage |
| Simple Query | 200-800ms | Direct retrieval + generation |
| Complex Query | 1-3s | Multi-step with sub-queries |
| Web Search Fallback | 2-5s | Corrective RAG when needed |
| Critic Validation | +500ms | Multi-Agent mode refinement |

## Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Setting up development environment
- Running tests and linting
- Code structure conventions
- Submitting contributions

## License

MIT License - see [LICENSE](LICENSE) for details

## Support

- **Issues**: [GitHub Issues](https://github.com/luongnv89/doc-bases/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: See CLI interactive mode for walkthrough

---

**Status**: Production-ready v2.0 with advanced RAG patterns and multi-agent orchestration
**Last Updated**: January 2026
