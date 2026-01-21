# DocBases v2.0 - Complete Features Guide

**Last Updated**: January 2026
**Version**: 2.0 (Production-Ready)

This document comprehensively covers all DocBases features, use cases, and configuration options.

---

## Table of Contents

1. [RAG Modes](#rag-modes)
2. [Document Processing](#document-processing)
3. [Knowledge Base Management](#knowledge-base-management)
4. [Multi-Agent System](#multi-agent-system)
5. [Retrieval & Generation](#retrieval--generation)
6. [Persistence & Observability](#persistence--observability)
7. [Configuration Guide](#configuration-guide)
8. [Use Cases & Examples](#use-cases--examples)

---

## RAG Modes

DocBases offers 4 distinct RAG (Retrieval-Augmented Generation) modes, each optimized for different scenarios:

### 1. Basic RAG

**Best For**: Simple factual queries, fast responses, minimal latency
**Latency**: 200-800ms
**Complexity**: Low
**API Calls**: 1 (retrieve + generate)

**How It Works**:
```
User Query → Vector Retrieval (K=5) → LLM Generation → Answer
```

**Characteristics**:
- Direct vector similarity search
- No validation or quality checks
- Fastest mode
- Suitable for well-indexed documents

**Configuration**:
```bash
docb config set rag.mode basic
```

**When to Use**:
- ✅ FAQ-style questions
- ✅ Factual lookups
- ✅ High-throughput scenarios
- ✅ When latency is critical
- ❌ Requires validation

---

### 2. Corrective RAG (CRAG)

**Best For**: When answer quality matters, hallucination concerns
**Latency**: 600-1600ms (without web search)
**Complexity**: Medium
**Key Feature**: Self-correcting with validation

**How It Works**:
```
Query
  ↓
Retrieve Documents (K=5)
  ↓
Grade Relevance (LLM-based scoring)
  ↓
[Decision: Are docs relevant enough?]
  ├─ YES (score ≥ 0.5) → Generate Answer
  └─ NO → Web Search → Combine Results
  ↓
Check for Hallucinations
  ↓
Return Validated Answer
```

**Key Components**:

1. **Relevance Grading**:
   - Scores each document (0.0-1.0)
   - Threshold: 0.5 (configurable)
   - Minimum docs: 1 (configurable)

2. **Web Search Fallback**:
   - Triggered if insufficient relevant docs
   - Uses DuckDuckGo (no API key needed)
   - Fetches 3 results
   - Combines with KB context

3. **Hallucination Detection**:
   - Validates if answer is grounded in sources
   - Identifies unsupported claims
   - Flags for user attention

**Configuration**:
```bash
docb config set rag.mode corrective
docb config set corrective_rag.relevance_threshold 0.6
docb config set corrective_rag.min_relevant_docs 2
```

**Environment Variables**:
```env
RAG_MODE=corrective
CORRECTIVE_RAG_RELEVANCE_THRESHOLD=0.5
CORRECTIVE_RAG_MIN_RELEVANT_DOCS=1
```

**When to Use**:
- ✅ Quality-critical applications
- ✅ Mixed KB quality
- ✅ Need current info (web search)
- ✅ Hallucination concerns
- ✅ Technical accuracy important

**Example Workflow**:
```
Q: "What's new in AI in 2026?"
KB: 2024 documentation

Flow:
  1. Retrieve → 5 docs (all 2024 info)
  2. Grade → Scores 0.2 (too old)
  3. Fallback → Web search
  4. Generate → "...2024 background (from KB)... 2026 trends (from web)..."
  5. Validate → Grounded ✅
```

---

### 3. Adaptive RAG

**Best For**: Mixed workloads, cost optimization, intelligent routing
**Latency**: 500-1700ms
**Complexity**: Medium-High
**Key Feature**: Query classification & routing

**How It Works**:
```
Query
  ↓
Classify Query Type
  ├─ Simple (factual, single lookup)
  ├─ Complex (reasoning, multi-step)
  └─ Web (current events, external)
  ↓
Route to Strategy
  ├─ SIMPLE: K=3, direct retrieval
  ├─ COMPLEX: K=6 + sub-query generation
  └─ WEB: K=2 local + web results
  ↓
Generate Answer
  ↓
Return
```

**Classification Examples**:

| Query | Type | Strategy |
|-------|------|----------|
| "What is Python?" | Simple | Direct K=3 retrieval |
| "Compare Python vs Go" | Complex | K=6 + 2 sub-queries |
| "AI trends 2026" | Web | Local K=2 + web search |
| "How do I fix this bug?" | Complex | Decompose + retrieve |

**Sub-Query Generation** (for complex):
```
Original: "How do machine learning models handle imbalanced datasets?"

Generated sub-queries:
1. "What are imbalanced datasets?"
2. "Machine learning techniques for imbalanced data"
3. "Sampling and weighting methods"

Flow:
  - Execute all sub-queries
  - Retrieve K=6 for each
  - Deduplicate results
  - Combine context
  - Generate comprehensive answer
```

**Configuration**:
```bash
docb config set rag.mode adaptive
```

**When to Use**:
- ✅ Varied query complexity
- ✅ Cost-conscious (routes appropriately)
- ✅ Mixed KB + external data
- ✅ Optimal performance per query
- ✅ Auto-adjusts retrieval depth

---

### 4. Multi-Agent RAG

**Best For**: Maximum quality, complex reasoning, iterative refinement
**Latency**: 2-6 seconds
**Complexity**: High
**Key Feature**: Specialized agents with supervision

**Agents**:

| Agent | Role | Capability |
|-------|------|-----------|
| **Retriever** | Document fetching | Query decomposition, relevance awareness |
| **Summarizer** | Condensing | Query-focused summarization, key extraction |
| **Generator** | Answer creation | Comprehensive, well-structured responses |
| **Critic** | Quality validation | Accuracy/completeness/clarity scoring |
| **Supervisor** | Orchestration | Routing, iteration control, state management |

**Workflow**:
```
Iteration 1:
  Supervisor → [Check state] → Retrieve (no docs yet)
             → [Check state] → Summarize (have docs)
             → [Check state] → Generate (have summary)
             → [Check state] → Critic (evaluate)
                              ↓
                        [Needs revision?]
                        ├─ YES + iterations < 3
                        │  ↓
Iteration 2:
  Supervisor → [Feedback from Critic]
             → Generate (revised)
             → Critic (re-evaluate)
                              ↓
                        [Continue or finish?]
                        └─ Final Answer

Iteration 3 (max):
  [Return final answer]
```

**Critic Scoring**:
Each dimension scored 0.0-1.0:
- **Accuracy**: Correctness of information
- **Completeness**: Coverage of all aspects
- **Clarity**: Quality of explanation

**Configuration**:
```bash
docb config set rag.mode multi_agent
docb config set multi_agent.max_iterations 3
```

**When to Use**:
- ✅ Complex research questions
- ✅ Need highest quality
- ✅ Value accuracy over speed
- ✅ Iterative refinement beneficial
- ✅ Quality metrics important

**Example Workflow**:
```
Q: "Explain machine learning pipeline design"

Iteration 1:
  Retriever → Finds 5 docs on ML pipelines
  Summarizer → Extracts key phases
  Generator → Initial answer
  Critic → Scores: accuracy 0.9, completeness 0.7, clarity 0.85
         → Feedback: "Missing data validation step"

Iteration 2:
  Supervisor → Routes back to Generator (add missing step)
  Generator → Revised answer (added validation)
  Critic → Scores: accuracy 0.95, completeness 0.95, clarity 0.9
         → Decision: "Good enough"

Return final answer
```

---

## Document Processing

### Source Types

DocBases supports multiple source types for knowledge base creation:

#### 1. Local Files
```bash
docb kb add file /path/to/document.pdf --name "My PDF"
docb kb add file /path/to/document.docx --name "My Word Doc"
docb kb add file /path/to/document.txt --name "My Text"
```

**Supported Formats**:
- PDF (with advanced parsing option)
- DOCX (Microsoft Word)
- PPTX (PowerPoint)
- XLSX (Excel)
- TXT, MD (Text formats)
- HTML, HTM (Web content)

#### 2. Local Folders
```bash
docb kb add folder /path/to/documents --name "All Docs"
```

**Processing**:
- Recursively finds all supported files
- Maintains directory structure in metadata
- Progress indication during indexing

#### 3. GitHub Repositories
```bash
docb kb add repo https://github.com/langchain-ai/langchain --name "LangChain Docs"
```

**Processing**:
- Clones repository
- Extracts documentation files
- Indexes code comments (optional)
- Tracks repository URL for updates

#### 4. Websites
```bash
docb kb add website https://python.org/docs --name "Python Docs"
```

**Processing**:
- Web scraping with HTML parsing
- Follows links up to depth limit
- Extracts text content
- Preserves URL references

#### 5. Download URLs
```bash
docb kb add download https://example.com/whitepaper.pdf --name "Whitepaper"
```

**Processing**:
- Downloads file temporarily
- Processes like local file
- Maintains URL reference

### Advanced Parsing Options

#### Docling Integration (Enhanced PDF Parsing)

**Enable**: `USE_DOCLING=true`

**Capabilities**:
- Table structure preservation (maintains rows/columns)
- Formula recognition (LaTeX, MathML)
- Layout analysis (detects columns, headers)
- Multi-page context
- Metadata extraction

**When to Use**:
- ✅ Technical PDFs with tables/formulas
- ✅ Academic papers
- ✅ Reports with complex layout
- ✅ Documents with figures/charts

**Example**:
```
Regular parsing:
"Col1 Col2 123 456..."  (loses structure)

Docling parsing:
[Table]
  Col1 | Col2
  -----|-----
   123 | 456  ✅ (preserves structure)
```

#### Semantic Chunking

**Enable**: `CHUNKING_STRATEGY=semantic`

**How It Works**:
- Calculates embedding similarity between potential chunks
- Identifies natural topic boundaries
- Creates chunks at semantic breaks (not arbitrary sizes)

**Advantages**:
- Better coherence in retrieved chunks
- Reduced information fragmentation
- Improved answer quality
- Slightly slower processing

**Example**:
```
Document excerpt:
"...detailed content on Machine Learning...
In conclusion, ML is powerful. Now let's discuss Deep Learning.
Deep Learning focuses on neural networks..."

Character-based split:
  "...ML is powerful. Now let's" [SPLIT]
  "discuss Deep Learning..."  ❌ Broken context

Semantic split:
  "...detailed content on ML...In conclusion, ML is powerful." [SPLIT]
  "Now let's discuss Deep Learning. Deep Learning focuses..." ✅ Natural boundary
```

**Configuration**:
```bash
docb config set chunking.strategy semantic

# Or environment variable
CHUNKING_STRATEGY=semantic
```

### File Change Detection

**Purpose**: Automatically detect and update knowledge bases when source files change

**How It Works**:

1. **Initial Creation**: Stores metadata (file size, modification time, hash)
2. **Subsequent Checks**: Compares current files against stored metadata
3. **Detection**:
   - **Added**: New files in source directory
   - **Modified**: Existing files with changed content
   - **Deleted**: Files no longer in source
4. **User Prompted**: On next query, user asked if should re-index
5. **Smart Update**: Only processes changed files

**Metadata Format**:
```json
{
  "kb_name": "My Docs",
  "source": "/path/to/docs",
  "source_type": "folder",
  "created_at": "2026-01-15T10:00:00Z",
  "last_updated": "2026-01-21T14:30:00Z",
  "files": [
    {
      "path": "guide.pdf",
      "size": 2048000,
      "mtime": 1674316800,
      "hash": "sha256:abc123..."
    },
    {
      "path": "docs/intro.md",
      "size": 15000,
      "mtime": 1674316900,
      "hash": "sha256:def456..."
    }
  ]
}
```

**Example Workflow**:
```bash
# Week 1
docb kb add folder ./docs --name "Project Docs"
# → Creates: knowledges/Project Docs/metadata.json

# Documents updated
# → guide.pdf: modified
# → new_feature.md: added

# Week 2: Query triggers detection
docb query interactive --kb "Project Docs"
# Output:
#   Checking for changes...
#   Found: 1 modified, 1 added
#   Update knowledge base? [y/n]:

# User enters 'y'
# → Re-indexes only changed files (fast)
# → Updates metadata.json
```

**Disable Auto-Check**:
```bash
docb config set kb.auto_check_changes false
```

**Manual Check**:
```bash
docb kb info "Project Docs"  # Shows change summary
```

---

## Knowledge Base Management

### Create Knowledge Base

```bash
# From file
docb kb add file document.pdf --name "My KB"

# From folder
docb kb add folder ./documents --name "All Docs"

# From GitHub
docb kb add repo https://github.com/user/repo --name "GitHub Repo"

# From website
docb kb add website https://docs.example.com --name "Website Docs"

# From download URL
docb kb add download https://example.com/file.pdf --name "Downloaded PDF"
```

### List Knowledge Bases

```bash
docb kb list
# Output:
# 1. Project Docs (folder) - 1,250 docs - Last updated: 2026-01-21
# 2. Python Docs (website) - 856 docs - Last updated: 2026-01-15
# 3. Whitepaper (file) - 42 docs - Last updated: 2026-01-20
```

### Get KB Information

```bash
docb kb info "Project Docs"
# Output:
# Name: Project Docs
# Type: folder
# Source: /path/to/docs
# Documents: 1,250
# Created: 2026-01-10
# Updated: 2026-01-21
# Size: 245 MB
# Status: ✅ Healthy
# Changes Detected: ⚠️ 3 files modified, 1 added
```

### Delete Knowledge Base

```bash
docb kb delete "Project Docs"
# Warning: This cannot be undone!
# Proceed? [y/n]: y
# ✅ Deleted "Project Docs"
```

### Configuration

```bash
# Set default KB
docb config set kb.default "Project Docs"

# Enable auto-change detection
docb config set kb.auto_check_changes true

# Set re-index strategy
docb config set kb.reindex_strategy full  # or 'partial'
```

---

## Multi-Agent System

### Agent Roles

**Retriever Agent**:
- Fetches documents from knowledge base
- Performs query decomposition for complex questions
- Maintains awareness of retrieval quality
- Tool: `retrieve_documents(query, k=4)`

**Summarizer Agent**:
- Condenses large document sets
- Query-focused summarization (extracts relevant parts)
- Map-reduce for very large contexts
- Preserves key information

**Generator Agent**:
- Creates comprehensive answers
- Structures responses
- Incorporates context appropriately
- Handles multi-source synthesis

**Critic Agent**:
- Evaluates answer quality
- Scores: accuracy (0.0-1.0), completeness, clarity
- Identifies missing information
- Suggests improvements
- Decides if revision needed

**Supervisor Agent**:
- Orchestrates workflow
- Routes between agents based on state
- Manages iteration count
- Makes high-level decisions

### Supervisor Routing Logic

```python
State Check:
├─ No documents yet?
│  └─ Route to: Retriever
├─ Have documents, no summary?
│  └─ Route to: Summarizer
├─ Have summary/docs, no answer?
│  └─ Route to: Generator
├─ Have answer, not validated?
│  └─ Route to: Critic
├─ Critic says needs revision, iterations < max?
│  └─ Route to: Generator (revise)
└─ Complete or max iterations reached?
   └─ Route to: Finalize & Return
```

### Configuration

```python
from src.agents.supervisor import MultiAgentSupervisor

supervisor = MultiAgentSupervisor(
    vectorstore=vs,
    llm=llm,
    checkpointer=checkpointer,
    max_iterations=3  # Max refinement loops
)
```

### Example Use Cases

**Research Paper Analysis**:
```
Q: "Summarize the methodology and findings"

Flow:
  Retriever → Gets paper sections
  Summarizer → Extracts methodology + findings
  Generator → Creates structured summary
  Critic → Validates completeness

Result: Comprehensive, validated summary
```

**Complex Problem Solving**:
```
Q: "Design a caching strategy for distributed systems"

Flow:
  Retriever → Finds consistency models, cache algorithms
  Summarizer → Extracts key approaches
  Generator → Creates design combining approaches
  Critic → Checks for completeness (found all considerations?)
  [If incomplete] → Generator revises, Critic re-checks

Result: Thorough, multi-perspective answer
```

---

## Retrieval & Generation

### LLM Providers

**Local** (Free, Private):
- Ollama (recommended)
- Configuration: `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1:8b`

**Cloud** (API-based):
- OpenAI (GPT-3.5, GPT-4)
- Google (Gemini)
- Groq (Ultra-fast)

**Configuration**:
```bash
# Local
docb config set llm.provider ollama
docb config set llm.model llama3.1:8b

# OpenAI
docb config set llm.provider openai
docb config set llm.model gpt-4
export OPENAI_API_KEY=sk-...

# Google
docb config set llm.provider google
docb config set llm.model gemini-1.5-pro
export GOOGLE_API_KEY=...

# Groq
docb config set llm.provider groq
docb config set llm.model mixtral-8x7b
export GROQ_API_KEY=...
```

### Embedding Providers

**Local**:
- Ollama (default)
- Models: nomic-embed-text (274MB, recommended)

**Cloud**:
- OpenAI (Dimensions: 1536)
- Google (Dimensions: 768)

**Configuration**:
```bash
# Local
docb config set emb.provider ollama
docb config set emb.model nomic-embed-text

# OpenAI
docb config set emb.provider openai
docb config set emb.model text-embedding-3-small
```

### Web Search Integration

**Enable**:
- Corrective RAG: Automatic when docs insufficient
- Adaptive RAG: When query classified as "web"
- Manual: Can be called directly

**Configuration**:
```bash
# Corrective RAG
docb config set corrective_rag.min_relevant_docs 2
# If < 2 relevant docs, triggers web search

# Adaptive RAG
# Web search triggered for queries classified as "web"

# Web search settings
docb config set web_search.num_results 3
docb config set web_search.timeout 5
```

**Provider**: DuckDuckGo (no API key required)

**Example**:
```bash
Q: "Latest developments in quantum computing 2026"
KB: 2024-2025 documentation

Flow:
  1. Retrieve (gets 3 docs - all 2024-2025)
  2. Grade (relevance score: 0.3 - outdated)
  3. Below threshold (0.5) → Web search
  4. Fetch current articles
  5. Generate: "...background from KB... latest from web..."
```

---

## Persistence & Observability

### Session Persistence

**Enable**: `USE_PERSISTENT_MEMORY=true`

**How It Works**:
- Stores conversation state in SQLite
- Thread ID = Session identifier
- Can resume session after restart
- Maintains context across messages

**Configuration**:
```bash
docb config set persistence.enabled true
docb config set persistence.db_path "knowledges/checkpoints.db"
```

**Usage**:
```bash
# Session automatically created with UUID
docb query interactive --kb "My KB"
# Session ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
# (Context saved)

# Close terminal

# Later: Resume same session
docb query interactive --kb "My KB" --session a1b2c3d4-e5f6-7890-abcd-ef1234567890
# (Previous context restored)
```

### Metrics Tracking

**Tracked**:
- Query latency (milliseconds)
- Retrieval count (documents retrieved)
- RAG mode used
- Success/failure status
- Timestamp
- Session ID

**Location**: `knowledges/metrics.db`

**View Metrics**:
```bash
docb health check --metrics
# Output:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics Summary (Last 7 days)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Total Queries: 1,245
# Avg Latency: 1,240ms
# Success Rate: 98.5%
#
# By Mode:
#   basic: 245 queries, 320ms avg
#   corrective: 412 queries, 1,200ms avg
#   adaptive: 623 queries, 1,450ms avg
#   multi_agent: 45 queries, 3,800ms avg
```

**Configuration**:
```bash
docb config set observability.metrics.enabled true
docb config set observability.metrics.retention_days 30  # Keep 30 days
```

### LangSmith Integration (Advanced)

**Enable**: `LANGSMITH_TRACING=true`

**Setup**:
```bash
export LANGSMITH_API_KEY=ls_...
export LANGSMITH_PROJECT="doc-bases"
docb config set observability.langsmith.enabled true
```

**Traces**:
- All LLM calls
- Tool executions
- Agent decisions
- Retrieval operations
- Timing information

**Use Cases**:
- Debugging query failures
- Performance profiling
- Identifying bottlenecks
- Optimizing prompts

---

## Configuration Guide

### Environment Variables

```env
# ============ LLM Configuration ============
LLM_PROVIDER=ollama              # Options: ollama, openai, google, groq
LLM_MODEL=llama3.1:8b
LLM_API_BASE=http://localhost:11434

# ============ Embedding Configuration ============
EMB_PROVIDER=ollama
EMB_MODEL=nomic-embed-text
EMB_API_BASE=http://localhost:11434

# ============ RAG Mode ============
RAG_MODE=adaptive                # Options: basic, corrective, adaptive, multi_agent

# ============ Document Processing ============
USE_DOCLING=true                 # Enhanced PDF parsing
CHUNKING_STRATEGY=semantic       # Options: recursive, semantic

# ============ Persistence ============
USE_PERSISTENT_MEMORY=true
CHECKPOINT_DB_PATH=knowledges/checkpoints.db
METRICS_DB_PATH=knowledges/metrics.db

# ============ Corrective RAG ============
CORRECTIVE_RAG_RELEVANCE_THRESHOLD=0.5
CORRECTIVE_RAG_MIN_RELEVANT_DOCS=1

# ============ Observability ============
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=doc-bases

# ============ Logging ============
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
```

### CLI Configuration Commands

```bash
# View all settings
docb config list

# View specific setting
docb config get rag.mode

# Change setting
docb config set rag.mode adaptive
docb config set llm.provider openai

# Reset to defaults
docb config reset

# Export configuration
docb config export config.json

# Import configuration
docb config import config.json
```

---

## Use Cases & Examples

### Use Case 1: Technical Documentation Query

**Scenario**: Users query technical documentation frequently updated

**Setup**:
```bash
# 1. Create KB from docs
docb kb add folder ./docs --name "Tech Docs"

# 2. Configure for quality
docb config set rag.mode adaptive
docb config set chunking.strategy semantic
docb config set use.docling true

# 3. Enable change detection
docb config set kb.auto_check_changes true
```

**Workflow**:
```bash
# Week 1: Initial setup
docb query interactive --kb "Tech Docs"

# Week 2: Doc updates detected automatically
docb query single --kb "Tech Docs" --query "How do I use feature X?"
# → "Checking for changes... 2 files modified"
# → "Update KB? [y/n]:" y
# → Re-indexes changed files
# → Answers question with latest info
```

### Use Case 2: Research Paper Analysis

**Scenario**: Analyze multiple research papers

**Setup**:
```bash
# Add papers
docb kb add file paper1.pdf --name "Paper1"
docb kb add file paper2.pdf --name "Paper2"

# Configure for quality
docb config set rag.mode multi_agent
docb config set multi_agent.max_iterations 5
```

**Example Query**:
```bash
docb query single --kb "Paper1" \
  --query "Compare methodology with Paper2"

# Multi-Agent flow:
#   Retriever → Gets methodology sections
#   Summarizer → Extracts key approaches
#   Generator → Creates comparison
#   Critic → Validates completeness
#   [If incomplete] → Refines
#   Final: Comprehensive comparison
```

### Use Case 3: Real-time Information

**Scenario**: Knowledge base outdated, need current info

**Setup**:
```bash
docb config set rag.mode corrective
docb config set corrective_rag.min_relevant_docs 2
```

**Example Query**:
```bash
docb query single --kb "News KB" \
  --query "Latest AI developments"

# Corrective RAG flow:
#   Retrieve → Gets KB docs (possibly outdated)
#   Grade → Scores low (old info)
#   Trigger → Web search
#   Combine → KB context + web results
#   Validate → Check grounding
#   Answer: Current + contextual
```

### Use Case 4: Interactive Research Session

**Scenario**: Multi-turn conversation with persistence

**Setup**:
```bash
docb config set use.persistent_memory true
docb config set rag.mode adaptive
```

**Session**:
```bash
docb query interactive --kb "Research KB"

Q1: "Explain machine learning basics"
A1: [Answer with context saved]

Q2: "How does backpropagation work?"
A2: [Uses Q1 context for follow-up understanding]

Q3: "Compare with evolutionary algorithms"
A3: [References prior ML discussion]

# Session saved - can resume later
```

---

## Performance Tuning

### For Speed
```bash
# Use basic RAG
docb config set rag.mode basic

# Character-based chunking
docb config set chunking.strategy recursive

# Smaller model
docb config set llm.model mistral

# Disable unnecessary features
docb config set use.docling false
```

**Expected latency**: 200-800ms

### For Quality
```bash
# Use multi-agent
docb config set rag.mode multi_agent
docb config set multi_agent.max_iterations 5

# Semantic chunking
docb config set chunking.strategy semantic

# Larger model
docb config set llm.model llama3.1:70b

# Enable all features
docb config set use.docling true
docb config set use.persistent_memory true
```

**Expected latency**: 2-6 seconds

### Balanced
```bash
# Adaptive mode
docb config set rag.mode adaptive

# Semantic chunking
docb config set chunking.strategy semantic

# Medium model
docb config set llm.model llama3.1:8b
```

**Expected latency**: 500-1700ms

---

## Troubleshooting

### "Connection refused" error
```
Solution:
1. Make sure Ollama is running: ollama serve
2. Check API base URL: docb config get llm.api_base
3. Verify localhost:11434 is accessible
```

### "Hallucination detected"
```
Solution (Corrective RAG):
1. Check source documents - may be contradictory
2. Disable web search temporarily
3. Switch to multi-agent for validation
4. Review the flagged claims
```

### Slow first query
```
Normal! Model loads into RAM (30-60 seconds)
Subsequent queries: 2-5 seconds
```

### Out of memory
```
Solution:
1. Close other applications
2. Use smaller model: docb config set llm.model mistral
3. Check available RAM: free -h (Linux/Mac)
4. Reduce KB size or split into smaller KBs
```

---

**For more information, see**:
- [Architecture Guide](ARCHITECTURE.md)
- [CLI Usage Guide](CLI_USAGE.md)
- [API Reference](API.md)
- [Deployment Guide](DEPLOYMENT.md)
