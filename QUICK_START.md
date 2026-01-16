# DocBases Quick Start Guide

## Your First Knowledge Base ✓ Created

You now have a sample knowledge base loaded and ready to query!

---

## Using DocBases - Common Commands

### Query Your Knowledge Base

**Interactive Mode** (Recommended - conversation style):
```bash
# Start interactive session
./docb query interactive

# With specific knowledge base
./docb query interactive --kb "DocBases Sample KB"
```

**Single Query** (One question, one answer):
```bash
./docb query single --query "What are the main features of DocBases?"
./docb query single --kb "DocBases Sample KB" --query "Tell me about RAG modes"
```

**Batch Queries** (Multiple questions from file):
```bash
# Create queries.txt with one question per line
./docb query batch queries.txt --output results.json
./docb query batch queries.txt --kb "DocBases Sample KB" --output results.json
```

---

## Managing Knowledge Bases

### List All Knowledge Bases
```bash
./docb kb list
```

### Get Info About a Knowledge Base
```bash
./docb kb info "DocBases Sample KB"
```

### Add More Knowledge Bases

**From a local file:**
```bash
./docb kb add file /path/to/document.pdf --name "My PDF KB"
./docb kb add file /path/to/document.txt --name "My Text KB"
./docb kb add file /path/to/document.docx --name "My Word Doc"
```

**From a folder (multiple files):**
```bash
./docb kb add folder /path/to/documents/ --name "My Documents"
```

**From GitHub:**
```bash
./docb kb add repo https://github.com/langchain-ai/langchain --name "LangChain Docs"
```

**From Website:**
```bash
./docb kb add website https://docs.python.org --name "Python Docs"
```

### Delete a Knowledge Base
```bash
./docb kb delete "DocBases Sample KB"
```

---

## Configuration & Settings

### View Current Configuration
```bash
./docb config list
```

### Change RAG Mode (Quality vs Speed)
```bash
# Basic (fastest) - good for simple queries
./docb config set rag.mode basic

# Corrective (balanced) - validates and refines results
./docb config set rag.mode corrective

# Adaptive (smarter) - routes queries intelligently
./docb config set rag.mode adaptive

# Multi-Agent (best quality) - uses specialized agents
./docb config set rag.mode multi_agent
```

### Change Chunking Strategy (Retrieval Quality)
```bash
# Recursive (faster) - default
./docb config set chunking.strategy recursive

# Semantic (better quality) - preserves meaning
./docb config set chunking.strategy semantic
```

### Change Models
```bash
# List available Ollama models
ollama list

# Use different LLM
./docb config set llm.model mistral
./docb config set llm.model llama3.1:70b

# Use different embedding model
./docb config set emb.model all-minilm
./docb config set emb.model mxbai-embed-large
```

---

## Performance Tips

### For Faster Responses
```bash
# Use basic RAG mode (default)
./docb config set rag.mode basic

# Use recursive chunking (faster document processing)
./docb config set chunking.strategy recursive

# Use smaller/faster model
./docb config set llm.model mistral
```

### For Better Quality
```bash
# Use corrective or adaptive mode
./docb config set rag.mode corrective

# Use semantic chunking (better retrieval)
./docb config set chunking.strategy semantic

# Use larger model
./docb config set llm.model llama3.1:70b
```

---

## Troubleshooting

### Problem: "Connection refused" error
```bash
# Make sure Ollama is running in another terminal
ollama serve
```

### Problem: Slow first query
This is normal! The model loads into RAM on first use (30-60 seconds).
Subsequent queries will be 2-5 seconds.

### Problem: Queries are timing out
```bash
# Try simpler RAG mode
./docb config set rag.mode basic

# Or use faster model
./docb config set llm.model mistral
```

### Problem: Out of memory errors
```bash
# Close other applications
# Or use smaller model
ollama pull mistral
./docb config set llm.model mistral
```

---

## Example Workflows

### Workflow 1: Research from GitHub Documentation
```bash
# 1. Load LangChain docs
./docb kb add repo https://github.com/langchain-ai/langchain --name "LangChain"

# 2. Query with corrective mode (best quality)
./docb config set rag.mode corrective

# 3. Ask questions
./docb query single --kb "LangChain" --query "How do I create a custom tool?"
```

### Workflow 2: Analyze Local Files
```bash
# 1. Add your documents
./docb kb add folder ~/Documents/research --name "My Research"

# 2. Use semantic chunking for better retrieval
./docb config set chunking.strategy semantic

# 3. Run batch analysis
./docb query batch questions.txt --kb "My Research" --output analysis.json
```

### Workflow 3: Interactive Research Session
```bash
# 1. Add multiple knowledge bases
./docb kb add file whitepaper.pdf --name "Whitepaper"
./docb kb add folder docs/ --name "Technical Docs"

# 2. Start interactive session
./docb query interactive --kb "Whitepaper"

# 3. Have a conversation (context is preserved across questions)
```

---

## Available Models

### LLM Models (via Ollama)

**Fast & Lightweight:**
- `mistral` (1B) - Ultra fast, good for quick answers
- `tinyllama:1.1b` (1GB) - Very lightweight

**Balanced (Recommended):**
- `llama3.1:8b` (4.9GB) - Recommended for most uses
- `qwen3:8b` (5.2GB) - Good alternative

**High Quality:**
- `llama3.1:70b` (70GB+) - Best quality (requires powerful machine)

### Embedding Models

**Fast:**
- `all-minilm` - Lightweight, good quality
- `embeddinggemma:300m` - 300M parameter model

**Recommended:**
- `nomic-embed-text` (274MB) - Excellent quality, current default

**High Quality:**
- `mxbai-embed-large` - Large embedding model

---

## Next Steps

1. **Load your own documents:**
   ```bash
   ./docb kb add folder /path/to/your/documents --name "My KB"
   ```

2. **Try different query modes:**
   ```bash
   ./docb query interactive
   ```

3. **Experiment with RAG modes:**
   ```bash
   ./docb config set rag.mode adaptive
   ```

4. **Batch process multiple questions:**
   ```bash
   ./docb query batch my_questions.txt --output results.json
   ```

---

## Useful Links

- **DocBases GitHub**: https://github.com/luongnv89/doc-bases
- **Ollama Models**: https://ollama.ai/library
- **CLI Help**:
  ```bash
  ./docb --help
  ./docb query --help
  ./docb kb --help
  ./docb config --help
  ```

---

## Remember

✓ Keep `ollama serve` running in Terminal 1
✓ First query takes 30-60 seconds (normal)
✓ Data stays on your machine (private & offline)
✓ No API costs (everything runs locally)

Happy querying! 🚀
