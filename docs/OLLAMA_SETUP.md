# Ollama Quick Start Guide

Get DocBases running locally with Ollama in **5 minutes**. No API keys needed. Completely free and private.

## Prerequisites

- **OS**: macOS, Linux, or Windows (with WSL2)
- **RAM**: Minimum 8GB (16GB recommended for better performance)
- **GPU** (Optional but recommended): NVIDIA CUDA, AMD ROCm, or Apple Metal for faster inference
- **Disk Space**: 10-15GB for models

## Installation Steps

### Step 1: Install Ollama (2 minutes)

Visit **[ollama.ai](https://ollama.ai)** and download for your OS:

- **macOS**: Direct download or `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download Windows installer (WSL2 backend)

After installation, verify it worked:

```bash
ollama --version
```

### Step 2: Clone & Setup DocBases (1 minute)

```bash
# Clone the repository
git clone https://github.com/luongnv89/doc-bases.git
cd doc-bases

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

**That's it!** The `.env` file is already configured for Ollama with the right defaults.

### Step 3: Start Ollama (30 seconds)

Open a terminal and start the Ollama server:

```bash
ollama serve
```

You should see output like:
```
2024/01/16 10:30:45 "POST /api/generate HTTP/1.1" 200 1234567
```

**Keep this terminal open** while using DocBases.

### Step 4: Pull Models (2 minutes)

Open a **new terminal** (keep Ollama running in the first one):

```bash
# Pull the LLM model (8GB)
ollama pull llama3.1:8b

# Pull the embedding model (335MB)
ollama pull nomic-embed-text
```

Wait for both downloads to complete. You'll see:
```
pulling digest: abc123...
pulling digest: def456...
success
```

### Step 5: Run DocBases (30 seconds)

Still in the new terminal:

```bash
# Start the application
./docb --help

# Or use the interactive CLI
./docb query interactive
```

You're done! 🎉

---

## Verifying Everything Works

### Check Ollama is Running

```bash
curl http://localhost:11434/api/tags
```

Should return JSON with available models.

### Check DocBases Connection

```bash
./docb health check
```

Should show ✓ for LLM and embeddings.

### Quick Test Query

```bash
# Create a sample document
echo "DocBases is an intelligent document querying system." > sample.txt

# Add knowledge base
./docb kb add local sample.txt

# Query it
./docb query single --query "What is DocBases?"
```

---

## Model Selection

### Quick Start (What we installed)

- **LLM**: `llama3.1:8b` (8GB RAM, balanced quality/speed)
- **Embeddings**: `nomic-embed-text` (335MB, excellent quality)

### Alternative Models

#### Faster (Lower Quality)
```bash
# 1GB model - Ultra fast
ollama pull mistral

# Replace in .env:
# LLM_MODEL=mistral
```

#### Slower (Higher Quality)
```bash
# 70GB model - Best quality (requires 70GB+ RAM)
ollama pull llama3.1:70b

# Replace in .env:
# LLM_MODEL=llama3.1:70b
```

#### Lightweight Embeddings
```bash
# Faster but slightly lower quality
ollama pull all-minilm

# Replace in .env:
# EMB_MODEL=all-minilm
```

---

## Common Issues & Solutions

### Problem: "Connection refused" at localhost:11434

**Solution**: Make sure Ollama is running in another terminal:
```bash
ollama serve
```

### Problem: Out of Memory (OOM) errors

**Solutions**:
1. Use a smaller model: `ollama pull mistral` (1GB instead of 8GB)
2. Close other applications to free up RAM
3. Check available memory: `free -h` (Linux) or `top` (macOS)

### Problem: Slow responses (>10 seconds)

**Reasons & Solutions**:
- First request loads the model into memory (normal, 30-60 seconds)
- Using CPU instead of GPU: Add NVIDIA/AMD drivers for GPU acceleration
- Model too large for RAM: Switch to smaller model
- Disk is slow: Move model cache to SSD if possible

### Problem: Model not found error

**Solution**: Make sure models are pulled:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# List available models
ollama list
```

### Problem: Port 11434 already in use

**Solution**: Change the port:
```bash
# Start on different port
ollama serve --host 127.0.0.1:11435

# Update .env:
# LLM_API_BASE=http://localhost:11435
# EMB_API_BASE=http://localhost:11435
```

---

## Performance Tips

### Enable GPU Acceleration

**NVIDIA (CUDA)**:
```bash
# Install NVIDIA CUDA toolkit
# Ollama will auto-detect and use CUDA
ollama serve
```

**AMD (ROCm)**:
```bash
# Linux with AMD GPU
ollama serve
# Set: OLLAMA_NUM_GPU=1
```

**Apple Silicon (Metal)**:
```bash
# macOS with Apple Silicon
# Automatic - no extra steps needed
ollama serve
```

### Configuration Tweaks

Edit `.env` to optimize:

```env
# Use corrective RAG for better quality (slower)
RAG_MODE=corrective

# Use semantic chunking for better retrieval (slower)
CHUNKING_STRATEGY=semantic

# Use larger embedding model for better semantic search
EMB_MODEL=mxbai-embed-large
```

---

## Next Steps

### 1. Load Your Documents

```bash
# Add GitHub repository
./docb kb add repo https://github.com/langchain-ai/langchain

# Add local files
./docb kb add local /path/to/documents/

# Add website
./docb kb add web https://docs.example.com/
```

### 2. Try Different RAG Modes

```bash
# See current RAG mode
./docb config list | grep rag

# Switch to corrective RAG (better quality)
./docb config set rag.mode corrective

# Try adaptive RAG (smart routing)
./docb config set rag.mode adaptive
```

### 3. Query Your Knowledge Base

```bash
# Interactive mode (recommended)
./docb query interactive

# Single query
./docb query single --query "Your question here?"

# Batch queries from file
./docb query batch queries.txt --output results.json
```

### 4. Monitor Performance

```bash
# View metrics
./docb metrics view

# Export metrics to file
./docb metrics export metrics.csv
```

---

## Switching to Cloud Providers

### If you want to try faster cloud models:

Edit `.env` and uncomment one of the provider examples:

**OpenAI**:
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMB_PROVIDER=openai
EMB_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-your-key-here
```

**Google Gemini**:
```env
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
EMB_PROVIDER=google
EMB_MODEL=models/text-embedding-004
GOOGLE_API_KEY=your-key-here
```

**Groq** (Fastest):
```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
EMB_PROVIDER=openai
EMB_MODEL=text-embedding-3-small
GROQ_API_KEY=gsk_your-key-here
OPENAI_API_KEY=sk-your-key-here
```

Restart DocBases and it will use the new provider.

---

## Troubleshooting Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Check DocBases configuration
./docb config list

# Test LLM connection
./docb health check

# View logs with verbose output
./docb --verbose query single --query "test"

# List all knowledge bases
./docb kb list

# Check system resources
free -h          # Linux
vm_stat          # macOS
Get-WmiObject    # Windows
```

---

## FAQ

**Q: Can I run without GPU?**
A: Yes, but it will be slower (5-10 seconds per response). Get a GPU for better experience.

**Q: Can I switch models without reinstalling?**
A: Yes, edit `.env` and restart. Ollama keeps all models cached locally.

**Q: How much disk space do I need?**
A: 8-15GB depending on models. `llama3.1:8b` + `nomic-embed-text` ≈ 8.5GB

**Q: Can I run multiple models at once?**
A: Yes, but they share RAM. First model loaded stays in memory.

**Q: Is my data private?**
A: Completely! Everything runs locally. No data sent to external servers.

**Q: Can I use this in production?**
A: Yes, but consider cloud providers for higher availability and auto-scaling.

---

## System Requirements Reference

| Model | RAM | GPU | Speed |
|-------|-----|-----|-------|
| `mistral` (1B) | 4GB | No | ⚡⚡⚡ Fast |
| `llama3.1:8b` | 8GB | Recommended | ⚡⚡ Medium |
| `llama3.1:70b` | 70GB | Required | ⚡ Slow |
| `nomic-embed-text` | 2GB | No | Fast |
| `mxbai-embed-large` | 2GB | No | Medium |

---

## Need Help?

- **DocBases Issues**: [GitHub Issues](https://github.com/luongnv89/doc-bases/issues)
- **Ollama Help**: [Ollama Discord](https://discord.gg/ollama)
- **Models**: [Ollama Library](https://ollama.ai/library)

---

## What's Next?

1. ✅ Ollama running locally
2. ✅ Models pulled and tested
3. 👉 [Load your first knowledge base](CLI_USAGE.md#adding-knowledge-bases)
4. 👉 [Explore advanced RAG modes](ARCHITECTURE.md#rag-patterns)
5. 👉 [Deploy to production](DEPLOYMENT.md)

Happy exploring! 🚀
