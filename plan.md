# DocBases Ollama Setup Plan

## Overview
- **Total Tasks**: 13
- **Estimated Duration**: 30-40 minutes
- **High-Risk Tasks**: 2 (Model downloads - large files)
- **Manual Download Required**: No (Ollama already installed)
- **API Keys Required**: No ✓

## Pre-Flight Checklist
- [x] Python 3.10+ available (have 3.11.14)
- [x] Ollama installed and verified (v0.14.1)
- [x] Internet connection (for model downloads)
- [ ] 10GB+ free disk space (CRITICAL - verify before starting)
- [ ] 8GB+ RAM available (currently 6.4GB free - may need to free up RAM)
- [ ] Can leave terminal running (for Ollama server)

## Task Breakdown

### Phase A: Environment Setup (5 minutes)
- Task 1: Verify Python Version
- Task 2: Create Virtual Environment
- Task 3: Activate Virtual Environment
- Task 4: Upgrade pip

### Phase B: Dependencies (10 minutes)
- Task 5: Install Production Dependencies

### Phase C: Configuration (1 minute)
- Task 6: Create .env File from Template

### Phase D: Ollama Models (20 minutes)
- Task 7: Start Ollama Server (in separate terminal)
- Task 8: Pull LLM Model (llama3.1:8b)
- Task 9: Pull Embedding Model (nomic-embed-text)

### Phase E: Verification (5 minutes)
- Task 10: Verify DocBases Import
- Task 11: Health Check

## Success Indicators
- ✓ Python virtual environment created and activated
- ✓ All Python packages installed from requirements.txt
- ✓ .env file created with Ollama defaults
- ✓ Ollama server running on http://localhost:11434
- ✓ Both models available (llama3.1:8b, nomic-embed-text)
- ✓ DocBases imports successfully
- ✓ Health check passes

## Terminal Setup Required
- **Terminal 1**: Keep Ollama server running (`ollama serve`)
- **Terminal 2**: Run DocBases setup and commands
- **Both must be open simultaneously** during setup

## Task Details

```
TASK 1: Verify Python Version
  Command: python3 --version
  Expected: Python 3.10+
  Success: Output shows 3.10 or higher
  Risk: LOW

TASK 2: Create Virtual Environment
  Command: python3 -m venv venv
  Expected: venv directory created
  Success: venv/ folder exists
  Risk: LOW

TASK 3: Activate Virtual Environment
  Command: source venv/bin/activate (macOS/Linux)
  Expected: Prompt shows (venv)
  Success: which python shows venv path
  Risk: LOW
  Note: This must be done in Terminal 2 before other commands

TASK 4: Upgrade pip
  Command: python -m pip install --upgrade pip
  Expected: pip upgraded
  Success: pip --version shows recent version
  Risk: LOW

TASK 5: Install Production Dependencies
  Command: pip install -r requirements.txt
  Expected: All packages installed (5-10 minutes)
  Verification: python -c "import langchain, chromadb, langchain_ollama"
  Risk: MEDIUM (time-consuming, ~10 minutes)

TASK 6: Create .env File
  Command: cp .env.example .env
  Expected: .env created with Ollama defaults
  Success: File exists with correct settings
  Risk: LOW
  Note: Already pre-configured for Ollama

TASK 7: Start Ollama Server (SEPARATE TERMINAL)
  Command: ollama serve
  Terminal: Terminal 1 (KEEP RUNNING)
  Expected: Server listening on http://localhost:11434
  Verification: In Terminal 2: curl http://localhost:11434/api/tags
  Risk: LOW (but essential - keep this terminal open)

TASK 8: Pull LLM Model
  Command: ollama pull llama3.1:8b
  Terminal: Terminal 2 (while ollama serve runs in Terminal 1)
  Expected: Model downloads (~8GB)
  Time: 5-10 minutes depending on internet speed
  Verification: ollama list | grep llama3.1
  Risk: MEDIUM (large download, network dependent)

TASK 9: Pull Embedding Model
  Command: ollama pull nomic-embed-text
  Terminal: Terminal 2
  Expected: Model downloads (~335MB)
  Time: <1 minute
  Verification: ollama list | grep nomic-embed-text
  Risk: LOW (small download)

TASK 10: Verify DocBases Import
  Command: python -c "from src.utils.document_loader import DocumentLoader; from src.models.embeddings import setup_embeddings; print('✓ DocBases modules imported')"
  Expected: Script runs without errors
  Success: Prints success message
  Risk: LOW (read-only test)

TASK 11: Health Check
  Command: ./docb health check
  Expected: All systems show ready
  Success: Shows ✓ for LLM and embeddings
  Risk: LOW (read-only test)
```

## Important Notes
- **Ollama server must remain running** while using DocBases
- **First query will be slow** (30-60 seconds - model loading into RAM)
- **Subsequent queries are faster** (2-5 seconds with model cached in memory)
- Models persist in Ollama cache (~8.5GB disk space after setup)

## RAM Warning
Currently detected: 6.4GB free
- Minimum required: 8GB
- Recommended: 16GB
- Action: If RAM is low, close other applications before starting

## What You'll Have After Setup
✓ Complete local RAG system
✓ Free inference (no API costs)
✓ Private (data never leaves your machine)
✓ Offline capable (after setup)
✓ Fast responses (with GPU acceleration)

## Rollback/Cleanup Steps (if needed)
```bash
# Remove virtual environment
rm -rf venv/

# Remove .env file
rm .env

# Remove Ollama models (optional)
ollama rm llama3.1:8b
ollama rm nomic-embed-text
```

## Next Step
Review this plan and provide approval to proceed to Phase 3: Execute installation tasks.
