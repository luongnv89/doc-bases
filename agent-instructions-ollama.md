# DocBases v2.0 - Ollama Setup Instructions for AI Agent

This document provides specialized instructions for an AI agent to set up and deploy DocBases using **Ollama for local, cost-free inference**. This is the recommended approach for most users.

> **Note:** This is a simplified alternative to `agent-instructions.md` that assumes Ollama usage and eliminates API key configuration complexity.

> **✓ COMPLETED:** Installation guide updated! See **[INSTALLATION.md](INSTALLATION.md)** for the quick setup process (5 minutes).

---

## Overview

**Project:** DocBases v2.0 - Retrieval-Augmented Generation (RAG) System
**Language:** Python 3.10+
**Type:** CLI Application with Ollama Backend
**Purpose:** Query documents from various sources using local LLMs (no API keys needed)
**LLM Provider:** Ollama (local inference)
**Models:** llama3.1:8b (LLM) + nomic-embed-text (embeddings)

---

# PHASE 1: RESEARCH & PREPARATION

## Phase 1 Objective

Gather information about the project and verify system readiness for Ollama-based setup. The agent should NOT execute any installation commands during this phase.

### Task 1.1: Analyze Project Structure

**What the agent must do:**
1. Explore the project directory structure
2. Identify project type and entry points
3. Locate configuration files
4. Read project documentation (especially docs/OLLAMA_SETUP.md)

**Files to examine:**
- `README.md` - Project overview
- `docs/OLLAMA_SETUP.md` - Ollama-specific setup guide ⭐
- `.env.example` - Environment variables (Ollama defaults)
- `requirements.txt` - Production dependencies
- `src/main.py` - Main entry point

**Information to extract:**
- Project type: CLI application for document querying
- Python version: 3.10+
- Entry point: `python src/main.py` or `./docb`
- Ollama is default: `LLM_PROVIDER=ollama`, `EMB_PROVIDER=ollama`
- No API keys needed for Ollama
- Required models: `llama3.1:8b` and `nomic-embed-text`

### Task 1.2: Detect System and Ollama Readiness

**What the agent must do:**
1. Detect operating system (macOS, Linux, Windows/WSL2)
2. Verify Python version (3.10+)
3. Check if Ollama is installed and accessible
4. Determine available system resources (RAM, GPU)

**System checks to perform (do NOT execute, just document):**

```bash
# OS Detection
uname -s  # Darwin (macOS), Linux, or MINGW64 (Windows)

# Python Version
python3 --version  # Need 3.10+

# Ollama Installation
ollama --version  # Check if installed

# System Resources
free -h            # Linux: RAM available
vm_stat             # macOS: memory info
Get-PhysicalMemory # Windows: RAM

# GPU Detection
nvidia-smi         # NVIDIA GPU
rocm-smi          # AMD GPU
```

**Information to extract:**
- OS Type: macOS, Linux, or Windows/WSL2
- Python Version: Must be 3.10 or higher
- Ollama Status: Installed or needs installation
- RAM Available: At least 8GB (16GB recommended)
- GPU Available: Yes/No (NVIDIA, AMD, or Apple Silicon)
- GPU Memory: If available

### Task 1.3: Analyze Dependencies for Ollama Setup

**What the agent must do:**
1. Read `requirements.txt` for production dependencies
2. Understand that Ollama setup requires fewer dependencies
3. Categorize dependencies by purpose

**Key dependencies (Ollama-specific):**

**Core Framework:**
- `langchain`, `langchain-core` - LLM framework
- `langchain-ollama` - Ollama integration ⭐
- `langgraph` - Agent workflow framework
- `chromadb` - Vector database (local)

**Utilities:**
- `python-dotenv` - Environment variable loading
- `rich` - Terminal UI
- `requests` - HTTP requests

**Document Processing:**
- `unstructured` - Document parsing
- `pypdf` - PDF handling
- `python-docx` - Word documents

**Note:** No cloud provider packages needed (no `langchain-openai`, `langchain-google-genai`, etc.) since we're using Ollama.

### Task 1.4: Understand Ollama Configuration Requirements

**What the agent must do:**
1. Analyze `.env.example` specifically for Ollama settings
2. Identify that Ollama defaults are already set
3. Understand model requirements

**Ollama Configuration (already pre-configured):**

```env
# LLM Configuration
LLM_PROVIDER=ollama              # Already set
LLM_MODEL=llama3.1:8b           # Already set (8GB model)
LLM_API_BASE=http://localhost:11434  # Already set

# Embedding Configuration
EMB_PROVIDER=ollama              # Already set
EMB_MODEL=nomic-embed-text      # Already set (335MB model)
EMB_API_BASE=http://localhost:11434  # Already set

# Optional: RAG Mode
RAG_MODE=basic                   # Options: basic, corrective, adaptive, multi_agent

# Optional: Document Processing
CHUNKING_STRATEGY=recursive      # Options: recursive, semantic
USE_DOCLING=false               # Optional advanced parsing
```

**Model Information:**
- **LLM:** `llama3.1:8b`
  - Size: ~8GB RAM
  - Quality: Good balance of quality and performance
  - Download: 2-5 minutes

- **Embeddings:** `nomic-embed-text`
  - Size: ~335MB RAM
  - Quality: Excellent for semantic search
  - Download: 30 seconds

**Total Requirements:**
- Disk Space: ~8.5GB for both models
- RAM: Minimum 8GB, recommended 16GB
- Setup Time: 10-15 minutes (first run loads models)

### Task 1.5: Verify Ollama Installation Requirements

**What the agent must do:**
1. Document Ollama installation requirements for the detected OS
2. List installation methods
3. Prepare download links

**OS-Specific Installation:**

**macOS:**
- Download: https://ollama.ai/download
- Or: `brew install ollama`
- Start: `ollama serve` (or automatic on installation)

**Linux:**
- Download: https://ollama.ai/download
- Or: `curl -fsSL https://ollama.ai/install.sh | sh`
- Start: `ollama serve`

**Windows/WSL2:**
- Download: https://ollama.ai/download
- Note: Requires WSL2 backend
- Start: `ollama serve`

### Task 1.6: Create research.md

**What the agent must do:**
1. Document all Phase 1 findings
2. Create file named `research.md` with findings
3. Include detected system info and model requirements

**research.md should include:**

```markdown
# DocBases Ollama Setup - Research Findings

## System Information
- OS Detected: [macOS/Linux/Windows]
- Python Version: [3.10+]
- RAM Available: [e.g., 16GB]
- GPU Available: [Yes/No - NVIDIA/AMD/Apple/None]

## Ollama Status
- Ollama Installed: [Yes/No]
- Ollama Version: [if installed]

## Required Models
- LLM Model: llama3.1:8b (8GB)
- Embedding Model: nomic-embed-text (335MB)
- Total Disk Space Needed: ~8.5GB
- Estimated Download Time: 10-15 minutes

## Configuration (Pre-configured)
- LLM Provider: ollama
- Embedding Provider: ollama
- API Base: http://localhost:11434
- No API Keys Required: ✓

## Installation Steps Required
1. Install Ollama (if not already installed)
2. Start Ollama server
3. Pull required models
4. Clone and setup DocBases
5. Verify installation

## Estimated Total Time
- If Ollama installed: 15-20 minutes
- If Ollama not installed: 25-35 minutes

## Next Step
Proceed to Phase 2: Create detailed installation plan
```

### Phase 1 User Verification

**What the agent must do:**
1. Summarize key findings
2. Direct user to review `research.md`
3. **Wait for explicit user approval before Phase 2**

**Agent should say to user:**

```
═══════════════════════════════════════════════════════════════
PHASE 1: RESEARCH COMPLETE ✓
═══════════════════════════════════════════════════════════════

System Analysis:
  • OS: [Detected OS]
  • Python: [Version] ✓
  • RAM: [Amount] (minimum 8GB needed)
  • GPU: [Status]
  • Ollama: [Installed/Not installed]

Models to Download:
  • LLM: llama3.1:8b (8GB)
  • Embeddings: nomic-embed-text (335MB)
  • Total: ~8.5GB disk space

Configuration:
  • Provider: Ollama (local, free, no API keys)
  • API Base: http://localhost:11434
  • Status: Pre-configured ✓

Estimated Time: 25-35 minutes

Please review research.md for complete details.

Ready to proceed to Phase 2? [yes / no]
```

---

# PHASE 2: PLANNING

## Phase 2 Objective

Create a comprehensive, detailed plan for Ollama setup with Docker or native installation. The agent should NOT execute any installation commands.

### Task 2.1: Create Installation Task Sequence

**What the agent must do:**
1. Define numbered tasks in logical order
2. Specify success criteria for each task
3. Define verification methods
4. Identify high-risk operations

**Task Sequence for Ollama Setup:**

```
TASK 1: Verify Python Version
  Command: python3 --version
  Success: Python 3.10+ installed
  Verification: Check version >= 3.10
  Risk: LOW

TASK 2: Clone DocBases Repository
  Command: git clone https://github.com/luongnv89/doc-bases.git
  Command: cd doc-bases
  Success: Repository cloned
  Verification: Check if .env.example exists
  Risk: LOW

TASK 3: Create Virtual Environment
  Command: python3 -m venv venv
  Success: venv directory created
  Verification: Check if venv/bin/activate exists
  Risk: LOW

TASK 4: Activate Virtual Environment
  Command: source venv/bin/activate (macOS/Linux)
           OR venv\Scripts\activate (Windows)
  Success: Virtual environment active
  Verification: Check $VIRTUAL_ENV variable
  Risk: LOW

TASK 5: Upgrade pip
  Command: python -m pip install --upgrade pip
  Success: pip upgraded to latest
  Verification: pip --version shows latest
  Risk: LOW

TASK 6: Install Production Dependencies
  Command: pip install -r requirements.txt
  Success: All packages installed
  Verification: python -c "import langchain, chromadb, langchain_ollama"
  Risk: MEDIUM (takes 5-10 minutes)

TASK 7: Create .env File from Template
  Command: cp .env.example .env
  Success: .env file created
  Verification: Check if .env exists with Ollama defaults
  Risk: LOW

TASK 8: Download and Install Ollama
  Requirement: Ollama not already installed
  Status: MANUAL - User downloads from https://ollama.ai/download
  Verification: ollama --version shows installed version
  Risk: MEDIUM (requires download + installation)

TASK 9: Start Ollama Server
  Requirement: Ollama installed
  Command: ollama serve
  Success: Server running on http://localhost:11434
  Verification: curl http://localhost:11434/api/tags
  Risk: LOW (keep running in separate terminal)

TASK 10: Pull LLM Model
  Command: ollama pull llama3.1:8b
  Success: Model downloaded and available
  Verification: ollama list | grep llama3.1:8b
  Risk: MEDIUM (large download, 5-10 minutes)

TASK 11: Pull Embedding Model
  Command: ollama pull nomic-embed-text
  Success: Model downloaded and available
  Verification: ollama list | grep nomic-embed-text
  Risk: LOW (small download, <1 minute)

TASK 12: Verify DocBases Installation
  Command: python -c "from src.utils.document_loader import DocumentLoader; print('✓')"
  Success: DocBases modules import successfully
  Verification: Script runs without errors
  Risk: LOW

TASK 13: Health Check
  Command: ./docb health check
  Success: LLM and embeddings respond correctly
  Verification: ✓ LLM provider: ollama
             ✓ Embeddings: ollama
  Risk: LOW
```

### Task 2.2: Alternative Paths (Optional)

**What the agent should document:**

1. **Docker Deployment (Optional)**
   - Pros: Isolated environment, reproducible
   - Cons: Requires Docker installation
   - Steps: Build Docker image, run container with Ollama

2. **Conda Alternative (Optional)**
   - Pros: Better conda package management
   - Cons: Heavier than venv
   - Steps: Create conda env, install from requirements

3. **Development Setup (Optional)**
   - Include: requirements-dev.txt for testing/linting
   - Task 7 Alternative: pip install -r requirements-dev.txt
   - Risk: LOW - optional for developers

### Task 2.3: Identify Risk Assessment

**What the agent must do:**
1. Categorize tasks by risk level
2. Mark high-risk operations

**Risk Categories:**

**HIGH-RISK:**
- Task 8: Ollama Installation (download, system modification)
- Task 10: LLM Download (large file, 5-10 minutes)
- Requires: User confirmation before proceeding

**MEDIUM-RISK:**
- Task 6: Dependency Installation (5-10 minutes, may conflict)
- Task 9: Ollama Server (network port 11434)
- Requires: Information, but can proceed

**LOW-RISK:**
- All other tasks (verification, environment setup)
- Requires: No special confirmation

### Task 2.4: Create plan.md

**What the agent must do:**
1. Document complete installation plan
2. Include all tasks with success criteria
3. Include verification methods
4. Include risk assessments

**plan.md structure:**

```markdown
# DocBases Ollama Setup Plan

## Overview
- Total Tasks: 13
- Estimated Duration: 30-40 minutes
- High-Risk Tasks: 2 (Ollama installation, model downloads)
- Manual Download Required: Yes (Ollama)
- API Keys Required: No ✓

## Pre-Flight Checklist
- [ ] Python 3.10+ available
- [ ] Internet connection (for downloads)
- [ ] 10GB+ free disk space
- [ ] 8GB+ RAM available
- [ ] Can leave terminal running (for Ollama server)

## Task Breakdown

### Phase A: Environment Setup (5 minutes)
Task 1-4: Setup and activate Python virtual environment

### Phase B: Dependencies (10 minutes)
Task 5-6: Install Python packages

### Phase C: Ollama Setup (20 minutes)
Task 8-11: Download, install, and configure Ollama + models
- Requires: Manual download of Ollama installer
- Download Time: Variable (depends on internet)

### Phase D: Verification (5 minutes)
Task 12-13: Verify everything works

## Success Indicators
- ✓ Python virtual environment active
- ✓ All Python packages installed
- ✓ Ollama running on http://localhost:11434
- ✓ Models available (llama3.1:8b, nomic-embed-text)
- ✓ DocBases imports successfully
- ✓ Health check passes

## Terminal Setup Required
- Terminal 1: Keep Ollama server running (ollama serve)
- Terminal 2: Run DocBases commands
- Both need to be open simultaneously

## Important Notes
- Ollama server must remain running while using DocBases
- First query will be slower (model loading into memory)
- Subsequent queries are faster (models cached in RAM)
- Models persist in Ollama cache (~8.5GB disk space)
```

### Task 2.5: Consolidate Manual Inputs

**What the agent must do:**
1. Identify all manual actions required
2. Create clear guide for user actions

**Manual Inputs Required:**

```markdown
# Manual Configuration for DocBases Ollama Setup

## Manual Action 1: Download Ollama
- Visit: https://ollama.ai/download
- Choose installer for your OS
- Run installer and follow prompts
- Verify: ollama --version should show installed version

## Manual Action 2: Configure Terminal Setup
You will need TWO terminals open simultaneously:

Terminal 1 (Keep Running):
  $ ollama serve
  (Keep this running - shows model loading messages)

Terminal 2 (Execute Commands):
  $ cd doc-bases
  $ source venv/bin/activate  # or venv\Scripts\activate on Windows
  $ ./docb query interactive

## Manual Action 3: First Query is Slow
- First query will take 30-60 seconds (loading model into RAM)
- Subsequent queries are faster (2-5 seconds)
- This is normal and expected

## Manual Action 4 (Optional): Choose RAG Mode
Edit .env if desired:
  RAG_MODE=basic          # Default: simple and fast
  RAG_MODE=corrective     # Better quality, slower
  RAG_MODE=adaptive       # Intelligent routing
  RAG_MODE=multi_agent    # Best quality, slowest

## Manual Action 5 (Optional): Add Documents
After setup:
  $ ./docb kb add local /path/to/documents
  $ ./docb query interactive
```

### Phase 2 User Verification

**Agent should say to user:**

```
═══════════════════════════════════════════════════════════════
PHASE 2: PLAN CREATED ✓
═══════════════════════════════════════════════════════════════

Installation Plan Summary:
  • Total Tasks: 13
  • Estimated Time: 30-40 minutes
  • Python Setup: 5 minutes
  • Dependencies: 10 minutes
  • Ollama + Models: 20 minutes
  • Verification: 5 minutes

Manual Actions Required:
  1. Download Ollama installer from ollama.ai
  2. Run Ollama installer (depends on your OS)
  3. Keep Ollama server running in Terminal 1
  4. Run DocBases in Terminal 2

What You'll Have:
  • Complete local RAG system
  • Free inference (no API costs)
  • Private (data never leaves your machine)
  • Offline capable (after setup)

Please review plan.md for complete details.

Ready to execute Phase 3? [yes / no]
```

---

# PHASE 3: EXECUTE

## Phase 3 Objective

Execute all planned tasks with proper verification and error handling. The agent executes sequentially and verifies each task before proceeding.

### Task 3.1: Pre-Execution Checklist

**What the agent must do:**
1. Confirm user approval for Phase 3
2. Verify all prerequisites are ready

**Pre-Flight Checklist:**
- [ ] Phase 2 plan.md reviewed and approved
- [ ] Internet connection available
- [ ] 10GB+ free disk space confirmed
- [ ] Willing to keep Ollama server running
- [ ] Two terminals available

**If checklist fails, STOP and ask user to review plan.md**

### Task 3.2: Sequential Task Execution

**Execution Template:**

```
═══════════════════════════════════════════════════════════════
TASK [N/13]: [Task Name]
═══════════════════════════════════════════════════════════════
Success Criterion: [Define success]

[If high-risk]:
⚠️  This is a high-risk operation.
    [Explain what will happen]
    Proceed? [yes / no]

Executing: [Command]
[Show command output]

Verifying: [Verification method]
[Show verification result]

✓ Task [N] completed successfully
```

### Task 3.3: Detailed Task Execution

**TASK 1: Verify Python Version**
```
Command: python3 --version
Expected: Python 3.10 or higher
Verification: Parse version number
Rollback: N/A (read-only)
```

**TASK 2: Clone Repository**
```
Command: git clone https://github.com/luongnv89/doc-bases.git
Expected: Repository cloned successfully
Verification: ls -la doc-bases/.env.example
Rollback: rm -rf doc-bases
```

**TASK 3: Create Virtual Environment**
```
Command: python3 -m venv venv
Expected: venv directory created
Verification: test -f venv/bin/activate
Rollback: rm -rf venv
```

**TASK 4: Activate Virtual Environment**
```
Action: User must activate in their terminal
Command: source venv/bin/activate (macOS/Linux)
      OR venv\Scripts\activate (Windows)
Expected: Shell prompt shows (venv)
Verification: which python shows venv path
Note: Agent cannot activate for user, only provide instructions
```

**TASK 5: Upgrade pip**
```
Command: python -m pip install --upgrade pip
Expected: pip upgraded successfully
Verification: pip --version shows recent version
Rollback: Reinstall in fresh venv if needed
Risk: LOW
```

**TASK 6: Install Dependencies**
```
Command: pip install -r requirements.txt
Expected: All packages installed
Verification: python -c "import langchain, chromadb, langchain_ollama"
Allow Time: 5-10 minutes
Rollback: pip uninstall -r requirements.txt -y && rm -rf venv
Risk: MEDIUM
Display: Show progress with estimated time
```

**TASK 7: Create .env File**
```
Command: cp .env.example .env
Expected: .env created with Ollama defaults
Verification: grep "LLM_PROVIDER=ollama" .env
Rollback: rm .env
Risk: LOW
Note: .env is already pre-configured for Ollama, no edits needed
```

**TASK 8: Ollama Installation [MANUAL]**
```
Action: User downloads and installs Ollama
Provide: https://ollama.ai/download
Expected: ollama --version works
Verification: User confirms installation
Risk: HIGH - requires manual download and installation
Note: Agent cannot automate this, provide clear instructions
Display: OS-specific installation instructions
```

**TASK 9: Start Ollama Server**
```
Important: Run in SEPARATE TERMINAL (Terminal 1)
Command: ollama serve
Expected: Server starts and listens on :11434
Verification: In Terminal 2: curl http://localhost:11434/api/tags
Note: Keep this terminal running for all DocBases operations
Risk: LOW - but essential to keep running
```

**TASK 10: Pull LLM Model**
```
Command: ollama pull llama3.1:8b
Expected: Model downloads and available
Verification: ollama list | grep llama3.1
Allow Time: 5-10 minutes depending on internet
Display: Show download progress
Note: First time download, cached for future use
Risk: MEDIUM - large download
```

**TASK 11: Pull Embedding Model**
```
Command: ollama pull nomic-embed-text
Expected: Model downloads and available
Verification: ollama list | grep nomic-embed-text
Allow Time: 30 seconds to 1 minute
Display: Show download progress
Risk: LOW - small download
```

**TASK 12: Verify DocBases Import**
```
Command: python -c "
from src.utils.document_loader import DocumentLoader
from src.models.embeddings import setup_embeddings
print('✓ DocBases modules imported successfully')
"
Expected: Script runs without errors
Verification: Check for error messages
Rollback: Reinstall requirements if needed
Risk: LOW - read-only test
```

**TASK 13: Health Check**
```
Command: ./docb health check
Expected: All systems report ready
Display:
  ✓ LLM Provider: ollama
  ✓ Embeddings: ollama
  ✓ Models available
  ✓ API endpoints responding
Rollback: Check Ollama server is running
Risk: LOW
```

### Task 3.4: Error Handling

**If a task fails:**

1. **STOP immediately**
2. **Document the failure:**
   - Task name and number
   - Command executed
   - Error message
   - When it occurred

3. **Present to user with options:**
```
❌ TASK FAILED: Task 10 - Pull LLM Model

Command: ollama pull llama3.1:8b

Error: Connection refused - is Ollama running?

This likely means:
  - Ollama server (Task 9) not running
  - Port 11434 blocked
  - Firewall issue

Options:
  1. Verify Ollama server is running in Terminal 1
  2. Check: curl http://localhost:11434/api/tags
  3. Retry this task after fixing

Proceed? [retry / skip / abort]
```

4. **User options:**
   - `retry`: Attempt command again
   - `skip`: Skip task (may cause issues)
   - `abort`: Stop and create report

### Task 3.5: Completion Report

**After all tasks complete:**

```
═══════════════════════════════════════════════════════════════
INSTALLATION REPORT
═══════════════════════════════════════════════════════════════

Status: ✓ COMPLETE
Timestamp: [Date/Time]
Total Tasks: 13/13 completed

COMPLETED TASKS:
✓ Task 1: Python Version Verified
✓ Task 2: Repository Cloned
✓ Task 3: Virtual Environment Created
✓ Task 4: Virtual Environment Activated
✓ Task 5: pip Upgraded
✓ Task 6: Dependencies Installed
✓ Task 7: .env File Created
✓ Task 8: Ollama Installed
✓ Task 9: Ollama Server Started
✓ Task 10: llama3.1:8b Downloaded
✓ Task 11: nomic-embed-text Downloaded
✓ Task 12: DocBases Verified
✓ Task 13: Health Check Passed

═══════════════════════════════════════════════════════════════
NEXT STEPS
═══════════════════════════════════════════════════════════════

1. Keep Ollama Running:
   Terminal 1 (keep running):
   $ ollama serve

2. Use DocBases:
   Terminal 2:
   $ cd doc-bases
   $ source venv/bin/activate
   $ ./docb query interactive

3. Load Your Documents:
   $ ./docb kb add local /path/to/documents

4. Query Your Documents:
   $ ./docb query single --query "Your question?"

═══════════════════════════════════════════════════════════════
IMPORTANT NOTES
═══════════════════════════════════════════════════════════════

✓ All software is running locally on your machine
✓ No API keys required
✓ No internet connection needed after initial setup
✓ First query takes 30-60 seconds (model loading)
✓ Subsequent queries are faster (2-5 seconds)

═══════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════

If Ollama server stops:
  $ ollama serve

If models are missing:
  $ ollama pull llama3.1:8b
  $ ollama pull nomic-embed-text

If imports fail:
  $ pip install -r requirements.txt

For more help:
  See: docs/OLLAMA_SETUP.md
  Or: ./docb --help
```

---

## Success Criteria

### Phase 1 Success ✓
- research.md created with system details
- Ollama requirements documented
- User understands setup process
- Explicit user approval for Phase 2

### Phase 2 Success ✓
- plan.md created with 13 tasks
- All tasks have success criteria and verification
- Risk assessments documented
- Manual actions clearly identified
- Explicit user approval for Phase 3

### Phase 3 Success ✓
- All 13 tasks execute in order
- Each task verifies before proceeding
- Ollama server running and responding
- Models available and downloadable
- DocBases imports successfully
- Health check passes
- User has clear next steps

---

## Quick Reference

### Critical Commands

```bash
# Start Ollama (Terminal 1 - KEEP RUNNING)
ollama serve

# Pull Models (Terminal 2)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Setup DocBases (Terminal 2)
git clone https://github.com/luongnv89/doc-bases.git
cd doc-bases
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Verify Setup
./docb health check

# Use DocBases
./docb query interactive
```

### Files Created During Setup

| File | Created By | Purpose |
|------|-----------|---------|
| `venv/` | Task 3 | Virtual environment |
| `.env` | Task 7 | Configuration (Ollama pre-configured) |
| `research.md` | Phase 1 | Research findings |
| `plan.md` | Phase 2 | Installation plan |

### Time Breakdown

| Phase | Activity | Time |
|-------|----------|------|
| Phase 1 | Research & Analysis | 5 minutes |
| Phase 2 | Planning | 5 minutes |
| Phase 3a | Environment Setup | 5 minutes |
| Phase 3b | Dependencies | 10 minutes |
| Phase 3c | Ollama Installation | 20 minutes |
| Phase 3d | Verification | 5 minutes |
| **Total** | **Complete Setup** | **30-40 minutes** |

---

## Implementation Notes for AI Agent

```python
# Phase 1: Research
- Detect OS (Darwin/Linux/Windows)
- Check Python 3.10+
- Check Ollama installed status
- Understand Ollama requirements
- Create research.md
- Wait for user approval

# Phase 2: Plan
- Create 13-task sequence
- Define success criteria for each
- Assess risks
- Create plan.md
- Document manual actions
- Wait for user approval

# Phase 3: Execute
- Pre-flight checklist
- Execute tasks 1-7 automatically
- Display instructions for Task 8 (manual Ollama install)
- Execute tasks 9-13 (Ollama operations)
- Display instructions for parallel terminals
- Generate completion report
- Provide next steps
```

---

## Final Notes

- **This is the Ollama-focused version** - No cloud API keys needed
- **Much simpler than agent-instructions.md** - Fewer configuration choices
- **Terminal management is critical** - Keep Ollama server running in Terminal 1
- **Verify each task before proceeding** - Don't skip verification steps
- **Handle large downloads gracefully** - Show progress for model downloads
- **Clear next steps** - User should know exactly what to do after setup

---

**This document provides specialized instructions for AI agents to setup DocBases with Ollama in 30-40 minutes with zero API key complexity.**
