# DocBases v2.0 - AI Agent Installation & Setup Instructions

This document provides comprehensive instructions for an AI agent to perform a complete, from-scratch installation and setup of the DocBases project. The process is divided into three phases: Research, Plan, and Execute.

> **📌 Important:** For **Ollama-based setup** (recommended, no API keys needed), see **[agent-instructions-ollama.md](agent-instructions-ollama.md)** instead. That version is simpler and focused on local inference.

> **✓ NEW:** Installation is now **fully automated**! See **[INSTALLATION.md](INSTALLATION.md)** for quick setup, or **[SETUP_REPORT.md](SETUP_REPORT.md)** for completed setup documentation.

---

## Overview

**Project:** DocBases v2.0 - Retrieval-Augmented Generation (RAG) System
**Language:** Python 3.11+
**Type:** CLI Application
**Purpose:** Query documents from various sources using natural language with an AI-powered RAG system

---

# PHASE 1: RESEARCH

## Phase 1 Objective

The agent must gather all necessary information about the project requirements, dependencies, and configuration to understand what needs to be installed and configured. The agent should NOT execute any installation commands during this phase.

### Task 1.1: Analyze Project Structure

**What the agent must do:**
1. Explore the project directory structure
2. Identify the project type and entry points
3. Locate all configuration files
4. Read and understand the project documentation

**Files to examine:**
- `README.md` - Project overview and features
- `INSTALLATION.md` - Basic installation instructions
- `pyproject.toml` - Project metadata and tool configurations
- `.env.example` - Environment variables and their purposes
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies
- `.pre-commit-config.yaml` - Code quality configuration
- `src/main.py` - Main entry point

**Information to extract:**
- Project type (CLI application)
- Python version requirement (3.11+)
- Main entry point (`python src/main.py`)
- Key directories (`src/`, `tests/`, `knowledges/`)
- Core features and dependencies

### Task 1.2: Identify System Requirements

**What the agent must do:**
1. Detect the operating system (Darwin/macOS, Linux, Windows)
2. Check Python version availability
3. Identify system-level dependencies

**System requirements to identify:**
- **Python:** 3.11 or higher (verify with `python3 --version` or `python --version`)
- **System Library:** libmagic1/libmagic (for file type detection)
  - macOS: Usually included, but can be installed via Homebrew
  - Ubuntu/Debian: Need to install via apt-get
  - Windows: Need to download DLL
- **Git:** For repository operations (verify with `git --version`)

**Commands the agent should verify (but not execute):**
```bash
python3 --version  # Check Python 3 availability
pip --version      # Check pip installation
git --version      # Check git installation
```

### Task 1.3: Analyze Dependencies

**What the agent must do:**
1. Read `requirements.txt` and understand production dependencies
2. Read `requirements-dev.txt` and understand development dependencies
3. Categorize dependencies by purpose

**Key dependencies to understand:**

**Core Framework:**
- `langchain`, `langchain-core`, `langchain-community` - LLM framework
- `langgraph`, `langgraph-checkpoint-sqlite` - Agent workflow framework
- `chromadb`, `langchain-chroma` - Vector database

**LLM Providers (at least one required):**
- `langchain-openai` - OpenAI integration
- `langchain-google-genai` - Google Gemini integration
- `langchain-groq` - Groq integration
- `langchain-ollama` - Local Ollama support

**Utilities:**
- `python-dotenv` - Environment variable loading
- `rich` - Terminal output formatting
- `requests` - HTTP requests

**Document Processing:**
- `unstructured` - Document parsing
- `pypdf` - PDF handling
- `python-docx` - Word document handling

**Development:**
- `pytest`, `pytest-cov`, `pytest-xdist` - Testing
- `black`, `ruff`, `flake8` - Code quality
- `mypy` - Type checking
- `pre-commit` - Git hooks

### Task 1.4: Understand Configuration Requirements

**What the agent must do:**
1. Analyze `.env.example` to understand all configuration options
2. Identify required vs. optional environment variables
3. Understand default values

**Configuration sections:**

1. **LLM Provider Configuration** (REQUIRED)
   - `LLM_PROVIDER`: openai, google, groq, or ollama
   - `LLM_MODEL`: Model name for the chosen provider
   - `LLM_API_BASE`: Optional, for custom/local APIs

2. **API Keys** (REQUIRED for cloud providers)
   - `OPENAI_API_KEY`: For OpenAI models
   - `GOOGLE_API_KEY`: For Google Gemini
   - `GROQ_API_KEY`: For Groq models
   - (Not needed for Ollama)

3. **Embedding Configuration** (REQUIRED)
   - `EMB_PROVIDER`: openai, google, or ollama
   - `EMB_MODEL`: Embedding model name
   - `EMB_API_BASE`: For custom/local embeddings

4. **RAG Mode** (OPTIONAL, defaults to 'basic')
   - Choices: basic, corrective, adaptive, multi_agent

5. **Document Processing** (OPTIONAL)
   - `CHUNKING_STRATEGY`: recursive or semantic
   - `USE_DOCLING`: true or false (advanced parsing)

6. **Persistence** (OPTIONAL)
   - `USE_PERSISTENT_MEMORY`: Enable/disable memory persistence
   - Database paths for checkpoints and metrics

7. **Observability** (OPTIONAL)
   - `LANGSMITH_TRACING`: Enable LangSmith monitoring
   - LangSmith API key and project name

### Task 1.5: Identify Ambiguities and Ask User for Clarification

**What the agent must do:**
1. Identify any configuration choices that require user input
2. Determine if user wants production or development setup
3. Ask about API key availability

**Key questions to ask the user:**

```
Question 1: Installation Type
- Full development setup (with testing, linting, pre-commit)
- Production setup (core dependencies only)
- Which would you prefer? [full / production]

Question 2: LLM Provider Selection
- User must choose ONE provider: openai, google, groq, or ollama
- Do you have an API key for any of these? [openai / google / groq / none/ollama]

Question 3: Embedding Provider (if not using Ollama for LLM)
- Same providers available as LLM
- Can it be the same as LLM provider or different?

Question 4: Platform Confirmation
- Detected OS: [Darwin/Linux/Windows]
- Is this correct? [yes / no]
```

### Task 1.6: Create research.md

**What the agent must do:**
1. Document all findings from Phase 1
2. Create a file named `research.md` with findings
3. Include detected system information, dependencies, configuration requirements, and identified questions

**research.md should include:**
- Detected OS and Python version
- System library requirements
- Dependency categories and counts
- Configuration sections and examples
- List of all identified questions for the user
- Recommendations for the installation approach

### Phase 1 User Verification

**What the agent must do:**
1. Summarize the key findings in a brief message
2. Direct user to read the detailed `research.md` file
3. Ask user to provide answers to any clarification questions
4. **Wait for explicit user approval before proceeding to Phase 2**

**Agent should say to user:**
```
PHASE 1 RESEARCH COMPLETE

I have analyzed the project and identified:
- [Count] production dependencies
- [Count] development dependencies
- [Count] configuration sections
- [Count] important questions

Please review the detailed findings in research.md and provide clarification on:
1. [Question 1]
2. [Question 2]
3. [Question 3]

Proceed to Phase 2? [yes / no]
```

---

# PHASE 2: PLAN

## Phase 2 Objective

Based on the approved research findings, the agent must create a comprehensive, detailed plan for installation and setup. The agent should NOT execute any installation commands during this phase.

### Task 2.1: Break Down Installation into Numbered Tasks

**What the agent must do:**
1. Create a sequence of numbered installation tasks
2. Define success criteria for each task
3. Define verification methods for each task
4. Identify high-risk operations

**Task Sequence Structure:**

```
Task 1: Verify Python Version
  - Command: python3 --version
  - Success Criterion: Python 3.11 or higher installed
  - Verification: Run command and parse version

Task 2: Install System Library (libmagic)
  - Command: [OS-specific: brew install libmagic or apt-get install libmagic1]
  - Success Criterion: libmagic development files installed
  - Verification: Check for libmagic.so or check python-magic availability

Task 3: Create Virtual Environment
  - Command: python3 -m venv venv
  - Success Criterion: Virtual environment directory created
  - Verification: Check if venv/bin/activate exists

Task 4: Activate Virtual Environment
  - Command: source venv/bin/activate (on Windows: venv\Scripts\activate)
  - Success Criterion: Virtual environment activated
  - Verification: Check $VIRTUAL_ENV or python location

Task 5: Upgrade pip
  - Command: python -m pip install --upgrade pip
  - Success Criterion: pip upgraded to latest version
  - Verification: pip --version shows latest

Task 6: Install Production Dependencies
  - Command: pip install -r requirements.txt
  - Success Criterion: All packages installed successfully
  - Verification: Import all core packages (langchain, chromadb, etc.)

Task 7: [Optional] Install Development Dependencies
  - Command: pip install -r requirements-dev.txt
  - Success Criterion: Dev tools installed
  - Verification: Run pre-commit --version, pytest --version

Task 8: [Optional] Set Up Pre-commit Hooks
  - Command: pre-commit install
  - Success Criterion: Git hooks installed
  - Verification: Check .git/hooks/pre-commit exists

Task 9: Create .env File from Template
  - Command: cp .env.example .env
  - Success Criterion: .env file created
  - Verification: Check if .env exists

Task 10: [Manual] Configure Environment Variables
  - Manual Input Required: User must edit .env file
  - Success Criterion: .env configured with valid API keys/settings
  - Verification: [Depends on configuration]

Task 11: Verify Installation
  - Command: python -c "from src.main import main; print('✓ Import successful')"
  - Success Criterion: Main module imports successfully
  - Verification: Script runs without errors

Task 12: [Optional] Run Tests
  - Command: pytest tests/ -v
  - Success Criterion: All tests pass (158/158)
  - Verification: Check test output for PASSED count
```

### Task 2.2: Define Success Criteria and Verification Methods

**What the agent must do:**
1. For each task, clearly define what "success" means
2. Provide specific verification commands that can be automated

**Example format:**

```
Task: Install Production Dependencies
  Success Criterion: All packages in requirements.txt installed
  Verification Method:
    python -c "
    import langchain, chromadb, langgraph
    import requests, python_dotenv, rich
    print('✓ All core packages installed')
    "
  Rollback Strategy: Delete venv directory and recreate
```

### Task 2.3: Identify Risk Assessment and Permission Gates

**What the agent must do:**
1. Identify tasks that require explicit user permission
2. Mark high-risk operations

**Risk Categories:**

1. **High-Risk Tasks** (require explicit permission):
   - Installing system packages (might require sudo)
   - Modifying .env file (contains sensitive API keys)
   - Pre-commit hook installation (modifies .git directory)

2. **Medium-Risk Tasks** (should confirm):
   - Virtual environment creation (modifies filesystem)
   - Package installation (might conflict with system packages)

3. **Low-Risk Tasks** (can proceed automatically):
   - pip upgrade (isolated to venv)
   - Verification commands (read-only)

**Permission gates format:**
```
Task 5: Install System Package (libmagic)
  Risk Level: HIGH
  Requires: sudo access
  Message to User: "This task requires installing a system library. Do you have sudo access? [yes/no]"
```

### Task 2.4: Create plan.md

**What the agent must do:**
1. Document the complete installation plan
2. Include all numbered tasks with success criteria
3. Include verification methods
4. Include risk assessments and permission gates
5. Include rollback strategies for critical tasks

**plan.md structure:**
```
# DocBases Installation Plan

## Overview
- Total Tasks: [number]
- Estimated Duration: [rough estimate]
- High-Risk Tasks: [count]
- Manual Input Required: [count]

## Pre-Flight Checklist
- [ ] Python 3.11+ available
- [ ] pip available
- [ ] git available
- [ ] User has appropriate permissions (sudo if needed)
- [ ] API keys available (if using cloud providers)

## Task Sequence
[Detailed numbered task list with criteria, verification, and rollback]

## Manual Input Summary
[Consolidated list of all manual inputs needed]

## Success Indicators
[How to know installation succeeded completely]
```

### Task 2.5: Consolidate Manual Input Requirements

**What the agent must do:**
1. Identify all manual inputs required from the user
2. Group them into a clear summary
3. Create a separate guide to help user complete them

**Manual inputs to consolidate:**

1. **API Keys** (if using cloud providers)
   - OPENAI_API_KEY (for OpenAI)
   - GOOGLE_API_KEY (for Google)
   - GROQ_API_KEY (for Groq)

2. **Configuration Choices**
   - LLM_PROVIDER selection
   - LLM_MODEL selection
   - EMB_PROVIDER selection
   - EMB_MODEL selection
   - RAG_MODE selection (optional, has default)
   - CHUNKING_STRATEGY (optional, has default)

3. **Optional Settings**
   - USE_PERSISTENT_MEMORY (optional, has default)
   - LANGSMITH_TRACING (optional, has default)

**human_tasks.md format:**
```
# Manual Configuration Tasks for DocBases

## Task 1: Get LLM API Key [USER CHOICE REQUIRED]
Choose one provider and get API key:
- OpenAI: Go to https://platform.openai.com/api-keys
- Google: Go to https://aistudio.google.com/app/apikey
- Groq: Go to https://console.groq.com/keys
- Ollama: Use local model (free, no key needed)

## Task 2: Get Embedding API Key [USER CHOICE REQUIRED]
Choose embedding provider and get API key...

## Task 3: Edit .env File [USER ACTION REQUIRED]
Fill in the following fields in .env:
- LLM_PROVIDER: [user's choice]
- LLM_MODEL: [user's choice]
- OPENAI_API_KEY: [if applicable]
... etc
```

### Phase 2 User Verification

**What the agent must do:**
1. Display a summary of the plan
2. Highlight key milestones and critical tasks
3. Show all high-risk operations requiring approval
4. Present the consolidated manual input requirements
5. **Wait for explicit user approval before proceeding to Phase 3**

**Agent should say to user:**
```
PHASE 2 PLAN CREATED

Installation plan includes:
- [X] system package installation (libmagic)
- [X] virtual environment setup
- [X] dependency installation ([count] packages)
- [X] environment configuration (requires manual API keys)
- [X] verification tests

HIGH-RISK OPERATIONS:
1. System package installation (requires sudo)
2. API key entry (sensitive data)

Please review plan.md and manual configuration requirements in human_tasks.md.

Before proceeding, you will need:
- [API keys or Ollama setup]
- [Other requirements]

Ready to execute Phase 3? [yes / no]
```

---

# PHASE 3: EXECUTE

## Phase 3 Objective

Execute all planned tasks from Phase 2 with proper verification, error handling, and user feedback. The agent should execute tasks sequentially and verify each one before proceeding.

### Task 3.1: Pre-Execution Checklist

**What the agent must do:**
1. Confirm user has approved the plan
2. Verify all manual inputs are ready
3. Confirm user understands high-risk operations

**Checklist:**
- [ ] Phase 2 plan.md reviewed and approved
- [ ] human_tasks.md completed (if required)
- [ ] API keys/credentials ready
- [ ] User confirms understanding of high-risk operations
- [ ] User has appropriate system access (sudo if needed)

**If checklist fails, STOP and ask user to reread plan.md**

### Task 3.2: Sequential Task Execution

**What the agent must do:**
1. Execute each task in order
2. For each task:
   - Display the task name and success criterion
   - If high-risk, request user permission
   - Execute the command
   - Run verification immediately after
   - Confirm success before proceeding
3. Never proceed past a failed task

**Execution Template for Each Task:**

```
═══════════════════════════════════════════════════════════════
TASK [N/M]: [Task Name]
═══════════════════════════════════════════════════════════════
Success Criterion: [Define what success means]

[If high-risk]:
⚠️  This is a high-risk operation.
    [Explain what will happen]
    Proceed? [yes / no]

Executing: [Command to run]
[Execute command and show output]

Verifying: [Verification command]
[Run verification and show result]

✓ Task [N] completed successfully
```

### Task 3.3: Task-Specific Execution Details

**For Each Task, the Agent Should:**

**Task 1: Verify Python Version**
```
Command: python3 --version
Verification: Check output contains "Python 3.1[1-9]" or higher
Rollback: N/A (read-only verification)
```

**Task 2: Install System Library (libmagic)**
```
Command (macOS): brew install libmagic
Command (Ubuntu): sudo apt-get install libmagic1
Command (Windows): Download from GitHub and add to PATH
Verification: python -c "import magic; print(magic.Magic())"
Rollback: brew uninstall libmagic OR sudo apt-get remove libmagic1
Risk: HIGH - requires sudo on Linux
```

**Task 3-4: Create and Activate Virtual Environment**
```
Command: python3 -m venv venv
Activation: source venv/bin/activate (or venv\Scripts\activate on Windows)
Verification: which python shows path inside venv/
Rollback: rm -rf venv
Risk: LOW
```

**Task 5: Upgrade pip**
```
Command: python -m pip install --upgrade pip
Verification: pip --version shows latest version
Rollback: Re-run within new venv
Risk: LOW
```

**Task 6: Install Production Dependencies**
```
Command: pip install -r requirements.txt
Verification:
  python -c "
  import langchain, langgraph, chromadb
  import requests, python_dotenv, rich
  print('✓ Core dependencies installed')
  "
Rollback: pip uninstall -r requirements.txt -y && rm -rf venv
Risk: MEDIUM - might take several minutes
Allow Installation Time: 5-10 minutes depending on system
```

**Task 7: Install Development Dependencies (optional)**
```
Command: pip install -r requirements-dev.txt
Verification: pytest --version && black --version && flake8 --version
Rollback: pip uninstall -r requirements-dev.txt -y
Risk: LOW (only if user chose full development setup)
```

**Task 8: Setup Pre-commit Hooks (optional)**
```
Command: pre-commit install
Verification: ls -la .git/hooks/pre-commit (should exist and be executable)
Rollback: rm .git/hooks/pre-commit
Risk: MEDIUM - modifies .git directory
Requires: Pre-commit to be installed (Task 7)
```

**Task 9: Create .env File**
```
Command: cp .env.example .env
Verification: ls -la .env (file should exist)
Rollback: rm .env
Risk: LOW
```

**Task 10: Configure Environment Variables (MANUAL)**
```
Action: User must edit .env file
Success Criterion: .env file contains valid configuration
Verification:
  - .env file exists
  - User confirms configuration complete
  - Optionally: python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('LLM_PROVIDER'))"
Risk: HIGH - contains sensitive API keys
Do NOT: Log the contents of .env file
```

**Task 11: Verify Installation**
```
Command: python -c "from src.main import main; print('✓ Import successful')"
Verification: Command runs without errors
Rollback: Check Python path, reactivate venv if needed
Risk: LOW - read-only test
```

**Task 12: Run Tests (optional)**
```
Command: pytest tests/ -v --tb=short
Success Criterion: 158 tests pass
Verification: Check output for "158 passed"
Allow Test Time: 10-15 seconds
Rollback: N/A (read-only tests)
Risk: LOW - if tests fail, they indicate environment issues
```

### Task 3.4: Error Handling and Recovery

**What the agent must do if a task fails:**

1. **Immediately STOP execution**
2. **Document the failure:**
   - What task failed
   - What command was executed
   - What the error was
   - When it occurred (task N of M)

3. **Present error to user with context:**
   ```
   ❌ TASK FAILED: Task 6 - Install Production Dependencies

   Command: pip install -r requirements.txt

   Error Output:
   [Show relevant error lines]

   This likely means:
   [Diagnosis]

   Options:
   1. Retry (try command again)
   2. Skip (skip this task, may cause later issues)
   3. Abort (stop installation)
   4. Troubleshoot (show diagnostic steps)
   ```

4. **Wait for user decision:**
   - `retry`: Execute the command again
   - `skip`: Skip this task and continue to next
   - `abort`: Stop execution and summarize progress
   - `troubleshoot`: Show diagnostic information

5. **If retry chosen:** Execute command again with updated output

6. **If skip chosen:** Warn user about potential issues and continue

7. **If abort chosen:** Create summary report and stop

### Task 3.5: Completion Report

**What the agent must do after all tasks complete (or abort):**

1. **Create Summary Report:**
   ```
   ═══════════════════════════════════════════════════════════════
   INSTALLATION REPORT
   ═══════════════════════════════════════════════════════════════

   Status: [COMPLETE / PARTIAL / FAILED]
   Timestamp: [When completed]
   Total Tasks: [X of Y completed]

   COMPLETED TASKS:
   ✓ Task 1: Python Version Verified
   ✓ Task 2: System Library Installed
   ✓ Task 3: Virtual Environment Created
   ... [full list]

   SKIPPED TASKS:
   ⊘ Task 7: Development Dependencies (user skipped)

   FAILED TASKS:
   ✗ Task X: [Description] - [Error]

   NEXT STEPS:
   1. Activate virtual environment: source venv/bin/activate
   2. Configure .env file with your API keys
   3. Start the application: python src/main.py

   VERIFICATION:
   Run: python -c "from src.main import main; print('✓ Ready')"
   ```

2. **Provide Usage Instructions:**
   - How to activate venv in future sessions
   - How to run the main application
   - How to run tests
   - How to modify .env configuration

3. **Offer Troubleshooting:**
   - Common issues and solutions
   - How to check installed packages
   - How to reinstall if needed

---

## Implementation Notes for AI Agent

### How to Implement Phase 1

```python
# Pseudo-code structure
class Phase1Research:
    def run(self):
        self.analyze_structure()           # Read README, pyproject.toml, etc.
        self.detect_system_info()          # Detect OS, Python version
        self.read_dependencies()           # Parse requirements.txt
        self.understand_configuration()    # Analyze .env.example
        self.identify_ambiguities()        # Ask clarification questions
        self.create_research_md()          # Document findings
        self.present_summary_to_user()     # Show summary, wait for approval
        return research_data
```

### How to Implement Phase 2

```python
# Pseudo-code structure
class Phase2Plan:
    def run(self, research_data):
        self.create_task_sequence()        # Break into numbered tasks
        self.define_success_criteria()     # For each task
        self.define_verification()         # Verification methods
        self.assess_risks()                # Identify high-risk tasks
        self.create_plan_md()              # Document plan
        self.consolidate_manual_inputs()   # Create human_tasks.md
        self.present_summary_to_user()     # Show plan, wait for approval
        return plan_data
```

### How to Implement Phase 3

```python
# Pseudo-code structure
class Phase3Execute:
    def run(self, plan_data):
        self.pre_flight_check()            # Verify readiness

        for task in plan_data.tasks:
            self.display_task_header(task)

            if task.is_high_risk:
                self.request_permission()  # Ask user

            result = self.execute_task(task)  # Run command

            if not result.success:
                result = self.handle_error(task, result)  # Ask user
                if not result.should_continue:
                    break

            self.verify_task(task, result)  # Verify success
            self.log_completion(task)

        self.generate_report()             # Create summary
        self.offer_next_steps()            # Usage instructions
```

---

## Success Criteria

### Phase 1 Success
- research.md file created with detailed findings
- All ambiguities identified and questions listed
- User has reviewed research.md and provided clarifications
- Explicit user approval to proceed to Phase 2

### Phase 2 Success
- plan.md file created with all tasks
- human_tasks.md created with manual input guide
- Success criteria and verification methods defined
- Risk assessments documented
- User has reviewed both files
- Explicit user approval to proceed to Phase 3

### Phase 3 Success
- All tasks execute in order
- Each task verifies successfully before proceeding
- .env file configured with valid settings
- Main module imports successfully
- Installation report generated
- User has clear instructions for next steps

---

## Troubleshooting Guide

### Common Issues During Installation

**Issue: "Python 3.11+ not found"**
- Solution: Install Python 3.11+ from python.org
- Verification: `python3 --version`

**Issue: "libmagic not found"**
- macOS: `brew install libmagic`
- Linux: `sudo apt-get install libmagic1`
- Windows: Download from GitHub or use Windows subsystem

**Issue: "pip install fails with permission error"**
- Solution: Ensure virtual environment is activated
- Verify: `which python` should show path inside venv/

**Issue: "API key invalid"**
- Solution: Get fresh API key from provider
- Verify: Paste key into .env exactly as provided
- Check: Don't include quotes or extra spaces

**Issue: "Module import fails"**
- Solution: Reinstall dependencies: `pip install -r requirements.txt`
- Verify: `python -c "import langchain"`

### How to Debug

1. **Check Python environment:**
   ```bash
   which python
   python --version
   pip --version
   ```

2. **Check virtual environment:**
   ```bash
   echo $VIRTUAL_ENV
   ls venv/bin/  # or venv\Scripts\ on Windows
   ```

3. **Check installed packages:**
   ```bash
   pip list
   pip show langchain
   ```

4. **Test imports:**
   ```bash
   python -c "import langchain; print(langchain.__version__)"
   ```

5. **Check .env configuration:**
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('LLM_PROVIDER'))"
   ```

---

## Quick Reference for Agent

### Files to Create/Modify During Installation

| File | Phase | Action | Purpose |
|------|-------|--------|---------|
| research.md | 1 | Create | Document research findings |
| plan.md | 2 | Create | Document installation plan |
| human_tasks.md | 2 | Create | Manual configuration guide |
| venv/ | 3 | Create | Virtual environment directory |
| .env | 3 | Create from template | Configuration file |

### Key Commands Reference

| Command | Purpose | Phase |
|---------|---------|-------|
| `python3 --version` | Check Python | 1 & 3 |
| `python3 -m venv venv` | Create venv | 3 |
| `source venv/bin/activate` | Activate venv | 3 |
| `pip install -r requirements.txt` | Install deps | 3 |
| `pre-commit install` | Setup hooks | 3 |
| `pytest tests/` | Run tests | 3 |
| `python src/main.py` | Start app | After 3 |

### Success Indicators

- ✓ research.md created with detailed findings
- ✓ plan.md created with complete task list
- ✓ Virtual environment successfully created and activated
- ✓ All dependencies installed without errors
- ✓ .env file configured with valid API keys
- ✓ Main module imports without errors
- ✓ Tests pass (if development setup)

---

## Final Notes

- **Do not skip Phase 1 or 2:** These phases are critical for understanding the project and planning properly
- **Always wait for user approval:** Each phase requires explicit user approval before proceeding
- **Document everything:** Create research.md and plan.md for future reference
- **Handle errors gracefully:** Never proceed past a failed task without user input
- **Provide clear feedback:** Users should always know what's happening and why
- **Test at each step:** Verify each task succeeds before continuing

---

**This document provides complete instructions for an AI agent to perform a full installation and setup of DocBases v2.0. Follow the three-phase model strictly for optimal results.**
