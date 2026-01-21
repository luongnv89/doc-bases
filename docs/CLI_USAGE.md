# DocBases CLI Usage Guide

## Overview

DocBases provides a professional command-line interface (CLI) named `docb` for managing your knowledge bases and executing queries. The CLI offers both interactive and non-interactive modes for different use cases.

## Installation

After installing dependencies, the CLI is immediately available:

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# View available commands
./docb --help
```

## Command Structure

The CLI is organized into command groups:

```
docb [global options] <command> [command options]
```

### Global Options

- `--version`, `-v` - Show version and exit
- `--verbose` - Enable verbose output
- `--config`, `-c <path>` - Use custom config file
- `--help` - Show help message

## Configuration Management

### List Configuration

```bash
# Show all current configuration
./docb config list

# Show all with sources (where values came from)
./docb config list --all

# Show including API keys (use with caution!)
./docb config list --secrets
```

### Get/Set Configuration

```bash
# Get a specific configuration value
./docb config get llm.provider
./docb config get llm.model

# Set a configuration value
./docb config set llm.provider openai
./docb config set llm.model gpt-4o-mini
./docb config set rag.mode adaptive

# Set without confirmation
./docb config set llm.provider openai --force
```

### Import/Export Configuration

```bash
# Export configuration to YAML
./docb config export
./docb config export --output my-config.yaml

# Import configuration from YAML
./docb config import my-config.yaml
./docb config import my-config.yaml --force

# Reset to defaults
./docb config reset
./docb config reset --force
```

### Validate Configuration

```bash
# Check if configuration is valid
./docb config validate
```

## Setup Commands

### Complete Setup

```bash
# Run full setup (venv, dependencies, config, directories, validation)
./docb setup init

# Setup with custom Python version
./docb setup init --python 3.11

# Skip certain steps
./docb setup init --skip-venv
./docb setup init --skip-install
```

### Individual Setup Steps

```bash
# Create virtual environment
./docb setup venv
./docb setup venv --python 3.10

# Install dependencies
./docb setup install
./docb setup install --dev    # Include dev dependencies

# Setup .env file
./docb setup env

# Create data directories
./docb setup dirs

# Validate installation
./docb setup validate
```

## Knowledge Base Management

### Add Knowledge Base

```bash
# Add from GitHub repository
./docb kb add repo https://github.com/langchain-ai/langchain

# Add from local file
./docb kb add file ./my-document.pdf
./docb kb add file ./my-document.md

# Add from folder
./docb kb add folder ./docs
./docb kb add folder ./research

# Add from website
./docb kb add website https://python.langchain.com/docs

# Add from download URL
./docb kb add url https://example.com/large-document.pdf

# Add with custom name
./docb kb add repo https://github.com/langchain-ai/langchain --name my-langchain-kb

# Overwrite existing KB
./docb kb add file ./docs --overwrite
./docb kb add file ./docs -o  # Short form
```

### List Knowledge Bases

```bash
# List all knowledge bases
./docb kb list
```

### Get Knowledge Base Info

```bash
# Show details about a knowledge base
./docb kb info my-kb-name
./docb kb info langchain
```

The `kb info` command displays:
- **Name**: Knowledge base identifier
- **Path**: Storage location
- **Vector Store**: ChromaDB status
- **Source Type**: How the KB was created (folder, file, repo, website, url)
- **Source Path**: Original source location
- **Last Sync**: When the KB was last indexed
- **Indexed Files**: Number of tracked source files
- **Storage Files**: Files in the KB directory
- **Size**: Total storage size

### Delete Knowledge Base

```bash
# Delete a knowledge base (with confirmation)
./docb kb delete my-kb-name

# Delete without confirmation
./docb kb delete my-kb-name --force
./docb kb delete my-kb-name -f  # Short form
```

## File Change Detection

DocBases automatically tracks source files and detects changes since the last sync. When you start a query session, the system checks if any source files have been added, modified, or deleted.

### How It Works

1. **On KB Creation**: Metadata is saved including file paths and modification times
2. **On Query Start**: Source files are scanned and compared to the indexed state
3. **If Changes Found**: You're prompted to continue or update the KB

### Change Detection Prompt

When changes are detected, you'll see:

```
Source files have changed since last sync:

  Added (2 files):
    • new-document.md
    • guide.txt

  Modified (1 file):
    • readme.md

  Deleted (1 file):
    • old-file.md

What would you like to do?
  [1] Continue without updating
  [2] Update KB now (shows re-index command)

Choice [1/2]:
```

### Supported Source Types

| Source Type | Change Detection |
|-------------|------------------|
| `folder` | Full tracking (added/modified/deleted) |
| `file` | Full tracking (modified) |
| `repo` | Limited (shows source URL only) |
| `website` | Limited (shows source URL only) |
| `url` | Limited (shows source URL only) |

### Re-indexing a Knowledge Base

If changes are detected and you want to update:

```bash
# The system shows the exact command to run
docb kb add folder /path/to/docs --name my-kb --overwrite
```

### Edge Cases

- **Old KBs without metadata**: Shows info message, continues normally
- **Missing source path**: Warning displayed, allows continuing with outdated data
- **Large change sets**: Truncated to first 10 files + count of remaining

## Querying Knowledge Bases

### Interactive Mode

```bash
# Start interactive session with first KB (or choose if multiple)
./docb query interactive

# Specify knowledge base
./docb query interactive --kb my-kb
./docb query interactive -k my-kb

# Specify RAG mode
./docb query interactive --mode adaptive
./docb query interactive -m corrective

# Specify session ID (for conversation history)
./docb query interactive --session my-session-123
./docb query interactive -s my-session-123
```

In interactive mode:
- Type natural language questions
- Type `exit` to quit
- Conversation history is automatically saved (if persistence enabled)

### Single Query

```bash
# Execute a single query
./docb query single --query "What is machine learning?"

# Specify knowledge base
./docb query single -q "What is this?" --kb my-kb

# With RAG mode
./docb query single -q "Explain the architecture" --mode adaptive
```

### Batch Queries

```bash
# Execute queries from file (one per line)
./docb query batch queries.txt

# Specify knowledge base
./docb query batch queries.txt --kb my-kb

# Save results to JSON
./docb query batch queries.txt --output results.json
./docb query batch queries.txt -o results.json
```

Example queries file:
```
What is the purpose of this?
How does this work?
Can you explain the architecture?
What are the main components?
```

## Testing

### Run Tests

```bash
# Run all tests
./docb test run

# Run with verbose output
./docb test run --verbose

# Run with coverage report
./docb test run --coverage

# Run specific module
./docb test run --module cli
./docb test run -m rag_utils

# Run only unit tests
./docb test run --unit

# Run only integration tests
./docb test run --integration
```

## Health Checks

### Full Health Check

```bash
# Run all health checks
./docb health check

# Check specific systems
./docb health check --llm              # LLM connectivity
./docb health check --embeddings       # Embeddings connectivity
./docb health check --vectorstore      # Vector store status
./docb health check --deps             # Dependencies

# Check multiple systems
./docb health check --llm --embeddings --deps
```

## Configuration Priority

Configuration is loaded in this order (highest to lowest priority):

1. **CLI Arguments** - `./docb config set llm.provider openai`
2. **Environment Variables** - `LLM_PROVIDER=openai ./docb query interactive`
3. **User Config** - `~/.docbases/config.yaml`
4. **Project Config** - `.docbases/config.yaml`
5. **Project .env** - `./.env`
6. **Defaults** - Built-in defaults

This means:
```bash
# Environment variables override config files
LLM_MODEL=gpt-4 ./docb query single -q "Question"

# But CLI args would take precedence if CLI supported them
```

## Configuration Files

### .env File

```env
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here

# Embedding Configuration
EMB_PROVIDER=openai
EMB_MODEL=text-embedding-3-small

# RAG Mode
RAG_MODE=adaptive

# Persistence
USE_PERSISTENT_MEMORY=true
```

### YAML Config Files

Location: `.docbases/config.yaml` or `~/.docbases/config.yaml`

```yaml
version: "1.0"

llm:
  provider: openai
  model: gpt-4o-mini
  api_base: null

embedding:
  provider: openai
  model: text-embedding-3-small

rag:
  mode: adaptive
  use_docling: true
  chunking_strategy: semantic

persistence:
  use_persistent_memory: true
  checkpoint_db_path: knowledges/checkpoints.db
```

## Common Workflows

### First-Time Setup

```bash
# 1. Complete setup
./docb setup init

# 2. Configure
./docb config set llm.provider openai
./docb config set llm.model gpt-4o-mini

# 3. Validate
./docb health check

# 4. Add knowledge base
./docb kb add repo https://github.com/langchain-ai/langchain

# 5. Start querying
./docb query interactive
```

### Using Multiple Knowledge Bases

```bash
# Add multiple KBs
./docb kb add repo https://github.com/langchain-ai/langchain --name langchain-docs
./docb kb add folder ./company-docs --name internal-docs

# List available KBs
./docb kb list

# Query specific KB
./docb query interactive --kb langchain-docs
./docb query single -q "What is this?" --kb internal-docs
```

### Development Workflow

```bash
# Make code changes...

# Run tests
./docb test run --coverage

# Check health
./docb health check

# Test with small KB
./docb kb add file ./test-document.md --name test
./docb query single -q "Test question" --kb test
```

### CI/CD Pipeline

```bash
#!/bin/bash
set -e

# Activate venv
source .venv/bin/activate

# Install
./docb setup install

# Test
./docb test run --coverage

# Health check
./docb health check --llm --embeddings

# Validate config
./docb config validate
```

## Troubleshooting

### LLM Connection Failed

```bash
# Check health
./docb health check --llm

# Verify configuration
./docb config get llm.provider
./docb config get llm.model

# Check API key
./docb config get llm.api_base
```

### Missing Knowledge Bases

```bash
# List existing KBs
./docb kb list

# Add new KB
./docb kb add repo <url>
```

### Configuration Issues

```bash
# Validate configuration
./docb config validate

# Check all settings
./docb config list --all

# Reset to defaults if needed
./docb config reset --force
```

## Backward Compatibility

The legacy menu-based CLI is still available:

```bash
# Run new CLI (default)
./docb query interactive

# Run legacy menu mode
python src/main.py --legacy
```

## Getting Help

```bash
# General help
./docb --help

# Command group help
./docb setup --help
./docb kb --help
./docb query --help

# Specific command help
./docb kb add --help
./docb query single --help
```

## Environment Variables

Set at runtime to override configuration:

```bash
# Override LLM provider
LLM_PROVIDER=groq ./docb query interactive

# Override RAG mode
RAG_MODE=corrective ./docb query interactive

# Override multiple settings
LLM_PROVIDER=openai RAG_MODE=adaptive ./docb kb add repo <url>
```

## Performance Tips

1. **Use single queries for one-off questions**: `./docb query single -q "..."`
2. **Use batch mode for many queries**: `./docb query batch queries.txt`
3. **Use adaptive RAG for mixed queries**: `./docb config set rag.mode adaptive`
4. **Enable metrics**: Check with `./docb health check`

## Security Notes

1. **Never commit `.env` files**: Add to `.gitignore`
2. **Use `--secrets` carefully**: Shows API keys in terminal
3. **Mask sensitive config**: Use environment variables instead of YAML files for secrets
4. **Restrict file permissions**: `.env` and `.docbases/` should be readable only by user

## Next Steps

- See [docs/ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [docs/DEVELOPMENT.md](DEVELOPMENT.md) for development guide
- See [README.md](../README.md) for overview
