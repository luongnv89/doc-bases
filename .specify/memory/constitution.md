<!--
================================================================================
SYNC IMPACT REPORT
================================================================================
Version change: N/A (initial) → 1.0.0
Modified principles: N/A (initial creation)
Added sections:
  - Core Principles (5 principles)
  - Code Quality & Testing section
  - Development Workflow section
  - Governance section
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md: N/A (already generic)
  - .specify/templates/spec-template.md: N/A (already generic)
  - .specify/templates/tasks-template.md: N/A (already generic)
Follow-up TODOs: None
================================================================================
-->

# DocBases Constitution

## Core Principles

### I. CLI-First User Experience

DocBases MUST provide an intuitive command-line interface as the primary interaction
method. All features MUST be accessible via CLI commands with clear, consistent
syntax. User prompts MUST guide users through multi-step operations with numbered
options. Error messages MUST be actionable and displayed using rich formatting for
clarity.

**Rationale**: The CLI is the primary interface for developers and power users
working with document knowledge bases. A consistent, well-designed CLI reduces
friction and improves productivity.

### II. Multi-Source Document Ingestion

The system MUST support loading documents from multiple source types:
- GitHub repositories
- Local files (various formats including PDF)
- Local folders
- Website URLs
- Downloadable file URLs

Each source type MUST have dedicated loading logic with appropriate error handling.
New source types SHOULD be added as independent modules that conform to the existing
loader interface.

**Rationale**: Users have documents scattered across different locations and formats.
Supporting multiple sources makes DocBases a comprehensive solution rather than
requiring users to manually consolidate their documents.

### III. Knowledge Base Isolation

Each knowledge base MUST be stored independently in the `knowledges/` directory.
Knowledge bases MUST be uniquely named based on their source. Operations on one
knowledge base MUST NOT affect others. Users MUST be able to list, query, and
delete knowledge bases independently.

**Rationale**: Isolation ensures that users can manage multiple document collections
without interference, enabling organized workflows for different projects or topics.

### IV. Conversational Context Preservation

The RAG system MUST maintain multi-turn conversation memory within a session.
Follow-up questions MUST have access to prior context. The system MUST use an
agentic approach that can reason over multiple steps and chain tool calls for
complex queries.

**Rationale**: Document querying is inherently iterative. Users refine their
questions based on previous answers, and the system must support this natural
interaction pattern.

### V. Observability & Logging

All significant operations MUST be logged with appropriate severity levels.
Logging MUST be toggleable by users at runtime. Log messages MUST include
contextual information (e.g., knowledge base names, source URLs, user actions).
Rich console output MUST use consistent theming and styling.

**Rationale**: Debugging RAG systems requires visibility into document loading,
embedding generation, and query processing. Toggleable logging allows users to
enable detailed output when troubleshooting without cluttering normal usage.

## Code Quality & Testing

### Python Environment

All Python code MUST be executed within a virtual environment. Dependencies MUST
be managed via `requirements.txt`. The project MUST NOT modify system-wide Python
packages.

### Code Organization

- Models (embeddings, LLM) MUST reside in `src/models/`
- Utility functions MUST reside in `src/utils/`
- The main entry point MUST be `src/main.py`
- Configuration MUST be loaded from environment variables via `.env` files

### Error Handling

- User-facing errors MUST be displayed with `[error]` styling
- Warnings MUST use `[warning]` styling
- Success messages MUST use `[success]` styling
- All exceptions in document loading MUST be caught and reported gracefully

## Development Workflow

### Feature Development

1. Features MUST be specified in `.specify/` before implementation
2. User stories MUST be prioritized (P1, P2, P3) with independent testability
3. Implementation MUST follow the task breakdown in `tasks.md`
4. Each user story SHOULD be deliverable as a standalone increment

### Code Review Requirements

- Changes to core RAG logic (`rag_utils.py`) require careful review
- Changes to document loaders require testing with actual sources
- CLI changes require validation of user interaction flow

### Commit Guidelines

- Commits MUST NOT include auto-generated co-author attributions
- Commit messages MUST describe the "why" not just the "what"
- Commits SHOULD be atomic and focused on a single change

## Governance

### Amendment Process

1. Proposed changes MUST be documented with rationale
2. Changes to Core Principles require version MAJOR bump
3. New sections or expanded guidance require version MINOR bump
4. Clarifications and typo fixes require version PATCH bump

### Compliance

- All PRs MUST verify compliance with this constitution
- Complexity beyond these principles MUST be explicitly justified
- The constitution supersedes conflicting practices elsewhere in documentation

### Versioning Policy

Version follows MAJOR.MINOR.PATCH format:
- MAJOR: Backward-incompatible principle changes or removals
- MINOR: New principles, sections, or material expansions
- PATCH: Clarifications, wording improvements, non-semantic refinements

**Version**: 1.0.0 | **Ratified**: 2025-12-01 | **Last Amended**: 2025-12-01
