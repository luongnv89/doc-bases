# GitHub RAG System

A Retrieval-Augmented Generation (RAG) system for querying GitHub repositories.

## Features
- Clone and parse GitHub repositories.
- Set up and query knowledge bases.
- Interactive CLI for querying repositories.

## Installation

### Prerequisites
- Python 3.9 or higher.
- [Poetry](https://python-poetry.org/) for dependency management.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/github-rag.git
   cd github-rag
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in the required keys.

## Usage

### Running the App
To start the app, run:
```bash
poetry run python src/main.py
```

### Commands
1. **Setup RAG**: Set up a new RAG system for a GitHub repository.
2. **List Knowledge Bases**: List all available knowledge bases.
3. **Delete Knowledge Base**: Delete a specific knowledge base.
4. **Interactive CLI**: Start an interactive CLI for querying the RAG system.
5. **Exit**: Exit the script.

## Development

### Installing Development Dependencies
To install development dependencies (e.g., `black`, `flake8`, `pytest`), run:
```bash
poetry install --with dev
```

### Running Tests
Run the test suite using:
```bash
poetry run pytest tests/
```

### Formatting and Linting
- Format the code using `black`:
  ```bash
  poetry run black src/ tests/
  ```

- Lint the code using `flake8`:
  ```bash
  poetry run flake8 src/ tests/
  ```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.