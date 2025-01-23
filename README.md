# A RAG System

A Retrieval-Augmented Generation (RAG) system for querying documents from various sources, including GitHub repositories, local files, folders, websites, and downloadable files. This system allows you to set up and interact with knowledge bases using an interactive CLI.

## Features
- **Multiple Document Sources**: Load documents from GitHub repositories, local files, folders, websites, and downloadable files.
- **Knowledge Base Management**: Create, list, and delete knowledge bases.
- **Interactive CLI**: Query knowledge bases interactively using natural language.
- **Logging**: Toggle logs on or off for debugging and monitoring.

## Installation

### Prerequisites
- **Python 3.9 or higher**: Ensure Python is installed on your system.
- **pip**: Python's package installer.
- **Install `libmagic`**:
   - On **Ubuntu/Debian**:
     ```bash
     sudo apt-get install libmagic1
     ```
   - On **macOS** (using Homebrew):
     ```bash
     brew install libmagic
     ```
   - On **Windows**:
     - Download and install the [DLL version of libmagic](https://github.com/pidydx/libmagicwin64).
     - Add the path to the DLL to your system's `PATH` environment variable.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/github-rag.git
   cd github-rag
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Fill in the required keys in the `.env` file (e.g., GitHub API token, OpenAI API key).

## Usage

### Running the App
To start the app, run:
```bash
python src/main.py
```

### Main Menu
When the app starts, you will see a welcome message and a list of available commands:
```
========================================================
    Welcome to RAG System! 🚀
    Your AI-powered assistant for querying documents.
    Version: <version>
========================================================
```

### Commands
1. **Setup RAG**: Set up a new RAG system by loading documents from a source.
   - Choose from the following sources:
     - Repository URL
     - Local File
     - Local Folder
     - Website URL
     - Download File URL
   - Provide the necessary input (e.g., URL or file path).
   - A knowledge base will be created with a generated name.

2. **List Knowledge Bases**: List all available knowledge bases.

3. **Delete Knowledge Base**: Delete a specific knowledge base by name.

4. **Interactive CLI**: Start an interactive CLI for querying the RAG system.
   - Enter natural language queries to retrieve relevant information.
   - After exiting the interactive CLI, you can choose to return to the main menu or exit the app.

5. **Toggle Logs**: Turn on or off all logs for debugging and monitoring.

6. **Exit**: Exit the application.

### Example Workflow
1. **Setup RAG**:
   - Choose "Setup RAG" from the main menu.
   - Select the source type (e.g., Repository URL).
   - Provide the repository URL.
   - The system will load the documents and create a knowledge base.

2. **Interactive CLI**:
   - Choose "Interactive CLI" from the main menu.
   - Enter queries to retrieve information from the knowledge base.

3. **Delete Knowledge Base**:
   - Choose "Delete Knowledge Base" from the main menu.
   - Enter the name of the knowledge base to delete.

## Development

### Installing Development Dependencies
To install development dependencies (e.g., `black`, `flake8`, `pytest`), run:
```bash
pip install -r requirements-dev.txt
```

### Running Tests
Run the test suite using:
```bash
pytest tests/
```

### Formatting and Linting
- Format the code using `black`:
  ```bash
  black src/ tests/
  ```

- Lint the code using `flake8`:
  ```bash
  flake8 src/ tests/
  ```

### Pre-commit Hooks
To ensure code quality before committing, set up pre-commit hooks:
1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```
2. Install the hooks:
   ```bash
   pre-commit install
   ```
3. The hooks will automatically run on every commit.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.