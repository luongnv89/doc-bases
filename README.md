# DocBases

DocBases is a powerful **Retrieval-Augmented Generation (RAG)** system designed to help you query documents from various sources, including GitHub repositories, local files, folders, websites, and downloadable files. With an intuitive CLI, you can easily set up, manage, and interact with knowledge bases to retrieve information using natural language queries.

---
![DocBases Screenshot](screenshot.png)

## Features

- **Multiple Document Sources**: Load documents from:
  - GitHub repositories
  - Local files
  - Local folders
  - Websites
  - Downloadable files
- **Knowledge Base Management**:
  - Create new knowledge bases from your documents.
  - List all available knowledge bases.
  - Delete knowledge bases you no longer need.
- **Interactive CLI**:
  - Query your knowledge bases using natural language.
  - Seamlessly switch between tasks or exit the app.
- **Logging**:
  - Toggle logs on or off for debugging and monitoring.

---

## How to Use

### Running the App
To start DocBases, run:
```bash
python src/main.py
```

### Main Menu
When the app starts, you’ll see a welcome message and a list of available commands:
```
========================================================
    Welcome to DocBases! 🚀
    Your AI-powered assistant for querying documents.
    Version: <version>
========================================================
```

### Commands
1. **Setup RAG**:
   - Set up a new RAG system by loading documents from a source.
   - Choose from the following sources:
     - Repository URL
     - Local File
     - Local Folder
     - Website URL
     - Download File URL
   - Provide the necessary input (e.g., URL or file path).
   - A knowledge base will be created with a generated name.

2. **List Knowledge Bases**:
   - List all available knowledge bases.

3. **Delete Knowledge Base**:
   - Delete a specific knowledge base by name.

4. **Interactive CLI**:
   - Start an interactive CLI for querying the RAG system.
   - Enter natural language queries to retrieve relevant information.
   - After exiting the interactive CLI, you can choose to return to the main menu or exit the app.

5. **Toggle Logs**:
   - Turn on or off all logs for debugging and monitoring.

6. **Exit**:
   - Exit the application.

---

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

---

## Installation and Development

For detailed installation instructions and development guidelines, please refer to the [INSTALLATION.md](INSTALLATION.md) file.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.