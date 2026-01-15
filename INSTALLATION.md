# Installation and Development

## Prerequisites
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

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/docbases.git
   cd docbases
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

---

## Contributing

We welcome contributions! If you'd like to contribute to DocBases, please follow these steps:
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
----
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
