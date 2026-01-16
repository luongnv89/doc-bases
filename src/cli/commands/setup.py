"""Setup and initialization commands"""

import platform
import subprocess
import sys
from pathlib import Path

import typer

from src.cli.config_manager import get_config_manager
from src.cli.utils import confirm, console, print_error, print_info, print_section, print_success, print_warning, prompt_for_value, run_command

app = typer.Typer(help="Setup and initialization")


@app.command()
def init(
    python_version: str | None = typer.Option(
        None,
        "--python",
        help="Python version for venv (default: current)",
    ),
    skip_venv: bool = typer.Option(
        False,
        "--skip-venv",
        help="Skip virtual environment creation",
    ),
    skip_install: bool = typer.Option(
        False,
        "--skip-install",
        help="Skip dependency installation",
    ),
) -> None:
    """Complete DocBases setup and initialization."""
    print_section("DocBases Setup")
    console.print("This will set up DocBases with all dependencies and configuration.\n")

    # Step 1: Create venv if needed
    if not skip_venv:
        _setup_venv(python_version)

    # Step 2: Install dependencies
    if not skip_install:
        _install_dependencies()

    # Step 3: Setup environment file
    _setup_env_file()

    # Step 4: Create data directories
    _create_directories()

    # Step 5: Validate installation
    _validate_installation()

    print_section("Setup Complete!")
    console.print("[green]✓[/green] DocBases is ready to use!")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("  1. Configure your settings: [cyan]docb config list[/cyan]")
    console.print("  2. Check health: [cyan]docb health check[/cyan]")
    console.print("  3. Add knowledge base: [cyan]docb kb add --help[/cyan]")
    console.print("  4. Start querying: [cyan]docb query[/cyan]")


@app.command()
def venv(
    python_version: str | None = typer.Option(
        None,
        "--python",
        help="Python version (default: current)",
    ),
) -> None:
    """Create virtual environment."""
    _setup_venv(python_version)


@app.command()
def install(
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Install development dependencies",
    ),
) -> None:
    """Install dependencies."""
    _install_dependencies(dev_deps=dev)


@app.command()
def env() -> None:
    """Setup .env environment file."""
    _setup_env_file()


@app.command()
def dirs() -> None:
    """Create necessary data directories."""
    _create_directories()


@app.command()
def validate() -> None:
    """Validate DocBases installation."""
    _validate_installation()


# Helper functions


def _setup_venv(python_version: str | None = None) -> None:
    """Create virtual environment."""
    print_section("Setting up Virtual Environment")

    venv_path = Path(".venv")

    if venv_path.exists():
        print_warning("Virtual environment already exists at .venv")
        if not confirm("Recreate it?", default=False):
            print_info("Skipping venv setup")
            return
        import shutil

        shutil.rmtree(venv_path)

    # Determine Python executable
    python_exe = sys.executable
    if python_version:
        python_exe = f"python{python_version}"

    print_info(f"Creating virtual environment with {python_exe}")

    try:
        run_command([python_exe, "-m", "venv", str(venv_path)])
        print_success("Virtual environment created")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        raise typer.Exit(1)

    # Print activation instructions
    print_info("\nTo activate the virtual environment, run:")
    if platform.system() == "Windows":
        console.print("[cyan].venv\\Scripts\\activate[/cyan]")
    else:
        console.print("[cyan]source .venv/bin/activate[/cyan]")


def _install_dependencies(dev_deps: bool = False) -> None:
    """Install project dependencies."""
    print_section("Installing Dependencies")

    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print_error("requirements.txt not found")
        raise typer.Exit(1)

    print_info(f"Installing from {requirements_file}")

    try:
        # Upgrade pip first
        print_info("Upgrading pip...")
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

        # Install requirements
        run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])

        print_success("Dependencies installed")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        raise typer.Exit(1)


def _setup_env_file() -> None:
    """Setup .env environment file."""
    print_section("Setting up Environment File")

    env_file = Path(".env")
    example_file = Path(".env.example")

    if env_file.exists():
        print_warning(".env file already exists")
        if not confirm("Overwrite it?", default=False):
            print_info("Skipping .env setup")
            return

    # Copy from example
    if example_file.exists():
        try:
            example_content = example_file.read_text()
            env_file.write_text(example_content)
            print_success(".env file created from .env.example")
        except Exception as e:
            print_error(f"Failed to create .env: {e}")
            raise typer.Exit(1)
    else:
        print_error(".env.example not found")
        raise typer.Exit(1)

    # Prompt for configuration
    console.print("\n[yellow]Configuration Wizard[/yellow]")
    console.print("Leave blank to keep default values\n")

    # Get LLM provider
    providers = ["openai", "google", "groq", "ollama"]
    console.print(f"LLM Providers: {', '.join(providers)}")
    llm_provider = prompt_for_value("LLM Provider", default="openai")

    if llm_provider == "openai":
        api_key = prompt_for_value("OpenAI API Key")
        if api_key:
            _update_env_var(env_file, "OPENAI_API_KEY", api_key)
    elif llm_provider == "google":
        api_key = prompt_for_value("Google API Key")
        if api_key:
            _update_env_var(env_file, "GOOGLE_API_KEY", api_key)
    elif llm_provider == "groq":
        api_key = prompt_for_value("Groq API Key")
        if api_key:
            _update_env_var(env_file, "GROQ_API_KEY", api_key)
    elif llm_provider == "ollama":
        api_base = prompt_for_value("Ollama API Base", default="http://localhost:11434")
        if api_base:
            _update_env_var(env_file, "LLM_API_BASE", api_base)

    # Update provider in .env
    _update_env_var(env_file, "LLM_PROVIDER", llm_provider)

    print_success(".env file configured")


def _create_directories() -> None:
    """Create necessary data directories."""
    print_section("Creating Data Directories")

    directories = [
        Path("knowledges"),
        Path("logs"),
        Path("temps"),
        Path(".docbases"),
    ]

    for directory in directories:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print_info(f"Created {directory}")
            except Exception as e:
                print_error(f"Failed to create {directory}: {e}")
                raise typer.Exit(1)
        else:
            print_info(f"Directory exists: {directory}")

    print_success("All directories created")


def _validate_installation() -> None:
    """Validate DocBases installation."""
    print_section("Validating Installation")

    errors = []

    # Check Python version
    print_info(f"Python version: {sys.version.split()[0]}")

    # Check required packages
    required_packages = [
        "typer",
        "pyyaml",
        "rich",
        "dotenv",
        "langchain",
        "langgraph",
        "chromadb",
    ]

    print_info("Checking required packages...")
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_info(f"  ✓ {package}")
        except ImportError:
            print_warning(f"  ✗ {package} (not installed)")
            errors.append(f"Package not found: {package}")

    # Check configuration
    print_info("Checking configuration...")
    config_mgr = get_config_manager()
    config_errors = config_mgr.validate()

    if config_errors:
        for error in config_errors:
            print_warning(f"  ✗ {error}")
            errors.append(error)
    else:
        print_info("  ✓ Configuration valid")

    # Check directories
    print_info("Checking directories...")
    for directory in ["knowledges", "logs", "temps", ".docbases"]:
        if Path(directory).exists():
            print_info(f"  ✓ {directory}/")
        else:
            errors.append(f"Directory not found: {directory}")

    # Summary
    if errors:
        console.print("\n[red]Validation failed with errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        raise typer.Exit(1)
    else:
        print_success("Installation validated successfully")


def _update_env_var(env_file: Path, key: str, value: str) -> None:
    """Update environment variable in .env file."""
    try:
        content = env_file.read_text()

        # Find and replace or add new variable
        lines = content.split("\n")
        found = False

        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                found = True
                break

        if not found:
            # Append to end
            lines.append(f"{key}={value}")

        env_file.write_text("\n".join(lines))
    except Exception as e:
        print_error(f"Failed to update .env: {e}")
        raise typer.Exit(1)
