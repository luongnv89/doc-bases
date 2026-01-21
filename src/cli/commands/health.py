"""Health check commands"""

from pathlib import Path

import typer

from src.cli.config_manager import get_config_manager
from src.cli.utils import console, print_error, print_info, print_section, print_success, print_warning

app = typer.Typer(help="Health checks")


@app.command(name="check")
def health_check(
    llm: bool = typer.Option(
        False,
        "--llm",
        help="Check LLM connectivity",
    ),
    embeddings: bool = typer.Option(
        False,
        "--embeddings",
        help="Check embeddings connectivity",
    ),
    vectorstore: bool = typer.Option(
        False,
        "--vectorstore",
        help="Check vector store",
    ),
    deps: bool = typer.Option(
        False,
        "--deps",
        help="Check dependencies",
    ),
) -> None:
    """Run health checks."""
    print_section("Health Check")

    # If no specific checks requested, run all
    if not any([llm, embeddings, vectorstore, deps]):
        llm = embeddings = vectorstore = deps = True

    all_ok = True

    # Check configuration
    print_info("Checking configuration...")
    config_mgr = get_config_manager()
    config_errors = config_mgr.validate()

    if config_errors:
        print_error("Configuration errors:")
        for error in config_errors:
            console.print(f"  • {error}")
        all_ok = False
    else:
        print_success("Configuration valid")

    # Check directories
    print_info("Checking directories...")
    dirs = ["knowledges", "logs", "temps", ".docbases"]
    for directory in dirs:
        if Path(directory).exists():
            print_success(f"{directory}/ exists")
        else:
            print_warning(f"{directory}/ not found")

    # Check LLM connectivity
    if llm:
        print_info("Checking LLM connectivity...")
        try:
            from src.models.llm import get_llm_model

            _ = get_llm_model()
            print_success(f"LLM OK: {config_mgr.get('llm.provider')} / {config_mgr.get('llm.model')}")

        except Exception as e:
            print_error(f"LLM connection failed: {e}")
            all_ok = False

    # Check embeddings connectivity
    if embeddings:
        print_info("Checking embeddings connectivity...")
        try:
            from src.models.embeddings import get_embedding_model

            _ = get_embedding_model()
            print_success(f"Embeddings OK: {config_mgr.get('embedding.provider')} / {config_mgr.get('embedding.model')}")

        except Exception as e:
            print_error(f"Embeddings connection failed: {e}")
            all_ok = False

    # Check vector store
    if vectorstore:
        print_info("Checking vector store...")
        try:
            from langchain_chroma import Chroma

            from src.models.embeddings import get_embedding_model

            embeddings_fn = get_embedding_model()
            # Try to connect to ChromaDB
            Chroma(persist_directory="knowledges/test", embedding_function=embeddings_fn)
            print_success("Vector store accessible")

        except Exception as e:
            print_error(f"Vector store check failed: {e}")
            all_ok = False

    # Check dependencies
    if deps:
        print_info("Checking dependencies...")
        # Map package names to their import names (when different)
        package_import_map = {
            "pyyaml": "yaml",
        }
        required_packages = [
            "typer",
            "pyyaml",
            "rich",
            "langchain",
            "langgraph",
            "chromadb",
            "pytest",
        ]

        missing = []
        for package in required_packages:
            try:
                import_name = package_import_map.get(package, package.replace("-", "_"))
                __import__(import_name)
                print_success(f"{package}")
            except ImportError:
                print_warning(f"{package} (missing)")
                missing.append(package)

        if missing:
            all_ok = False

    # Summary
    console.print()
    if all_ok:
        print_success("Health check passed")
    else:
        print_error("Health check failed. See above for details")
        raise typer.Exit(1)
