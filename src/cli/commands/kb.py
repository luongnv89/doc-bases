"""Knowledge Base management commands"""

from pathlib import Path

import typer
from rich.table import Table

from src.cli.utils import confirm, console, print_error, print_info, print_section, print_success
from src.utils.document_loader import DocumentLoader
from src.utils.rag_utils import delete_knowledge_base, list_knowledge_bases, setup_rag
from src.utils.utilities import generate_knowledge_base_name

app = typer.Typer(help="Knowledge base management")


@app.command()
def add(
    source_type: str = typer.Argument(
        ...,
        help="Source type: repo, file, folder, website, or url",
    ),
    source: str = typer.Argument(..., help="Source location (URL, file path, etc.)"),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Custom knowledge base name",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-o",
        help="Overwrite if KB already exists",
    ),
) -> None:
    """Add a new knowledge base from various sources."""
    print_section("Adding Knowledge Base")

    # Initialize document loader
    doc_loader = DocumentLoader()

    # Load documents based on source type
    print_info(f"Loading documents from {source_type}: {source}")

    docs = None
    try:
        if source_type == "repo":
            docs = doc_loader.load_documents_from_repo(source)
        elif source_type == "file":
            docs = doc_loader.load_documents_from_file(source)
        elif source_type == "folder":
            docs = doc_loader.load_documents_from_directory(source)
        elif source_type == "website":
            docs = doc_loader.load_documents_from_website(source)
        elif source_type == "url":
            docs = doc_loader.load_documents_from_url(source)
        else:
            print_error(f"Unknown source type: {source_type}")
            raise typer.Exit(1)

        if not docs:
            print_error("No documents loaded from source")
            raise typer.Exit(1)

        print_success(f"Loaded {len(docs)} documents")

    except Exception as e:
        print_error(f"Failed to load documents: {e}")
        raise typer.Exit(1)

    # Generate KB name if not provided
    kb_name = name or generate_knowledge_base_name(source)
    print_info(f"Knowledge base name: {kb_name}")

    # Check if KB already exists
    existing_kbs = list_knowledge_bases()
    if kb_name in existing_kbs and not overwrite and not confirm(f"Knowledge base '{kb_name}' already exists. Overwrite?", default=False):
        print_info("Cancelled")
        raise typer.Exit(0)

    # Setup RAG
    print_info("Setting up knowledge base with RAG...")

    try:
        _ = setup_rag(docs, kb_name)
        print_success(f"Knowledge base '{kb_name}' created successfully")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print(f"  Query: [cyan]docb query --kb {kb_name}[/cyan]")
        console.print(f"  Info: [cyan]docb kb info {kb_name}[/cyan]")

    except Exception as e:
        print_error(f"Failed to setup RAG: {e}")
        raise typer.Exit(1)


@app.command()
def list_cmd() -> None:
    """List all knowledge bases."""
    print_section("Knowledge Bases")

    try:
        kbs = list_knowledge_bases()

        if not kbs:
            print_info("No knowledge bases found")
            return

        # Create table
        table = Table(title=f"Available Knowledge Bases ({len(kbs)})", header_style="bold cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="green")

        for kb_name in kbs:
            kb_path = Path("knowledges") / kb_name
            table.add_row(kb_name, str(kb_path))

        console.print(table)

        console.print("\n[cyan]Tip:[/cyan] Use 'docb kb info <name>' to see details")

    except Exception as e:
        print_error(f"Failed to list knowledge bases: {e}")
        raise typer.Exit(1)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Knowledge base name"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Don't ask for confirmation",
    ),
) -> None:
    """Delete a knowledge base."""
    print_section("Delete Knowledge Base")

    kb_path = Path("knowledges") / name

    if not kb_path.exists():
        print_error(f"Knowledge base not found: {name}")
        raise typer.Exit(1)

    # Confirm deletion
    if not force:
        console.print("\n[red]WARNING:[/red] This will permanently delete:")
        console.print(f"  • {kb_path}/")

        if not confirm("Delete this knowledge base?", default=False):
            print_info("Cancelled")
            raise typer.Exit(0)

    try:
        delete_knowledge_base(name)
        print_success(f"Knowledge base '{name}' deleted")
    except Exception as e:
        print_error(f"Failed to delete knowledge base: {e}")
        raise typer.Exit(1)


@app.command()
def info(name: str = typer.Argument(..., help="Knowledge base name")) -> None:
    """Show knowledge base information."""
    print_section(f"Knowledge Base: {name}")

    kb_path = Path("knowledges") / name

    if not kb_path.exists():
        print_error(f"Knowledge base not found: {name}")
        raise typer.Exit(1)

    try:
        # Get KB info
        vector_store_path = kb_path / "vector_store"
        checkpoint_db = Path("knowledges/checkpoints.db")
        metrics_db = Path("knowledges/metrics.db")

        # Create info table
        table = Table(title="Knowledge Base Information", header_style="bold cyan")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Name", name)
        table.add_row("Path", str(kb_path))
        table.add_row("Vector Store", "ChromaDB" if vector_store_path.exists() else "Not initialized")

        # Count files in KB
        kb_files = list(kb_path.glob("**/*"))
        table.add_row("Files", str(len([f for f in kb_files if f.is_file()])))

        # Size
        total_size = sum(f.stat().st_size for f in kb_files if f.is_file())
        size_mb = total_size / (1024 * 1024)
        table.add_row("Size", f"{size_mb:.2f} MB")

        table.add_row("Persistence", "Enabled" if checkpoint_db.exists() else "Disabled")
        table.add_row("Metrics", "Enabled" if metrics_db.exists() else "Disabled")

        console.print(table)

        console.print("\n[cyan]Usage:[/cyan]")
        console.print(f"  Query: [cyan]docb query --kb {name}[/cyan]")
        console.print(f"  Delete: [cyan]docb kb delete {name}[/cyan]")

    except Exception as e:
        print_error(f"Failed to get knowledge base info: {e}")
        raise typer.Exit(1)


# Rename 'list' to 'list_cmd' to avoid conflict with Python builtin
@app.command(name="list")
def list_command() -> None:
    """List all knowledge bases."""
    list_cmd()
