"""Knowledge Base management commands"""

from pathlib import Path

import typer
from rich.table import Table

from src.cli.utils import confirm, console, print_error, print_info, print_section, print_success
from src.utils.document_loader import DocumentLoader
from src.utils.kb_metadata import KBMetadataManager, collect_file_info, collect_single_file_info
from src.utils.rag_utils import create_vector_store, delete_knowledge_base, list_knowledge_bases
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
    # Map string source_type to integer for generate_knowledge_base_name
    source_type_map = {"repo": 1, "file": 2, "folder": 3, "website": 4, "url": 5}
    kb_name = name or generate_knowledge_base_name(source_type_map.get(source_type, 0), source)
    print_info(f"Knowledge base name: {kb_name}")

    # Check if KB already exists
    existing_kbs = list_knowledge_bases()
    if kb_name in existing_kbs and not overwrite and not confirm(f"Knowledge base '{kb_name}' already exists. Overwrite?", default=False):
        print_info("Cancelled")
        raise typer.Exit(0)

    # Create vector store (no RAG agent needed for kb add)
    print_info("Creating vector store...")

    try:
        create_vector_store(docs, kb_name)
        print_success(f"Knowledge base '{kb_name}' created successfully")

        # Save metadata for change detection
        _save_kb_metadata(kb_name, source_type, source)

        console.print("\n[cyan]Next steps:[/cyan]")
        console.print(f"  Query: [cyan]docb query --kb {kb_name}[/cyan]")
        console.print(f"  Info: [cyan]docb kb info {kb_name}[/cyan]")

    except Exception as e:
        print_error(f"Failed to setup RAG: {e}")
        raise typer.Exit(1)


def _save_kb_metadata(kb_name: str, source_type: str, source: str) -> None:
    """Save metadata for a knowledge base after creation."""
    manager = KBMetadataManager(kb_name)

    # Collect file info based on source type
    if source_type == "folder":
        source_path = str(Path(source).resolve())
        indexed_files = collect_file_info(source_path)
    elif source_type == "file":
        source_path = str(Path(source).resolve())
        indexed_files = collect_single_file_info(source_path)
    else:
        # For repo, website, url - store source but no file tracking
        source_path = source
        indexed_files = []

    if manager.save_metadata(source_type, source_path, indexed_files):
        if indexed_files:
            print_info(f"Tracking {len(indexed_files)} source file(s) for change detection")
    else:
        print_info("Note: Could not save metadata for change tracking")


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
def rename(
    old_name: str = typer.Argument(..., help="Current knowledge base name"),
    new_name: str = typer.Argument(..., help="New knowledge base name"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Don't ask for confirmation",
    ),
) -> None:
    """Rename a knowledge base."""
    print_section("Rename Knowledge Base")

    old_path = Path("knowledges") / old_name
    new_path = Path("knowledges") / new_name

    # Check if old KB exists
    if not old_path.exists():
        print_error(f"Knowledge base not found: {old_name}")
        raise typer.Exit(1)

    # Check if new name already exists
    if new_path.exists():
        print_error(f"Knowledge base already exists: {new_name}")
        raise typer.Exit(1)

    # Confirm rename
    if not force:
        console.print(f"\nRename: [cyan]{old_name}[/cyan] -> [green]{new_name}[/green]")
        if not confirm("Proceed with rename?", default=True):
            print_info("Cancelled")
            raise typer.Exit(0)

    try:
        old_path.rename(new_path)
        print_success(f"Knowledge base renamed: '{old_name}' -> '{new_name}'")
        console.print("\n[cyan]Usage:[/cyan]")
        console.print(f"  Query: [cyan]docb query --kb {new_name}[/cyan]")
        console.print(f"  Info: [cyan]docb kb info {new_name}[/cyan]")
    except Exception as e:
        print_error(f"Failed to rename knowledge base: {e}")
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

        # Load metadata if available
        manager = KBMetadataManager(name)
        metadata = manager.load_metadata()

        if metadata:
            table.add_row("Source Type", metadata.get("source_type", "Unknown"))
            table.add_row("Source Path", metadata.get("source_path", "Unknown"))
            table.add_row("Last Sync", metadata.get("last_sync_at", "Unknown"))
            indexed_files = metadata.get("indexed_files", [])
            table.add_row("Indexed Files", str(len(indexed_files)))

        # Count files in KB
        kb_files = list(kb_path.glob("**/*"))
        table.add_row("Storage Files", str(len([f for f in kb_files if f.is_file()])))

        # Size
        total_size = sum(f.stat().st_size for f in kb_files if f.is_file())
        size_mb = total_size / (1024 * 1024)
        table.add_row("Size", f"{size_mb:.2f} MB")

        table.add_row("Persistence", "Enabled" if checkpoint_db.exists() else "Disabled")
        table.add_row("Metrics", "Enabled" if metrics_db.exists() else "Disabled")

        console.print(table)

        if not metadata:
            console.print("\n[dim]Note: No metadata found (older KB without change tracking)[/dim]")

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
