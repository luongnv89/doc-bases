# src/main.py
"""
DocBases Main Entry Point

This module now delegates to the new Typer-based CLI framework.
The old menu-based interface can still be accessed with the --legacy flag.
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from src.utils.logger import custom_theme, setup_logger

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = setup_logger()

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)


def main():
    """Main entry point - delegates to new CLI framework."""

    # Check if legacy mode is requested
    if "--legacy" in sys.argv:
        logger.info("Starting in legacy menu mode")
        console.print("[yellow]⚠ Legacy mode[/yellow] - Consider using the new CLI: [cyan]docb --help[/cyan]\n")
        sys.argv.remove("--legacy")
        _run_legacy_cli()
    else:
        # Use new CLI framework
        from src.cli.main import app

        logger.info("Starting DocBases CLI")
        try:
            app()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            console.print("\n[yellow]Interrupted[/yellow]")
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise


def _run_legacy_cli():
    """Run the old menu-based CLI for backward compatibility."""
    from rich.table import Table

    from src.observability.metrics import get_metrics_tracker
    from src.utils.document_loader import DocumentLoader
    from src.utils.logger import toggle_logs
    from src.utils.rag_utils import delete_knowledge_base, interactive_cli, list_knowledge_bases, setup_rag
    from src.utils.utilities import generate_knowledge_base_name, get_version_from_git

    def print_welcome_slogan():
        """Prints a welcome slogan and version number when the app starts."""
        version = get_version_from_git()
        welcome_message = f"""
        Welcome to DocBases! 🚀
        Your AI-powered assistant for querying documents.
        Version: {version}
        """
        console.print(
            Panel.fit(
                welcome_message,
                title="[header]DocBases[/header]",
                border_style="bold cyan",
            )
        )
        logger.info("Application started.")
        logger.info(f"Version: {version}")

    def cli_helper():
        """Displays a help message explaining how to use the script."""
        table = Table(title="[header]DocBases CLI Helper[/header]", border_style="bold cyan")
        table.add_column("Command", style="info")
        table.add_column("Description", style="info")

        table.add_row("0", "Back to Main Menu")
        table.add_row("1", "Add Knowledge Base: Setup a new knowledge base.")
        table.add_row("2", "List Knowledge Bases: List all available knowledge bases.")
        table.add_row("3", "Delete Knowledge Base: Delete a specific knowledge base.")
        table.add_row("4", "Interactive CLI: Start an interactive CLI for querying the DocBases.")
        table.add_row("5", "Toggle Logs: Turn on or off all logs.")
        table.add_row("6", "View Metrics: Display performance metrics dashboard.")
        table.add_row("7", "Exit: Exit the script.")

        console.print(table)

    def setup_rag_cli(doc_loader: DocumentLoader):
        """Prompts the user for the source of documents to add a new Knowledge Database."""
        logger.info("Add new Knowledge Database CLI")
        console.print("\n[header]Choose the source for adding a new Knowledge Database:[/header]")
        console.print("0. Back to Main Menu")
        console.print("1. Repository URL")
        console.print("2. Local File")
        console.print("3. Local Folder")
        console.print("4. Website URL")
        console.print("5. Download File URL")
        console.print("6. Toggle Logs")
        console.print("7. Exit")

        try:
            source_choice = int(input("Enter your choice (0-7): ").strip())
            logger.info(f"User chose source type: {source_choice}")
        except ValueError:
            console.print("[error]Invalid input. Please enter a number between 0 and 7.[/error]")
            logger.error("Invalid input for source choice.")
            return

        if source_choice == 0:
            return
        elif source_choice == 6:
            toggle_logs()
            return
        elif source_choice == 7:
            console.print("[success]Exiting...[/success]")
            logger.info("Exiting application.")
            exit()

        docs = None
        input_str = None

        if source_choice == 1:
            input_str = input("Enter repository URL: ").strip()
            logger.info(f"User provided repo URL: {input_str}")
            docs = doc_loader.load_documents_from_repo(input_str)
        elif source_choice == 2:
            input_str = input("Enter file path: ").strip()
            logger.info(f"User provided file path: {input_str}")
            docs = doc_loader.load_documents_from_file(input_str)
        elif source_choice == 3:
            input_str = input("Enter folder path: ").strip()
            logger.info(f"User provided folder path: {input_str}")
            docs = doc_loader.load_documents_from_directory(input_str)
        elif source_choice == 4:
            input_str = input("Enter website URL: ").strip()
            logger.info(f"User provided website URL: {input_str}")
            docs = doc_loader.load_documents_from_website(input_str)
        elif source_choice == 5:
            input_str = input("Enter download file URL: ").strip()
            logger.info(f"User provided download URL: {input_str}")
            docs = doc_loader.load_documents_from_url(input_str)
        else:
            console.print("[error]Invalid choice.[/error]")
            logger.error(f"Invalid source choice: {source_choice}")
            return

        if docs:
            kb_name = generate_knowledge_base_name(source_choice, input_str)
            logger.info(f"Generated knowledge base name: {kb_name}")
            existing_kbs = list_knowledge_bases()
            if kb_name in existing_kbs:
                overwrite_choice = input(f"Knowledge base '{kb_name}' already exists. Do you want to overwrite it? (yes/no): ").strip()
                logger.info(f"User chose to overwrite: {overwrite_choice}")
                if overwrite_choice.lower() not in ("yes", "y"):
                    console.print("[error]Cancelled.[/error]")
                    logger.info("User cancelled overwrite.")
                    return

            setup_rag(docs, kb_name)
        else:
            console.print("[error]Failed to load documents.[/error]")
            logger.error("Failed to load documents.")

    # Start legacy menu loop
    print_welcome_slogan()

    os.makedirs("knowledges", exist_ok=True)
    doc_loader = DocumentLoader()

    while True:
        cli_helper()
        try:
            action = int(input("Enter your choice (0-7): ").strip())
            logger.info(f"User chose action: {action}")
        except ValueError:
            console.print("[error]Invalid input. Please enter a number between 0 and 7.[/error]")
            logger.error("Invalid input for main menu.")
            continue

        if action == 0:
            continue
        elif action == 1:
            setup_rag_cli(doc_loader)
        elif action == 2:
            logger.info("Listing knowledge bases.")
            console.print()
            kbs = list_knowledge_bases()
            if kbs:
                table = Table(title="Available Knowledge Bases", border_style="bold cyan")
                table.add_column("Name", style="info")
                for kb in kbs:
                    table.add_row(kb)
                console.print(table)
            else:
                console.print("[warning]No knowledge bases found.[/warning]")
            logger.info(f"Found {len(kbs)} knowledge bases.")
        elif action == 3:
            kb_name = input("Enter knowledge base name to delete: ").strip()
            logger.info(f"User chose to delete knowledge base: {kb_name}")
            delete_knowledge_base(kb_name)
        elif action == 4:
            logger.info("Starting interactive CLI.")
            interactive_cli()
        elif action == 5:
            toggle_logs()
        elif action == 6:
            days = input("Enter number of days to display metrics for (default: 7): ").strip()
            try:
                days = int(days) if days else 7
            except ValueError:
                days = 7
            metrics = get_metrics_tracker()
            metrics.display_dashboard(days=days)
            logger.info(f"Displayed metrics dashboard for {days} days")
        elif action == 7:
            console.print("[success]Exiting...[/success]")
            logger.info("Exiting application.")
            break
        else:
            console.print("[error]Invalid action. Please choose a number between 0 and 7.[/error]")
            logger.warning(f"Invalid action: {action}")


if __name__ == "__main__":
    main()
