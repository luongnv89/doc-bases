# src/main.py
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.document_loader import DocumentLoader
from src.utils.logger import custom_theme, setup_logger, toggle_logs
from src.utils.rag_utils import delete_knowledge_base, interactive_cli, list_knowledge_bases, setup_rag
from src.utils.utilities import generate_knowledge_base_name, get_version_from_git

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = setup_logger()

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)


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
    table.add_row("6", "Exit: Exit the script.")

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
        input_str = input("Enter the file path: ").strip()
        logger.info(f"User provided file path: {input_str}")
        docs = doc_loader.load_documents_from_file(input_str)
    elif source_choice == 3:
        input_str = input("Enter the folder path: ").strip()
        # Strip quotes from the input path
        input_str = input_str.strip("'\"")  # Remove both single and double quotes
        logger.info(f"User provided folder path: {input_str}")
        docs = doc_loader.load_documents_from_directory(input_str)
    elif source_choice == 4:
        input_str = input("Enter the website URL: ").strip()
        logger.info(f"User provided website URL: {input_str}")
        docs = doc_loader.load_documents_from_website(input_str)
    elif source_choice == 5:
        input_str = input("Enter the download file URL: ").strip()
        logger.info(f"User provided download file URL: {input_str}")
        docs = doc_loader.load_documents_from_url(input_str)
    else:
        console.print("[error]Invalid source choice.[/error]")
        logger.error(f"Invalid source choice: {source_choice}")
        return

    if not docs:
        console.print("[error]Failed to load document from the specified source[/error]")
        logger.error("Failed to load document from the specified source")
        return

    knowledge_base_name = generate_knowledge_base_name(source_choice, input_str)
    logger.info(f"Generated knowledge base name: {knowledge_base_name}")
    if knowledge_base_name in list_knowledge_bases():
        console.print(f"[warning]A knowledge base with the name '{knowledge_base_name}' already exists.[/warning]")
        logger.warning(f"Knowledge base '{knowledge_base_name}' already exists.")
        choice = input("Do you want to overwrite it (o), or cancel (c)? ").strip().lower()
        if choice == "c":
            console.print("[info]Setup cancelled.[/info]")
            logger.info("Adding Knowledge Database cancelled by user.")
            return
        elif choice == "o":
            console.print("[info]Overwriting existing knowledge base[/info]")
            logger.info(f"User chose to overwrite knowledge base '{knowledge_base_name}'")
            if source_choice == 1:
                docs = doc_loader.load_documents_from_repo(input_str, overwrite=True)
            elif source_choice == 5:
                docs = doc_loader.load_documents_from_url(input_str, overwrite=True)
            else:
                console.print("[warning]Overwriting is only supported for Repo and Downloaded Files[/warning]")
                logger.warning("Overwriting is only supported for Repo and Downloaded Files")
                return

            if docs:
                setup_rag(docs, knowledge_base_name)
                console.print("[success]Add a new Knowledge Database completed.[/success]")
                logger.info(f"Add a new Knowledge Database completed with overwrite option for '{knowledge_base_name}'.")
            else:
                console.print("[error]Failed to load documents with overwrite option[/error]")
                logger.error(f"Failed to load documents with overwrite option for '{knowledge_base_name}'")
            return
        else:
            console.print("[error]Invalid choice. Skipping setup.[/error]")
            logger.warning(f"Invalid choice. Skipping setup for '{knowledge_base_name}'")
            return

    setup_rag(docs, knowledge_base_name)
    console.print("[success]Add a new Knowledge Database completed.[/success]")
    logger.info(f"Add a new Knowledge Database completed for '{knowledge_base_name}'.")


def main():
    """Main function to run the DocBases."""
    print_welcome_slogan()
    # Ensure the knowledges directory exists
    if not os.path.exists("knowledges"):
        os.makedirs("knowledges")
        logger.info("Created 'knowledges' directory.")

    doc_loader = DocumentLoader()
    logger.info("DocumentLoader initialized.")

    while True:
        cli_helper()
        try:
            action = int(input("Choose an action (0-6): ").strip())
            logger.info(f"User chose action: {action}")
        except ValueError:
            console.print("[error]Invalid input. Please enter a number between 0 and 6.[/error]")
            logger.error("Invalid action input.")
            continue

        if action == 0:
            continue
        elif action == 1:
            setup_rag_cli(doc_loader)
        elif action == 2:
            knowledge_bases = list_knowledge_bases()
            table = Table(
                title="[header]Available Knowledge Bases[/header]",
                border_style="bold cyan",
            )
            table.add_column("Knowledge Base", style="info")
            for kb in knowledge_bases:
                table.add_row(kb)
            console.print(table)
            logger.info(f"Listed available knowledge bases: {knowledge_bases}")
        elif action == 3:
            knowledge_base_name = input("Enter knowledge base name to delete: ").strip()
            logger.info(f"User requested deletion of knowledge base: {knowledge_base_name}")
            delete_knowledge_base(knowledge_base_name)
        elif action == 4:
            interactive_cli()
            # After exiting interactive CLI, ask if the user wants to return to the task list or exit
            while True:
                choice = input("\nDo you want to return to the task list? (yes/no): ").strip().lower()
                if choice in ["yes", "no"]:
                    break
                console.print("[error]Invalid input. Please enter 'yes' or 'no'.[/error]")
                logger.warning("Invalid input for return to task list or exit")
            if choice == "no":
                console.print("[success]Exiting...[/success]")
                logger.info("Exiting application from interactive cli")
                break
        elif action == 5:
            toggle_logs()
        elif action == 6:
            console.print("[success]Exiting...[/success]")
            logger.info("Exiting application.")
            break
        else:
            console.print("[error]Invalid action. Please choose a number between 0 and 6.[/error]")
            logger.warning(f"Invalid action: {action}")


if __name__ == "__main__":
    main()
