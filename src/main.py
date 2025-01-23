# src/main.py
import os
from dotenv import load_dotenv
from src.utils.document_loader import DocumentLoader
from src.utils.rag_utils import (
    setup_rag,
    list_knowledge_bases,
    delete_knowledge_base,
    interactive_cli,
)
from utils.utilities import get_version_from_git, generate_knowledge_base_name
from src.utils.logger import setup_logger, toggle_logs


# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = setup_logger()


def print_welcome_slogan():
    """Prints a welcome slogan and version number when the app starts."""
    version = get_version_from_git()
    slogan = f"""
    ========================================================
        Welcome to RAG System! 🚀
        Your AI-powered assistant for querying documents.
        {version}
    ========================================================
    """
    print(slogan)
    logger.info("Application started.")
    logger.info(f"Version: {version}")


def cli_helper():
    """Displays a help message explaining how to use the script."""
    print(
        """
    Repository RAG System CLI Helper
    ---------------------------------------
    Commands:
    0. Back to Main Menu: Return to the main menu.
    1. Setup RAG: Set up a new RAG system.
    2. List Knowledge Bases: List all available knowledge bases.
    3. Delete Knowledge Base: Delete a specific knowledge base.
    4. Interactive CLI: Start an interactive CLI for querying the RAG system.
    5. Toggle Logs: Turn on or off all logs.
    6. Exit: Exit the script.

    Usage:
    - Run the script and follow the prompts.
    """
    )


def setup_rag_cli(doc_loader: DocumentLoader):
    """Prompts the user for the source of documents to setup RAG."""
    logger.info("Starting RAG setup CLI")
    print("\nChoose the source for setting up RAG:")
    print("0. Back to Main Menu")
    print("1. Repository URL")
    print("2. Local File")
    print("3. Local Folder")
    print("4. Website URL")
    print("5. Download File URL")
    print("6. Toggle Logs")
    print("7. Exit")

    try:
        source_choice = int(input("Enter your choice (0-7): ").strip())
        logger.info(f"User chose source type: {source_choice}")
    except ValueError:
        print("Invalid input. Please enter a number between 0 and 7.")
        logger.error("Invalid input for source choice.")
        return

    if source_choice == 0:
        return
    elif source_choice == 6:
        toggle_logs()
        return
    elif source_choice == 7:
        print("Exiting...")
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
        logger.info(f"User provided folder path: {input_str}")
        docs = doc_loader.load_documents_from_directory(input_str)
    elif source_choice == 4:
        input_str = input("Enter the website url: ").strip()
        logger.info(f"User provided website url: {input_str}")
        docs = doc_loader.load_documents_from_website(input_str)
    elif source_choice == 5:
        input_str = input("Enter the download file url: ").strip()
        logger.info(f"User provided download file url: {input_str}")
        docs = doc_loader.load_documents_from_url(input_str)
    else:
        print("Invalid source choice.")
        logger.error(f"Invalid source choice: {source_choice}")
        return

    if not docs:
        print("Failed to load document from the specified source")
        logger.error("Failed to load document from the specified source")
        return

    knowledge_base_name = generate_knowledge_base_name(source_choice, input_str)
    logger.info(f"Generated knowledge base name: {knowledge_base_name}")
    if knowledge_base_name in list_knowledge_bases():
        print(f"A knowledge base with the name '{knowledge_base_name}' already exists.")
        logger.warning(f"Knowledge base '{knowledge_base_name}' already exists.")
        choice = (
            input("Do you want to overwrite it (o), or cancel (c)? ").strip().lower()
        )
        if choice == "c":
            print("Setup cancelled.")
            logger.info("RAG setup cancelled by user.")
            return
        elif choice == "o":
            print("Overwriting existing knowledge base")
            logger.info(
                f"User chose to overwrite knowledge base '{knowledge_base_name}'"
            )
            if source_choice == 1:
                docs = doc_loader.load_documents_from_repo(input_str, overwrite=True)
            elif source_choice == 5:
                docs = doc_loader.load_documents_from_url(input_str, overwrite=True)
            else:
                print("Overwriting is only supported for Repo and Downloaded Files")
                logger.warning(
                    "Overwriting is only supported for Repo and Downloaded Files"
                )
                return

            if docs:
                setup_rag(docs, knowledge_base_name)
                print("RAG system setup completed.")
                logger.info(
                    f"RAG system setup completed with overwrite option for '{knowledge_base_name}'."
                )
            else:
                print("Failed to load documents with overwrite option")
                logger.error(
                    f"Failed to load documents with overwrite option for '{knowledge_base_name}'"
                )
            return
        else:
            print("Invalid choice. Skipping setup.")
            logger.warning(
                f"Invalid choice. Skipping setup for '{knowledge_base_name}'"
            )
            return

    setup_rag(docs, knowledge_base_name)
    print("RAG system setup completed.")
    logger.info(f"RAG system setup completed for '{knowledge_base_name}'.")


def main():
    """Main function to run the RAG system."""
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
            print("Invalid input. Please enter a number between 0 and 6.")
            logger.error("Invalid action input.")
            continue

        if action == 0:
            continue
        elif action == 1:
            setup_rag_cli(doc_loader)
        elif action == 2:
            knowledge_bases = list_knowledge_bases()
            print("Available Knowledge Bases:")
            for kb in knowledge_bases:
                print(f"- {kb}")
            logger.info(f"Listed available knowledge bases: {knowledge_bases}")
        elif action == 3:
            knowledge_base_name = input("Enter knowledge base name to delete: ").strip()
            logger.info(
                f"User requested deletion of knowledge base: {knowledge_base_name}"
            )
            delete_knowledge_base(knowledge_base_name)
        elif action == 4:
            interactive_cli()
            # After exiting interactive CLI, ask if the user wants to return to the task list or exit
            while True:
                choice = (
                    input("\nDo you want to return to the task list? (yes/no): ")
                    .strip()
                    .lower()
                )
                if choice in ["yes", "no"]:
                    break
                print("Invalid input. Please enter 'yes' or 'no'.")
                logger.warning("Invalid input for return to task list or exit")
            if choice == "no":
                print("Exiting...")
                logger.info("Exiting application from interactive cli")
                break
        elif action == 5:
            toggle_logs()
        elif action == 6:
            print("Exiting...")
            logger.info("Exiting application.")
            break
        else:
            print("Invalid action. Please choose a number between 0 and 6.")
            logger.warning(f"Invalid action: {action}")


if __name__ == "__main__":
    main()
