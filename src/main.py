# src/main.py
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from src.utils.github_utils import clone_and_parse_repo, get_version_from_git
from src.utils.rag_utils import (
    setup_rag,
    list_knowledge_bases,
    delete_knowledge_base,
    interactive_cli,
)

# Load environment variables from .env file
load_dotenv()


def print_welcome_slogan():
    """Prints a welcome slogan and version number when the app starts."""
    version = get_version_from_git()
    slogan = f"""
    ========================================================
      Welcome to GitHub RAG System! 🚀
      Your AI-powered assistant for querying GitHub repos.
      {version}
    ========================================================
    """
    print(slogan)


def thinking_animation():
    """Displays a thinking animation with dots."""
    for i in range(3):
        sys.stdout.write("\rgithub_rag is thinking" + "." * (i + 1))
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write("\r" + " " * 30 + "\r")  # Clear the line


def super_cli_helper():
    """Displays a help message explaining how to use the script."""
    print(
        """
    GitHub Repository RAG System CLI Helper
    ---------------------------------------
    Commands:
    1. Setup RAG: Set up a new RAG system for a GitHub repository.
    2. List Knowledge Bases: List all available knowledge bases.
    3. Delete Knowledge Base: Delete a specific knowledge base.
    4. Interactive CLI: Start an interactive CLI for querying the RAG system.
    5. Exit: Exit the script.

    Usage:
    - Run the script and follow the prompts.
    """
    )


def main():
    """Main function to run the RAG system."""
    print_welcome_slogan()
    # Ensure the knowledges directory exists
    if not os.path.exists("knowledges"):
        os.makedirs("knowledges")

    while True:
        super_cli_helper()
        try:
            action = int(input("Choose an action (1-5): ").strip())
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
            continue

        if action == 1:
            repo_url = input("Enter GitHub repository URL: ")
            docs = clone_and_parse_repo(repo_url)
            if docs:
                repo_name = repo_url.split("/")[-1].replace(".git", "")
                if repo_name in list_knowledge_bases():
                    print(
                        f"A knowledge base with the name '{repo_name}' already exists."
                    )
                    choice = (
                        input(
                            "Do you want to overwrite it (o), or keep both versions (k)? "
                        )
                        .strip()
                        .lower()
                    )
                    if choice == "k":
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        repo_name = f"{repo_name}_{timestamp}"
                        print(f"Creating new knowledge base with name: {repo_name}")
                    elif choice != "o":
                        print("Invalid choice. Skipping setup.")
                        continue

                qa_chain = setup_rag(docs, repo_name)
                print("RAG system setup completed.")
        elif action == 2:
            print("Available Knowledge Bases:", list_knowledge_bases())
        elif action == 3:
            knowledge_base_name = input("Enter knowledge base name to delete: ")
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
            if choice == "no":
                print("Exiting...")
                break
        elif action == 5:
            print("Exiting...")
            break
        else:
            print("Invalid action. Please choose a number between 1 and 5.")


if __name__ == "__main__":
    main()
