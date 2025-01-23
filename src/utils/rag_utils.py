# src/utils/rag_utils.py
import os
import shutil
import time
import sys
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from src.models.llm import get_llm_model
from src.models.embeddings import get_embedding_model


def create_qa_chain(retriever):
    """Creates a RetrievalQA chain with the given retriever and LLM model.

    Args:
        retriever: The retriever object for fetching relevant documents.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.
    """
    # Define prompt template
    prompt = PromptTemplate.from_template(
        "Answer questions about the GitHub repository based on the following context:\n\n{context}\n\nQuestion: {question}"
    )

    # Set up LLM
    llm = get_llm_model()

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def setup_rag(documents, knowledge_base_name: str):
    """Sets up the RAG system with the given documents and saves the vector store to disk.

    Args:
        documents: List of documents to index.
        knowledge_base_name (str): Name of the knowledge base.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.
    """
    # Define the persist directory based on the knowledge base name
    persist_directory = os.path.join("knowledges", f"chroma_db_{knowledge_base_name}")

    # Create embeddings
    embeddings = get_embedding_model()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents, embeddings, persist_directory=persist_directory
    )

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Create QA chain
    return create_qa_chain(retriever)


def list_knowledge_bases():
    """Lists all available knowledge bases.

    Returns:
        list: A list of knowledge base names.
    """
    # Ensure the knowledges directory exists
    if not os.path.exists("knowledges"):
        return []

    # List all directories starting with "chroma_db_"
    knowledge_bases = [
        d for d in os.listdir("knowledges") if d.startswith("chroma_db_")
    ]
    return [kb.replace("chroma_db_", "") for kb in knowledge_bases]


def delete_knowledge_base(knowledge_base_name: str):
    """Deletes a knowledge base from disk.

    Args:
        knowledge_base_name (str): Name of the knowledge base to delete.
    """
    persist_directory = os.path.join("knowledges", f"chroma_db_{knowledge_base_name}")
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Deleted knowledge base: {knowledge_base_name}")
    else:
        print(f"Knowledge base '{knowledge_base_name}' does not exist.")


def thinking_animation():
    """Displays a thinking animation with dots."""
    for i in range(3):
        sys.stdout.write("\rgithub_rag is thinking" + "." * (i + 1))
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write("\r" + " " * 30 + "\r")  # Clear the line


def interactive_cli():
    """Starts an interactive CLI for querying the RAG system."""
    knowledge_bases = list_knowledge_bases()
    if not knowledge_bases:
        print("No knowledge bases available. Please set up a RAG system first.")
        return

    # If there's only one knowledge base, use it directly
    if len(knowledge_bases) == 1:
        knowledge_base_name = knowledge_bases[0]
    else:
        print("Available Knowledge Bases:")
        for i, kb in enumerate(knowledge_bases, 1):
            print(f"{i}. {kb}")

        try:
            selected = int(input("Select a knowledge base by number: ").strip())
            if selected < 1 or selected > len(knowledge_bases):
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input. Please enter a number.")
            return

        knowledge_base_name = knowledge_bases[selected - 1]

    persist_directory = os.path.join("knowledges", f"chroma_db_{knowledge_base_name}")

    # Load embeddings
    embeddings = get_embedding_model()

    # Load vector store
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Create QA chain
    qa_chain = create_qa_chain(retriever)

    while True:
        query = input("\n\033[94mYou:\033[0m ")  # Blue color for user input
        if query.lower() == "exit":
            break

        # Show thinking animation
        thinking_animation()

        # Get the answer
        result = qa_chain.invoke({"query": query})

        # Display the answer
        print(
            f"\033[92mgithub_rag:\033[0m {result['result']}"
        )  # Green color for AI response
