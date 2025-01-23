"""
This module provides utility functions for setting up and interacting with a
Retrieval-Augmented Generation (RAG) system. It includes functionalities for
creating QA chains, setting up the RAG system, listing available knowledge
bases, deleting knowledge bases, and providing an interactive CLI for querying
the RAG system.
"""

import os
import shutil
import time
from typing import List, Optional

from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.schema import Document
from src.models.llm import get_llm_model
from src.models.embeddings import get_embedding_model
from src.utils.logger import get_logger, custom_theme  # Import custom_theme

# Setup logging
logger = get_logger()

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)

KNOWLEDGE_BASE_DIR = "knowledges"


def _create_qa_chain(retriever, llm):
    """Creates a RetrievalQA chain with the given retriever and LLM model.

    Args:
        retriever: The retriever object for fetching relevant documents.
        llm: The LLM model to use.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.
    """
    logger.info("Creating QA chain.")
    # Define prompt template
    prompt = PromptTemplate.from_template(
        "Answer questions about the given content based on the following context:\n\n{context}\n\nQuestion: {question}"
    )
    logger.debug(f"Prompt Template: {prompt}")
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    logger.info("QA chain created.")
    return qa_chain


def setup_rag(
    documents: List[Document], knowledge_base_name: str, llm=None
) -> RetrievalQA:
    """Sets up the RAG system with the given documents and saves the vector store to disk.

    Args:
        documents: List of documents to index.
        knowledge_base_name (str): Name of the knowledge base.
        llm (LLM, optional): LLM model to be used. if None, will use the default model

    Returns:
        RetrievalQA: The configured RetrievalQA chain.
    """
    logger.info(f"Setting up RAG for knowledge base: '{knowledge_base_name}'.")
    # Define the persist directory based on the knowledge base name
    persist_directory = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)
    logger.debug(f"Persist directory: {persist_directory}")

    # Create embeddings
    embeddings = get_embedding_model()
    logger.debug(f"Embedding model: {embeddings}")

    # Create vector store
    console.print(f"[header]Creating vector store for '{knowledge_base_name}'...[/header]")
    logger.info(f"Creating vector store for '{knowledge_base_name}'.")
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    logger.info(f"Starting embedding process for '{knowledge_base_name}'.")
    with tqdm(total=len(documents), desc="Embedding data") as pbar:
        for doc in documents:
            vectorstore.add_documents([doc])
            pbar.update(1)
    logger.info(f"Embedding process for '{knowledge_base_name}' completed.")

    console.print(f"[success]Vector store for '{knowledge_base_name}' created successfully.[/success]")
    logger.info(f"Vector store for '{knowledge_base_name}' created successfully.")

    # Create retriever
    retriever = vectorstore.as_retriever()
    logger.debug("Retriever created.")
    if not llm:
        llm = get_llm_model()

    # Create QA chain
    qa_chain = _create_qa_chain(retriever, llm)
    logger.info(f"RAG setup completed for knowledge base: '{knowledge_base_name}'.")
    return qa_chain


def list_knowledge_bases() -> List[str]:
    """Lists all available knowledge bases.

    Returns:
        list: A list of knowledge base names.
    """
    logger.info("Listing available knowledge bases.")
    # Ensure the knowledges directory exists
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        logger.warning(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
        return []

    # List all directories in knowledges/
    knowledge_bases = [
        d
        for d in os.listdir(KNOWLEDGE_BASE_DIR)
        if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))
    ]
    logger.info(f"Found knowledge bases: {knowledge_bases}")
    return knowledge_bases


def delete_knowledge_base(knowledge_base_name: str) -> None:
    """Deletes a knowledge base from disk.

    Args:
        knowledge_base_name (str): Name of the knowledge base to delete.
    """
    logger.info(f"Deleting knowledge base: {knowledge_base_name}")
    persist_directory = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        console.print(f"[error]Deleted knowledge base: {knowledge_base_name}[/error]")
        logger.info(f"Deleted knowledge base: {knowledge_base_name}")
    else:
        console.print(f"[warning]Knowledge base '{knowledge_base_name}' does not exist.[/warning]")
        logger.warning(f"Knowledge base '{knowledge_base_name}' does not exist.")


def load_rag_chain(knowledge_base_name: str, llm=None) -> Optional[RetrievalQA]:
    """Loads a RAG chain from disk using a knowledge base name.

    Args:
        knowledge_base_name (str): Name of the knowledge base to load.
        llm (LLM, optional): LLM model to be used. if None, will use the default model

    Returns:
        RetrievalQA: The configured RetrievalQA chain, or None if loading fails.
    """
    logger.info(f"Loading RAG chain for knowledge base: {knowledge_base_name}")
    persist_directory = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)

    if not os.path.exists(persist_directory):
        console.print(f"[error]Knowledge base '{knowledge_base_name}' not found.[/error]")
        logger.warning(f"Knowledge base '{knowledge_base_name}' not found.")
        return None

    # Load embeddings
    embeddings = get_embedding_model()
    logger.debug(f"Embedding model: {embeddings}")

    # Load vector store
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    logger.debug("Vector store loaded.")

    # Create retriever
    retriever = vectorstore.as_retriever()
    logger.debug("Retriever created.")

    # Create QA chain
    if not llm:
        llm = get_llm_model()
    qa_chain = _create_qa_chain(retriever, llm)
    logger.info(f"RAG chain loaded for knowledge base: {knowledge_base_name}")
    return qa_chain


def interactive_cli() -> None:
    """Starts an interactive CLI for querying the RAG system."""
    logger.info("Starting interactive CLI.")
    knowledge_bases = list_knowledge_bases()
    if not knowledge_bases:
        console.print("[error]No knowledge bases available. Please set up a RAG system first.[/error]")
        logger.warning("No knowledge bases available for interactive CLI.")
        return

    # If there's only one knowledge base, use it directly
    if len(knowledge_bases) == 1:
        knowledge_base_name = knowledge_bases[0]
        logger.info(f"Only one knowledge base found, using: {knowledge_base_name}")
    else:
        console.print("[header]Available Knowledge Bases:[/header]")
        for i, kb in enumerate(knowledge_bases, 1):
            console.print(f"{i}. {kb}")

        try:
            selected = int(input("Select a knowledge base by number: ").strip())
            if selected < 1 or selected > len(knowledge_bases):
                console.print("[error]Invalid selection.[/error]")
                logger.warning(f"Invalid knowledge base selection: {selected}")
                return
            logger.info(
                f"User selected knowledge base: {knowledge_bases[selected - 1]}"
            )
        except ValueError:
            console.print("[error]Invalid input. Please enter a number.[/error]")
            logger.error("Invalid input for knowledge base selection.")
            return

        knowledge_base_name = knowledge_bases[selected - 1]

    llm = get_llm_model()
    if not llm:
        logger.error("Failed to load LLM model for interactive CLI.")
        return
    qa_chain = load_rag_chain(knowledge_base_name, llm=llm)
    if not qa_chain:
        logger.error("Failed to load QA chain for interactive CLI.")
        return

    console.print("[success]Interactive CLI started. Type 'exit' to quit.[/success]")
    while True:
        query = console.input("[info]You: [/info]")
        if query.lower() == "exit":
            logger.info("Exiting interactive CLI.")
            break

        # Show thinking animation
        with Live(Spinner("dots"), refresh_per_second=20) as live:
            time.sleep(1)  # Simulate thinking time
            result = qa_chain.invoke({"query": query})
            logger.info(f"Query: {query}, Result: {result}")

        # Display the answer
        console.print(Panel.fit(
            result['result'],
            title="[success]DocBases[/success]",
            border_style="green",
        ))