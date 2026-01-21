"""
This module provides utility functions for setting up and interacting with a
Retrieval-Augmented Generation (RAG) system. It includes functionalities for
creating QA chains, setting up the RAG system, listing available knowledge
bases, deleting knowledge bases, and providing an interactive CLI for querying
the RAG system.

Phase 3 enhancements:
- Multiple RAG modes: basic, corrective, adaptive
- Configurable via RAG_MODE environment variable

Phase 4 enhancements:
- Multi-agent orchestration mode
- Specialized agents: Retriever, Summarizer, Critic
- Supervisor-based coordination

Phase 5 enhancements:
- Persistent memory with SQLite checkpointer
- LangSmith tracing integration
- Query metrics tracking
"""

import asyncio
import os
import shutil
import time
import uuid
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from tqdm import tqdm

# Phase 4: Multi-agent orchestration
from src.agents.supervisor import MultiAgentSupervisor

# Phase 5: Persistent memory and observability
from src.checkpointing.sqlite_saver import get_async_checkpointer, get_checkpointer
from src.graphs.adaptive_rag import AdaptiveRAGGraph

# Phase 3: Advanced RAG patterns
from src.graphs.corrective_rag import CorrectiveRAGGraph
from src.models.embeddings import get_embedding_model
from src.models.llm import get_llm_model
from src.observability.langsmith_tracer import setup_langsmith_tracing
from src.observability.metrics import get_metrics_tracker
from src.utils.logger import custom_theme, get_logger  # Import custom_theme
from src.utils.utilities import format_image_error_message, validate_file_paths

# Setup logging
logger = get_logger()

# Initialize LangSmith tracing if configured
setup_langsmith_tracing()

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)

KNOWLEDGE_BASE_DIR = "knowledges"


def create_vector_store(documents: list[Document], knowledge_base_name: str) -> Chroma:
    """
    Create a vector store from documents without initializing a RAG agent.

    This function only creates embeddings and stores them in ChromaDB.
    Use this for `kb add` operations where you don't need the full RAG agent.

    Args:
        documents: List of documents to embed and store.
        knowledge_base_name: Name of the knowledge base.

    Returns:
        The created Chroma vector store.
    """
    logger.info(f"Creating vector store for knowledge base: '{knowledge_base_name}'")
    persist_directory = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)
    logger.debug(f"Persist directory: {persist_directory}")

    embeddings = get_embedding_model()
    logger.debug(f"Embedding model: {embeddings}")

    console.print(f"[header]Creating vector store for '{knowledge_base_name}'...[/header]")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    logger.info(f"Starting embedding process for '{knowledge_base_name}'.")
    with tqdm(total=len(documents), desc="Embedding data") as pbar:
        for doc in documents:
            vectorstore.add_documents([doc])
            pbar.update(1)

    logger.info(f"Embedding process for '{knowledge_base_name}' completed.")
    console.print(f"[success]Vector store for '{knowledge_base_name}' created successfully.[/success]")

    return vectorstore


def get_rag_mode() -> str:
    """Get configured RAG mode from environment."""
    return os.getenv("RAG_MODE", "basic").lower()


def get_retriever_tool(vectorstore):
    @tool
    def retrieve_context(query: str):
        """Retrieve relevant documents from the knowledge base."""
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return retrieve_context


def setup_rag(documents: list[Document], knowledge_base_name: str, llm=None):
    """
    Setup RAG with selected mode.

    Modes (configured via RAG_MODE env var):
    - basic: Standard ReAct agent (default)
    - corrective: Self-correcting RAG with relevance grading
    - adaptive: Query-routed RAG with strategy selection
    - multi_agent: Multi-agent orchestration with specialized agents
    """
    logger.info(f"Setting up Agentic RAG for knowledge base: '{knowledge_base_name}'.")
    persist_directory = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)
    logger.debug(f"Persist directory: {persist_directory}")
    embeddings = get_embedding_model()
    logger.debug(f"Embedding model: {embeddings}")
    console.print(f"[header]Creating vector store for '{knowledge_base_name}'...[/header]")
    logger.info(f"Creating vector store for '{knowledge_base_name}'.")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    logger.info(f"Starting embedding process for '{knowledge_base_name}'.")
    with tqdm(total=len(documents), desc="Embedding data") as pbar:
        for doc in documents:
            vectorstore.add_documents([doc])
            pbar.update(1)
    logger.info(f"Embedding process for '{knowledge_base_name}' completed.")
    console.print(f"[success]Vector store for '{knowledge_base_name}' created successfully.[/success]")
    logger.info(f"Vector store for '{knowledge_base_name}' created successfully.")

    if not llm:
        llm = get_llm_model()

    rag_mode = get_rag_mode()

    # Use async checkpointer for async agents, sync for basic mode
    if rag_mode in ["corrective", "adaptive", "multi_agent"]:
        memory = get_async_checkpointer()
    else:
        memory = get_checkpointer()

    agent: Any
    if rag_mode == "corrective":
        logger.info("Setting up Corrective RAG mode")
        console.print("[info]RAG Mode: Corrective[/info]")
        agent = CorrectiveRAGGraph(vectorstore, llm, checkpointer=memory)
    elif rag_mode == "adaptive":
        logger.info("Setting up Adaptive RAG mode")
        console.print("[info]RAG Mode: Adaptive[/info]")
        agent = AdaptiveRAGGraph(vectorstore, llm, checkpointer=memory)
    elif rag_mode == "multi_agent":
        logger.info("Setting up Multi-Agent RAG mode")
        console.print("[info]RAG Mode: Multi-Agent[/info]")
        agent = MultiAgentSupervisor(vectorstore, llm, checkpointer=memory)
    else:
        logger.info("Setting up Basic RAG mode")
        console.print("[info]RAG Mode: Basic[/info]")
        retrieve_tool = get_retriever_tool(vectorstore)
        agent = create_react_agent(llm, [retrieve_tool], checkpointer=memory)

    logger.info(f"Agentic RAG with memory setup completed for knowledge base: '{knowledge_base_name}'.")
    return agent


def list_knowledge_bases() -> list[str]:
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
    knowledge_bases = [d for d in os.listdir(KNOWLEDGE_BASE_DIR) if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))]
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


def load_rag_chain(knowledge_base_name: str, llm=None):
    """
    Load RAG chain with selected mode.

    Modes (configured via RAG_MODE env var):
    - basic: Standard ReAct agent (default)
    - corrective: Self-correcting RAG with relevance grading
    - adaptive: Query-routed RAG with strategy selection
    - multi_agent: Multi-agent orchestration with specialized agents
    """
    logger.info(f"Loading Agentic RAG for knowledge base: {knowledge_base_name}")
    persist_directory = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)
    if not os.path.exists(persist_directory):
        console.print(f"[error]Knowledge base '{knowledge_base_name}' not found.[/error]")
        logger.warning(f"Knowledge base '{knowledge_base_name}' not found.")
        return None
    embeddings = get_embedding_model()
    logger.debug(f"Embedding model: {embeddings}")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    logger.debug("Vector store loaded.")

    if not llm:
        llm = get_llm_model()

    rag_mode = get_rag_mode()

    # Use async checkpointer for async agents, sync for basic mode
    if rag_mode in ["corrective", "adaptive", "multi_agent"]:
        memory = get_async_checkpointer()
    else:
        memory = get_checkpointer()

    agent: Any
    if rag_mode == "corrective":
        logger.info("Loading Corrective RAG mode")
        console.print("[info]RAG Mode: Corrective[/info]")
        agent = CorrectiveRAGGraph(vectorstore, llm, checkpointer=memory)
    elif rag_mode == "adaptive":
        logger.info("Loading Adaptive RAG mode")
        console.print("[info]RAG Mode: Adaptive[/info]")
        agent = AdaptiveRAGGraph(vectorstore, llm, checkpointer=memory)
    elif rag_mode == "multi_agent":
        logger.info("Loading Multi-Agent RAG mode")
        console.print("[info]RAG Mode: Multi-Agent[/info]")
        agent = MultiAgentSupervisor(vectorstore, llm, checkpointer=memory)
    else:
        logger.info("Loading Basic RAG mode")
        console.print("[info]RAG Mode: Basic[/info]")
        retrieve_tool = get_retriever_tool(vectorstore)
        agent = create_react_agent(llm, [retrieve_tool], checkpointer=memory)

    logger.info(f"Agentic RAG with memory loaded for knowledge base: {knowledge_base_name}")
    return agent


def interactive_cli(knowledge_base_name: str | None = None, session_id: str | None = None) -> None:
    """
    Interactive CLI with support for multiple RAG modes.

    Handles both sync (basic) and async (corrective, adaptive) agents.
    Includes metrics tracking for query performance monitoring.

    Args:
        knowledge_base_name: Optional KB name. If provided, skips KB selection.
        session_id: Optional session ID. If provided, skips session ID prompt.
    """
    logger.info("Starting interactive CLI.")

    # Initialize metrics tracker
    metrics = get_metrics_tracker()

    # Select knowledge base if not provided
    if not knowledge_base_name:
        knowledge_bases = list_knowledge_bases()
        if not knowledge_bases:
            console.print("[error]No knowledge bases available. Please set up a RAG system first.[/error]")
            logger.warning("No knowledge bases available for interactive CLI.")
            return
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
                logger.info(f"User selected knowledge base: {knowledge_bases[selected - 1]}")
            except ValueError:
                console.print("[error]Invalid input. Please enter a number.[/error]")
                logger.error("Invalid input for knowledge base selection.")
                return
            knowledge_base_name = knowledge_bases[selected - 1]

    # Get RAG mode for handling async vs sync
    rag_mode = get_rag_mode()
    is_async_agent = rag_mode in ["corrective", "adaptive", "multi_agent"]

    # For async agents, create a persistent event loop BEFORE loading the agent
    # This ensures the aiosqlite connection is created in the same loop we'll use for queries
    if is_async_agent:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        loop = None

    llm = get_llm_model()
    if not llm:
        logger.error("Failed to load LLM model for interactive CLI.")
        if loop:
            loop.close()
        return
    agent = load_rag_chain(knowledge_base_name, llm=llm)
    if not agent:
        logger.error("Failed to load Agentic RAG for interactive CLI.")
        if loop:
            loop.close()
        return

    # Ask for or generate a session/thread id for conversation memory if not provided
    if not session_id:
        session_id = input("Enter a session id for this conversation (leave blank to auto-generate): ").strip()
        if not session_id:
            session_id = str(uuid.uuid4())
            console.print(f"[info]Generated session id: {session_id}[/info]")
        else:
            console.print(f"[info]Using session id: {session_id}[/info]")
    else:
        console.print(f"[info]Using session id: {session_id}[/info]")
    config = {"configurable": {"thread_id": session_id}}
    console.print(f"[success]Interactive CLI started (Mode: {rag_mode}). Type 'exit' to quit.[/success]")
    messages = []

    try:
        while True:
            query = console.input("[info]You: [/info]")
            if query.lower() == "exit":
                logger.info("Exiting interactive CLI.")
                break

            # Validate query for image files before processing
            is_valid, invalid_files = validate_file_paths(query)
            if not is_valid:
                error_message = format_image_error_message(invalid_files)
                console.print(
                    Panel.fit(
                        error_message,
                        title="[error]Unsupported Image Input[/error]",
                        border_style="red",
                    )
                )
                logger.warning(f"User query contained unsupported image files: {invalid_files}")
                continue

            messages.append({"role": "user", "content": query})

            # Track query metrics
            start_time = time.time()
            success = True
            error_msg = None
            answer = ""

            try:
                with Live(Spinner("dots"), refresh_per_second=20):
                    time.sleep(0.5)
                    if is_async_agent:
                        # Async agents (corrective, adaptive) - use persistent event loop
                        assert loop is not None, "Event loop must be initialized for async agents"
                        answer = loop.run_until_complete(agent.invoke(query, config=config))
                        logger.info(f"Query: {query}, Answer: {answer[:100]}...")
                    else:
                        # Basic ReAct agent
                        result = agent.invoke({"messages": messages}, config=config)
                        logger.info(f"Query: {query}, Result: {result}")
                        # Extract the final answer from the agent's output
                        answer = result["messages"][-1].content if "messages" in result and result["messages"] else str(result)
            except Exception as e:
                success = False
                error_msg = str(e)
                answer = f"Error processing query: {e}"
                logger.error(f"Query failed: {e}")

            # Log metrics
            latency_ms = int((time.time() - start_time) * 1000)
            metrics.log_query(
                query=query,
                latency_ms=latency_ms,
                rag_mode=rag_mode,
                knowledge_base=knowledge_base_name,
                session_id=session_id,
                success=success,
                error=error_msg,
            )

            messages.append({"role": "assistant", "content": answer})
            markdown_content = Markdown(answer)
            console.print(
                Panel.fit(
                    markdown_content,
                    title=f"[success]DocBases ({knowledge_base_name})[/success]",
                    border_style="green",
                )
            )
    finally:
        # Clean up the event loop for async agents
        if loop is not None:
            loop.close()
            logger.info("Closed async event loop")
