"""Query commands for knowledge bases"""

from pathlib import Path

import typer

from src.cli.utils import print_error, print_info, print_section
from src.utils.logger import get_logger
from src.utils.rag_utils import interactive_cli, list_knowledge_bases, load_rag_chain

logger = get_logger()

app = typer.Typer(help="Query knowledge bases")


@app.command(name="interactive")
def interactive(
    kb: str | None = typer.Option(
        None,
        "--kb",
        "-k",
        help="Knowledge base name (auto-select if only one exists)",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        "-m",
        help="RAG mode (basic, corrective, adaptive, multi_agent)",
    ),
    session: str | None = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID for conversation history",
    ),
) -> None:
    """Start interactive querying session."""
    print_section("DocBases Interactive Query")

    # If no KB specified, list available ones
    if not kb:
        kbs = list_knowledge_bases()
        if not kbs:
            print_error("No knowledge bases found. Create one with: docb kb add --help")
            raise typer.Exit(1)
        elif len(kbs) == 1:
            kb = kbs[0]
            print_info(f"Using knowledge base: {kb}")
        else:
            print_error("Multiple knowledge bases found. Specify one with --kb")
            print_info("Available KBs:")
            for kb_name in kbs:
                print_info(f"  - {kb_name}")
            raise typer.Exit(1)

    # Load RAG chain
    print_info(f"Loading knowledge base: {kb}")
    agent = load_rag_chain(kb)

    if not agent:
        print_error(f"Failed to load knowledge base: {kb}")
        raise typer.Exit(1)

    print_info("Ready for queries. Type 'exit' to quit.\n")

    # Call existing interactive CLI (which handles the query loop)
    try:
        interactive_cli()
    except KeyboardInterrupt:
        print_info("\nInteractive session ended")
        raise typer.Exit(0)
    except Exception as e:
        logger.exception(f"Error in interactive session: {e}")
        print_error(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def single(
    query: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="Query text",
    ),
    kb: str | None = typer.Option(
        None,
        "--kb",
        "-k",
        help="Knowledge base name",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        "-m",
        help="RAG mode",
    ),
) -> None:
    """Execute a single query."""
    print_section("Single Query")

    # Select knowledge base
    if not kb:
        kbs = list_knowledge_bases()
        if not kbs:
            print_error("No knowledge bases found")
            raise typer.Exit(1)
        elif len(kbs) == 1:
            kb = kbs[0]
        else:
            print_error("Multiple KBs found. Specify with --kb")
            raise typer.Exit(1)

    print_info(f"KB: {kb}")
    print_info(f"Query: {query}")

    # Load RAG chain
    agent = load_rag_chain(kb)
    if not agent:
        print_error(f"Failed to load knowledge base: {kb}")
        raise typer.Exit(1)

    # Execute query
    try:
        import asyncio
        import inspect
        import uuid

        # Generate thread ID for checkpointer
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Prepare input - try with HumanMessage first
        try:
            from langchain_core.messages import HumanMessage

            input_data = {"messages": [HumanMessage(content=query)]}
        except Exception:
            input_data = {"messages": [{"role": "user", "content": query}]}

        # Invoke agent
        if inspect.iscoroutinefunction(agent.invoke):
            result = asyncio.run(agent.invoke(input_data, config))
        else:
            result = agent.invoke(input_data, config)

        # Extract answer from result
        if isinstance(result, dict):
            # Try to find the answer in various possible locations
            if "output" in result:
                answer = result["output"]
            elif "answer" in result:
                answer = result["answer"]
            elif "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                # Handle both Message objects and dicts
                if hasattr(last_msg, "content"):
                    answer = last_msg.content
                elif isinstance(last_msg, dict):
                    answer = last_msg.get("content", str(result))
                else:
                    answer = str(last_msg)
            else:
                answer = str(result)
        elif isinstance(result, str):
            answer = result
        else:
            answer = str(result)

        print_info(f"\nAnswer:\n{answer}")

    except Exception as e:
        logger.exception(f"Error executing query: {e}")
        print_error(f"Error executing query: {e}")
        raise typer.Exit(1)


@app.command()
def batch(
    file: str = typer.Argument(..., help="File with queries (one per line)"),
    kb: str | None = typer.Option(
        None,
        "--kb",
        "-k",
        help="Knowledge base name",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON)",
    ),
) -> None:
    """Execute batch queries from a file."""
    print_section("Batch Queries")

    # Read queries from file
    try:
        file_path = Path(file)
        queries = file_path.read_text().strip().split("\n")
        queries = [q.strip() for q in queries if q.strip()]

        if not queries:
            print_error(f"No queries found in {file}")
            raise typer.Exit(1)

        print_info(f"Found {len(queries)} queries")

    except FileNotFoundError:
        print_error(f"File not found: {file}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        raise typer.Exit(1)

    # Select knowledge base
    if not kb:
        kbs = list_knowledge_bases()
        if not kbs:
            print_error("No knowledge bases found")
            raise typer.Exit(1)
        elif len(kbs) == 1:
            kb = kbs[0]
        else:
            print_error("Multiple KBs found. Specify with --kb")
            raise typer.Exit(1)

    print_info(f"KB: {kb}")

    # Load RAG chain
    agent = load_rag_chain(kb)
    if not agent:
        print_error(f"Failed to load knowledge base: {kb}")
        raise typer.Exit(1)

    # Execute queries
    results = []
    for i, query_text in enumerate(queries, 1):
        print_info(f"Query {i}/{len(queries)}: {query_text[:50]}...")

        try:
            import asyncio
            import inspect
            import uuid

            # Generate thread ID for checkpointer
            thread_id = f"{kb}-{uuid.uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}

            # Prepare input
            try:
                from langchain_core.messages import HumanMessage

                input_data = {"messages": [HumanMessage(content=query_text)]}
            except Exception:
                input_data = {"messages": [{"role": "user", "content": query_text}]}

            # Invoke agent
            if inspect.iscoroutinefunction(agent.invoke):
                result = asyncio.run(agent.invoke(input_data, config))
            else:
                result = agent.invoke(input_data, config)

            # Extract answer
            if isinstance(result, dict):
                if "output" in result:
                    answer = result["output"]
                elif "answer" in result:
                    answer = result["answer"]
                elif "messages" in result and result["messages"]:
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, "content"):
                        answer = last_msg.content
                    elif isinstance(last_msg, dict):
                        answer = last_msg.get("content", str(result))
                    else:
                        answer = str(last_msg)
                else:
                    answer = str(result)
            else:
                answer = str(result)

            results.append({"query": query_text, "answer": answer, "status": "success"})

        except Exception as e:
            logger.exception(f"Error on query {i}: {e}")
            results.append({"query": query_text, "error": str(e), "status": "error"})

    # Save results if requested
    if output:
        try:
            import json

            output_path = Path(output)
            output_path.write_text(json.dumps(results, indent=2))
            print_info(f"Results saved to {output_path}")

        except Exception as e:
            print_error(f"Failed to save results: {e}")
            raise typer.Exit(1)

    # Print summary
    successes = len([r for r in results if r["status"] == "success"])
    errors = len([r for r in results if r["status"] == "error"])

    print_info(f"\nBatch completed: {successes} successful, {errors} errors")

    if errors > 0:
        raise typer.Exit(1)
