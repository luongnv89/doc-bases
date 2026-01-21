"""Query commands for knowledge bases"""

from pathlib import Path

import typer

from src.cli.utils import console, print_error, print_info, print_section
from src.utils.kb_metadata import detect_file_changes, format_change_report, get_reindex_command
from src.utils.logger import get_logger
from src.utils.rag_utils import interactive_cli, list_knowledge_bases, load_rag_chain

logger = get_logger()

app = typer.Typer(help="Query knowledge bases")


def check_and_prompt_for_changes(kb: str) -> bool:
    """
    Check for source file changes and prompt user for action.

    Args:
        kb: Knowledge base name

    Returns:
        True if the session should continue, False to exit for re-indexing
    """
    report = detect_file_changes(kb)

    # Handle special cases
    if report.error == "no_metadata":
        console.print("[dim]Note: This KB does not have change tracking metadata.[/dim]")
        console.print("[dim]Re-index with 'docb kb add' to enable change detection.[/dim]\n")
        return True

    if report.error == "source_not_found":
        console.print(f"[yellow]Warning:[/yellow] Source path no longer exists: {report.source_path}")
        console.print("[dim]The knowledge base may contain outdated information.[/dim]\n")
        return True

    if report.error == "remote_source":
        # Remote sources (repo, website, url) don't support change detection
        return True

    if not report.has_changes:
        return True

    # Display change report
    formatted = format_change_report(report, kb)
    if formatted:
        console.print(formatted)
        console.print()

    # Prompt user for action
    console.print("[cyan]What would you like to do?[/cyan]")
    console.print("  [1] Continue without updating")
    console.print("  [2] Update KB now (shows re-index command)")

    try:
        choice = console.input("\n[cyan]Choice [1/2]: [/cyan]").strip()
    except KeyboardInterrupt:
        print_info("\nCancelled")
        raise typer.Exit(0)

    if choice == "2":
        reindex_cmd = get_reindex_command(kb, report.source_type, report.source_path or "")
        console.print("\n[cyan]Run this command to update the knowledge base:[/cyan]")
        console.print(f"  [green]{reindex_cmd}[/green]\n")
        return False

    return True


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

    # If no KB specified, check available ones
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

    # Check for file changes
    if not check_and_prompt_for_changes(kb):
        raise typer.Exit(0)

    # Call interactive CLI with the KB name and session
    try:
        interactive_cli(knowledge_base_name=kb, session_id=session)
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

    # Check for file changes
    if not check_and_prompt_for_changes(kb):
        raise typer.Exit(0)

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

        # Prepare input with HumanMessage
        from langchain_core.messages import HumanMessage

        input_data = {"messages": [HumanMessage(content=query)]}

        # Invoke agent
        if inspect.iscoroutinefunction(agent.invoke):  # noqa: SIM108
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

    # Check for file changes
    if not check_and_prompt_for_changes(kb):
        raise typer.Exit(0)

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

            # Prepare input with HumanMessage
            from langchain_core.messages import HumanMessage

            input_data = {"messages": [HumanMessage(content=query_text)]}

            # Invoke agent
            if inspect.iscoroutinefunction(agent.invoke):  # noqa: SIM108
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
