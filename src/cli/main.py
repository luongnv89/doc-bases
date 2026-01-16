"""
DocBases Professional CLI Application
Main Typer application with command registration
"""

import os

import typer
from rich.console import Console
from rich.panel import Panel

from src.utils.logger import custom_theme
from src.utils.utilities import get_version_from_git

# Initialize Typer app
app = typer.Typer(
    name="docb",
    help="DocBases - Intelligent Answers from Any Document",
    add_completion=True,
    rich_markup_mode="rich",
)

# Initialize rich console with custom theme
console = Console(theme=custom_theme)


def print_welcome() -> None:
    """Print welcome message."""
    version = get_version_from_git()
    welcome_message = f"""
    [bold cyan]DocBases[/bold cyan] 🚀
    Intelligent Answers from Any Document
    [yellow]Version:[/yellow] {version}
    """
    console.print(
        Panel.fit(
            welcome_message,
            title="[bold cyan]Welcome[/bold cyan]",
            border_style="bold cyan",
        )
    )


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to custom config file",
    ),
):
    """
    DocBases CLI - Professional command-line interface for document querying.

    Use 'docb [command] --help' for command-specific help.
    """
    if version:
        version_str = get_version_from_git()
        console.print(f"[cyan]DocBases[/cyan] version [green]{version_str}[/green]")
        raise typer.Exit()

    if verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"

    if config:
        os.environ["DOCB_CONFIG"] = config

    # Print welcome only if no command is being invoked
    if ctx.invoked_subcommand is None:
        print_welcome()
        console.print("\n[yellow]Tip:[/yellow] Use 'docb --help' to see available commands")


@app.command()
def version() -> None:
    """Show DocBases version."""
    version_str = get_version_from_git()
    console.print(f"[cyan]DocBases[/cyan] version [green]{version_str}[/green]")


# Register command groups
from src.cli.commands import config as config_commands
from src.cli.commands import health as health_commands
from src.cli.commands import kb as kb_commands
from src.cli.commands import query as query_commands
from src.cli.commands import setup as setup_commands
from src.cli.commands import test as test_commands

app.add_typer(config_commands.app, name="config", help="Configuration management")
app.add_typer(setup_commands.app, name="setup", help="Setup and initialization")
app.add_typer(kb_commands.app, name="kb", help="Knowledge base management")
app.add_typer(query_commands.app, name="query", help="Query knowledge bases")
app.add_typer(test_commands.app, name="test", help="Run tests")
app.add_typer(health_commands.app, name="health", help="Health checks")


if __name__ == "__main__":
    app()
