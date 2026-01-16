"""CLI utilities and helper functions"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

# Initialize console for CLI
console = Console()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_venv_path() -> Path | None:
    """Find active or available virtual environment."""
    project_root = get_project_root()

    # Check common venv locations
    venv_paths = [
        project_root / ".venv",
        project_root / "venv",
        project_root / "env",
        project_root / "virtualenv",
    ]

    for venv_path in venv_paths:
        if venv_path.exists():
            return venv_path

    return None


def is_venv_active() -> bool:
    """Check if a virtual environment is currently active."""
    return hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)


def run_command(
    command: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a shell command safely.

    Args:
        command: List of command parts (not shell string)
        cwd: Working directory
        check: Raise on non-zero exit code
        capture_output: Capture stdout/stderr

    Returns:
        CompletedProcess result
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd or get_project_root(),
            check=check,
            capture_output=capture_output,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed: {' '.join(command)}[/red]")
        if e.stderr:
            console.print(f"[red]{e.stderr}[/red]")
        raise


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓ {message}[/green]")


def print_error(message: str, hint: str | None = None) -> None:
    """Print an error message with optional hint."""
    console.print(f"[red]✗ {message}[/red]")
    if hint:
        console.print(f"[yellow]  Hint: {hint}[/yellow]")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[cyan]ℹ {message}[/cyan]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠ {message}[/yellow]")


def print_section(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("[cyan]" + "─" * 60 + "[/cyan]")


def print_panel(content: str, title: str | None = None) -> None:
    """Print content in a rich panel."""
    console.print(
        Panel(
            content,
            title=f"[bold cyan]{title}[/bold cyan]" if title else None,
            border_style="cyan",
            padding=(1, 2),
        )
    )


def confirm(prompt: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.

    Args:
        prompt: The confirmation prompt
        default: Default answer if user just presses Enter

    Returns:
        User's choice
    """
    choices = "Y/n" if default else "y/N"
    response = input(f"{prompt} ({choices}): ").strip().lower()

    if not response:
        return default

    return response in ("y", "yes")


def prompt_for_value(
    prompt: str,
    default: str | None = None,
    required: bool = False,
) -> str:
    """
    Prompt user for a value.

    Args:
        prompt: The prompt text
        default: Default value if user just presses Enter
        required: Whether a non-empty value is required

    Returns:
        User's input or default
    """
    display_prompt = prompt
    if default:
        display_prompt += f" [{default}]"
    display_prompt += ": "

    while True:
        value = input(display_prompt).strip()

        if not value:
            if default:
                return default
            if required:
                console.print("[red]Value is required[/red]")
                continue

        return value


def get_env_var(key: str, default: str | None = None) -> str | None:
    """Get environment variable safely."""
    return os.environ.get(key, default)


def print_config_table(config: dict[str, Any], mask_secrets: bool = True) -> None:
    """Print configuration as a table."""
    from rich.table import Table

    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    def flatten_config(cfg: dict[str, Any], prefix: str = "") -> None:
        for key, value in cfg.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flatten_config(value, full_key)
            else:
                # Mask sensitive values
                if mask_secrets and any(secret in full_key.lower() for secret in ["key", "token", "password", "api"]):
                    display_value = "***" if value else ""
                else:
                    display_value = str(value)

                table.add_row(full_key, display_value)

    flatten_config(config)
    console.print(table)
