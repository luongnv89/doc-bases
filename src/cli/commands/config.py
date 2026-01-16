"""Configuration management commands"""

from pathlib import Path

import typer
from rich.table import Table

from src.cli.config_manager import get_config_manager
from src.cli.utils import confirm, console, print_config_table, print_error, print_section, print_success

app = typer.Typer(help="Configuration management")


@app.command()
def list(
    all: bool = typer.Option(False, "--all", "-a", help="Show all including defaults"),
    secrets: bool = typer.Option(
        False,
        "--secrets",
        "-s",
        help="Show API keys and secrets (be careful!)",
    ),
) -> None:
    """List all configuration values."""
    config_mgr = get_config_manager()
    config = config_mgr.get_all()

    if all:
        # Show with sources
        config_with_sources = config_mgr.get_all_with_sources()
        table = Table(title="Configuration (with sources)", show_header=True, header_style="bold cyan")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Source", style="yellow")

        def add_rows(cfg: dict, prefix: str = "") -> None:
            for key, item in cfg.items():
                if isinstance(item, dict) and "value" in item:
                    full_key = f"{prefix}.{key}" if prefix else key
                    value = item["value"]
                    source = item["source"]

                    # Mask secrets
                    if not secrets and any(secret in full_key.lower() for secret in ["key", "token", "password", "api"]) and value:
                        display_value = "***"
                    else:
                        display_value = str(value)

                    table.add_row(full_key, display_value, source)

        add_rows(config_with_sources)
        console.print(table)
    else:
        # Show just values in a simple format
        print_config_table(config, mask_secrets=not secrets)


@app.command()
def get(key: str = typer.Argument(..., help="Configuration key (e.g., 'llm.provider')")) -> None:
    """Get a specific configuration value."""
    config_mgr = get_config_manager()
    value = config_mgr.get(key)

    if value is None:
        print_error(f"Configuration key not found: {key}")
        raise typer.Exit(1)

    # Mask secrets
    display_value = "***" if any(secret in key.lower() for secret in ["key", "token", "password", "api"]) and value else value

    console.print(f"[cyan]{key}[/cyan] = [green]{display_value}[/green]")


@app.command()
def set(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Don't ask for confirmation",
    ),
) -> None:
    """Set a configuration value."""
    config_mgr = get_config_manager()

    # Get current value for confirmation
    old_value = config_mgr.get(key)
    is_secret = any(secret in key.lower() for secret in ["key", "token", "password", "api"])

    # Format display values
    if is_secret:
        display_old = "***" if old_value else "(not set)"
        display_new = "***"
    else:
        display_old = str(old_value) if old_value is not None else "(not set)"
        display_new = value

    if not force:
        console.print(f"\n[yellow]Current value:[/yellow] {display_old}")
        console.print(f"[yellow]New value:[/yellow] {display_new}")

        if not confirm("Update this value?", default=False):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Convert value to appropriate type
    typed_value = config_mgr._convert_value(value, key)
    config_mgr.set(key, typed_value)

    # Save to project config file
    _save_config_to_file(config_mgr)

    print_success(f"Configuration updated: {key}")


@app.command()
def validate() -> None:
    """Validate configuration."""
    config_mgr = get_config_manager()
    errors = config_mgr.validate()

    if errors:
        print_error("Configuration validation failed:")
        for error in errors:
            console.print(f"  • {error}")
        raise typer.Exit(1)
    else:
        print_success("Configuration is valid")


@app.command()
def export(
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )
) -> None:
    """Export configuration to YAML."""
    config_mgr = get_config_manager()
    yaml_content = config_mgr.export_yaml()

    if output:
        output_path = Path(output)
        try:
            output_path.write_text(yaml_content)
            print_success(f"Configuration exported to {output_path}")
        except Exception as e:
            print_error(f"Failed to export configuration: {e}")
            raise typer.Exit(1)
    else:
        print_section("Configuration (YAML)")
        console.print(yaml_content)


@app.command()
def import_config(
    file: str = typer.Argument(..., help="YAML configuration file to import"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Don't ask for confirmation",
    ),
) -> None:
    """Import configuration from YAML file."""
    file_path = Path(file)

    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        raise typer.Exit(1)

    try:
        yaml_content = file_path.read_text()
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        raise typer.Exit(1)

    if not force:
        console.print(f"\n[yellow]Importing from:[/yellow] {file_path}")
        console.print(yaml_content)

        if not confirm("Import this configuration?", default=False):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    config_mgr = get_config_manager()

    try:
        config_mgr.import_yaml(yaml_content)
        _save_config_to_file(config_mgr)
        print_success(f"Configuration imported from {file_path}")
    except Exception as e:
        print_error(f"Failed to import configuration: {e}")
        raise typer.Exit(1)


@app.command()
def reset(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Don't ask for confirmation",
    )
) -> None:
    """Reset configuration to defaults."""
    if not force:
        console.print("\n[red]WARNING:[/red] This will reset all configuration to defaults")
        console.print("Your API keys and settings will be lost (unless stored in .env)")

        if not confirm("Reset configuration?", default=False):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    config_mgr = get_config_manager()
    config_mgr.reset_to_defaults()

    # Remove config files
    project_config = Path(".docbases/config.yaml")
    if project_config.exists():
        project_config.unlink()

    print_success("Configuration reset to defaults")


def _save_config_to_file(config_mgr) -> None:
    """Save configuration to project config file."""
    config_dir = Path(".docbases")
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.yaml"
    yaml_content = config_mgr.export_yaml()

    try:
        config_file.write_text(yaml_content)
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        raise typer.Exit(1)
