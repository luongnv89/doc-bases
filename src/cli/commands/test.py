"""Test runner commands"""

import subprocess
import sys

import typer

from src.cli.utils import print_error, print_info, print_section, print_success

app = typer.Typer(help="Run tests")


@app.command(name="run")
def run_tests(
    unit: bool = typer.Option(
        False,
        "--unit",
        help="Run only unit tests",
    ),
    integration: bool = typer.Option(
        False,
        "--integration",
        help="Run only integration tests",
    ),
    coverage: bool = typer.Option(
        False,
        "--coverage",
        help="Generate coverage report",
    ),
    module: str = typer.Option(
        None,
        "--module",
        "-m",
        help="Run tests for specific module",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
) -> None:
    """Run tests."""
    print_section("Running Tests")

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Add path or module
    if module:
        cmd.append(f"tests/{module}")
    else:
        cmd.append("tests")

    # Add options
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    # Add markers
    if unit and not integration:
        cmd.extend(["-m", "unit"])
    elif integration and not unit:
        cmd.extend(["-m", "integration"])

    print_info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print_success("All tests passed")
        else:
            print_error("Some tests failed")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Failed to run tests: {e}")
        raise typer.Exit(1)
