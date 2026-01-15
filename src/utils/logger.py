# src/utils/logger.py
import logging
import os
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Define a custom theme for rich console output
custom_theme = Theme(
    {
        "info": "bold blue",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "header": "bold cyan",
    }
)

# Initialize rich console with custom_theme
console = Console(theme=custom_theme)

# Ensure the logs directory exists
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
    console.print(f"[info]Created logs directory: {LOGS_DIR}[/info]")

# Create a log file with a timestamp in the filename
LOG_FILE = os.path.join(LOGS_DIR, f"doc_bases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def setup_logger(level=logging.INFO):
    """Sets up and returns a logger instance with a timestamped log file."""
    logger = logging.getLogger("doc_bases")
    logger.setLevel(level)

    if not logger.hasHandlers():
        # Formatter for file handler
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

        # File handler for logging to a file
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)

        # Rich console handler for styled output
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            level=level,
            rich_tracebacks=True,
        )

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(rich_handler)

    console.print(f"[info]Logging to file: {LOG_FILE}[/info]")
    return logger


def get_logger():
    """Return the logger instance."""
    return logging.getLogger("doc_bases")


def toggle_logs():
    """Toggles logging on and off."""
    logger = logging.getLogger("doc_bases")
    current_level = logger.level
    if current_level == logging.INFO:
        logger.setLevel(logging.CRITICAL)
        console.print("[info]Logging is now disabled.[/info]")
    else:
        logger.setLevel(logging.INFO)
        console.print("[info]Logging is now enabled.[/info]")
        logger.info("Logging is now enabled.")
