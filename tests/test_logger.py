# tests/test_logger.py
import logging
import os
import tempfile

import pytest

from src.utils.logger import get_logger, setup_logger, toggle_logs


# Helper to create a fresh logger with a temp log file
def create_fresh_logger():
    """Create a fresh logger with a temp log file and return (logger, log_file_path)."""
    # Create temp log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        temp_log_path = f.name

    # Get and clear the logger
    logger = logging.getLogger("doc_bases")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)

    # Manually set up handlers with our temp file
    logger.setLevel(logging.INFO)

    # Formatter for file handler
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

    # File handler for logging to temp file
    file_handler = logging.FileHandler(temp_log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # We'll skip RichHandler in tests to simplify - just add a StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, temp_log_path


# Fixtures
@pytest.fixture
def fresh_logger():
    """Fixture to set up and return a fresh logger instance with temp file."""
    logger, temp_log_path = create_fresh_logger()
    yield logger, temp_log_path

    # Cleanup
    logger.handlers.clear()
    if os.path.exists(temp_log_path):
        os.remove(temp_log_path)


# Tests
def test_setup_logger():
    """Test that the logger is correctly configured."""
    # Clear any existing handlers first
    logger = logging.getLogger("doc_bases")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)

    # Also need to disable propagation temporarily to prevent hasHandlers() from
    # returning True due to parent handlers
    original_propagate = logger.propagate
    logger.propagate = False

    # Call the real setup_logger
    result_logger = setup_logger()

    try:
        assert result_logger.level == logging.INFO
        assert len(result_logger.handlers) == 2  # File and Rich handlers
        # Check handler types
        handler_types = [type(h).__name__ for h in result_logger.handlers]
        assert "FileHandler" in handler_types
        assert "RichHandler" in handler_types
    finally:
        # Cleanup
        result_logger.handlers.clear()
        logger.propagate = original_propagate


def test_get_logger(fresh_logger):
    """Test that get_logger returns the same logger instance."""
    logger, _ = fresh_logger
    logger_from_get = get_logger()
    assert logger is logger_from_get


def test_toggle_logs(fresh_logger):
    """Test that toggle_logs correctly toggles the logging level."""
    logger, _ = fresh_logger
    # Initial state: INFO
    assert logger.level == logging.INFO

    # Toggle to CRITICAL
    toggle_logs()
    assert logger.level == logging.CRITICAL

    # Toggle back to INFO
    toggle_logs()
    assert logger.level == logging.INFO


def test_logging_messages(fresh_logger):
    """Test that log messages are correctly written to the log file."""
    logger, log_file_path = fresh_logger
    # Log a message
    test_message = "This is a test log message."
    logger.info(test_message)

    # Flush handlers to ensure message is written
    for handler in logger.handlers:
        handler.flush()

    # Check log file exists and contains the message
    assert os.path.exists(log_file_path)
    with open(log_file_path) as log_file:
        log_content = log_file.read()
        assert test_message in log_content


def test_logger_handlers_not_duplicated():
    """Test that handlers are not duplicated when setup_logger is called multiple times."""
    # Clear any existing handlers first
    logger = logging.getLogger("doc_bases")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)

    # Call setup_logger first time
    setup_logger()
    initial_handler_count = len(logger.handlers)

    # Call setup_logger again
    setup_logger()

    try:
        # Ensure handlers are not duplicated
        assert len(logger.handlers) == initial_handler_count
    finally:
        # Cleanup
        logger.handlers.clear()
