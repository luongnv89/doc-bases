# tests/test_logger.py
import os
import logging
import pytest
from src.utils.logger import setup_logger, get_logger, toggle_logs, LOG_FILE


# Fixtures
@pytest.fixture
def cleanup_log_file():
    """Fixture to clean up the log file after tests."""
    yield
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)


@pytest.fixture
def logger():
    """Fixture to set up and return the logger."""
    return setup_logger()


# Tests
def test_setup_logger(logger, cleanup_log_file):
    """Test that the logger is correctly configured."""
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2  # File and Stream handlers
    assert isinstance(logger.handlers[0], logging.FileHandler)
    assert isinstance(logger.handlers[1], logging.StreamHandler)


def test_get_logger(logger, cleanup_log_file):
    """Test that get_logger returns the same logger instance."""
    logger_from_get = get_logger()
    assert logger is logger_from_get


def test_toggle_logs(logger, cleanup_log_file, capsys):
    """Test that toggle_logs correctly toggles the logging level."""
    # Initial state: INFO
    assert logger.level == logging.INFO

    # Toggle to CRITICAL
    toggle_logs()
    assert logger.level == logging.CRITICAL
    captured = capsys.readouterr()
    assert "Logging is now disabled." in captured.out

    # Toggle back to INFO
    toggle_logs()
    assert logger.level == logging.INFO
    captured = capsys.readouterr()
    assert "Logging is now enabled." in captured.out


def test_logging_messages(logger, cleanup_log_file, capsys):
    """Test that log messages are correctly written to the log file and console."""
    # Log a message
    test_message = "This is a test log message."
    logger.info(test_message)

    # Check console output
    captured = capsys.readouterr()
    assert test_message in captured.out

    # Check log file
    with open(LOG_FILE, "r") as log_file:
        log_content = log_file.read()
        assert test_message in log_content


def test_logger_handlers_not_duplicated(logger, cleanup_log_file):
    """Test that handlers are not duplicated when setup_logger is called multiple times."""
    # Call setup_logger again
    setup_logger()

    # Ensure only 2 handlers exist (File and Stream)
    assert len(logger.handlers) == 2
