# src/utils/logger.py

import logging

LOG_FILE = "rag_system.log"

def setup_logger(level=logging.INFO):
    """Sets up and returns a logger instance."""
    logger = logging.getLogger("rag_system")
    logger.setLevel(level)

    if not logger.hasHandlers():
      formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

      file_handler = logging.FileHandler(LOG_FILE)
      file_handler.setFormatter(formatter)

      stream_handler = logging.StreamHandler()
      stream_handler.setFormatter(formatter)

      logger.addHandler(file_handler)
      logger.addHandler(stream_handler)

    return logger

def get_logger():
    """Return the logger instance"""
    return logging.getLogger("rag_system")

def toggle_logs():
    """Toggles logging on and off."""
    logger = logging.getLogger("rag_system")
    current_level = logger.level
    if current_level == logging.INFO:
        logger.setLevel(logging.CRITICAL)
        print("Logging is now disabled.")
    else:
        logger.setLevel(logging.INFO)
        print("Logging is now enabled.")
        logger.info("Logging is now enabled.")