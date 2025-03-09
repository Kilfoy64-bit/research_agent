"""
Logging configuration for the research agent.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: The name of the module requesting the logger

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Configure the logging system for the entire application.

    Args:
        log_level: The log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to a log file, or None to log to stderr only
        log_format: Format string for log messages, or None to use default
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get the numeric log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Configure the root logger
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set LangChain logging to WARNING unless we're in DEBUG mode
    if numeric_level > logging.DEBUG:
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("langgraph").setLevel(logging.WARNING)


def get_log_level_from_env() -> str:
    """
    Get the log level from environment variable or return the default.

    Returns:
        The log level string (INFO, DEBUG, etc.)
    """
    return os.getenv("LOG_LEVEL", "INFO").upper()
