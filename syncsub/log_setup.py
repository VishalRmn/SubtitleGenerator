"""Logging configuration for SyncSub."""

import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from .utils import ensure_dir_exists # Use relative import

DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = "syncsub.log",
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = 10 * 1024 * 1024, # 10 MB
    backup_count: int = 5
) -> None:
    """
    Configures logging for the application.

    Sets up logging to both the console (stdout) and a rotating file.

    Args:
        log_level: The minimum logging level (e.g., logging.INFO, logging.DEBUG).
        log_dir: The directory to store log files.
        log_file: The name of the log file.
        log_format: The format string for log messages.
        date_format: The format string for timestamps in logs.
        max_bytes: Maximum size of a log file before rotation.
        backup_count: Number of backup log files to keep.
    """
    logger = logging.getLogger() # Get root logger
    if logger.hasHandlers():
        # Avoid adding handlers multiple times if called again
        # logger.warning("Logger already configured. Skipping setup.")
        # return # Or clear existing handlers if re-configuration is desired
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console Handler (Stream to stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level) # Console can have its own level if needed
    logger.addHandler(stream_handler)

    # File Handler (Rotating)
    try:
        ensure_dir_exists(log_dir)
        log_path = os.path.join(log_dir, log_file)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging initialized. Log file: {log_path}")
    except Exception as e:
        # Log to console if file handler fails
        logger.error(f"Failed to set up file logging handler at {log_dir}/{log_file}: {e}", exc_info=True)

    # Suppress overly verbose logs from dependencies if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # logging.getLogger("whisper").setLevel(logging.WARNING) # Uncomment if whisper logs too much
    # logging.getLogger("transformers").setLevel(logging.WARNING) # Uncomment if transformers logs too much