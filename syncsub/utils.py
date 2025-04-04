"""Utility functions for SyncSub."""

import os
import logging
from .exceptions import FileSystemError

logger = logging.getLogger(__name__)

def ensure_dir_exists(dir_path: str) -> None:
    """
    Ensures that a directory exists. Creates it if it doesn't.

    Args:
        dir_path: The path to the directory.

    Raises:
        FileSystemError: If the directory cannot be created due to permissions
                         or if the path exists but is not a directory.
    """
    if not dir_path:
        raise ValueError("Directory path cannot be empty.")
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
        elif not os.path.isdir(dir_path):
            raise FileSystemError(f"Path exists but is not a directory: {dir_path}")
    except OSError as e:
        logger.error(f"Error creating or accessing directory {dir_path}: {e}", exc_info=True)
        raise FileSystemError(f"Could not create or access directory {dir_path}: {e}") from e

def format_time_srt(seconds: float) -> str:
    """
    Formats seconds into SRT time format HH:MM:SS,ms.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 0:
        seconds = 0.0 # Ensure non-negative time
    milliseconds = round(seconds * 1000)
    hrs = milliseconds // 3600000
    milliseconds %= 3600000
    mins = milliseconds // 60000
    milliseconds %= 60000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{milliseconds:03d}"