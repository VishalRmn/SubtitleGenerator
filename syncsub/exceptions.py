"""Custom Exceptions for the SyncSub application."""

class SyncSubError(Exception):
    """Base class for exceptions in this module."""
    pass

class ConfigurationError(SyncSubError):
    """Exception raised for errors in configuration loading."""
    pass

class AudioExtractionError(SyncSubError):
    """Exception raised for errors during audio extraction."""
    pass

class TranscriptionError(SyncSubError):
    """Exception raised for errors during transcription."""
    pass

class TranslationError(SyncSubError):
    """Exception raised for errors during translation."""
    pass

class FormattingError(SyncSubError):
    """Exception raised for errors during subtitle formatting."""
    pass

class FileSystemError(SyncSubError):
    """Exception raised for file system related errors (permissions, not found etc)."""
    pass