"""Data models for SyncSub."""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Segment:
    """Represents a single timed chunk of text."""
    start_time: float
    end_time: float
    text: str

@dataclass
class TranscriptionResult:
    """Holds the structured output from the ASR process."""
    language: Optional[str]
    segments: List[Segment] = field(default_factory=list)
    original_audio_path: Optional[str] = None # Keep track of source if needed