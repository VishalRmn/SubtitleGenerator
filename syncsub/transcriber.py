"""Handles Speech-to-Text transcription using Whisper."""

import whisper
import logging
import torch
from abc import ABC, abstractmethod
from typing import Optional
import os

from .models import TranscriptionResult, Segment
from .exceptions import TranscriptionError

logger = logging.getLogger(__name__)

class Transcriber(ABC):
    """Abstract base class for transcription services."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribes the given audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            A TranscriptionResult object containing segments and language.

        Raises:
            TranscriptionError: If transcription fails.
            FileNotFoundError: If the audio file doesn't exist.
        """
        pass

class WhisperTranscriber(Transcriber):
    """Implements transcription using OpenAI's Whisper model."""

    def __init__(self, model_name: str = "medium.en", device: str = "cuda", fp16: bool = True):
        """
        Initializes the WhisperTranscriber.

        Args:
            model_name: The name of the Whisper model to use (e.g., "base", "medium.en").
            device: The device to run the model on ("cuda" or "cpu").
            fp16: Whether to use float16 precision (faster on compatible GPUs).

        Raises:
            ValueError: If the specified device is invalid or unavailable.
            TranscriptionError: If the model fails to load.
        """
        self.model_name = model_name
        self.device = device
        self.fp16 = fp16

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            self.device = "cpu"
        elif self.device not in ["cuda", "cpu"]:
             raise ValueError(f"Invalid device specified: {self.device}. Choose 'cuda' or 'cpu'.")

        logger.info(f"Initializing WhisperTranscriber with model '{self.model_name}' on device '{self.device}' (FP16: {self.fp16})")
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_name}': {e}", exc_info=True)
            raise TranscriptionError(f"Failed to load Whisper model '{self.model_name}': {e}") from e

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribes the audio file using the loaded Whisper model.

        Args:
            audio_path: Path to the audio file (WAV format recommended).

        Returns:
            A TranscriptionResult object.

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            TranscriptionError: If transcription fails during processing.
        """
        logger.info(f"Starting transcription for: {audio_path}")
        if not os.path.exists(audio_path):
             raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Note: Whisper's 'language' parameter enforces decoding in that language.
            # Omit it to let Whisper auto-detect (might be slower, less accurate if you KNOW it's English)
            # Use fp16=False if encountering precision issues, though self.fp16 controls it globally here.
            # verbose=None uses default logging, verbose=False suppresses Whisper's tqdm/prints
            result = self.model.transcribe(
                audio_path,
                language='en', # Assuming English source as per plan
                fp16=self.fp16 if self.device == "cuda" else False, # FP16 only works on CUDA
                verbose=None # Let Whisper handle its progress bars if desired, or set False
            )
            logger.info(f"Transcription completed. Detected language: {result.get('language', 'N/A')}")

            segments = []
            if 'segments' in result:
                for seg_data in result['segments']:
                    # Whisper provides 'start', 'end', 'text'
                    if 'start' in seg_data and 'end' in seg_data and 'text' in seg_data:
                        segment = Segment(
                            start_time=float(seg_data['start']),
                            end_time=float(seg_data['end']),
                            text=seg_data['text'].strip() # Remove leading/trailing whitespace
                        )
                        segments.append(segment)
                    else:
                        logger.warning(f"Skipping incomplete segment data: {seg_data}")
            else:
                 logger.warning("Transcription result did not contain 'segments'.")


            transcription_result = TranscriptionResult(
                language=result.get('language'),
                segments=segments,
                original_audio_path=audio_path
            )
            logger.info(f"Processed {len(segments)} segments from transcription.")
            return transcription_result

        except FileNotFoundError as e: # Should be caught earlier, but double-check
            logger.error(f"Audio file disappeared during transcription?: {audio_path}", exc_info=True)
            raise e # Re-raise specific FileNotFoundError
        except Exception as e:
            logger.error(f"Error during Whisper transcription process for {audio_path}: {e}", exc_info=True)
            raise TranscriptionError(f"Whisper transcription failed for {audio_path}: {e}") from e