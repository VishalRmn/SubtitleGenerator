"""Handles audio extraction from video files using ffmpeg."""

import ffmpeg
import os
import logging
from .exceptions import AudioExtractionError, FileSystemError
from typing import Optional
from .utils import ensure_dir_exists

logger = logging.getLogger(__name__)

class AudioExtractor:
    """Extracts audio track from video files."""

    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initializes the AudioExtractor.

        Args:
            ffmpeg_path: Optional path to the ffmpeg executable.
                         If None, assumes ffmpeg is in the system PATH.
        """
        self.ffmpeg_cmd = ffmpeg_path or 'ffmpeg'
        logger.info(f"Using ffmpeg command: {self.ffmpeg_cmd}")

    def extract_audio(self, video_filepath: str, output_audio_dir: str, output_filename: Optional[str] = None) -> str:
        """
        Extracts the audio stream from a video file to a WAV file.

        Args:
            video_filepath: Path to the input video file.
            output_audio_dir: Directory to save the extracted audio file.
            output_filename: Optional base name for the output audio file (without extension).
                             If None, uses the video filename.

        Returns:
            The full path to the extracted audio file (WAV format).

        Raises:
            FileNotFoundError: If the input video file does not exist.
            AudioExtractionError: If ffmpeg fails to extract the audio.
            FileSystemError: If the output directory cannot be created/accessed.
        """
        logger.info(f"Starting audio extraction for: {video_filepath}")
        if not os.path.exists(video_filepath):
            raise FileNotFoundError(f"Input video file not found: {video_filepath}")

        ensure_dir_exists(output_audio_dir) # Ensure output dir exists

        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(video_filepath))[0]
        else:
             base_name = os.path.splitext(output_filename)[0] # Ensure no ext in filename either

        output_audio_path = os.path.join(output_audio_dir, f"{base_name}.wav")

        logger.debug(f"Output audio path set to: {output_audio_path}")

        # Check if output file already exists, potentially overwrite or skip
        if os.path.exists(output_audio_path):
            logger.warning(f"Output audio file already exists, overwriting: {output_audio_path}")
            try:
                os.remove(output_audio_path)
            except OSError as e:
                 raise FileSystemError(f"Could not remove existing audio file {output_audio_path}: {e}") from e


        try:
            logger.info(f"Running ffmpeg to extract audio to {output_audio_path}...")
            # Use ffmpeg-python bindings
            # acodec='pcm_s16le' -> standard WAV format
            # ar=16000 -> Resample to 16kHz, common for ASR models like Whisper
            # ac=1 -> Mono channel
            (
                ffmpeg
                .input(video_filepath)
                .output(output_audio_path, acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output() # Already handled manually, but good practice
                .run(cmd=self.ffmpeg_cmd, capture_stdout=True, capture_stderr=True, quiet=False) # Set quiet=False to see ffmpeg output in logs if needed
            )
            logger.info(f"Successfully extracted audio to: {output_audio_path}")
            return output_audio_path
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg error during audio extraction for {video_filepath}", exc_info=True)
            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
            logger.error(f"ffmpeg stderr: {stderr_output}")
            # Attempt cleanup if extraction failed mid-way
            if os.path.exists(output_audio_path):
                 try:
                     os.remove(output_audio_path)
                 except OSError:
                     logger.warning(f"Could not clean up partially created audio file: {output_audio_path}")
            raise AudioExtractionError(f"ffmpeg failed: {stderr_output}") from e
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}", exc_info=True)
            raise AudioExtractionError(f"An unexpected error occurred: {e}") from e