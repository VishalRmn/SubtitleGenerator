"""Orchestrates the subtitle generation pipeline."""

import logging
import os
import time

from .audio_extractor import AudioExtractor
from .transcriber import Transcriber
from .translator import Translator
from .subtitle_formatter import SubtitleFormatter, SRTFormatter #, VTTFormatter # Import VTT when implemented
from .models import TranscriptionResult, Segment
from .exceptions import SyncSubError, FileSystemError
from .utils import ensure_dir_exists
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class SubtitleGenerator:
    """
    Manages the end-to-end process of generating subtitles for a video file.
    """

    def __init__(
        self,
        config: dict,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        translator: Translator,
        # subtitle_formatter: SubtitleFormatter # Use specific formatters for now
    ):
        """
        Initializes the SubtitleGenerator.

        Args:
            config: A dictionary containing configuration settings.
            audio_extractor: An instance of AudioExtractor.
            transcriber: An instance of Transcriber.
            translator: An instance of Translator.
            # subtitle_formatter: An instance of SubtitleFormatter.
        """
        self.config = config
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.translator = translator
        # self.subtitle_formatter = subtitle_formatter # Choose formatter based on config later

        # Validate essential config paths
        self.temp_dir = config.get('temp_dir')
        if not self.temp_dir:
            raise SyncSubError("Configuration missing 'temp_dir'.")
        try:
            # Ensure temp dir exists and is writable early on
             ensure_dir_exists(self.temp_dir)
             # Test writability (optional, but good check)
             test_file = os.path.join(self.temp_dir, f".syncsub_write_test_{int(time.time())}")
             with open(test_file, "w") as f: f.write("test")
             os.remove(test_file)
        except (FileSystemError, OSError, ValueError) as e:
             raise SyncSubError(f"Temporary directory '{self.temp_dir}' is invalid or not writable: {e}") from e


        self.output_format = config.get('output_format', 'srt').lower()
        if self.output_format == 'srt':
             self.subtitle_formatter = SRTFormatter()
        # elif self.output_format == 'vtt':
        #      self.subtitle_formatter = VTTFormatter() # Use when implemented
        else:
             raise SyncSubError(f"Unsupported output format '{self.output_format}' specified in config.")

        # Get segmentation rules from config
        self.max_chars = config.get('max_chars_per_segment', 80)
        self.max_duration = config.get('max_duration_seconds', 7.0)
        self.max_lines = config.get('max_lines_per_block', 2)


    def _get_output_paths(self, video_path: str, output_dir: str) -> Tuple[str, str, str]:
        """Determines output filenames based on video path and config."""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        ext = self.output_format # e.g., 'srt'
        en_subtitle_path = os.path.join(output_dir, f"{base_name}.en.{ext}")
        ml_subtitle_path = os.path.join(output_dir, f"{base_name}.ml.{ext}")
        temp_audio_filename = f"{base_name}_{int(time.time())}.wav" # Add timestamp for uniqueness
        temp_audio_path = os.path.join(self.temp_dir, temp_audio_filename)
        return en_subtitle_path, ml_subtitle_path, temp_audio_path

    def _cleanup_temp_files(self, *file_paths: str) -> None:
        """Removes temporary files specified."""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary file {file_path}: {e}", exc_info=False) # Don't need full stack for cleanup warning


    def generate(self, video_path: str, output_dir: str) -> None:
        """
        Executes the full subtitle generation pipeline for a single video.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save the final subtitle files.

        Raises:
            SyncSubError: For any configuration or processing errors in the pipeline.
            FileNotFoundError: If the input video is not found.
            FileSystemError: If output/temp directories are invalid/unwritable.
        """
        start_time = time.time()
        logger.info(f"--- Starting SyncSub process for: {video_path} ---")
        ensure_dir_exists(output_dir) # Ensure output dir exists

        en_subtitle_path, ml_subtitle_path, temp_audio_path = self._get_output_paths(video_path, output_dir)
        extracted_audio_path = None # Keep track of created temp file

        try:
            # 1. Extract Audio
            logger.info("Step 1: Extracting Audio...")
            extracted_audio_path = self.audio_extractor.extract_audio(video_path, self.temp_dir, os.path.basename(temp_audio_path))
            logger.info(f"Audio extracted to: {extracted_audio_path}")

            # 2. Transcribe (English)
            logger.info("Step 2: Transcribing Audio (English)...")
            english_result: TranscriptionResult = self.transcriber.transcribe(extracted_audio_path)
            if not english_result or not english_result.segments:
                 raise SyncSubError("Transcription produced no segments. Cannot proceed.")
            logger.info(f"Transcription complete. Found {len(english_result.segments)} segments.")

            # 3. Format English Subtitles
            logger.info(f"Step 3: Formatting English Subtitles ({self.output_format.upper()})...")
            self.subtitle_formatter.format_subtitles(
                english_result,
                en_subtitle_path,
                max_chars_per_segment=self.max_chars,
                max_duration_seconds=self.max_duration,
                max_lines_per_block=self.max_lines
            )
            logger.info(f"English subtitles saved to: {en_subtitle_path}")

            # 4. Translate to Malayalam
            logger.info("Step 4: Translating Segments to Malayalam...")
            malayalam_segments: List[Segment] = []
            total_segments = len(english_result.segments)
            for i, segment in enumerate(english_result.segments):
                try:
                    # Preserve original timing, translate text
                    translated_text = self.translator.translate(segment.text, source_lang='en', target_lang='ml')
                    malayalam_segments.append(
                        Segment(
                            start_time=segment.start_time,
                            end_time=segment.end_time,
                            text=translated_text
                        )
                    )
                    if (i + 1) % 20 == 0 or i == total_segments - 1: # Log progress periodically
                        logger.info(f"Translated segment {i + 1}/{total_segments}")
                except TranslationError as e:
                    logger.warning(f"Failed to translate segment {i+1} ('{segment.text[:30]}...'): {e}. Skipping segment.")
                    # Optionally add a placeholder or keep original English? For now, skip.
                    # malayalam_segments.append(Segment(start_time=segment.start_time, end_time=segment.end_time, text="[Translation Failed]"))


            if not malayalam_segments:
                logger.warning("Translation resulted in no valid segments. Skipping Malayalam subtitle generation.")
            else:
                malayalam_result = TranscriptionResult(
                    language='ml', # Set target language
                    segments=malayalam_segments,
                    original_audio_path=extracted_audio_path # Reference original source
                )

                # 5. Format Malayalam Subtitles
                logger.info(f"Step 5: Formatting Malayalam Subtitles ({self.output_format.upper()})...")
                self.subtitle_formatter.format_subtitles(
                    malayalam_result,
                    ml_subtitle_path,
                    max_chars_per_segment=self.max_chars,
                    max_duration_seconds=self.max_duration,
                    max_lines_per_block=self.max_lines
                    )
                logger.info(f"Malayalam subtitles saved to: {ml_subtitle_path}")

            end_time = time.time()
            logger.info(f"--- SyncSub process completed successfully in {end_time - start_time:.2f} seconds ---")

        except (SyncSubError, FileNotFoundError, FileSystemError) as e:
            # Catch specific, known errors and log them cleanly
            logger.error(f"SyncSub process failed: {e}", exc_info=False) # No stack needed for expected errors
            raise # Re-raise to be caught by CLI handler
        except Exception as e:
            # Catch unexpected errors and log with stack trace
            logger.critical(f"An unexpected critical error occurred during subtitle generation: {e}", exc_info=True)
            # Wrap in SyncSubError for consistent handling upstream
            raise SyncSubError(f"An unexpected critical error occurred: {e}") from e
        finally:
            # 6. Cleanup
            logger.info("Step 6: Cleaning up temporary files...")
            self._cleanup_temp_files(extracted_audio_path) # Pass the actual path used