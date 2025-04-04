#!/usr/bin/env python3
"""
SyncSub Batch Processing Entry Point

Processes all MP4 videos in a specified directory, ordered by size,
generating English and Malayalam subtitles in structured subfolders.
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Tuple

# Progress bar library
from tqdm import tqdm

# Import necessary components from the syncsub package
from syncsub.config_loader import ConfigLoader
from syncsub.log_setup import setup_logging
from syncsub.audio_extractor import AudioExtractor
from syncsub.transcriber import WhisperTranscriber
from syncsub.translator import HuggingFaceTranslator
from syncsub.subtitle_generator import SubtitleGenerator
from syncsub.exceptions import SyncSubError, ConfigurationError, FileSystemError
from syncsub.utils import ensure_dir_exists

# Initialize logger for this script
logger = logging.getLogger(__name__)

def find_and_sort_videos(input_dir: str) -> List[Tuple[str, int]]:
    """
    Finds all .mp4 files in the input directory and sorts them by size.

    Args:
        input_dir: The directory to search for video files.

    Returns:
        A list of tuples, where each tuple is (filepath, filesize),
        sorted by filesize in ascending order.

    Raises:
        FileNotFoundError: If the input directory doesn't exist.
        ValueError: If the input path is not a directory.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    videos = []
    logger.info(f"Scanning directory for MP4 files: {input_dir}")
    for filename in os.listdir(input_dir):
        # Case-insensitive check for .mp4 extension
        if filename.lower().endswith(".mp4"):
            filepath = os.path.join(input_dir, filename)
            try:
                if os.path.isfile(filepath): # Ensure it's actually a file
                    filesize = os.path.getsize(filepath)
                    videos.append((filepath, filesize))
            except OSError as e:
                logger.warning(f"Could not access file {filepath}: {e}. Skipping.")

    # Sort videos by file size (the second element in the tuple)
    videos.sort(key=lambda item: item[1])
    logger.info(f"Found {len(videos)} MP4 files. Sorted by size (smallest first).")
    return videos


def run_batch_processing():
    """Parses arguments, sets up, and runs the batch subtitle generation."""
    parser = argparse.ArgumentParser(
        description="SyncSub Batch: Generate EN/ML subtitles for all MP4s in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Directory containing the input MP4 video files."
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--temp-dir",
        default=None, # Default taken from config file
        help="Override the temporary directory specified in the config file. Must exist."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for console and file output."
    )
    parser.add_argument(
        "--device",
        default=None, # Default taken from config
        choices=["cuda", "cpu"],
        help="Override the processing device (cuda or cpu) specified in config."
    )

    args = parser.parse_args()

    # --- Setup Logging (Initial) ---
    log_level_name = args.log_level.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    setup_logging(log_level=log_level, log_dir='logs', log_file='syncsub_batch_init.log')

    # --- Load Configuration ---
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
    except (ConfigurationError, FileNotFoundError) as e:
        logger.critical(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    # --- Re-configure Logging (Final) ---
    log_dir = config.get('log_dir', 'logs')
    log_file = config.get('log_file', 'syncsub_batch.log') # Use a different log file name for batch
    setup_logging(log_level=log_level, log_dir=log_dir, log_file=log_file)
    logger.info("Logging re-configured with settings from config file for batch processing.")

    # --- Apply CLI Overrides ---
    if args.temp_dir:
        logger.info(f"Overriding temp_dir from config with CLI argument: {args.temp_dir}")
        config['temp_dir'] = args.temp_dir
    if args.device:
        logger.info(f"Overriding device from config with CLI argument: {args.device}")
        config['device'] = args.device

    # --- Find and Sort Videos ---
    try:
        sorted_videos_with_size = find_and_sort_videos(args.input_dir)
        if not sorted_videos_with_size:
            logger.warning(f"No .mp4 files found in {args.input_dir}. Exiting.")
            sys.exit(0)
        # Extract just the paths for processing
        sorted_video_paths = [item[0] for item in sorted_videos_with_size]
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Input directory error: {e}")
        sys.exit(1)

    # --- Setup Output Directories ---
    base_subs_dir = os.path.join(args.input_dir, "Subs")
    english_subs_dir = os.path.join(base_subs_dir, "English")
    malayalam_subs_dir = os.path.join(base_subs_dir, "Malayalam")

    try:
        ensure_dir_exists(base_subs_dir)
        ensure_dir_exists(english_subs_dir)
        ensure_dir_exists(malayalam_subs_dir)
        logger.info(f"Ensured output directories exist: {english_subs_dir}, {malayalam_subs_dir}")
    except FileSystemError as e:
        logger.critical(f"Could not create output directories: {e}")
        sys.exit(1)

    # --- Initialize Components (ONCE) ---
    # This is the key efficiency gain for batch processing
    generator = None
    try:
        logger.info("Initializing SyncSub components for batch processing...")
        device = config.get('device', 'cuda')

        audio_extractor = AudioExtractor(ffmpeg_path=config.get('ffmpeg_path'))
        transcriber = WhisperTranscriber(
            model_name=config.get('whisper_model', 'medium.en'),
            device=device,
            fp16=config.get('whisper_fp16', True) if device == 'cuda' else False
        )
        translator = HuggingFaceTranslator(
            model_name=config.get('translation_model', 'Helsinki-NLP/opus-mt-en-ml'),
            device=device
        )
        generator = SubtitleGenerator(
            config=config,
            audio_extractor=audio_extractor,
            transcriber=transcriber,
            translator=translator
        )
        logger.info("Components initialized successfully.")

    except SyncSubError as e:
        logger.critical(f"Failed to initialize SyncSub components: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during component initialization: {e}", exc_info=True)
        sys.exit(1)


    # --- Process Videos Sequentially ---
    total_files = len(sorted_video_paths)
    files_processed = 0
    files_failed = 0
    batch_start_time = time.time()

    logger.info(f"--- Starting Batch Subtitle Generation for {total_files} files ---")

    # Use tqdm for the progress bar
    # We iterate through paths, but display filename in description
    with tqdm(total=total_files, unit="video", desc="Starting Batch") as pbar:
        for video_path in sorted_video_paths:
            video_filename = os.path.basename(video_path)
            pbar.set_description(f"Processing: {video_filename[:30]}...") # Update description

            # Define expected output paths *before* generation for moving later
            base_name = os.path.splitext(video_filename)[0]
            ext = config.get('output_format', 'srt')
            # Path where generator will place the files initially (within base Subs dir)
            expected_gen_en_path = os.path.join(base_subs_dir, f"{base_name}.en.{ext}")
            expected_gen_ml_path = os.path.join(base_subs_dir, f"{base_name}.ml.{ext}")
            # Final destination paths
            final_en_path = os.path.join(english_subs_dir, f"{base_name}.en.{ext}")
            final_ml_path = os.path.join(malayalam_subs_dir, f"{base_name}.ml.{ext}")

            try:
                logger.info(f"--- Processing video: {video_path} ---")
                video_start_time = time.time()

                # Generate subtitles - they will land in base_subs_dir first
                generator.generate(video_path, base_subs_dir) # Pass base dir

                video_end_time = time.time()
                logger.info(f"Subtitle generation for {video_filename} successful ({video_end_time - video_start_time:.2f}s).")

                # Move generated files to final directories
                moved_en = False
                if os.path.exists(expected_gen_en_path):
                    try:
                        os.rename(expected_gen_en_path, final_en_path)
                        logger.info(f"Moved English subtitle to: {final_en_path}")
                        moved_en = True
                    except OSError as move_err:
                        logger.error(f"Failed to move {expected_gen_en_path} to {final_en_path}: {move_err}")
                else:
                     logger.warning(f"Expected English subtitle file not found after generation: {expected_gen_en_path}")

                moved_ml = False
                if os.path.exists(expected_gen_ml_path):
                     try:
                        os.rename(expected_gen_ml_path, final_ml_path)
                        logger.info(f"Moved Malayalam subtitle to: {final_ml_path}")
                        moved_ml = True
                     except OSError as move_err:
                        logger.error(f"Failed to move {expected_gen_ml_path} to {final_ml_path}: {move_err}")
                else:
                     logger.warning(f"Expected Malayalam subtitle file not found after generation: {expected_gen_ml_path}")

                if moved_en or moved_ml: # Count as success if at least one sub was generated and moved
                     files_processed += 1
                else:
                     logger.error(f"No subtitle files could be generated or moved for {video_filename}.")
                     files_failed += 1


            except SyncSubError as e:
                logger.error(f"SyncSub failed for video '{video_filename}': {e}")
                files_failed += 1
            except KeyboardInterrupt:
                 logger.warning("Batch process interrupted by user (Ctrl+C). Exiting.")
                 sys.exit(1)
            except Exception as e:
                logger.error(f"An unexpected error occurred processing '{video_filename}': {e}", exc_info=True)
                files_failed += 1
            finally:
                 pbar.update(1) # Increment progress bar regardless of success/failure

    batch_end_time = time.time()
    logger.info(f"--- Batch Subtitle Generation Finished ---")
    logger.info(f"Total time: {batch_end_time - batch_start_time:.2f} seconds")
    logger.info(f"Successfully processed: {files_processed}/{total_files} videos")
    logger.info(f"Failed: {files_failed}/{total_files} videos")

    if files_failed > 0:
        sys.exit(1) # Indicate partial failure with exit code
    else:
        sys.exit(0) # Indicate success


if __name__ == "__main__":
    # Basic check for minimal Python version if necessary
    if sys.version_info < (3, 7):
        sys.stderr.write("SyncSub requires Python 3.7 or later.\n")
        sys.exit(1)

    run_batch_processing()