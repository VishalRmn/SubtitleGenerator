"""Command-Line Interface handler for SyncSub."""

import argparse
import logging
import os
import sys
import time

from .config_loader import ConfigLoader
from .log_setup import setup_logging
from .audio_extractor import AudioExtractor
from .transcriber import WhisperTranscriber
from .translator import HuggingFaceTranslator
from .subtitle_generator import SubtitleGenerator
from .exceptions import SyncSubError, ConfigurationError

logger = logging.getLogger(__name__) # Get logger for this module

class CLIHandler:
    """Parses arguments and orchestrates the SyncSub process."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Creates the argument parser for the CLI."""
        parser = argparse.ArgumentParser(
            description="SyncSub: Generate synchronized English and Malayalam subtitles for local videos.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
        )
        parser.add_argument(
            "-v", "--video",
            required=True,
            help="Path to the input video file."
        )
        parser.add_argument(
            "-o", "--output-dir",
            required=True,
            help="Directory to save the generated subtitle files (.srt)."
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
        # Add more overrides if needed, e.g., --whisper-model, --device
        parser.add_argument(
            "--device",
            default=None, # Default taken from config
            choices=["cuda", "cpu"],
            help="Override the processing device (cuda or cpu) specified in config."
        )

        return parser

    def run(self) -> None:
        """Parses arguments, sets up logging, loads config, and runs the generator."""
        args = self.parser.parse_args()

        # --- Setup Logging ---
        # Get log level from argparse first
        log_level_name = args.log_level.upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        # Temporarily setup basic logging to catch config loading errors
        setup_logging(log_level=log_level, log_dir='logs', log_file='syncsub_init.log') # Use a temp init log

        # --- Load Configuration ---
        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(args.config)
        except ConfigurationError as e:
            logger.critical(f"Failed to load configuration from {args.config}: {e}", exc_info=True)
            sys.exit(1)
        except FileNotFoundError:
             logger.critical(f"Configuration file not found: {args.config}", exc_info=True)
             sys.exit(1)

        # --- Re-configure Logging with settings from Config ---
        log_dir = config.get('log_dir', 'logs')
        log_file = config.get('log_file', 'syncsub.log')
        setup_logging(log_level=log_level, log_dir=log_dir, log_file=log_file) # Re-init with final paths/names
        logger.info("Logging re-configured with settings from config file.")


        # --- Apply CLI Overrides ---
        if args.temp_dir:
            logger.info(f"Overriding temp_dir from config with CLI argument: {args.temp_dir}")
            config['temp_dir'] = args.temp_dir
        if args.device:
             logger.info(f"Overriding device from config with CLI argument: {args.device}")
             config['device'] = args.device
        # Add more overrides here...

        # --- Validate Input/Output Paths ---
        if not os.path.isfile(args.video):
            logger.critical(f"Input video file not found or is not a file: {args.video}")
            sys.exit(1)
        # Output directory existence checked within SubtitleGenerator/utils


        # --- Instantiate Components ---
        try:
            logger.info("Initializing SyncSub components...")
            device = config.get('device', 'cuda') # Get potentially overridden device

            audio_extractor = AudioExtractor(
                ffmpeg_path=config.get('ffmpeg_path') # None if not specified
            )
            transcriber = WhisperTranscriber(
                model_name=config.get('whisper_model', 'medium.en'),
                device=device,
                fp16=config.get('whisper_fp16', True) if device == 'cuda' else False
            )
            translator = HuggingFaceTranslator(
                model_name=config.get('translation_model', 'Helsinki-NLP/opus-mt-en-ml'),
                device=device
            )
            # Formatter chosen inside generator based on config['output_format']

            generator = SubtitleGenerator(
                config=config,
                audio_extractor=audio_extractor,
                transcriber=transcriber,
                translator=translator
                # formatter instantiated inside generator
            )
            logger.info("Components initialized successfully.")

            # --- Run Generation ---
            generator.generate(args.video, args.output_dir)
            logger.info("SyncSub finished successfully.")
            sys.exit(0)

        except SyncSubError as e:
             # Catch errors originating from our application logic
             logger.error(f"A SyncSub error occurred: {e}")
             # Potentially log stack trace if it wasn't logged deeper down
             # logger.exception("Stack trace:", exc_info=True) # Uncomment for debugging
             sys.exit(1)
        except KeyboardInterrupt:
             logger.warning("Process interrupted by user (Ctrl+C). Exiting.")
             sys.exit(1)
        except Exception as e:
             # Catch any other unexpected errors
             logger.critical(f"An unexpected critical error occurred at the top level: {e}", exc_info=True)
             sys.exit(2) # Use a different exit code for unexpected crashes