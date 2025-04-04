"""Handles loading configuration from YAML files."""

import yaml
import os
import logging
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Loads configuration settings from a YAML file."""

    def load_config(self, config_path: str) -> dict:
        """
        Loads configuration from the specified YAML file path.

        Args:
            config_path: The path to the YAML configuration file.

        Returns:
            A dictionary containing the loaded configuration settings.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ConfigurationError: If the file cannot be parsed as YAML or
                              if there are other reading errors.
        """
        logger.info(f"Attempting to load configuration from: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found at path: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.isfile(config_path):
             logger.error(f"Configuration path is not a file: {config_path}")
             raise ConfigurationError(f"Configuration path is not a file: {config_path}")


        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    # Handle cases where YAML loads something other than a dictionary (e.g., just a string)
                    logger.error(f"Configuration file {config_path} did not load as a dictionary (root object).")
                    raise ConfigurationError(f"Invalid YAML structure in {config_path}. Root must be a mapping (dictionary).")
                logger.info(f"Configuration loaded successfully from {config_path}")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {config_path}: {e}", exc_info=True)
            raise ConfigurationError(f"Invalid YAML format in {config_path}: {e}") from e
        except IOError as e:
            logger.error(f"Error reading configuration file {config_path}: {e}", exc_info=True)
            raise ConfigurationError(f"Could not read configuration file {config_path}: {e}") from e
        except Exception as e:
             logger.error(f"An unexpected error occurred while loading configuration from {config_path}: {e}", exc_info=True)
             raise ConfigurationError(f"Unexpected error loading config: {e}") from e