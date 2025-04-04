"""Handles text translation using Hugging Face models."""

import logging
import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

from .exceptions import TranslationError

logger = logging.getLogger(__name__)

class Translator(ABC):
    """Abstract base class for translation services."""

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text from source to target language.

        Args:
            text: The text to translate.
            source_lang: Source language code (e.g., 'en').
            target_lang: Target language code (e.g., 'ml').

        Returns:
            The translated text.

        Raises:
            TranslationError: If translation fails.
        """
        pass

class HuggingFaceTranslator(Translator):
    """Implements translation using Hugging Face Transformers models."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ml", device: str = "cuda"):
        """
        Initializes the HuggingFaceTranslator.

        Args:
            model_name: The name of the Hugging Face translation model.
            device: The device to run the model on ("cuda" or "cpu").

        Raises:
            ValueError: If the specified device is invalid or unavailable.
            TranslationError: If the model or tokenizer fails to load.
        """
        self.model_name = model_name
        self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(f"CUDA device requested but not available for translation. Falling back to CPU.")
            self.device = "cpu"
        elif self.device not in ["cuda", "cpu"]:
             raise ValueError(f"Invalid device specified: {self.device}. Choose 'cuda' or 'cpu'.")

        logger.info(f"Initializing HuggingFaceTranslator with model '{self.model_name}' on device '{self.device}'")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            logger.info(f"Hugging Face translation model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load translation model or tokenizer '{self.model_name}': {e}", exc_info=True)
            raise TranslationError(f"Failed to load translation model/tokenizer '{self.model_name}': {e}") from e

    def translate(self, text: str, source_lang: str = 'en', target_lang: str = 'ml') -> str:
        """
        Translates a single string of text.

        Args:
            text: The text to translate.
            source_lang: Source language code (used for logging/potential future model logic).
            target_lang: Target language code (used for logging/potential future model logic).

        Returns:
            The translated text.

        Raises:
            TranslationError: If the translation process fails.
        """
        if not text:
            return "" # Handle empty input gracefully

        logger.debug(f"Translating (en->ml): '{text[:50]}...'") # Log snippet
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) # Adjust max_length if needed
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move tensors to the correct device

            # Generate translation using the model
            with torch.no_grad(): # Disable gradient calculation for inference
                translated_tokens = self.model.generate(**inputs)

            # Decode the generated tokens back to text
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

            logger.debug(f"Translation result: '{translated_text[:50]}...'")
            return translated_text

        except Exception as e:
            logger.error(f"Error during translation of text '{text[:50]}...': {e}", exc_info=True)
            raise TranslationError(f"Hugging Face translation failed: {e}") from e

    # Potential Optimization: Batch Translation
    # def translate_batch(self, texts: List[str], source_lang: str = 'en', target_lang: str = 'ml') -> List[str]:
    #     # Implement batching for potentially faster translation of multiple segments
    #     # Tokenize all texts together with padding
    #     # Generate translations in one go
    #     # Decode results
    #     # Handle potential errors carefully
    #     pass