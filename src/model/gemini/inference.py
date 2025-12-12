"""
Gemini Inference wrapper for PhoneBench integration.

This module provides an inference wrapper that adapts GeminiClient to the
project's distributed_inference.py workflow, handling prompt templating
and response post-processing.
"""

import json
import string
import unicodedata
from pathlib import Path
from typing import Any, Optional

from src.model.gemini.client import GeminiClient


class GeminiInference:
    """
    Inference wrapper for Gemini model integration with PhoneBench.

    This class adapts the GeminiClient to work with the distributed_inference.py
    workflow by implementing the expected __call__ interface and handling
    prompt templating and response cleaning.
    """

    def __init__(
        self,
        client_config: dict[str, Any],
        prompt_config: dict[str, Any],
        clean_response: bool = False,
        output_key: Optional[str] = None,
        device: Optional[str] = None,  # Ignored for API-based model
    ) -> None:
        """
        Initialize the Gemini inference wrapper.

        Args:
            client_config: Configuration for GeminiClient. Expected keys:
                - model_name (str): Gemini model identifier
                - api_key (str, optional): API key for authentication
                - temperature (float, optional): Sampling temperature
                - response_schema (dict, optional): Schema for structured JSON output
                - retry_config (dict, optional): Retry configuration
            prompt_config: Configuration for prompt handling. Expected keys:
                - system_prompt (str): System instruction for the model
                - user_prompt_template (str): Template string with {placeholders}
                - default_user_prompt (str, optional): Fallback prompt if template fails
            clean_response: If True, normalize the response text (remove spaces,
                            punctuation, etc.). Useful for IPA transcription tasks.
            output_key: Key to extract from JSON response when using structured output.
                If None and structured output is used, returns the raw JSON string.
            device: Ignored parameter for API compatibility with distributed_inference.
        """
        # Initialize the client
        self.client = GeminiClient(**client_config)

        # Store prompt configuration
        self.system_prompt = prompt_config.get("system_prompt", "")
        self.user_prompt_template = prompt_config.get("user_prompt_template", "{prompt}")
        self.default_user_prompt = prompt_config.get("default_user_prompt", "")

        # Store post-processing options
        self.clean_response = clean_response
        self.output_key = output_key

    def __call__(self, wav_path: str | Path, **kwargs: Any) -> str:
        """
        Run inference on an audio file.

        This method implements the interface expected by distributed_inference.py.
        It formats the prompt using the template and kwargs, then calls the
        Gemini client for generation.

        Args:
            wav_path: Path to the audio file to process.
            **kwargs: Additional fields from the dataset item. These can be used
                      in the prompt template (e.g., language, speaker_id).

        Returns:
            Model's response text, optionally cleaned.
        """
        # 1. Format user prompt using template and kwargs
        try:
            user_prompt = self.user_prompt_template.format(**kwargs)
        except KeyError:
            # Fall back to default prompt if template keys are missing
            user_prompt = self.default_user_prompt or self.user_prompt_template

        # 2. Call client with formatted prompt
        raw_response = self.client.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt if self.system_prompt else None,
            files=wav_path,
        )

        # 3. Parse JSON response if output_key is specified
        if self.output_key:
            raw_response = self._parse_json_response(raw_response, self.output_key)

        # 4. Optionally clean the response
        if self.clean_response:
            return self._clean_response(raw_response)

        return raw_response

    @staticmethod
    def _parse_json_response(response: str, key: str) -> str:
        """
        Parse JSON response and extract value for the specified key.

        Args:
            response: JSON string response from the model.
            key: Key to extract from the JSON object.

        Returns:
            Extracted value as string, or original response if parsing fails.
        """
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and key in parsed:
                return str(parsed[key])
            return response
        except json.JSONDecodeError:
            return response

    @staticmethod
    def _clean_response(text: str) -> str:
        """
        Clean and normalize response text.

        This method removes spaces, punctuation, and normalizes unicode characters.
        Useful for IPA transcription comparison.

        Reference: src/api/tasks/pr_tusom_eval.py:PhoneRecognitionEvaluator.clean_text

        Args:
            text: Raw response text from the model.

        Returns:
            Cleaned and normalized text.
        """
        # Remove whitespace
        text = text.replace(" ", "")

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Unicode normalization (NFD form)
        text = unicodedata.normalize("NFD", text)

        # Replace common IPA variants
        text = text.replace("g", "ɡ")  # ASCII 'g' to IPA 'ɡ'

        return text.strip()
