"""
Gemini Inference wrapper for PhoneBench integration.

This module provides an inference wrapper that adapts GeminiClient to the
project's distributed_inference.py workflow, handling prompt templating
and response post-processing.
"""

import json
import os
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
        cache_path: Optional[str | Path] = None,
        resume: bool = True,
        cache_key_field: str = "metadata_idx",
        error_log_path: Optional[str | Path] = None,
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
            clean_response: If True, normalize the response text (remove spaces,
                            punctuation, etc.). Useful for IPA transcription tasks.
            output_key: Key to extract from JSON response when using structured output.
                If None and structured output is used, returns the raw JSON string.
            device: Ignored parameter for API compatibility with distributed_inference.
            cache_path: Optional path to a JSONL cache file for per-sample checkpointing.
                If set, each successful prediction is appended as one JSON line.
            resume: If True and cache_path exists, reuse cached predictions to skip
                already-processed samples (best-effort).
            cache_key_field: Field name to use as cache key (default: "metadata_idx").
                If missing, falls back to audio_path string.
            error_log_path: Optional JSONL file to append per-sample errors for debugging.
        """
        # Initialize the client
        self.client = GeminiClient(**client_config)

        # Store prompt configuration
        self.system_prompt = prompt_config.get("system_prompt", "")
        self.user_prompt_template = prompt_config.get("user_prompt_template", "{prompt}")

        # Store post-processing options
        self.clean_response = clean_response
        self.output_key = output_key

        # Resume / caching options (does not depend on distributed_inference internals)
        self.cache_path = Path(cache_path) if cache_path else None
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.resume = resume
        self.cache_key_field = cache_key_field
        self._cache: dict[str, Any] = {}
        if self.cache_path and self.resume:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load existing JSONL cache into memory (best-effort)."""
        assert self.cache_path is not None
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    k = rec.get("key", None)
                    pred = rec.get("pred", None)
                    if k is None or pred is None:
                        continue
                    self._cache[str(k)] = pred
        except Exception:
            return

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        """Append one JSON record as a line (best-effort)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass

    @staticmethod
    def _is_error_pred(pred: Any) -> bool:
        """Heuristic: treat a prediction as error if pred[0] has an 'error' field."""
        if not isinstance(pred, list) or not pred:
            return True
        first = pred[0]
        return isinstance(first, dict) and ("error" in first)

    def __call__(self, audio_path: str | Path, **kwargs: Any) -> Any:
        """
        Run inference on an audio file.

        This method implements the interface expected by distributed_inference.py.
        It formats the prompt using the template and kwargs, then calls the
        Gemini client for generation.

        Args:
            audio_path: Path to the audio file to process.
            **kwargs: Additional fields from the dataset item. These can be used
                      in the prompt template (e.g., language, speaker_id).

        Returns:
            Dict containing raw and processed transcripts.
        """
        # 1. Format user prompt using template and kwargs
        try:
            user_prompt = self.user_prompt_template.format(**kwargs)
        except Exception:
            # Best-effort: fall back to the raw template if formatting fails
            user_prompt = self.user_prompt_template

        cache_key: Optional[str] = None
        if self.cache_path:
            if self.cache_key_field in kwargs:
                cache_key = str(kwargs.get(self.cache_key_field))
            else:
                cache_key = str(audio_path)
            if self.resume and cache_key in self._cache:
                return self._cache[cache_key]

        try:
            raw_model_response = self.client.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt if self.system_prompt else None,
                files=audio_path,
            )
        except Exception as e:
            err_code = getattr(e, "code", None)
            err_status = getattr(e, "status", None)
            err_msg = str(e)
            pred = [
                {
                    "processed_transcript": "",
                    "predicted_transcript": "",
                    "raw_model_response": "",
                    "error": {
                        "type": type(e).__name__,
                        "code": err_code,
                        "status": err_status,
                        "message": err_msg,
                    },
                }
            ]
            if self.error_log_path:
                self._append_jsonl(
                    self.error_log_path,
                    {
                        "key": cache_key or str(audio_path),
                        "audio_path": str(audio_path),
                        "error": pred[0]["error"],
                    },
                )
            return pred

        # 3. Parse JSON response if output_key is specified
        raw_transcript = raw_model_response
        if self.output_key:
            raw_transcript = self._parse_json_response(raw_model_response, self.output_key)

        # 4. Optionally clean the response
        processed_transcript = (
            self._clean_response(raw_transcript) if self.clean_response else raw_transcript
        )

        # Match the common naming used by other inference wrappers
        # (e.g., wav2vec2phoneme_inference.py) for easier downstream handling.
        pred = [
            {
                "processed_transcript": processed_transcript,
                "predicted_transcript": raw_transcript,
                "raw_model_response": raw_model_response,
            }
        ]
        if self.cache_path and cache_key and not self._is_error_pred(pred):
            self._cache[cache_key] = pred
            self._append_jsonl(self.cache_path, {"key": cache_key, "pred": pred})
        return pred

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

        Args:
            text: Raw response text from the model.

        Returns:
            Cleaned and normalized text.
        """
        # Remove whitespace (including newlines/tabs)
        text = "".join(text.split())

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Unicode normalization (NFD form)
        text = unicodedata.normalize("NFD", text)

        # Replace common IPA variants
        text = text.replace("g", "ɡ")  # ASCII 'g' to IPA 'ɡ'

        return text.strip()
