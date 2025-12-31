"""
Gemini Direct Prompt Inference wrapper for PhoneBench.

This module provides an inference wrapper for direct task prediction (classification,
regression, geolocation) as opposed to transcription. It follows the same
distributed_inference.py contract as GeminiInference but returns structured
prediction values instead of transcription-focused outputs.

Key differences from GeminiInference:
- Returns single prediction value (dict) instead of list with transcript fields
- Error details go only to errors.jsonl, main pred is null on failure
- Designed for classification/regression/geolocation tasks

Reference: src/model/gemini/transcribe.py
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from src.model.gemini.client import GeminiClient


class DirectPromptInference:
    """
    Inference wrapper for Gemini direct-prompt tasks (classification, regression, geolocation).

    This class implements the distributed_inference.py contract:
    - Hydra instantiate compatible
    - `device` argument accepted (ignored for API-based model)
    - `__call__(audio_path, **kwargs)` interface

    Returns a single prediction dict per sample:
    - Classification: {"class_id": int}
    - Regression: {"score": number}
    - Geolocation: {"lat": number, "lon": number}  (decimal degrees)

    On error, returns None for pred and logs details to errors.jsonl only.
    """

    def __init__(
        self,
        client_config: dict[str, Any],
        prompt_config: dict[str, Any],
        device: Optional[str] = None,  # Ignored for API-based model
        cache_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
        cache_key_field: str = "metadata_idx",
        error_log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the Direct Prompt inference wrapper.

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
            device: Ignored parameter for API compatibility with distributed_inference.
            cache_path: Optional path to a JSONL cache file for per-sample checkpointing.
                If set, each successful prediction is appended as {"key": ..., "pred": ...}.
            resume: If True (default) and cache_path exists, reuse cached predictions
                to skip already-processed samples.
            cache_key_field: Field name to use as cache key (default: "metadata_idx").
                If missing from kwargs, falls back to audio_path string.
            error_log_path: Optional JSONL file to append per-sample errors for debugging.
        """
        # Initialize the client
        self.client = GeminiClient(**client_config)

        # Store prompt configuration
        self.system_prompt = prompt_config.get("system_prompt", "")
        self.user_prompt_template = prompt_config.get("user_prompt_template", "{prompt}")
        self.default_user_prompt = prompt_config.get("default_user_prompt", "")

        # Resume / caching options
        self.cache_path = Path(cache_path) if cache_path else None
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.resume = resume
        self.cache_key_field = cache_key_field
        self._cache: dict[str, Any] = {}

        # Load existing cache if resuming
        if self.cache_path and self.resume:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load existing JSONL cache into memory (best-effort)."""
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    k = rec.get("key")
                    pred = rec.get("pred")
                    if k is not None and pred is not None:
                        self._cache[str(k)] = pred
        except Exception:
            # Best-effort: if cache loading fails, start fresh
            pass

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        """Append one JSON record as a line to the specified file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass

    def _get_cache_key(self, audio_path: Union[str, Path], kwargs: dict[str, Any]) -> str:
        """Determine cache key from kwargs or audio_path."""
        if self.cache_key_field in kwargs:
            return str(kwargs[self.cache_key_field])
        return str(audio_path)

    def __call__(self, audio_path: Union[str, Path], **kwargs: Any) -> Any:
        """
        Run inference on an audio file for direct task prediction.

        This method implements the interface expected by distributed_inference.py.
        It formats the prompt using the template and kwargs, calls the Gemini client,
        and returns the structured prediction.

        Args:
            audio_path: Path to the audio file to process.
            **kwargs: Additional fields from the dataset item. These can be used
                      in the prompt template (e.g., language, speaker_id).

        Returns:
            Prediction dict on success:
                - Classification: {"class_id": int}
                - Regression: {"score": number}
                - Geolocation: {"lat": number, "lon": number}  (decimal degrees)
            None on error (error details logged to errors.jsonl).
        """
        # TODO(Yoonjae): Need to maintain consistency for dealing with cache_key (with transcribe.py).
        cache_key = self._get_cache_key(audio_path, kwargs)

        # Check cache first (if resuming)
        if self.cache_path and self.resume and cache_key in self._cache:
            return self._cache[cache_key]

        # Format user prompt using template and kwargs
        try:
            user_prompt = self.user_prompt_template.format(**kwargs)
        except KeyError:
            # Fall back to default prompt if template keys are missing
            user_prompt = self.default_user_prompt or self.user_prompt_template

        # Call Gemini API
        try:
            raw_response = self.client.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt if self.system_prompt else None,
                files=audio_path,
            )
        except Exception as e:
            # Log error to errors.jsonl, return None for pred
            self._log_error(cache_key, audio_path, e)
            return None

        # Parse the JSON response
        pred = self._parse_response(raw_response, cache_key, audio_path)

        # Cache successful prediction
        if pred is not None and self.cache_path:
            self._cache[cache_key] = pred
            self._append_jsonl(self.cache_path, {"key": cache_key, "pred": pred})

        return pred

    def _parse_response(
        self, raw_response: str, cache_key: str, audio_path: Union[str, Path]
    ) -> Optional[dict[str, Any]]:
        """
        Parse JSON response from Gemini into structured prediction.

        Args:
            raw_response: Raw JSON string from the model.
            cache_key: Cache key for error logging.
            audio_path: Audio path for error logging.

        Returns:
            Parsed prediction dict, or None if parsing fails.
        """
        try:
            parsed = json.loads(raw_response)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed).__name__}")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            # Log parse error
            self._log_error(
                cache_key,
                audio_path,
                e,
                extra={"raw_response": raw_response[:500] if raw_response else ""},
            )
            return None

    def _log_error(
        self,
        cache_key: str,
        audio_path: Union[str, Path],
        error: Exception,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log error details to errors.jsonl."""
        if not self.error_log_path:
            return

        error_record = {
            "key": cache_key,
            "audio_path": str(audio_path),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "code": getattr(error, "code", None),
                "status": getattr(error, "status", None),
            },
        }
        if extra:
            error_record.update(extra)

        self._append_jsonl(self.error_log_path, error_record)
