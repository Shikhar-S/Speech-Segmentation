"""
Gemini API client for multimodal processing.

This module provides a clean wrapper around the Gemini API for various tasks.
It handles file upload, inference, and cleanup with retry logic.
"""

import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from google import genai
from google.genai import types


@dataclass
class UploadedFile:
    """Wrapper for an uploaded file with its metadata."""

    file: Any  # The uploaded file object from Gemini
    original_path: Optional[Path] = None


# Type alias for thinking level options
ThinkingLevel = Literal["low", "high"]


class GeminiClient:
    """
    Gemini API client for multimodal content generation.

    This class handles low-level API communication including file upload,
    content generation, and resource cleanup. It is designed to be task-agnostic
    and can process any file type supported by Gemini (audio, image, video, PDF, etc.).
    """

    def __init__(
        self,
        model_name: str = "gemini-3-pro-preview",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        seed: int = 0,
        thinking_level: ThinkingLevel = "low",
        retry_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the Gemini client.

        Args:
            model_name: Identifier of the Gemini model to use.
            api_key: API key for authentication. If None, falls back to
                     GEMINI_API_KEY or GOOGLE_API_KEY environment variables.
            temperature: Sampling temperature for generation (default: 1.0).
            top_p: Top-p (nucleus) sampling parameter (default: 1.0).
            top_k: Top-k sampling parameter (default: 1).
            seed: Random seed for reproducibility (default: 0).
            thinking_level: Thinking level for reasoning models (default: "low").
                Options: "none", "low", "medium", "high".
            retry_config: Configuration for retry logic. Keys:
                - max_retries (int): Maximum retry attempts (default: 5)
                - initial_delay (float): Initial delay in seconds (default: 1.0)
                - backoff_factor (float): Multiplier for delay (default: 2.0)
        """
        # Resolve API key
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key is not configured. Please provide api_key or set "
                "the GEMINI_API_KEY / GOOGLE_API_KEY environment variables."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.thinking_level = thinking_level
        self.client = genai.Client(api_key=key)

        # Retry configuration
        default_retry = {"max_retries": 5, "initial_delay": 1.0, "backoff_factor": 2.0}
        self.retry_config = {**default_retry, **(retry_config or {})}

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        files: Optional[list[str | Path] | str | Path] = None,
        anonymize: bool = True,
    ) -> str:
        """
        Generate content from prompt and optional files.

        Args:
            prompt: User prompt for generation.
            system_prompt: Optional system instruction for the model.
            files: Optional file(s) to include. Accepts None (text-only),
                   single path (str/Path), or list of paths.
            anonymize: If True, anonymize filenames before upload.

        Returns:
            Model's response text.
        """
        uploaded_files: list[UploadedFile] = []

        try:
            # 1. Upload files if provided
            if files is not None:
                # Normalize to list
                files_to_process = (
                    [files] if isinstance(files, (str, Path)) else list(files)
                )
                for file_path in files_to_process:
                    uploaded = self._upload_file(Path(file_path), anonymize=anonymize)
                    uploaded_files.append(uploaded)

            # 2. Generate content
            response_text = self._generate_content(
                uploaded_files=uploaded_files,
                prompt=prompt,
                system_prompt=system_prompt,
            )

            return response_text

        finally:
            # 3. Cleanup uploaded files
            for uploaded in uploaded_files:
                self._delete_file(uploaded)

    def _upload_file(self, path: Path, anonymize: bool = True) -> UploadedFile:
        """
        Upload a file to Gemini API.

        Args:
            path: Path to the file to upload.
            anonymize: If True, copy file to temp location with random name.

        Returns:
            UploadedFile object containing the uploaded file reference.

        Raises:
            RuntimeError: If upload fails after max retries.
        """
        temp_file: Optional[Path] = None
        upload_path = path

        # Anonymize filename to prevent information leakage
        if anonymize:
            suffix = path.suffix or ".bin"
            random_name = f"{uuid.uuid4().hex}{suffix}"
            temp_dir = tempfile.gettempdir()
            temp_file = Path(temp_dir) / random_name
            shutil.copy2(path, temp_file)
            upload_path = temp_file

        uploaded = None
        last_error: Optional[Exception] = None
        delay = self.retry_config["initial_delay"]

        for attempt in range(self.retry_config["max_retries"]):
            try:
                uploaded = self.client.files.upload(file=upload_path)
                break
            except Exception as e:
                last_error = e
                if attempt < self.retry_config["max_retries"] - 1:
                    time.sleep(delay)
                    delay *= self.retry_config["backoff_factor"]

        # Clean up temp file after upload attempt
        if temp_file and temp_file.exists():
            temp_file.unlink()

        if uploaded is None:
            raise RuntimeError(
                f"Failed to upload file after {self.retry_config['max_retries']} attempts"
            ) from last_error

        return UploadedFile(file=uploaded, original_path=path)

    def _delete_file(self, uploaded_file: UploadedFile) -> None:
        """
        Delete an uploaded file from Gemini.

        Args:
            uploaded_file: The UploadedFile object to delete.
        """
        try:
            self.client.files.delete(name=uploaded_file.file.name)
        except Exception:
            # Silently ignore deletion errors
            pass

    def _generate_content(
        self,
        uploaded_files: list[UploadedFile],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate content using Gemini API.

        Args:
            uploaded_files: List of uploaded files to include in the request.
            prompt: User prompt text.
            system_prompt: Optional system instruction.

        Returns:
            Model's response text.
        """
        # Build content parts
        parts: list[types.Part] = []

        # Add file parts
        for uploaded in uploaded_files:
            file_part = types.Part(
                file_data=types.FileData(
                    file_uri=uploaded.file.uri,
                    mime_type=uploaded.file.mime_type,
                )
            )
            parts.append(file_part)

        # Add text prompt
        parts.append(types.Part(text=prompt))

        # Build contents
        contents = [types.Content(role="user", parts=parts)]

        # Build generation config
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "candidate_count": 1,
            "response_modalities": ["TEXT"],
            "thinking_config": types.ThinkingConfig(thinking_level=self.thinking_level),
        }

        # Add system instruction if provided
        if system_prompt:
            config_kwargs["system_instruction"] = types.Content(
                role="system", parts=[types.Part(text=system_prompt)]
            )

        config = types.GenerateContentConfig(**config_kwargs)

        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        return (response.text or "").strip()
