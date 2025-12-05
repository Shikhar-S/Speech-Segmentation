"""Base tokenizer interface for downstream tasks.

This module defines the common interface that tokenizers must implement
for text/label encoding and decoding in downstream tasks.

Note: src/core/ipa_utils.py is deprecated and should not be used.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers.

    Tokenizers are responsible for:
    - Converting text to token IDs (encode)
    - Converting token IDs back to text (decode)
    - Mapping labels to IDs and vice versa (for classification tasks)

    The actual implementation will be provided by concrete subclasses.
    DataModule and recipe should reference this interface via type hints
    to establish pipeline connections.

    TODO: Implement concrete tokenizer classes after data format is finalized.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Return the padding token ID."""
        raise NotImplementedError

    @property
    @abstractmethod
    def unk_id(self) -> int:
        """Return the unknown token ID."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Args:
            text: Input text string to encode.

        Returns:
            List of integer token IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs back into text.

        Args:
            token_ids: Sequence of integer token IDs.

        Returns:
            Decoded text string.
        """
        raise NotImplementedError

    @abstractmethod
    def label_to_id(self, label: str) -> int:
        """Convert a label string to its corresponding ID.

        Args:
            label: Label string (e.g., class name).

        Returns:
            Integer ID for the label.
        """
        raise NotImplementedError

    @abstractmethod
    def id_to_label(self, idx: int) -> str:
        """Convert a label ID back to its string representation.

        Args:
            idx: Integer label ID.

        Returns:
            Label string.
        """
        raise NotImplementedError

