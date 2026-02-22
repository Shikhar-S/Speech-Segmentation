"""Character-level tokenizer for IPA transcripts.

This module provides a simple character-based tokenizer that treats each
character as a separate token. The vocabulary is built from training data and
can be stored as a JSON file.

Usage:
    # Build vocabulary from training transcripts
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(train_texts, min_freq=1)

    # Encode/decode
    ids = tokenizer.encode("həˈloʊ")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from src.core.tokenizer.base_tokenizer import BaseTokenizer


# Default special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

DEFAULT_SPECIALS: Tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer for IPA transcripts.

    Each character in the input text is treated as a separate token.
    Special tokens (<pad>, <unk>, etc.) are reserved at the beginning
    of the vocabulary.

    Attributes:
        vocab: Mapping from token string to integer ID.
        ids_to_tokens: Reverse mapping from ID to token string.
    """

    def __init__(
        self,
        vocab_path: Union[str, Path] = None,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
    ) -> None:
        """Initialize the CharacterTokenizer.
        Args:
            pad_token: Padding token string.
            unk_token: Unknown token string.
        """
        super().__init__()
        self._pad_token = pad_token
        self._unk_token = unk_token
        if vocab_path is not None:
            self._vocab = self._load_vocab(Path(vocab_path))
            self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
            assert pad_token in self._vocab, f"pad_token '{pad_token}' not in vocab"
            assert unk_token in self._vocab, f"unk_token '{unk_token}' not in vocab"
        else:
            self._ids_to_tokens = None
            self._vocab = None

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the vocabulary mapping (token -> id)."""
        return self._vocab

    @property
    def ids_to_tokens(self) -> Dict[int, str]:
        """Return the reverse mapping (id -> token)."""
        return self._ids_to_tokens

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        """Return the padding token ID."""
        return self._vocab[self._pad_token]

    @property
    def unk_id(self) -> int:
        """Return the unknown token ID."""
        return self._vocab[self._unk_token]

    @property
    def pad_token(self) -> str:
        """Return the padding token string."""
        return self._pad_token

    @property
    def unk_token(self) -> str:
        """Return the unknown token string."""
        return self._unk_token

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Each character is mapped to its ID. Unknown characters are mapped
        to the <unk> token ID.

        Args:
            text: Input text string to encode.

        Returns:
            List of integer token IDs.
        """
        return [self._vocab.get(char, self.unk_id) for char in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs back into text.

        Special tokens (<pad>, <unk>, etc.) are included in the output.
        Use decode_clean() to remove them.

        Args:
            token_ids: Sequence of integer token IDs.

        Returns:
            Decoded text string.
        """
        return "".join(
            self._ids_to_tokens.get(idx, self._unk_token) for idx in token_ids
        )

    def decode_clean(self, token_ids: Sequence[int], skip_special: bool = True) -> str:
        """Decode token IDs, optionally removing special tokens.

        Args:
            token_ids: Sequence of integer token IDs.
            skip_special: If True, skip special tokens in output.

        Returns:
            Decoded text string.
        """
        special_ids = {self.pad_id, self.unk_id}
        # Add BOS/EOS if present
        if BOS_TOKEN in self._vocab:
            special_ids.add(self._vocab[BOS_TOKEN])
        if EOS_TOKEN in self._vocab:
            special_ids.add(self._vocab[EOS_TOKEN])

        chars = []
        for idx in token_ids:
            if skip_special and idx in special_ids:
                continue
            chars.append(self._ids_to_tokens.get(idx, ""))
        return "".join(chars)

    def label_to_id(self, label: str) -> int:
        """Convert a label string to its corresponding ID.

        For CharacterTokenizer, this is equivalent to looking up a single
        character or token in the vocabulary.

        Args:
            label: Label string (single character or token).

        Returns:
            Integer ID for the label.
        """
        return self._vocab.get(label, self.unk_id)

    def id_to_label(self, idx: int) -> str:
        """Convert a label ID back to its string representation.

        Args:
            idx: Integer label ID.

        Returns:
            Label string.
        """
        return self._ids_to_tokens.get(idx, self._unk_token)

    def build_vocab(
        self,
        texts: Iterable[str],
        min_freq: int = 1,
        specials: Tuple[str, ...] = DEFAULT_SPECIALS,
    ) -> Dict[str, int]:
        """Build vocabulary from a collection of texts.
        Args:
            texts: Iterable of text strings (e.g., IPA transcripts).
            min_freq: Minimum frequency for a character to be included.
            specials: Tuple of special tokens to prepend to vocabulary.
        """
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)
        vocab: Dict[str, int] = {}
        for idx, token in enumerate(specials):
            vocab[token] = idx
        next_id = len(specials)
        for char, freq in sorted(counter.items()):
            if freq >= min_freq and char not in vocab:
                vocab[char] = next_id
                next_id += 1
        self._vocab = vocab
        self._ids_to_tokens = {v: k for k, v in vocab.items()}

    @staticmethod
    def save_vocab(vocab: Dict[str, int], path: Union[str, Path]) -> None:
        """Save vocabulary to a JSON file.

        Args:
            vocab: Vocabulary dictionary (token -> id).
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(
        cls,
        vocab_path: Union[str, Path],
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
    ) -> "CharacterTokenizer":
        """Create a CharacterTokenizer from a vocabulary file.

        This is a convenience factory method equivalent to:
            CharacterTokenizer(vocab_path=path)

        Args:
            vocab_path: Path to JSON vocabulary file.
            pad_token: Padding token string.
            unk_token: Unknown token string.

        Returns:
            Initialized CharacterTokenizer instance.
        """
        return cls(vocab_path=vocab_path, pad_token=pad_token, unk_token=unk_token)

    @staticmethod
    def _load_vocab(path: Path) -> Dict[str, int]:
        """Load vocabulary from JSON file."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    # Quick sanity check
    sample_texts = [
        "həˈloʊ",
        "wɝld",
        "ˈtɛstɪŋ",
    ]

    print("Building vocabulary from sample texts...")
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(sample_texts)
    print(f"Vocabulary size: {len(tokenizer._vocab)}")
    print(f"Vocabulary: {tokenizer._vocab}")

    print("\nCreating tokenizer from vocab dict...")
    tokenizer = CharacterTokenizer(vocab=tokenizer._vocab)

    test_text = "həˈloʊ"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Match:    {test_text == decoded}")

    # Test unknown character handling
    unknown_text = "xyz"
    encoded_unk = tokenizer.encode(unknown_text)
    print(f"\nUnknown text: {unknown_text}")
    print(f"Encoded (should have unk_id={tokenizer.unk_id}): {encoded_unk}")
