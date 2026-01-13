"""IPA Segment-level tokenizer.

This module provides a tokenizer that first segments IPA strings using panphon
(which handles digraphs, affricates, and modifiers intelligently) and then
treats each segment as a separate token.

Usage:
    python -m src.core.tokenizer.ipasegment_tokenizer
    # Build vocabulary from training transcripts
    tokenizer = IPASegmentTokenizer()
    tokenizer.build_vocab(train_texts, min_freq=1)

    # Encode/decode
    # Input: "t͡ʃaɪ" (ch ai)
    # Segments: ['t͡ʃ', 'a', 'ɪ'] (vs ['t', '͡', 'ʃ', 'a', 'ɪ'] in char tokenizer)
    ids = tokenizer.encode("t͡ʃaɪ")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import panphon
from src.core.tokenizer.base_tokenizer import BaseTokenizer
import tqdm


# Default special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

DEFAULT_SPECIALS: Tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


class IPASegmentTokenizer(BaseTokenizer):
    """Segment-level tokenizer for IPA transcripts using panphon.

    Unlike the CharacterTokenizer which splits "t͡ʃ" into ['t', '͡', 'ʃ'],
    this tokenizer uses panphon to respect IPA segments, keeping valid
    phones and modifiers together.

    Attributes:
        vocab: Mapping from token string to integer ID.
        ids_to_tokens: Reverse mapping from ID to token string.
    """

    def __init__(
        self,
        vocab_path: Union[str, Path, None] = None,
        vocab: Dict[str, int] = None,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
    ) -> None:
        """Initialize the IPASegmentTokenizer.

        Args:
            vocab_path: Path to JSON vocabulary file.
            vocab: Direct dictionary injection (optional).
            pad_token: Padding token string.
            unk_token: Unknown token string.
        """
        super().__init__()
        self._pad_token = pad_token
        self._unk_token = unk_token

        # Initialize panphon feature table for segmentation
        self._ft = panphon.FeatureTable()

        if vocab is not None:
            self._vocab = vocab
            self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        elif vocab_path is not None:
            self._vocab = self._load_vocab(Path(vocab_path))
            self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
            # Validation
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
        return len(self._vocab) if self._vocab else 0

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

    def _segment_text(self, text: str) -> List[str]:
        """Segment text using panphon's compiled regex.

        This iterates through the string finding the longest matching IPA segments
        defined by panphon. Any characters not matched by panphon (like 
        punctuation or spaces) are treated as individual single-character tokens.
        """
        segments = []
        last_end = 0

        # finditer returns match objects in order
        for match in self._ft.seg_regex.finditer(text):
            start, end = match.span()
            
            # 1. Handle the gap (non-IPA characters like spaces/punctuation)
            if start > last_end:
                gap = text[last_end:start]
                segments.extend(list(gap)) # Treat gap as individual chars

            # 2. Add the valid IPA segment (e.g., "t͡ʃ")
            segments.append(match.group())
            last_end = end

        # 3. Handle any remaining characters at the end
        if last_end < len(text):
            segments.extend(list(text[last_end:]))

        return segments

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Text is first segmented into phones/units via panphon, then mapped to IDs.

        Args:
            text: Input text string to encode.

        Returns:
            List of integer token IDs.
        """
        if self._vocab is None:
            raise ValueError(
                "Vocabulary not initialized. Call build_vocab or load from file."
            )

        segments = self._segment_text(text)
        return [self._vocab.get(seg, self.unk_id) for seg in segments]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs back into text.

        Args:
            token_ids: Sequence of integer token IDs.

        Returns:
            Decoded text string (joined segments).
        """
        if self._ids_to_tokens is None:
            raise ValueError("Vocabulary not initialized.")

        return "".join(
            self._ids_to_tokens.get(idx, self._unk_token) for idx in token_ids
        )

    def decode_clean(self, token_ids: Sequence[int], skip_special: bool = True) -> str:
        """Decode token IDs, removing special tokens.

        Args:
            token_ids: Sequence of integer token IDs.
            skip_special: If True, skip special tokens in output.

        Returns:
            Decoded text string.
        """
        special_ids = {self.pad_id, self.unk_id}
        if self._vocab and BOS_TOKEN in self._vocab:
            special_ids.add(self._vocab[BOS_TOKEN])
        if self._vocab and EOS_TOKEN in self._vocab:
            special_ids.add(self._vocab[EOS_TOKEN])

        segments = []
        for idx in token_ids:
            if skip_special and idx in special_ids:
                continue
            segments.append(self._ids_to_tokens.get(idx, ""))
        return "".join(segments)

    def label_to_id(self, label: str) -> int:
        """Convert a label string to its corresponding ID."""
        return self._vocab.get(label, self.unk_id)

    def id_to_label(self, idx: int) -> str:
        """Convert a label ID back to its string representation."""
        return self._ids_to_tokens.get(idx, self._unk_token)

    def build_vocab(
        self,
        texts: Iterable[str],
        min_freq: int = 1,
        specials: Tuple[str, ...] = DEFAULT_SPECIALS,
    ) -> Dict[str, int]:
        """Build vocabulary from a collection of texts using segmentation.

        Args:
            texts: Iterable of text strings.
            min_freq: Minimum frequency.
            specials: Special tokens.
        """
        counter: Counter[str] = Counter()

        # Process all texts
        for text in tqdm.tqdm(texts):
            segments = self._segment_text(text)
            counter.update(segments)

        # Construct vocab
        vocab: Dict[str, int] = {}
        for idx, token in enumerate(specials):
            vocab[token] = idx

        next_id = len(specials)
        for seg, freq in sorted(counter.items()):
            if freq >= min_freq and seg not in vocab:
                vocab[seg] = next_id
                next_id += 1

        self._vocab = vocab
        self._ids_to_tokens = {v: k for k, v in vocab.items()}
        return self._vocab

    @staticmethod
    def save_vocab(vocab: Dict[str, int], path: Union[str, Path]) -> None:
        """Save vocabulary to a JSON file."""
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
    ) -> "IPASegmentTokenizer":
        """Create a tokenizer instance from a vocabulary file."""
        return cls(vocab_path=vocab_path, pad_token=pad_token, unk_token=unk_token)

    @staticmethod
    def _load_vocab(path: Path) -> Dict[str, int]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    # Sanity check
    sample_texts = [
        "həˈloʊ",
        "t͡ʃaɪ",  # Affricate: t͡ʃ should be kept together
        "kʰæt",  # Aspiration: kʰ should be kept together
    ]

    print("Building vocabulary from sample texts...")
    tokenizer = IPASegmentTokenizer()
    tokenizer.build_vocab(sample_texts)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Vocabulary: {tokenizer.vocab}")

    # Verify 't͡ʃ' is a single token, not 't' and '͡' and 'ʃ'
    test_text = "t͡ʃaɪ"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")

    # Check segment length
    # 't͡ʃaɪ' -> [t͡ʃ, a, ɪ] -> Length should be 3, not 5
    print(f"Token count: {len(encoded)} (Expected 3)")

    assert test_text == decoded
    print("Match:    True")
