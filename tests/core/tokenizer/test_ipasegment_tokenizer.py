"""Tests for IPASegmentTokenizer."""

import pytest

from src.core.tokenizer.ipasegment_tokenizer import (
    IPASegmentTokenizer,
    PAD_TOKEN,
    UNK_TOKEN,
)


def test_dict_vocab_validates_pad_unk():
    """Passing a vocab dict missing pad_token must raise AssertionError
    (bug M6 fix)."""
    bad_vocab = {"a": 0, "b": 1}  # no <pad> or <unk>
    with pytest.raises(AssertionError, match="pad_token"):
        IPASegmentTokenizer(vocab=bad_vocab)


def test_dict_vocab_valid():
    """A vocab dict containing pad and unk tokens should initialise
    without error."""
    good_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, "a": 2}
    tok = IPASegmentTokenizer(vocab=good_vocab)
    assert tok.vocab_size == 3
    assert tok.pad_id == 0
    assert tok.unk_id == 1


def test_dict_vocab_missing_unk():
    """Vocab dict with pad but missing unk must also raise."""
    vocab = {PAD_TOKEN: 0, "a": 1}
    with pytest.raises(AssertionError, match="unk_token"):
        IPASegmentTokenizer(vocab=vocab)
