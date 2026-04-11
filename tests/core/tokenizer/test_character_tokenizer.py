"""Tests for CharacterTokenizer."""

from src.core.tokenizer.character_tokenizer import (
    CharacterTokenizer,
    DEFAULT_SPECIALS,
)


def _make_tokenizer(*texts: str) -> CharacterTokenizer:
    """Build a tokenizer from sample texts."""
    tok = CharacterTokenizer()
    tok.build_vocab(texts)
    return tok


def test_build_vocab_returns_dict():
    """build_vocab must return a Dict[str, int] (bug L4 fix)."""
    tok = CharacterTokenizer()
    result = tok.build_vocab(["abc", "def"])
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) for v in result.values())


def test_build_vocab_includes_specials():
    """All default special tokens must appear in the vocab."""
    tok = CharacterTokenizer()
    vocab = tok.build_vocab(["hello"])
    for special in DEFAULT_SPECIALS:
        assert special in vocab


def test_encode_decode_roundtrip():
    """Encoding then decoding should recover the original text."""
    tok = _make_tokenizer("hello", "world")
    text = "hello"
    assert tok.decode(tok.encode(text)) == text


def test_unknown_char_uses_unk_token():
    """Characters absent from the vocab should map to unk_id."""
    tok = _make_tokenizer("abc")
    ids = tok.encode("xyz")
    assert all(i == tok.unk_id for i in ids)
