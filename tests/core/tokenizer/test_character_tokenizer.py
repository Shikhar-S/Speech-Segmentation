"""Tests for CharacterTokenizer."""

import json
import pytest
import tempfile
from pathlib import Path

from src.core.tokenizer.character_tokenizer import (
    CharacterTokenizer,
    DEFAULT_SPECIALS,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)


class TestCharacterTokenizerInit:
    """Tests for CharacterTokenizer initialization."""

    def test_init_without_vocab(self):
        """Test initialization without vocabulary."""
        tokenizer = CharacterTokenizer()
        
        assert tokenizer._vocab is None
        assert tokenizer._ids_to_tokens is None

    def test_init_with_custom_special_tokens(self):
        """Test initialization with custom special tokens."""
        tokenizer = CharacterTokenizer(
            pad_token="<pad>",
            unk_token="<unk>",
        )
        
        assert tokenizer._pad_token == "<pad>"
        assert tokenizer._unk_token == "<unk>"

    def test_init_without_vocab_raises_on_encode(self):
        """Test that encoding raises when vocab is not initialized."""
        tokenizer = CharacterTokenizer()
        
        with pytest.raises(AttributeError):
            tokenizer.encode("test")


class TestCharacterTokenizerVocab:
    """Tests for vocabulary building and loading."""

    def test_build_vocab_basic(self):
        """Test basic vocabulary building."""
        tokenizer = CharacterTokenizer()
        texts = ["hello", "world", "test"]
        
        tokenizer.build_vocab(texts)
        
        assert tokenizer.vocab is not None
        assert tokenizer.vocab_size > 0
        assert PAD_TOKEN in tokenizer.vocab
        assert UNK_TOKEN in tokenizer.vocab
        assert BOS_TOKEN in tokenizer.vocab
        assert EOS_TOKEN in tokenizer.vocab

    def test_build_vocab_with_min_freq(self):
        """Test vocabulary building with min_freq."""
        tokenizer = CharacterTokenizer()
        texts = ["aaa", "bbb", "ccc"]  # All chars appear 3 times
        
        tokenizer.build_vocab(texts, min_freq=2)
        
        # All characters should be in vocab
        assert "a" in tokenizer.vocab
        assert "b" in tokenizer.vocab
        assert "c" in tokenizer.vocab

    def test_build_vocab_min_freq_filters(self):
        """Test that min_freq filters out rare characters."""
        tokenizer = CharacterTokenizer()
        texts = ["aaa", "bbb"]  # 'c' doesn't appear
        
        tokenizer.build_vocab(texts, min_freq=2)
        
        # Characters from texts should be in vocab
        assert "a" in tokenizer.vocab
        assert "b" in tokenizer.vocab

    def test_build_vocab_with_custom_specials(self):
        """Test vocabulary building with custom special tokens."""
        tokenizer = CharacterTokenizer()
        texts = ["hello"]
        custom_specials = ("<pad>", "<unk>")
        
        tokenizer.build_vocab(texts, specials=custom_specials)
        
        assert "<pad>" in tokenizer.vocab
        assert "<unk>" in tokenizer.vocab

    def test_save_and_load_vocab(self):
        """Test saving and loading vocabulary."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello", "world"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            CharacterTokenizer.save_vocab(tokenizer.vocab, vocab_path)
            
            # Load into new tokenizer
            new_tokenizer = CharacterTokenizer(vocab_path=vocab_path)
            
            assert new_tokenizer.vocab_size == tokenizer.vocab_size

    def test_from_file_classmethod(self):
        """Test from_file class method."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello", "world"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            CharacterTokenizer.save_vocab(tokenizer.vocab, vocab_path)
            
            loaded_tokenizer = CharacterTokenizer.from_file(vocab_path)
            
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


class TestCharacterTokenizerEncode:
    """Tests for encoding functionality."""

    def test_encode_simple_text(self):
        """Test encoding simple text."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello"])
        
        ids = tokenizer.encode("hello")
        
        assert len(ids) == 5
        assert all(isinstance(i, int) for i in ids)

    def test_encode_with_unk_characters(self):
        """Test encoding with unknown characters."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])  # Only a, b, c in vocab
        
        ids = tokenizer.encode("abcd")  # 'd' is unknown
        
        assert len(ids) == 4
        assert ids[-1] == tokenizer.unk_id

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello"])
        
        ids = tokenizer.encode("")
        
        assert ids == []


class TestCharacterTokenizerDecode:
    """Tests for decoding functionality."""

    def test_decode_simple_ids(self):
        """Test decoding simple IDs."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello"])
        
        ids = [tokenizer.vocab[c] for c in "hello"]
        decoded = tokenizer.decode(ids)
        
        assert decoded == "hello"

    def test_decode_with_unk_ids(self):
        """Test decoding with unknown IDs."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["ab"])
        
        # Create IDs that include an unknown one (large ID)
        ids = [tokenizer.vocab["a"], 99999, tokenizer.vocab["b"]]
        decoded = tokenizer.decode(ids)
        
        assert "a" in decoded
        assert "b" in decoded
        assert tokenizer._unk_token in decoded

    def test_decode_empty_ids(self):
        """Test decoding empty list."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello"])
        
        decoded = tokenizer.decode([])
        
        assert decoded == ""


class TestCharacterTokenizerDecodeClean:
    """Tests for decode_clean functionality."""

    def test_decode_clean_default(self):
        """Test decode_clean with default skip_special=True."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello"])
        
        # Get IDs including special tokens
        pad_id = tokenizer.pad_id
        ids = [tokenizer.vocab["h"], pad_id, tokenizer.vocab["e"]]
        
        decoded = tokenizer.decode_clean(ids)
        
        # Special tokens should be skipped
        assert decoded == "he"

    def test_decode_clean_no_skip(self):
        """Test decode_clean with skip_special=False."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello"])
        
        pad_id = tokenizer.pad_id
        ids = [tokenizer.vocab["h"], pad_id, tokenizer.vocab["e"]]
        
        decoded = tokenizer.decode_clean(ids, skip_special=False)
        
        # Special tokens should be included
        assert PAD_TOKEN in decoded


class TestCharacterTokenizerLabelId:
    """Tests for label_to_id and id_to_label."""

    def test_label_to_id_known(self):
        """Test label_to_id with known label."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])
        
        label_id = tokenizer.label_to_id("a")
        
        assert label_id == tokenizer.vocab["a"]

    def test_label_to_id_unknown(self):
        """Test label_to_id with unknown label."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])
        
        label_id = tokenizer.label_to_id("z")  # Not in vocab
        
        assert label_id == tokenizer.unk_id

    def test_id_to_label_known(self):
        """Test id_to_label with known ID."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])
        
        label = tokenizer.id_to_label(tokenizer.vocab["a"])
        
        assert label == "a"

    def test_id_to_label_unknown(self):
        """Test id_to_label with unknown ID."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])
        
        label = tokenizer.id_to_label(99999)
        
        assert label == tokenizer._unk_token


class TestCharacterTokenizerProperties:
    """Tests for tokenizer properties."""

    def test_vocab_size_property(self):
        """Test vocab_size property."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])
        
        assert tokenizer.vocab_size == len(tokenizer.vocab)

    def test_pad_id_property(self):
        """Test pad_id property."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])
        
        assert tokenizer.pad_id == tokenizer.vocab[PAD_TOKEN]

    def test_unk_id_property(self):
        """Test unk_id property."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["abc"])
        
        assert tokenizer.unk_id == tokenizer.vocab[UNK_TOKEN]

    def test_pad_token_property(self):
        """Test pad_token property."""
        tokenizer = CharacterTokenizer(pad_token="<custom_pad>")
        
        assert tokenizer.pad_token == "<custom_pad>"

    def test_unk_token_property(self):
        """Test unk_token property."""
        tokenizer = CharacterTokenizer(unk_token="<custom_unk>")
        
        assert tokenizer.unk_token == "<custom_unk>"


class TestCharacterTokenizerRoundTrip:
    """Tests for round-trip encoding/decoding."""

    def test_roundtrip_simple(self):
        """Test encode -> decode preserves content."""
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(["hello world"])
        
        original = "hello"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        
        assert original == decoded

    def test_roundtrip_with_ipa(self):
        """Test encode -> decode with IPA characters."""
        tokenizer = CharacterTokenizer()
        texts = ["həˈloʊ", "wɝld", "ˈtɛstɪŋ"]
        tokenizer.build_vocab(texts)
        
        for text in texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert text == decoded


class TestCharacterTokenizerValidation:
    """Tests for vocabulary validation."""

    def test_validate_pad_token_in_vocab(self):
        """Test that pad_token must be in vocab when loading from file."""
        vocab = {"a": 2, "b": 3}  # Missing pad_token
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab, f)
            
            with pytest.raises(AssertionError):
                CharacterTokenizer(vocab_path=vocab_path)

    def test_validate_unk_token_in_vocab(self):
        """Test that unk_token must be in vocab when loading from file."""
        vocab = {"<pad>": 0}  # Missing unk_token
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab, f)
            
            with pytest.raises(AssertionError):
                CharacterTokenizer(vocab_path=vocab_path)
