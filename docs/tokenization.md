# Tokenization Workflow

This guide explains how to build and use tokenizers for IPA transcripts in PhoneBench.

## Overview

PhoneBench provides a `CharacterTokenizer` that treats each character (including IPA symbols) as a separate token. This is a simple baseline approach suitable for:

- IPA transcript processing
- Phoneme-level or character-level downstream tasks
- Quick experimentation before adopting more sophisticated tokenization

## Workflow Summary

```
1. Prepare training transcripts
       ↓
2. Build vocabulary (build_vocab)
       ↓
3. Save vocabulary to JSON file
       ↓
4. Configure Hydra (vocab_path in config)
       ↓
5. Train / Inference
```

## Step-by-Step Guide

### 1. Build Vocabulary

Before training, you must build a vocabulary from your training transcripts:

```python
from src.core.tokenizer import CharacterTokenizer

# Collect all training transcripts
train_texts = [
    "həˈloʊ",
    "wɝld",
    "ˈtɛstɪŋ",
    # ... all your training transcripts
]

# Build vocabulary (characters appearing at least min_freq times)
vocab = CharacterTokenizer.build_vocab(
    texts=train_texts,
    min_freq=1,  # minimum frequency threshold
)

# Save to file
CharacterTokenizer.save_vocab(vocab, "artifacts/ipa_vocab.json")
```

**Output format** (`artifacts/ipa_vocab.json`):
```json
{
  "<pad>": 0,
  "<unk>": 1,
  "<s>": 2,
  "</s>": 3,
  "h": 4,
  "ə": 5,
  ...
}
```

### 2. Configure Hydra

Set the vocabulary path in your experiment config or via CLI override.

**Option A: In experiment config**

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - override /data/tokenizer: char

data:
  tokenizer:
    vocab_path: artifacts/ipa_vocab.json
```

**Option B: CLI override**

```bash
python src/main.py experiment=my_experiment \
    data/tokenizer=char \
    data.tokenizer.vocab_path=artifacts/ipa_vocab.json
```

### 3. DataModule Integration

In your DataModule, initialize the tokenizer after vocabulary is available:

```python
from src.core.tokenizer import CharacterTokenizer

class MyDataModule(LightningDataModule):
    def __init__(self, tokenizer: CharacterTokenizer, ...):
        self.tokenizer = tokenizer
        # Hydra will instantiate CharacterTokenizer from config

    def setup(self, stage=None):
        # tokenizer is ready to use
        self.train_dataset = MyDataset(
            ...,
            tokenizer=self.tokenizer,
        )
```

### 4. Encode/Decode Usage

```python
# Encoding
text = "həˈloʊ"
token_ids = tokenizer.encode(text)  # [4, 5, 6, 7, 8, 9]

# Decoding
decoded = tokenizer.decode(token_ids)  # "həˈloʊ"

# Clean decode (skip special tokens)
decoded_clean = tokenizer.decode_clean(token_ids, skip_special=True)
```

## Configuration Reference

### Hydra Config (`configs/data/tokenizer/char.yaml`)

```yaml
_target_: src.core.tokenizer.CharacterTokenizer

# REQUIRED: Path to vocabulary JSON file
vocab_path: ???

# Optional: Override special token strings
pad_token: "<pad>"
unk_token: "<unk>"
```

**Important**: `vocab_path: ???` means this is a mandatory parameter. Hydra will raise `MissingMandatoryValue` error if not specified.

### Special Tokens

| Token | Default | Index | Purpose |
|-------|---------|-------|---------|
| `<pad>` | `<pad>` | 0 | Padding for batch alignment |
| `<unk>` | `<unk>` | 1 | Unknown/OOV characters |
| `<s>` | `<s>` | 2 | Beginning of sequence (optional) |
| `</s>` | `</s>` | 3 | End of sequence (optional) |

## Error Handling

### Missing Vocabulary File

If `vocab_path` points to a non-existent file:

```
FileNotFoundError: Vocabulary file not found: artifacts/ipa_vocab.json
Please build the vocabulary first using:
  vocab = CharacterTokenizer.build_vocab(train_texts)
  CharacterTokenizer.save_vocab(vocab, 'artifacts/ipa_vocab.json')
```

### Missing Config Parameter

If `vocab_path` is not set in config:

```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: data.tokenizer.vocab_path
```

## Best Practices

1. **Version your vocabulary**: Include vocab files in your experiment artifacts or version control.

2. **Use consistent vocab for train/val/test**: Always use the vocabulary built from training data.

3. **Check OOV rate**: Monitor how many `<unk>` tokens appear in validation/test sets.

4. **min_freq tuning**: Start with `min_freq=1` and increase if vocabulary is too large.

## API Reference

### `CharacterTokenizer`

```python
class CharacterTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ) -> None: ...

    # Properties
    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def unk_id(self) -> int: ...

    # Core methods
    def encode(self, text: str) -> List[int]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...
    def decode_clean(self, token_ids: Sequence[int], skip_special: bool = True) -> str: ...

    # Static utilities
    @staticmethod
    def build_vocab(
        texts: Iterable[str],
        min_freq: int = 1,
        specials: Tuple[str, ...] = ("<pad>", "<unk>", "<s>", "</s>"),
    ) -> Dict[str, int]: ...

    @staticmethod
    def save_vocab(vocab: Dict[str, int], path: str) -> None: ...

    @classmethod
    def from_file(cls, vocab_path: str, ...) -> "CharacterTokenizer": ...
```

## See Also

- [Features & Capabilities](features.md) - General Hydra/Lightning features
- [Running Inference](running_inference.md) - Inference workflow guide


