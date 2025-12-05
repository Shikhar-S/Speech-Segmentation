"""CMU L2Arctic IPA Dataset and DataModule for L1 Classification.

This module provides a LightningDataModule for L1 (native language) classification
using pre-extracted IPA transcripts from JSON files (e.g., PoWSM or Wav2Vec2Phoneme output).

The JSON file format is expected to be:
{
    "<sample_id>": {
        "pred": [{"processed_transcript": "...IPA text..."}],
        "passthrough": {
            "l1_label": "en",
            "metadata_idx": "...",
            "speaker_id": "...",
            "split": "train"|"val"|"test",
            "utt_id": "..."
        }
    },
    ...
}

Usage:
    python -m src.data.cmu_l2arctic.ipa_data \
        --json_path /path/to/cmu_l2arctic_powsm_out.json \
        --batch_size 32
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.core.tokenizer.character_tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)

# Fixed L1 label mapping (alphabetical order)
L1_LABELS: Tuple[str, ...] = ("ar", "en", "es", "hi", "ko", "vi", "zh")
L1_LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(L1_LABELS)}
L1_ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(L1_LABELS)}


def ipa_collate_fn(
    batch: List[Dict[str, Any]], pad_id: int = 0
) -> Dict[str, Any]:
    """Collate function for batching variable-length IPA sequences.

    Args:
        batch: List of dicts from CmuL2ArcticIPADataset.__getitem__
        pad_id: Padding token ID (default: 0)

    Returns:
        dict with batched tensors:
            - ipa_ids: (B, T_max) padded token IDs
            - lengths: (B,) actual lengths
            - label: (B,) L1 class IDs
            - l1_label: List[str] of length B (original string labels)
            - split: List[str] of length B
            - metadata_idx: List[str] of length B
            - speaker_id: List[str] of length B
            - utt_id: List[str] of length B
    """
    # Extract IPA token sequences
    ipa_seqs = [torch.tensor(b["ipa_ids"], dtype=torch.long) for b in batch]
    lengths = torch.tensor([len(seq) for seq in ipa_seqs], dtype=torch.long)

    # Pad sequences to max length in batch
    ipa_padded = pad_sequence(ipa_seqs, batch_first=True, padding_value=pad_id)

    # Stack labels
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {
        "ipa_ids": ipa_padded,  # (B, T_max)
        "lengths": lengths,  # (B,)
        "label": labels,  # (B,)
        "l1_label": [b["l1_label"] for b in batch],  # List[str]
        "split": [b["split"] for b in batch],  # List[str]
        "metadata_idx": [b["metadata_idx"] for b in batch],  # List[str]
        "speaker_id": [b["speaker_id"] for b in batch],  # List[str]
        "utt_id": [b["utt_id"] for b in batch],  # List[str]
    }


class CmuL2ArcticIPADataset(Dataset):
    """PyTorch Dataset for CMU + L2Arctic L1 classification from IPA transcripts.

    Loads IPA transcripts from a JSON file and converts them to token IDs
    using a CharacterTokenizer.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: CharacterTokenizer,
    ) -> None:
        """Initialize the dataset.

        Args:
            samples: List of sample dicts with keys:
                - processed_transcript: IPA text
                - l1_label: L1 class string
                - metadata_idx, speaker_id, split, utt_id
            tokenizer: CharacterTokenizer instance for encoding IPA text
        """
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.

        Returns:
            dict with:
                - ipa_ids: List[int], token IDs for IPA text
                - label: int, L1 class ID
                - l1_label: str, original L1 label string
                - split: str, data split
                - metadata_idx: str, original metadata index
                - speaker_id: str, speaker identifier
                - utt_id: str, utterance identifier
        """
        sample = self.samples[idx]

        # Encode IPA transcript to token IDs
        ipa_ids = self.tokenizer.encode(sample["processed_transcript"])

        # Convert L1 label string to integer ID
        label = L1_LABEL_TO_ID[sample["l1_label"]]

        return {
            "ipa_ids": ipa_ids,
            "label": label,
            "l1_label": sample["l1_label"],
            "split": sample["split"],
            "metadata_idx": sample["metadata_idx"],
            "speaker_id": sample["speaker_id"],
            "utt_id": sample["utt_id"],
        }


class CmuL2ArcticIPADataModule(LightningDataModule):
    """LightningDataModule for CMU + L2Arctic L1 classification from IPA transcripts.

    This DataModule:
        1. Loads IPA transcripts from a JSON file
        2. Loads vocabulary from the provided vocab_path file
        3. Creates train/val/test datasets based on the 'split' field
        4. Provides dataloaders with proper batching (padding)
    """

    def __init__(
        self,
        json_path: str,
        vocab_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        min_vocab_freq: int = 1,
    ) -> None:
        """Initialize the DataModule.

        Args:
            json_path: Path to JSON file with IPA transcripts
            vocab_path: Path to vocabulary JSON file (required)
            batch_size: Batch size (will be divided by world_size in distributed mode)
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory for GPU transfer
            min_vocab_freq: Minimum frequency for a character to be included in vocab
                (unused, kept for backward compatibility)
        """
        super().__init__()
        self.save_hyperparameters()

        self.json_path = Path(json_path)
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.min_vocab_freq = min_vocab_freq

        self.tokenizer: Optional[CharacterTokenizer] = None
        self.train_samples: List[Dict[str, Any]] = []
        self.val_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []

        self.ds_train: Optional[CmuL2ArcticIPADataset] = None
        self.ds_val: Optional[CmuL2ArcticIPADataset] = None
        self.ds_test: Optional[CmuL2ArcticIPADataset] = None
        self.bs_dev = batch_size

    @property
    def num_classes(self) -> int:
        """Return the number of L1 classes."""
        return len(L1_LABELS)

    @property
    def id_to_label(self) -> Tuple[str, ...]:
        """Return the L1 label tuple (index -> label string)."""
        return L1_LABELS

    @property
    def label_to_id(self) -> Dict[str, int]:
        """Return the L1 label mapping (label string -> index)."""
        return L1_LABEL_TO_ID

    def prepare_data(self) -> None:
        """Load JSON and load vocabulary from file.

        This method is called only on rank 0 in distributed training.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        # Load and parse JSON
        logger.info("Loading JSON from %s", self.json_path)
        with open(self.json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate vocab_path before processing data
        if not self.vocab_path:
            raise ValueError("vocab_path must be provided")
        
        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {self.vocab_path}. "
                "Please create the vocabulary file before running training."
            )

        # Parse samples and split by 'split' field
        for sample_id, sample_data in data.items():
            # Extract fields
            pred = sample_data.get("pred", [{}])
            processed_transcript = pred[0].get("processed_transcript", "") if pred else ""
            passthrough = sample_data.get("passthrough", {})

            sample = {
                "sample_id": sample_id,
                "processed_transcript": processed_transcript,
                "l1_label": passthrough.get("l1_label", ""),
                "metadata_idx": str(passthrough.get("metadata_idx", "")),
                "speaker_id": passthrough.get("speaker_id", ""),
                "split": passthrough.get("split", ""),
                "utt_id": passthrough.get("utt_id", ""),
            }

            # Skip samples with missing required fields
            if not sample["processed_transcript"] or not sample["l1_label"]:
                continue

            # Skip samples with unknown L1 labels
            if sample["l1_label"] not in L1_LABEL_TO_ID:
                logger.warning(
                    "Unknown L1 label '%s' for sample %s, skipping",
                    sample["l1_label"],
                    sample_id,
                )
                continue

            split = sample["split"]
            if split == "train":
                self.train_samples.append(sample)
            elif split == "val":
                self.val_samples.append(sample)
            elif split == "test":
                self.test_samples.append(sample)
            else:
                logger.warning("Unknown split '%s' for sample %s", split, sample_id)

        logger.info(
            "Loaded %d train, %d val, %d test samples",
            len(self.train_samples),
            len(self.val_samples),
            len(self.test_samples),
        )

        # Load vocabulary
        logger.info("Loading vocabulary from %s", self.vocab_path)
        self.tokenizer = CharacterTokenizer(vocab_path=self.vocab_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for train/val/test splits."""
        # Adjust batch size for distributed training
        if self.trainer:
            if self.batch_size % self.trainer.world_size:
                raise RuntimeError(
                    f"batch_size ({self.batch_size}) not divisible by "
                    f"world_size ({self.trainer.world_size})"
                )
            self.bs_dev = self.batch_size // self.trainer.world_size

        # Ensure prepare_data has been called
        if self.tokenizer is None:
            self.prepare_data()

        assert self.tokenizer is not None

        # Create datasets
        if self.ds_train is None:
            self.ds_train = CmuL2ArcticIPADataset(
                samples=self.train_samples,
                tokenizer=self.tokenizer,
            )
            self.ds_val = CmuL2ArcticIPADataset(
                samples=self.val_samples,
                tokenizer=self.tokenizer,
            )
            self.ds_test = CmuL2ArcticIPADataset(
                samples=self.test_samples,
                tokenizer=self.tokenizer,
            )

    def _dl(self, ds: Dataset, shuffle: bool) -> DataLoader:
        """Helper to create DataLoader with common settings."""
        assert self.tokenizer is not None
        return DataLoader(
            ds,
            batch_size=self.bs_dev,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            collate_fn=lambda batch: ipa_collate_fn(batch, pad_id=self.tokenizer.pad_id),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        assert self.ds_train is not None
        return self._dl(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        assert self.ds_val is not None
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        assert self.ds_test is not None
        return self._dl(self.ds_test, shuffle=False)


def _test_datamodule() -> None:
    """Simple test to verify DataModule works."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to JSON file with IPA transcripts",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Create DataModule
    dm = CmuL2ArcticIPADataModule(
        json_path=args.json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Setup and test
    dm.prepare_data()
    dm.setup()

    print(f"\n=== DataModule Info ===")
    print(f"Number of classes: {dm.num_classes}")
    print(f"Labels: {dm.id_to_label}")
    print(f"Vocabulary size: {dm.tokenizer.vocab_size if dm.tokenizer else 'N/A'}")

    print("\n=== Testing train_dataloader ===")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"ipa_ids shape: {batch['ipa_ids'].shape}")
    print(f"lengths: {batch['lengths']}")
    print(f"label: {batch['label']}")
    print(f"l1_label: {batch['l1_label']}")
    print(f"split: {batch['split']}")

    print("\n=== Sanity check passed! ===")


if __name__ == "__main__":
    _test_datamodule()

