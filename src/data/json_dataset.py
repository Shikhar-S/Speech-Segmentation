"""A datamodule to read outputs from a json file obtained via distributed inference.

The JSON file format is expected to be:
{
    "<sample_id>": {
        "pred": [{"processed_transcript": "...text..."}],
        "passthrough": {
            "target": 1,
            "split": "train"|"val"|"test",
            "utt_id": "unique_across_dataset"
        }
    },
    ...
}

Usage:
    python -m src.data.json_dataset \
        --json_path x
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.core.tokenizer.character_tokenizer import CharacterTokenizer
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


def collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    text_ids = [torch.tensor(b["text_ids"], dtype=torch.long) for b in batch]
    lengths = torch.tensor([len(seq) for seq in text_ids], dtype=torch.long)
    text_padded = pad_sequence(text_ids, batch_first=True, padding_value=pad_id)
    # (B,) or (B,2) - for geolocation
    target = torch.tensor([b["target"] for b in batch], dtype=torch.float)
    return {
        "text": text_padded,
        "lengths": lengths,
        "target": target,
        "utt_id": [b["utt_id"] for b in batch],
        "sample_index": [b["sample_index"] for b in batch],
        "metadata_idx": [b.get("metadata_idx", None) for b in batch],
    }


class TranscriptionDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: CharacterTokenizer,
    ) -> None:
        self.samples = samples
        self.max_length = 512
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        text_ids: List[int] of token IDs
        target: int or List of 2 floats (for geolocation)
        utt_id: str
        sample_index: str
        """
        sample = self.samples[idx]
        transcript = sample["processed_transcript"]
        if len(transcript) == 0:
            transcript = "?"
        if len(transcript) > self.max_length:
            transcript = transcript[: self.max_length]
        textids = self.tokenizer.encode(transcript)
        return {
            "text_ids": textids,
            "target": sample["target"],
            "utt_id": sample["utt_id"],
            "sample_index": sample["sample_index"],
            "metadata_idx": sample.get("metadata_idx", None),
        }


class TranscriptionDataModule(LightningDataModule):
    """LightningDataModule for reading phonenbench style transcripts.

    This DataModule:
        1. Loads IPA transcripts from a JSON file
        2. Loads vocabulary from the provided vocab_path file
        3. Creates train/val/test datasets based on the 'split' field
        4. Provides dataloaders with proper batching (padding)
    """

    def __init__(
        self,
        json_path: str,
        num_classes: int,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        min_vocab_freq: int = 1,
    ) -> None:
        """Initialize the DataModule.

        Args:
            json_path: Path to JSON file with IPA transcripts
            num_classes: Number of target classes
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
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.min_vocab_freq = min_vocab_freq

        self.tokenizer: Optional[CharacterTokenizer] = None
        self.train_samples: List[Dict[str, Any]] = []
        self.val_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []

        self.ds_train: Optional[TranscriptionDataset] = None
        self.ds_val: Optional[TranscriptionDataset] = None
        self.ds_test: Optional[TranscriptionDataset] = None

    def prepare_data(self) -> None:
        """Load JSON and load vocabulary from file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        log.info(f"Loading JSON from {self.json_path}")
        with open(self.json_path, encoding="utf-8") as f:
            data = json.load(f)

        for ix, datum in data.items():
            if "pred" not in datum or "passthrough" not in datum:
                raise ValueError(
                    f"Invalid datum format for sample {ix}, skipping because 'pred' or 'passthrough' key is missing."
                )
            pred = datum["pred"]
            passthrough = datum["passthrough"]
            if {"processed_transcript"} - set(pred[0].keys()):
                raise ValueError(
                    f"Missing the key 'processed_transcript' in predictions for sample {ix}!"
                )
            # TODO(shikhar): enable this check later
            # if {
            #     "target",
            #     "split",
            #     "utt_id",
            # } - set(passthrough.keys()):
            #     raise ValueError(
            #         f"Invalid passthrough format for sample {ix}, skipping."
            #     )
            sample = {
                "sample_index": ix,
                "processed_transcript": pred[0]["processed_transcript"],
                "target": passthrough["target"],
                "split": passthrough["split"],
                "utt_id": passthrough.get(
                    "utt_id", passthrough.get("metadata_idx", ix)
                ),
                "metadata_idx": passthrough.get("metadata_idx", None),
            }
            if "predicted_transcript" in pred[0]:
                sample["predicted_transcript"] = pred[0]["predicted_transcript"]

            split = sample["split"]
            if split == "train":
                self.train_samples.append(sample)
            elif split in ("val", "dev", "validation", "valid"):
                self.val_samples.append(sample)
            elif split == "test":
                self.test_samples.append(sample)
            else:
                log.warning("Unknown split '%s' for sample %s", split, ix)

        log.info(
            f"Loaded {len(self.train_samples)} train, {len(self.val_samples)} val, {len(self.test_samples)} test samples."
        )
        self.tokenizer = CharacterTokenizer()
        texts = [s["processed_transcript"] for s in self.train_samples]
        self.tokenizer.build_vocab(
            texts=texts,
            min_freq=self.min_vocab_freq,
        )
        assert len(self.train_samples) > 0, "No training samples found in the dataset."
        assert len(self.test_samples) > 0, "No test samples found in the dataset."

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for train/val/test splits."""
        if self.trainer:
            if self.batch_size % self.trainer.world_size:
                raise RuntimeError(
                    f"batch_size ({self.batch_size}) not divisible by "
                    f"world_size ({self.trainer.world_size})"
                )
            self.batch_size = self.batch_size // self.trainer.world_size

        assert (
            self.tokenizer is not None
        ), "Tokenizer not initialized. Call prepare_data() first."
        if self.ds_train is None:
            self.ds_train = TranscriptionDataset(
                samples=self.train_samples,
                tokenizer=self.tokenizer,
            )
            self.ds_val = TranscriptionDataset(
                samples=self.val_samples,
                tokenizer=self.tokenizer,
            )
            self.ds_test = TranscriptionDataset(
                samples=self.test_samples,
                tokenizer=self.tokenizer,
            )

    def _dl(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, pad_id=self.tokenizer.pad_id),
        )

    def train_dataloader(self) -> DataLoader:
        return self._dl(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dl(self.ds_test, shuffle=False)


def _test_datamodule() -> None:
    """Simple test to verify DataModule works."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to JSON file",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    # Create DataModule
    dm = TranscriptionDataModule(
        json_path=args.json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    dm.prepare_data()
    dm.setup()

    print(f"\n=== DataModule Info ===")
    print(f"Vocabulary size: {dm.tokenizer.vocab_size}")
    print("\n=== Testing train_dataloader ===")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, values={v} ...")
        else:
            print(f"  {k}: {v[:2]} ...")
    print("\n=== Testing val_dataloader ===")
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    print(f"Batch keys: {batch.keys()}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, values={v} ...")
        else:
            print(f"  {k}: {v[:2]} ...")
    print("\n=== Testing test_dataloader ===")
    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))
    print(f"Batch keys: {batch.keys()}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, values={v} ...")
        else:
            print(f"  {k}: {v[:2]} ...")
    print("\n=== Sanity check passed! ===")


if __name__ == "__main__":
    _test_datamodule()
