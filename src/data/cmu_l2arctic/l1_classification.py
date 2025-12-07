"""CMU L2Arctic Dataset and DataModule for L1 Classification.

This module provides a LightningDataModule for L1 (native language) classification
using the combined CMU Arctic and L2-ARCTIC corpora.

Usage:
    python -m src.data.cmu_l2arctic.l1_classification \
        --data_dir /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/cmu_l2arctic \
        --metadata_path /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/cmu_l2arctic_cache/metadata.csv \
        --batch_size 2
"""

import logging
import os
from typing import Dict, Optional, List
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
from src.core.utils import resample_dataset

logger = logging.getLogger(__name__)


def pad_collate(batch):
    """
    Collate function for batching variable-length audio sequences.

    Args:
        batch: List of dicts from CmuL2ArcticL1Dataset.__getitem__

    Returns:
        dict with batched tensors:
            - speech: (B, T_max) padded audio
            - speech_length: (B,) actual lengths
            - label: List[str] of length B
            - split: List[str] of length B
            - metadata_idx: List[int] of length B
            - speaker_id: List[str] of length B
            - utt_id: List[str] of length B
    """
    # Get maximum length in batch
    L = [b["speech"].shape[-1] for b in batch]
    M = max(L)

    # Pad all sequences to max length
    A = [
        torch.nn.functional.pad(b["speech"], (0, M - b["speech"].shape[-1]))
        for b in batch
    ]
    T = torch.tensor([b["target"] for b in batch], dtype=torch.long)

    return {
        "speech": torch.stack(A, 0),  # (B, T_max)
        "speech_length": torch.tensor(L, dtype=torch.long),  # (B,)
        "target": T,  # (B,)
        "label": [b["label"] for b in batch],  # List[str]
        "split": [b["split"] for b in batch],  # List[str]
        "metadata_idx": [b["metadata_idx"] for b in batch],  # List[int]
        "speaker_id": [b["speaker_id"] for b in batch],  # List[str]
        "utt_id": [b["utt_id"] for b in batch],  # List[str]
    }


class CmuL2ArcticL1Dataset(Dataset):
    """
    PyTorch Dataset for CMU + L2Arctic L1 classification.

    Loads audio and metadata from a CSV file with the following columns:
        - audio_path: relative path from data_dir
        - label: L1 class (e.g., 'en', 'ko', 'zh', 'ar', 'hi', 'es', 'vi')
        - split: 'train', 'val', or 'test'
        - speaker_id: speaker identifier
        - utt_id: utterance identifier
    """

    def __init__(
        self,
        metadata_path: str,
        split: str,  # 'train', 'val', or 'test'
        data_dir: str,
        label_to_ids: Dict[str, int],
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
    ):
        """
        Args:
            metadata_path: Path to metadata CSV file
            split: Which split to load ('train', 'val', or 'test')
            data_dir: Root directory for audio files (audio_path is relative to this)
            target_sr: Target sample rate for audio (default: 16000)
            max_duration_sec: Maximum audio duration in seconds (for truncation)
        """
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.split = split
        self.label_to_ids = label_to_ids

        # Load metadata and filter by split
        metadata = (
            pd.read_csv(metadata_path)
            .reset_index()
            .rename(columns={"index": "metadata_idx"})
        )
        self.metadata = metadata[metadata["split"] == split].reset_index(drop=True)

        logger.info("Loaded %d samples for split '%s'", len(self.metadata), split)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - speech: Tensor of shape (T,), float32
                - speech_length: int, actual audio length in samples
                - label: str, L1 class label
                - split: str, data split
                - metadata_idx: int, index in metadata CSV
                - speaker_id: str, speaker identifier
                - utt_id: str, utterance identifier
        """
        row = self.metadata.iloc[idx]
        # Load audio from resampled data!!
        audio_path = os.path.join(
            self.data_dir, f"resampled_{self.target_sr}Hz", row["audio_path"]
        )
        waveform, sr = torchaudio.load(audio_path)
        # Resample if necessary
        assert (
            sr == self.target_sr
        ), f"Expected sample rate {self.target_sr}, but got {sr}"
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Truncate if necessary
        if self.max_duration_sec is not None:
            max_samples = int(self.max_duration_sec * sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        # Squeeze to (T,)
        waveform = waveform.squeeze(0)

        return {
            "speech": waveform,
            "speech_length": waveform.shape[0],
            "label": row["l1_label"],
            "target": self.label_to_ids[row["l1_label"]],
            "split": row["split"],
            "metadata_idx": row["metadata_idx"],
            "speaker_id": row["speaker_id"],
            "utt_id": row["utt_id"],
        }


class CmuL2ArcticL1Classification(LightningDataModule):
    """
    LightningDataModule for CMU + L2Arctic L1 classification.

    This DataModule:
        1. Loads metadata from a CSV file
        2. Creates train/val/test datasets based on the 'split' column
        3. Provides dataloaders with proper batching (zero-padding)
        4. Supports distributed training (batch_size is divided by world_size)
    """

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        target_sr: int = 16000,
        num_classes: int = 7,
        id_to_label: List[str] = None,
        max_duration_sec: Optional[float] = None,
    ):
        """
        Args:
            data_dir: Root directory for audio files
            metadata_path: Path to metadata CSV
            batch_size: Batch size (will be divided by world_size in distributed mode)
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory for GPU transfer
            target_sr: Target sample rate
            num_classes: Number of L1 classes
            max_duration_sec: Maximum audio duration in seconds
        """
        super().__init__()
        self.save_hyperparameters()
        self.id_to_label = id_to_label
        self.label_to_ids = {label: i for i, label in enumerate(id_to_label)}
        self.ds_train = self.ds_val = self.ds_test = None
        self.batch_size = batch_size

    def prepare_data(self):
        """Prepare data by resampling audio."""
        tgt_dir = os.path.join(
            self.hparams.data_dir, f"resampled_{self.hparams.target_sr}Hz"
        )
        resample_dataset(
            metadata_df=pd.read_csv(self.hparams.metadata_path),
            path_key="audio_path",
            src_data_dir=self.hparams.data_dir,
            src_sr=44100,
            tgt_data_dir=tgt_dir,
            tgt_sr=self.hparams.target_sr,
            force_resample=False,
        )

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test splits."""
        # Adjust batch size for distributed training
        if self.trainer:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError(
                    f"batch_size ({self.hparams.batch_size}) not divisible by "
                    f"world_size ({self.trainer.world_size})"
                )
            self.batch_size = self.hparams.batch_size // self.trainer.world_size

        # Create datasets if not already created
        if self.ds_train is None:
            self.ds_train = CmuL2ArcticL1Dataset(
                metadata_path=self.hparams.metadata_path,
                split="train",
                data_dir=self.hparams.data_dir,
                target_sr=self.hparams.target_sr,
                label_to_ids=self.label_to_ids,
                max_duration_sec=self.hparams.max_duration_sec,
            )
            self.ds_val = CmuL2ArcticL1Dataset(
                metadata_path=self.hparams.metadata_path,
                split="val",
                data_dir=self.hparams.data_dir,
                target_sr=self.hparams.target_sr,
                label_to_ids=self.label_to_ids,
                max_duration_sec=self.hparams.max_duration_sec,
            )
            self.ds_test = CmuL2ArcticL1Dataset(
                metadata_path=self.hparams.metadata_path,
                split="test",
                data_dir=self.hparams.data_dir,
                target_sr=self.hparams.target_sr,
                label_to_ids=self.label_to_ids,
                max_duration_sec=self.hparams.max_duration_sec,
            )
            logger.info(
                "Dataset split into train: %d, val: %d, test: %d",
                len(self.ds_train),
                len(self.ds_val),
                len(self.ds_test),
            )

    def _dl(self, ds, shuffle):
        """Helper to create DataLoader with common settings."""
        logger.info(
            'Constructing DataLoader for split="%s", shuffle=%s', ds.split, shuffle
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=pad_collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        """Return training dataloader."""
        return self._dl(self.ds_train, shuffle=True)

    def val_dataloader(self):
        """Return validation dataloader."""
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self):
        """Return test dataloader."""
        return self._dl(self.ds_test, shuffle=False)

    def predict_dataloader(self):
        """
        Return prediction dataloader.

        Concatenates all splits for inference, allowing the inference runner
        to transcribe all data at once. The metadata_idx field allows tracking
        which split each sample came from.
        """
        return self._dl(
            ConcatDataset([self.ds_train, self.ds_val, self.ds_test]),
            shuffle=False,
        )


def _test_datamodule():
    """Simple test to verify DataModule works."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory for audio files",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to metadata CSV",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    # Create DataModule
    dm = CmuL2ArcticL1Classification(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        target_sr=16000,
    )

    # Setup and test
    dm.setup()

    print("\n=== Testing train_dataloader ===")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"speech shape: {batch['speech'].shape}")
    print(f"speech_length shape: {batch['speech_length'].shape}")
    print(f"speech_length values: {batch['speech_length']}")
    print(f"label: {batch['label']}")
    print(f"split: {batch['split']}")
    print(f"metadata_idx: {batch['metadata_idx']}")
    print(f"speaker_id: {batch['speaker_id']}")

    print("\n=== Testing predict_dataloader ===")
    predict_loader = dm.predict_dataloader()
    print(f"Total batches in predict_dataloader: {len(predict_loader)}")
    batch = next(iter(predict_loader))
    print(f"First batch - label: {batch['label']}")
    print(f"First batch - split: {batch['split']}")

    print("\n=== Sanity check passed! ===")


if __name__ == "__main__":
    _test_datamodule()
