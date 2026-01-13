"""CMU L2Arctic Dataset and DataModule for L1 Classification.

This module provides a LightningDataModule for L1 (native language) classification
using the combined CMU Arctic and L2-ARCTIC corpora.

Loads data from HuggingFace Hub parquet dataset with embedded audio bytes.
Audio is decoded and resampled to cache_dir/resampled_{target_sr}Hz/ on first run.

Usage:
    python -m src.data.cmu_l2arctic.l1_classification \
        --hf_repo y00njaekim/cmul2arctic-l1cls \
        --cache_dir exp/cache/cmu_l2arctic \
        --batch_size 2
"""

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
from datasets import Audio as HFAudio
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

logger = logging.getLogger(__name__)


def load_cmul2arctic_hf_data(
    hf_repo: str,
    split: str,
    cache_dir: Optional[str] = None,
):
    """Load CMU+L2ARCTIC dataset from HuggingFace Hub.

    The HF dataset has parquet shards organized as:
        - cmu/{train,val,test}/*.parquet
        - l2arctic/{train,val,test}/*.parquet

    This function loads the requested split using the dataset's pre-defined split
    metadata (stored in the dataset card / config on the Hub).

    Args:
        hf_repo: HuggingFace dataset repository (e.g., 'y00njaekim/cmul2arctic-l1cls')
        split: Which split to load ('train', 'val', or 'test')
        cache_dir: Local cache directory for HF datasets

    Returns:
        HF Dataset with combined cmu + l2arctic data for the given split
    """
    ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
    ds = ds.cast_column("audio", HFAudio(decode=False))
    return ds


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

    HuggingFace Dataset and loads audio from cache_dir/resampled_{sr}Hz/.
    The resampled wav files are expected to be prepared by prepare_data().

    HF Dataset columns used:
        - audio.path: relative path (e.g., 'cmu/cmu_us_bdl_arctic/wav/arctic_a0001.wav')
        - audio.bytes: embedded audio bytes (wav bytes)
        - l1_label: L1 class (e.g., 'en', 'ko', 'zh', 'ar', 'hi', 'es', 'vi')
        - speaker_id: speaker identifier
        - utt_id: utterance identifier
    """

    def __init__(
        self,
        hf_dataset,
        split: str,
        cache_dir: Union[str, Path],
        label_to_ids: Dict[str, int],
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
    ):
        """
        Args:
            hf_dataset: HuggingFace Dataset for this split
            split: Which split ('train', 'val', or 'test')
            cache_dir: Root cache directory (resampled wavs at cache_dir/resampled_{sr}Hz/)
            label_to_ids: Mapping from label string to integer
            target_sr: Target sample rate for audio (default: 16000)
            max_duration_sec: Maximum audio duration in seconds (for truncation)
        """
        self.hf_ds = hf_dataset
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.label_to_ids = label_to_ids
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec

        logger.info("Created dataset for split '%s' with %d samples", split, len(self))

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - speech: Tensor of shape (T,), float32
                - speech_length: int, actual audio length in samples
                - audio_path: str, path to the audio file (for API-based models)
                - label: str, L1 class label
                - split: str, data split
                - metadata_idx: int, index in dataset
                - speaker_id: str, speaker identifier
                - utt_id: str, utterance identifier
        """
        sample = self.hf_ds[idx]

        # Get audio path from HF row (e.g., 'cmu/cmu_us_bdl_arctic/wav/arctic_a0001.wav')
        audio_rel_path = sample["audio"]["path"]
        resampled_dir = self.cache_dir / f"resampled_{self.target_sr}Hz"
        audio_path = resampled_dir / audio_rel_path

        # Load resampled audio
        waveform, sr = torchaudio.load(str(audio_path))
        assert sr == self.target_sr, f"Expected sr={self.target_sr}, got {sr}"

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
            "audio_path": str(audio_path),
            "lang_sym": "<eng>",  # for powsm
            "label": sample["l1_label"],
            "target": self.label_to_ids[sample["l1_label"]],
            "split": self.split,
            "metadata_idx": idx,
            "speaker_id": sample["speaker_id"],
            "utt_id": sample["utt_id"],
        }


class CmuL2ArcticL1Classification(LightningDataModule):
    """
    LightningDataModule for CMU + L2Arctic L1 classification.

    This DataModule:
        1. Loads data from HuggingFace Hub (parquet with embedded audio bytes)
        2. Decodes audio bytes and resamples to target_sr, caching to cache_dir
        3. Creates train/val/test datasets wrapping the HF datasets
        4. Provides dataloaders with proper batching (zero-padding)
        5. Supports distributed training (batch_size is divided by world_size)
    """

    def __init__(
        self,
        hf_repo: str,
        cache_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        target_sr: int = 16000,
        num_classes: int = 7,
        id_to_label: List[str] = None,
        max_duration_sec: Optional[float] = None,
        predict_splits: Optional[List[str]] = None,
    ):
        """
        Args:
            hf_repo: HuggingFace dataset repository (e.g., 'y00njaekim/cmul2arctic-l1cls')
            cache_dir: Cache directory for resampled audio
            batch_size: Batch size (will be divided by world_size in distributed mode)
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory for GPU transfer
            target_sr: Target sample rate
            num_classes: Number of L1 classes
            id_to_label: List of label strings in order (default: alphabetical)
            max_duration_sec: Maximum audio duration in seconds
            predict_splits: List of splits to use for predict_dataloader (default: all)
        """
        super().__init__()
        self.save_hyperparameters()

        self.id_to_label = id_to_label
        self.label_to_ids = {label: i for i, label in enumerate(id_to_label)}

        self.ds_train = self.ds_val = self.ds_test = None
        self.hf_train = self.hf_val = self.hf_test = None
        self.batch_size = batch_size
        self.predict_splits = predict_splits or ["train", "val", "test"]

    def prepare_data(self):
        """Prepare data by downloading from HF and resampling audio to cache_dir."""
        cache_dir = Path(self.hparams.cache_dir)
        target_sr = self.hparams.target_sr
        resampled_dir = cache_dir / f"resampled_{target_sr}Hz"

        # Check if already prepared
        if resampled_dir.exists():
            existing_files = list(resampled_dir.rglob("*.wav"))
            if len(existing_files) > 0:
                logger.info(
                    f"Found {len(existing_files)} resampled files in {resampled_dir}, "
                    "skipping prepare_data."
                )
                return

        logger.info(f"Preparing data from HuggingFace: {self.hparams.hf_repo}")

        # Load all splits and resample
        for split in ["train", "val", "test"]:
            logger.info(f"Processing split: {split}")
            ds = load_cmul2arctic_hf_data(
                hf_repo=self.hparams.hf_repo,
                split=split,
                cache_dir=str(cache_dir),
            )

            # Resample and save each audio file
            for i, sample in enumerate(ds):
                audio_rel_path = sample["audio"]["path"]
                audio_bytes = sample["audio"]["bytes"]

                # Target path
                target_path = resampled_dir / audio_rel_path
                if target_path.exists():
                    continue

                # Decode audio from bytes
                waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

                # Resample if needed
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)

                # Save resampled audio
                target_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(target_path), waveform, target_sr)

                if (i + 1) % 1000 == 0:
                    logger.info(f"  Processed {i + 1}/{len(ds)} samples")

            logger.info(f"  Completed {split}: {len(ds)} samples")

        logger.info("prepare_data completed.")

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
            cache_dir = Path(self.hparams.cache_dir)

            # Load HF datasets
            self.hf_train = load_cmul2arctic_hf_data(
                hf_repo=self.hparams.hf_repo,
                split="train",
                cache_dir=str(cache_dir),
            )
            self.hf_val = load_cmul2arctic_hf_data(
                hf_repo=self.hparams.hf_repo,
                split="val",
                cache_dir=str(cache_dir),
            )
            self.hf_test = load_cmul2arctic_hf_data(
                hf_repo=self.hparams.hf_repo,
                split="test",
                cache_dir=str(cache_dir),
            )

            # Wrap in PyTorch Datasets
            self.ds_train = CmuL2ArcticL1Dataset(
                hf_dataset=self.hf_train,
                split="train",
                cache_dir=cache_dir,
                label_to_ids=self.label_to_ids,
                target_sr=self.hparams.target_sr,
                max_duration_sec=self.hparams.max_duration_sec,
            )
            self.ds_val = CmuL2ArcticL1Dataset(
                hf_dataset=self.hf_val,
                split="val",
                cache_dir=cache_dir,
                label_to_ids=self.label_to_ids,
                target_sr=self.hparams.target_sr,
                max_duration_sec=self.hparams.max_duration_sec,
            )
            self.ds_test = CmuL2ArcticL1Dataset(
                hf_dataset=self.hf_test,
                split="test",
                cache_dir=cache_dir,
                label_to_ids=self.label_to_ids,
                target_sr=self.hparams.target_sr,
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

        Concatenates splits specified in predict_splits for inference.
        The metadata_idx field allows tracking which split each sample came from.
        """
        datasets = []
        if "train" in self.predict_splits and self.ds_train:
            datasets.append(self.ds_train)
        if "val" in self.predict_splits and self.ds_val:
            datasets.append(self.ds_val)
        if "test" in self.predict_splits and self.ds_test:
            datasets.append(self.ds_test)
        if not datasets:
            datasets = [self.ds_test]  # fallback to test
        return self._dl(ConcatDataset(datasets), shuffle=False)


def _test_datamodule():
    """Simple test to verify DataModule works."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="y00njaekim/cmul2arctic-l1cls",
        help="HuggingFace dataset repository",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Cache directory for resampled audio",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    # Create DataModule
    dm = CmuL2ArcticL1Classification(
        hf_repo=args.hf_repo,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        target_sr=16000,
        id_to_label=[
            "ar",
            "en",
            "es",
            "hi",
            "ko",
            "vi",
            "zh",
        ],  # Fixed L1 labels (alphabetical order)
    )

    # Setup and test
    dm.prepare_data()
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
