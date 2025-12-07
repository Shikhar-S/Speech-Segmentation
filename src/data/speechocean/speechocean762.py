"""speechocean762 Dataset and DataModule for L2 Assessment.

Fields returned per sample:
    speech: (T,) float32 tensor
    speech_length: int
    speaker_id: str
    utt_id: str
    scores: dict
    split: str
    metadata_idx: int

Usage:
    python -m src.data.speechocean.speechocean762 \
        --data_dir /path/to/data_root \
        --metadata_path /path/to/metadata.csv \
        --batch_size 32
"""

import os
import math
from typing import Optional, Dict, Any, List

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
import pandas as pd


def pad_collate(batch: List[Dict[str, Any]]):
    L = [b["speech"].shape[-1] for b in batch]
    M = max(L)
    A = [
        torch.nn.functional.pad(b["speech"], (0, M - b["speech"].shape[-1]))
        for b in batch
    ]

    return {
        "speech": torch.stack(A, dim=0),   # (B, T_max)
        "speech_length": torch.tensor(L, dtype=torch.long),
        "scores": [b["scores"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "utt_id": [b["utt_id"] for b in batch],
        "split": [b["split"] for b in batch],
        "metadata_idx": [b["metadata_idx"] for b in batch],
    }


class SpeechOceanDataset(Dataset):

    def __init__(
        self,
        metadata_path: str,
        split: str,
        data_dir: str,
        target_sr: int = 16000,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.target_sr = target_sr

        metadata = (
            pd.read_csv(metadata_path)
            .reset_index()
            .rename(columns={"index": "metadata_idx"})
        )
        self.metadata = metadata[metadata["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        """
        Returns:
            dict:
                speech: Tensor (T,)
                speech_length: int
                speaker_id: str
                scores: dict{accuracy, completeness, fluency, prosodic, total}
                split: 'train' / 'val' / 'test'
                metadata_idx: idx
        """
        row = self.metadata.iloc[i]

        audio_path = os.path.join(self.data_dir, row["audio_path"])
        wav, sr = torchaudio.load(audio_path)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            wav = resampler(wav)
        
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze(0)  # (T,)

        scores = {
            "accuracy": float(row["accuracy"]),
            "completeness": float(row["completeness"]),
            "fluency": float(row["fluency"]),
            "prosodic": float(row["prosodic"]),
            "total": float(row["total"]),
        }

        return {
            "speech": wav,
            "speech_length": wav.shape[0],
            "scores": scores,
            "speaker_id": str(row["speaker_id"]),
            "utt_id": str(row["utt_id"]),
            "split": row["split"],
            "metadata_idx": row["metadata_idx"],
        }


class SpeechOceanDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = self.ds_val = self.ds_test = None
        self.bs_dev = batch_size

    def setup(self, stage: Optional[str] = None):
        if self.trainer:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError(
                    f"batch_size ({self.hparams.batch_size}) not divisible by world_size ({self.trainer.world_size})"
                )
            self.bs_dev = self.hparams.batch_size // self.trainer.world_size

        if self.ds_train is None:
            # speachocean does NOT provide val split
            # -> we split train 90/10 or provide train/val identical
            self.ds_train = SpeechOceanDataset(
                metadata_path=self.hparams.metadata_path,
                split="train",
                data_dir=self.hparams.data_dir,
                target_sr=self.hparams.target_sr
            )
            self.ds_val = SpeechOceanDataset(
                metadata_path=self.hparams.metadata_path,
                split="val",
                data_dir=self.hparams.data_dir,
                target_sr=self.hparams.target_sr
            )
            self.ds_test = SpeechOceanDataset(
                metadata_path=self.hparams.metadata_path,
                split="test",
                data_dir=self.hparams.data_dir,
                target_sr=self.hparams.target_sr
            )

            print(
                f"Dataset split into train: {len(self.ds_train)}, val: {len(self.ds_val)}, test: {len(self.ds_test)}"
            )

    def _dl(self, ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=self.bs_dev,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=pad_collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dl(self.ds_train, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.ds_test, shuffle=False)

    def predict_dataloader(self):
        return self._dl(
            ConcatDataset([self.ds_train, self.ds_val, self.ds_test]),
            shuffle=False,
        )


def _test_datamodule():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()

    dm = SpeechOceanDataModule(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        target_sr=16000,
    )
    dm.setup()

    print("=== Testing train_dataloader ===")
    train_loader = dm.train_dataloader()
    print("Total batches:", len(train_loader))
    
    batch = next(iter(train_loader))
    print(f"batch 0 (size={len(batch)}): {batch.keys()}")
    print(f"speech shape: {batch['speech'].shape}")
    print(f"speech_length shape: {batch['speech_length'].shape}")
    print(f"speech_length values: {batch['speech_length']}")
    print(f"speaker_id: {batch['speaker_id']}")
    print(f"utt_id: {batch['utt_id']}")
    print(f"scores: {batch['scores']}")
    print(f"split: {batch['split']}")
    print(f"metadata_idx: {batch['metadata_idx']}")

    print("=== Testing predict_dataloader ===")
    predict_loader = dm.predict_dataloader()
    print("Total batches:", len(predict_loader))
    
    batch = next(iter(predict_loader))
    print(f"batch 0 (size={len(batch)}): {batch.keys()}")
    print(f"speech shape: {batch['speech'].shape}")
    print(f"speech_length shape: {batch['speech_length'].shape}")
    print(f"scores: {batch['scores']}")
    print(f"split: {batch['split']}")


if __name__ == "__main__":
    _test_datamodule()
