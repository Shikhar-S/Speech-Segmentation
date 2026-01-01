"""Ultrasuite Datamodule for child speech atypicality classification.

HF Dataset characteristics:
- Speech dataset
- Labels embedded directly in Parquet files
- Metadata fields per sample:
    - audio (HF Audio, bytes-encoded)
    - label (int)
    - subject (string)
    - filename (string)

Usage:
    python -m src.data.ultrasuite_benchmark.childspeech \
        --hf_repo kgrosero14/ultrasuite-benchmark \
        --cache_dir exp/cache/ultrasuite
"""

import io
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L
from datasets import load_dataset, Audio as HFAudio
import os


import io
import argparse
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import lightning as L
from datasets import load_dataset, Audio as HFAudio


def _keep_not_too_short(example):
    return not example["too_short"]


def _resample_and_markshort(example):
    MIN_LENGTH = 8000  # 0.5 second at 16kHz
    audio = example["audio"]
    wav, sr = torchaudio.load(io.BytesIO(example["audio"]["bytes"]))
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        buf = io.BytesIO()
        torchaudio.save(buf, wav, 16000, format="wav")
        audio["bytes"] = buf.getvalue()
        audio["sampling_rate"] = 16000
        example["audio"] = audio
    example["too_short"] = wav.shape[1] < MIN_LENGTH
    return example


def load_ultrasuite_data(
    hf_repo: str,
    split: str,
    cache_dir: Optional[str] = None,
):
    ds = load_dataset(
        hf_repo,
        split=split,
        cache_dir=cache_dir,
    )
    ds = ds.cast_column("audio", HFAudio(decode=False))
    ds = ds.map(_resample_and_markshort)
    ds = ds.filter(_keep_not_too_short)
    ds = ds.with_format(None)
    return ds


class UltrasuiteDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        cache_dir: Union[Path, str],
        target_sr: int = 16000,
        split: str = "train",
        max_duration_sec: Optional[float] = None,
    ):
        self.hf_ds = hf_ds
        self.cache_dir = Path(cache_dir)
        self.target_sr = target_sr
        self.split = split
        self.max_duration_sec = max_duration_sec

        # Cache resamplers per source SR
        self._resamplers = {}

    def __len__(self):
        return len(self.hf_ds)

    def _cache_audio(self, waveform, sr, target_path):
        if os.path.exists(target_path):
            return
        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torchaudio.save(target_path, waveform, sr)

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        waveform, sr = torchaudio.load(io.BytesIO(sample["audio"]["bytes"]))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        assert sr == self.target_sr, f"Expected sr={self.target_sr}, got sr={sr}"

        if self.max_duration_sec is not None:
            max_samples = int(self.max_duration_sec * self.target_sr)
            waveform = waveform[:, :max_samples]

        target_path = self.cache_dir / "saved" / self.split / f"{sample['filename']}"
        self._cache_audio(waveform, sr, target_path)

        waveform = waveform.squeeze(0)

        return {
            "utt_id": sample["filename"],
            "audio_path": str(target_path),
            "split": self.split,
            "speech": waveform,
            "speech_length": waveform.shape[0],
            "lang_sym": "<eng>",  # TODO(shikhar,karen): confirm with karen
            "target": int(sample["label"]),
            "subject": sample["subject"],
            "orig_sr": sr,
        }


def collate_fn(batch):
    if not batch:
        raise ValueError("Empty batch in collate_fn")

    max_len = max(x["speech_length"] for x in batch)
    B = len(batch)

    speech = torch.zeros(B, max_len, dtype=torch.float32)
    speech_length = torch.zeros(B, dtype=torch.long)

    for i, x in enumerate(batch):
        L = x["speech_length"]
        speech[i, :L] = x["speech"]
        speech_length[i] = L

    return {
        "utt_id": [x["utt_id"] for x in batch],
        "split": [x["split"] for x in batch],
        "speech": speech,
        "speech_length": speech_length,
        "target": torch.tensor([x["target"] for x in batch], dtype=torch.long),
        "subject": [x["subject"] for x in batch],
        "orig_sr": [x["orig_sr"] for x in batch],
    }


class UltrasuiteDataModule(L.LightningDataModule):
    def __init__(
        self,
        hf_repo: str,
        cache_dir: str,
        num_classes: int = 2,
        target_sr: int = 16000,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_duration_sec: Optional[float] = None,
        predict_splits: Optional[list] = None,  # Splits for predict_dataloader, default: all
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hf_repo = hf_repo
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.target_sr = target_sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_duration_sec = max_duration_sec
        self.predict_splits = predict_splits or ["train", "validation", "test"]

    def prepare_data(self):
        for split in ["train", "validation", "test"]:
            load_ultrasuite_data(
                self.hf_repo,
                split=split,
                cache_dir=str(self.cache_dir),
            )

    def _ds(self, split: str):
        return UltrasuiteDataset(
            load_ultrasuite_data(
                self.hf_repo,
                split=split,
                cache_dir=str(self.cache_dir),
            ),
            target_sr=self.target_sr,
            cache_dir=self.cache_dir,
            split=split,
            max_duration_sec=self.max_duration_sec,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_ds = self._ds("train")
        self.val_ds = self._ds("validation")
        self.test_ds = self._ds("test")

    def _dl(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self._dl(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.test_ds, shuffle=False)

    def predict_dataloader(self):
        datasets = []
        if "train" in self.predict_splits and self.train_ds:
            datasets.append(self.train_ds)
        if "validation" in self.predict_splits and self.val_ds:
            datasets.append(self.val_ds)
        if "test" in self.predict_splits and self.test_ds:
            datasets.append(self.test_ds)
        if not datasets:
            datasets = [self.test_ds]  # fallback to test
        return self._dl(ConcatDataset(datasets), shuffle=False)


# Main
def main():
    parser = argparse.ArgumentParser(description="Ultrasuite Benchmark HF DataModule")
    parser.add_argument(
        "--hf_repo",
        type=str,
        required=True,
        default="kgrosero14/ultrasuite-benchmark",
        help="Hugging Face dataset repo",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Local cache directory for Hugging Face datasets",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sampling rate after resampling (default: 16000)",
    )

    args = parser.parse_args()

    dm = UltrasuiteDataModule(
        hf_repo=args.hf_repo,
        cache_dir=args.cache_dir,
        target_sr=args.target_sr,
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        max_duration_sec=20.0,
    )

    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.val_dataloader()))
    print("Loaded batch keys:", batch.keys())
    print("Speech:", batch["speech"])
    print("Targets:", batch["target"])


if __name__ == "__main__":
    main()
