"""FLEURS Dataset and DataModule.

Usage:
    python -m src.data.fleurs.common_datamodule
"""

import os
from pathlib import Path
from typing import Optional

import torch, io
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
from datasets import Audio as HFAudio, load_dataset


def load_fleurs_data(hf_repo: str, split: str, cache_dir: Optional[str] = None):
    ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
    ds = ds.cast_column("audio", HFAudio(decode=False))
    return ds


def pad_collate(batch):
    max_len = max(b["speech"].shape[-1] for b in batch)
    padded = [
        torch.nn.functional.pad(b["speech"], (0, max_len - b["speech"].shape[-1]))
        for b in batch
    ]
    padded_text = None
    padded_text_length = None
    if "text" in batch[0]:
        max_text_len = max(len(b["text"]) for b in batch)
        padded_text = torch.zeros((len(batch), max_text_len), dtype=torch.long)
        padded_text_length = torch.zeros((len(batch),), dtype=torch.long)
        for i, b in enumerate(batch):
            text_len = len(b["text"])
            padded_text[i, :text_len] = b["text"]
            padded_text_length[i] = text_len

    return_dict = {
        "speech": torch.stack(padded),
        "speech_length": torch.tensor([b["speech"].shape[-1] for b in batch]),
        "sr": batch[0]["sr"],
        "language": [b["language"] for b in batch],
        "target": torch.tensor([b["target"] for b in batch]),
        "split": [b.get("split", "none") for b in batch],
        "metadata_idx": [b["metadata_idx"] for b in batch],
    }
    if "text" in batch[0]:
        return_dict["text"] = padded_text
        return_dict["text_length"] = padded_text_length
    return return_dict


class FleursLanguageIdDataset(Dataset):
    def __init__(
        self,
        dataset,
        split: str,
        target_sr: int = 16000,
        max_audio_length: float = 20.0,
        tokenizer=None,
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.split = split
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_audio_length)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_audio(self, waveform: torch.Tensor, sr: int, target_path: Path) -> None:
        target_path = Path(target_path)
        if target_path.exists():
            return
        os.makedirs(target_path.parent, exist_ok=True)
        torchaudio.save(str(target_path), waveform, sr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample = self.dataset[i]

        waveform, sr = torchaudio.load(io.BytesIO(sample["audio"]["bytes"]))
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        assert sr == self.target_sr, f"Expected sr={self.target_sr}, got sr={sr}"

        hf_audio_path = None
        if isinstance(sample.get("audio"), dict):
            hf_audio_path = sample["audio"].get("path")

        assert hf_audio_path, "FLEURS audio path not found in dataset sample"
        audio_path = hf_audio_path
        name = f"{Path(hf_audio_path).stem}.wav"
        target_path = self.cache_dir / "saved" / self.split / name
        self._cache_audio(waveform, sr, target_path)
        audio_path = str(target_path)

        waveform = waveform.squeeze(0)  # (T,)
        if self.tokenizer:
            text = sample["transcription"]
            text = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Truncate
        if waveform.shape[-1] > self.max_len:
            waveform = waveform[: self.max_len]

        return_dict = {
            "speech": waveform.to(torch.float32),
            "sr": self.target_sr,
            "language": sample["language"],
            "lang_sym": "<unk>",  # out of domain for powsm
            "target": sample["target"],
            "split": self.split,
            "metadata_idx": i,
            "utt_id": f"{self.split}_{i}",
            "audio_path": audio_path,
        }
        if self.tokenizer:
            return_dict["text"] = text
        return return_dict


class FleursLanguageId(LightningDataModule):
    def __init__(
        self,
        hf_repo: str,
        num_classes: int = 24,
        target_sr: int = 16000,
        tokenizer: Optional[object] = None,
        max_audio_length: float = 20.0,
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
        predict_splits: Optional[list] = None,  # Splits for predict_dataloader, default: all
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.ds_train = self.ds_val = self.ds_test = None
        self.num_classes = self.hparams.num_classes
        self.bs_dev = batch_size
        self.predict_splits = predict_splits or ["train", "validation", "test"]

    def prepare_data(self):
        # first call here to download/cache
        for split in ["train", "validation", "test"]:
            load_fleurs_data(
                hf_repo=self.hparams.hf_repo,
                split=split,
                cache_dir=self.hparams.cache_dir,
            )

    def _ds(self, split):
        return FleursLanguageIdDataset(
            dataset=load_fleurs_data(
                hf_repo=self.hparams.hf_repo,
                split=split,
                cache_dir=self.hparams.cache_dir,
            ),
            split=split,
            target_sr=self.hparams.target_sr,
            max_audio_length=self.hparams.max_audio_length,
            tokenizer=self.tokenizer,
            cache_dir=self.hparams.cache_dir,
        )

    def setup(self, stage: Optional[str] = None):
        if self.trainer and self.trainer.world_size > 1:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError("batch_size not divisible by world_size")
            self.bs_dev = self.hparams.batch_size // self.trainer.world_size
        self.ds_train = self._ds("train")
        self.ds_val = self._ds("validation")
        self.ds_test = self._ds("test")

    def _dl(self, ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=self.bs_dev,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            collate_fn=pad_collate,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
        )

    def train_dataloader(self):
        return self._dl(self.ds_train, True)

    def val_dataloader(self):
        return self._dl(self.ds_val, False)

    def test_dataloader(self):
        return self._dl(self.ds_test, False)

    def predict_dataloader(self):
        datasets = []
        if "train" in self.predict_splits and self.ds_train:
            datasets.append(self.ds_train)
        if "validation" in self.predict_splits and self.ds_val:
            datasets.append(self.ds_val)
        if "test" in self.predict_splits and self.ds_test:
            datasets.append(self.ds_test)
        if not datasets:
            datasets = [self.ds_test]  # fallback to test
        return self._dl(ConcatDataset(datasets), False)


def test_datamodule():
    from src.core.tokenizer.character_tokenizer import CharacterTokenizer

    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(["abcdefghijklmnopqrstuvwxyz"])

    dm = FleursLanguageId(
        hf_repo="shikhar7ssu/fleurs24-lid",
        num_classes=24,
        batch_size=2,
        cache_dir="exp/cache/fleurs",
        tokenizer=tokenizer,
    )
    dm.prepare_data()
    dm.setup()

    for batch in dm.train_dataloader():
        print(batch)
        break


if __name__ == "__main__":
    test_datamodule()
