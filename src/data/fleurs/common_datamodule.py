"""FLEURS Dataset and DataModule.
TODO(shikhar): use fleurs-11 config to use this for geolocation as well.

Usage:
    python -m src.data.fleurs.common_datamodule
"""

from typing import Optional, List

import torch, io
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
from datasets import load_dataset, concatenate_datasets, Audio as HFAudio


def load_fleurs_data(
    split: str,
    language_subset: List[str],
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """Load FLEURS parquet split for a set of languages, with audio kept as file paths."""
    datasets = []
    samples_per_lang = (
        max(1, max_samples // len(language_subset)) if max_samples else None
    )

    for lang in language_subset:
        ds = load_dataset(
            "google/fleurs",
            data_dir=lang,
            split=split,
            revision="refs/convert/parquet",
            cache_dir=cache_dir,
        )
        if samples_per_lang:
            ds = ds.select(range(min(samples_per_lang, len(ds))))
        datasets.append(ds)

    dataset = concatenate_datasets(datasets)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Keep audio as file paths / metadata, to avoid torchcodec
    dataset = dataset.cast_column("audio", HFAudio(decode=False))

    # Remap lang_ids to 0-indexed
    unique_langs = sorted(set(dataset["language"]))
    lang_to_id = {lang: idx for idx, lang in enumerate(unique_langs)}

    def add_lang_id(example):
        example["lang_id"] = lang_to_id[example["language"]]
        return example

    dataset = dataset.map(add_lang_id)

    return dataset, len(unique_langs)


def pad_collate(batch):
    max_len = max(b["speech"].shape[-1] for b in batch)
    padded = [
        torch.nn.functional.pad(b["speech"], (0, max_len - b["speech"].shape[-1]))
        for b in batch
    ]
    return {
        "speech": torch.stack(padded),
        "speech_length": torch.tensor([b["speech"].shape[-1] for b in batch]),
        "sr": batch[0]["sr"],
        "lang_id": torch.tensor([b["lang_id"] for b in batch]),
        "language": [b["language"] for b in batch],
        "split": [b.get("split", "none") for b in batch],
        "metadata_idx": [b["metadata_idx"] for b in batch],
    }


class FleursLanguageIdDataset(Dataset):
    def __init__(
        self,
        dataset,
        split: str,
        target_sr: int = 16000,
        max_audio_length: float = 20.0,
    ):
        self.dataset = dataset
        self.split = split
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_audio_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample = self.dataset[i]

        waveform, sr = torchaudio.load(io.BytesIO(sample["audio"]["bytes"]))
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.squeeze(0)  # (T,)

        # Truncate
        if waveform.shape[-1] > self.max_len:
            waveform = waveform[: self.max_len]

        return {
            "speech": waveform.to(torch.float32),
            "sr": self.target_sr,
            "lang_id": sample["lang_id"],
            "language": sample["language"],
            "split": self.split,
            "metadata_idx": i,
        }


class FleursLanguageId(LightningDataModule):
    def __init__(
        self,
        language_subset: list,
        batch_size: int = 64,
        num_workers: int = 4,
        target_sr: int = 16000,
        max_audio_length: float = 20.0,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = self.ds_val = self.ds_test = None
        self.num_classes = None
        self.bs_dev = batch_size

    def prepare_data(self):
        for split in ["train", "validation", "test"]:
            load_fleurs_data(
                split,
                self.hparams.language_subset,
                self.hparams.max_samples,
                self.hparams.cache_dir,
            )

    def setup(self, stage: Optional[str] = None):
        if self.trainer and self.trainer.world_size > 1:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError("batch_size not divisible by world_size")
            self.bs_dev = self.hparams.batch_size // self.trainer.world_size

        if self.ds_train is None:
            train_data, self.num_classes = load_fleurs_data(
                "train",
                self.hparams.language_subset,
                self.hparams.max_samples,
                self.hparams.cache_dir,
            )
            val_data, _ = load_fleurs_data(
                "validation",
                self.hparams.language_subset,
                self.hparams.max_samples,
                self.hparams.cache_dir,
            )
            test_data, _ = load_fleurs_data(
                "test",
                self.hparams.language_subset,
                self.hparams.max_samples,
                self.hparams.cache_dir,
            )

            self.ds_train = FleursLanguageIdDataset(
                train_data,
                "train",
                self.hparams.target_sr,
                self.hparams.max_audio_length,
            )
            self.ds_val = FleursLanguageIdDataset(
                val_data,
                "validation",
                self.hparams.target_sr,
                self.hparams.max_audio_length,
            )
            self.ds_test = FleursLanguageIdDataset(
                test_data,
                "test",
                self.hparams.target_sr,
                self.hparams.max_audio_length,
            )

    def _dl(self, ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=self.bs_dev,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            collate_fn=pad_collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dl(self.ds_train, True)

    def val_dataloader(self):
        return self._dl(self.ds_val, False)

    def test_dataloader(self):
        return self._dl(self.ds_test, False)

    def predict_dataloader(self):
        return self._dl(
            ConcatDataset([self.ds_train, self.ds_val, self.ds_test]), False
        )


if __name__ == "__main__":
    dm = FleursLanguageId(
        batch_size=2,
        num_workers=2,
        language_subset=["en_us", "hi_in"],
        max_samples=50,
        cache_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/fleurs_cache",
    )
    dm.setup()

    for i, batch in enumerate(dm.train_dataloader()):
        print(f"Batch {i}: {batch['speech'].shape}, lang_ids={batch['lang_id']}")
        if i >= 2:
            break
