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
    powsm_lang_sym_map: Optional[dict] = None,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """Load FLEURS parquet split for a set of languages, with audio kept as file paths."""
    datasets = []
    samples_per_lang = (
        max(1, max_samples // len(language_subset)) if max_samples else None
    )

    langnames = []
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
        langnames.append(set(ds["language"]).pop())  # hack
        datasets.append(ds)

    dataset = concatenate_datasets(datasets)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Keep audio as file paths / metadata, to avoid torchcodec
    dataset = dataset.cast_column("audio", HFAudio(decode=False))

    # Remap lang_ids to 0-indexed
    lang_to_id = {lang: idx for idx, lang in enumerate(langnames)}

    def add_lang_ids(example):
        example["lang_id"] = lang_to_id[example["language"]]
        example["powsm_lang_sym"] = (
            powsm_lang_sym_map.get(example["language"], "<unk>")
            if powsm_lang_sym_map
            else "<unk>"
        )
        return example

    dataset = dataset.map(add_lang_ids)
    return dataset


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
    ):
        self.tokenizer = tokenizer
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
        assert sr == self.target_sr, f"Expected sr={self.target_sr}, got sr={sr}"
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
            "lang_sym": sample["powsm_lang_sym"],
            "target": sample["lang_id"],
            "split": self.split,
            "metadata_idx": i,
            "utt_id": f"{self.split}_{i}",
        }
        if self.tokenizer:
            return_dict["text"] = text
        return return_dict


class FleursLanguageId(LightningDataModule):
    def __init__(
        self,
        id_to_label: list,
        num_classes: int = 102,
        max_samples: Optional[int] = None,
        target_sr: int = 16000,
        powsm_lang_sym_map: Optional[dict] = None,
        tokenizer: Optional[object] = None,
        max_audio_length: float = 20.0,
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        """
        Args:
            id_to_label: List of language codes to use from FLEURS (e.g., ["en_us", "hi_in"])
        """
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.ds_train = self.ds_val = self.ds_test = None
        self.num_classes = self.hparams.num_classes
        assert self.num_classes == len(self.hparams.id_to_label), (
            f"num_classes={self.num_classes} does not match the number of languages "
            f"in id_to_label={len(self.hparams.id_to_label)}"
        )
        self.bs_dev = batch_size

    def prepare_data(self):
        # first call here to download
        for split in ["train", "validation", "test"]:
            load_fleurs_data(
                split=split,
                language_subset=self.hparams.id_to_label,
                powsm_lang_sym_map=self.hparams.powsm_lang_sym_map,
                max_samples=self.hparams.max_samples,
                cache_dir=self.hparams.cache_dir,
            )

    def _ds(self, split):
        return FleursLanguageIdDataset(
            dataset=load_fleurs_data(
                split=split,
                language_subset=self.hparams.id_to_label,
                powsm_lang_sym_map=self.hparams.powsm_lang_sym_map,
                max_samples=self.hparams.max_samples,
                cache_dir=self.hparams.cache_dir,
            ),
            split=split,
            target_sr=self.hparams.target_sr,
            max_audio_length=self.hparams.max_audio_length,
            tokenizer=self.tokenizer,
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
        return self._dl(
            ConcatDataset([self.ds_train, self.ds_val, self.ds_test]), False
        )


def test_datamodule():
    from src.core.tokenizer.character_tokenizer import CharacterTokenizer

    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(["abcdefghijklmnopqrstuvwxyz"])
    # tokenizer.save_vocab(tokenizer.vocab, "exp/cache/envocab.json")
    dm = FleursLanguageId(
        id_to_label=["en_us"],
        num_classes=1,
        max_samples=500,
        batch_size=8,
        cache_dir="/scratch/sbharad2/PhoneBench/exp/cache/fleurs",
        tokenizer=tokenizer,
    )
    dm.prepare_data()
    dm.setup()

    for batch in dm.train_dataloader():
        print(batch)
        break


if __name__ == "__main__":
    test_datamodule()
