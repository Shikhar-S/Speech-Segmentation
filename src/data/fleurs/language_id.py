"""FLEURS Language Identification Dataset and DataModule.

Usage:
    python -m src.data.fleurs.language_id
"""

import os
import torch
from typing import Optional
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
from datasets import load_dataset


def pad_collate(batch):
    L = [b["speech"].shape[-1] for b in batch]
    M = max(L)
    A = [
        torch.nn.functional.pad(b["speech"], (0, M - b["speech"].shape[-1]))
        for b in batch
    ]
    return {
        "speech": torch.stack(A, 0),
        "speech_length": torch.tensor(L),
        "sr": batch[0]["sr"],
        "lang_id": torch.tensor([b["lang_id"] for b in batch]),
        "language": [b["language"] for b in batch],
        "split": [b.get("split", "none") for b in batch],
        "metadata_idx": [b["metadata_idx"] for b in batch],
    }


class FleursLanguageIdDataset(Dataset):
    def __init__(
        self,
        split: str,  # 'train', 'validation', or 'test'
        target_sr: Optional[int] = 16000,
        max_audio_length: Optional[float] = 20.0,  # seconds
        config_name: str = "all",  # "all" for all languages, or specific language config
    ):
        """
        Initialize FLEURS Language Identification Dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            target_sr: Target sampling rate
            max_audio_length: Maximum audio length in seconds
            config_name: FLEURS config name. Use "all" to load all languages for LangID task.
        """
        self.target_sr = target_sr
        self.max_audio_length = max_audio_length
        self.split = split
        
        # Load FLEURS dataset
        # For language identification, we use "all" config to get all languages merged
        # According to FLEURS docs: "We simply create a single train/valid/test for LangID by merging all"
        # When split is specified, load_dataset returns just that split's Dataset
        # trust_remote_code=True is required because FLEURS uses a dataset script (fleurs.py)
        # Use streaming=False for now, but can be set to True for memory efficiency
        self.dataset = load_dataset(
            "google/fleurs", 
            config_name, 
            split=split, 
            streaming=False,
            trust_remote_code=True  # Required for FLEURS dataset script
        )
        
        # Get language names for mapping
        # According to FLEURS docs, lang_id is a ClassLabel with names attribute
        if hasattr(self.dataset.features["lang_id"], "names"):
            self.lang_names = self.dataset.features["lang_id"].names
        else:
            # Fallback: get unique languages from the dataset
            # This should not happen with FLEURS, but handle gracefully
            self.lang_names = sorted(list(set(self.dataset["language"])))
            print(f"Warning: lang_id names not found, using {len(self.lang_names)} unique languages")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample = self.dataset[i]
        
        # Load audio
        audio = sample["audio"]
        audio_array = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]
        
        # Resample if needed
        if self.target_sr and sr != self.target_sr:
            audio_array = audio_array.unsqueeze(0)  # Add channel dimension
            audio_array = torchaudio.functional.resample(audio_array, sr, self.target_sr)
            audio_array = audio_array.squeeze(0)  # Remove channel dimension
            sr = self.target_sr
        
        # Trim to max length
        max_len = int(sr * self.max_audio_length)
        if audio_array.shape[-1] > max_len:
            audio_array = audio_array[:max_len]
        
        # Get language ID
        lang_id = sample["lang_id"]
        language = sample.get("language", "unknown")
        
        return {
            "speech": audio_array,
            "speech_length": audio_array.shape[-1],
            "sr": sr,
            "lang_id": lang_id,
            "language": language,
            "split": self.split,
            "metadata_idx": i,
        }


class FleursLanguageId(LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        target_sr=16000,
        max_audio_length=20.0,
        config_name="all",  # "all" for all languages
    ):
        """
        FLEURS Language Identification DataModule.
        
        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            target_sr: Target sampling rate
            max_audio_length: Maximum audio length in seconds
            config_name: FLEURS config name. Use "all" to load all languages for LangID task.
        """
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = self.ds_val = self.ds_test = None
        self.bs_dev = batch_size

    def setup(self, stage: Optional[str] = None):
        if self.trainer:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError("batch_size not divisible by world_size")
            self.bs_dev = self.hparams.batch_size // self.trainer.world_size

        if self.ds_train is None:
            self.ds_train = FleursLanguageIdDataset(
                split="train",
                target_sr=self.hparams.target_sr,
                max_audio_length=self.hparams.max_audio_length,
                config_name=self.hparams.config_name,
            )
            self.ds_val = FleursLanguageIdDataset(
                split="validation",
                target_sr=self.hparams.target_sr,
                max_audio_length=self.hparams.max_audio_length,
                config_name=self.hparams.config_name,
            )
            self.ds_test = FleursLanguageIdDataset(
                split="test",
                target_sr=self.hparams.target_sr,
                max_audio_length=self.hparams.max_audio_length,
                config_name=self.hparams.config_name,
            )
            
            # Get number of classes from training set
            if hasattr(self.ds_train.dataset.features["lang_id"], "names"):
                self.num_classes = len(self.ds_train.dataset.features["lang_id"].names)
            else:
                # Fallback: count unique lang_ids
                unique_lang_ids = set()
                for i in range(len(self.ds_train)):
                    unique_lang_ids.add(self.ds_train[i]["lang_id"])
                self.num_classes = len(unique_lang_ids)
            
            print(
                f"Dataset split into train: {len(self.ds_train)}, val: {len(self.ds_val)}, test: {len(self.ds_test)}"
            )
            print(f"Number of language classes: {self.num_classes}")

    def _dl(self, ds, shuffle):
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
            ConcatDataset([self.ds_train, self.ds_val, self.ds_test]), shuffle=False
        )


if __name__ == "__main__":
    # Test the dataset
    from tqdm import tqdm

    dm = FleursLanguageId(
        batch_size=2,
        num_workers=1,
        target_sr=16000,
        max_audio_length=20.0,
        config_name="all",
    )
    dm.setup()
    
    print(f"Number of classes: {dm.num_classes}")
    
    # Test a few batches
    for i, batch in enumerate(tqdm(dm.train_dataloader(), desc="Train batches", unit="batch")):
        print(f"Batch {i}:")
        print(f"  Speech shape: {batch['speech'].shape}")
        print(f"  Lang IDs: {batch['lang_id']}")
        print(f"  Languages: {batch['language']}")
        if i >= 2:
            break

