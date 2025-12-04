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
        max_samples: Optional[int] = None,  # Limit number of samples (useful for testing)
        language_subset: Optional[list] = None,  # List of language codes to use (e.g., ["en_us", "hi_in"])
    ):
        """
        Initialize FLEURS Language Identification Dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            target_sr: Target sampling rate
            max_audio_length: Maximum audio length in seconds
            config_name: FLEURS config name. Use "all" to load all languages for LangID task.
                        Or use specific language codes like "en_us", "hi_in", etc. for subset.
            max_samples: Maximum number of samples to use (None = use all). Useful for testing.
            language_subset: List of language codes to filter (e.g., ["en_us", "hi_in"]).
                           If provided, filters dataset to only these languages.
        """
        self.target_sr = target_sr
        self.max_audio_length = max_audio_length
        self.split = split
        
        # Load FLEURS dataset
        # If language_subset is provided, load individual language configs and concatenate
        # IMPORTANT: We apply max_samples per language BEFORE concatenating to minimize downloads
        if language_subset is not None:
            print(f"Loading individual language configs: {language_subset}")
            from datasets import concatenate_datasets
            
            # Calculate samples per language if max_samples is specified
            # Distribute max_samples evenly across languages
            samples_per_lang = None
            if max_samples is not None:
                samples_per_lang = max(1, max_samples // len(language_subset))
                print(f"Limiting to ~{samples_per_lang} samples per language (total target: {max_samples})")
            
            datasets_list = []
            for lang in language_subset:
                try:
                    print(f"Loading {lang}...")
                    # Try streaming first to minimize downloads
                    try:
                        ds_streaming = load_dataset(
                            "google/fleurs", 
                            lang, 
                            split=split, 
                            streaming=True,
                            trust_remote_code=True
                        )
                        # Take only what we need if max_samples is specified
                        if samples_per_lang is not None:
                            ds_streaming = ds_streaming.take(samples_per_lang)
                        # Convert streaming dataset to regular dataset
                        ds = ds_streaming.to_list()
                        from datasets import Dataset
                        ds = Dataset.from_list(ds)
                        print(f"Successfully loaded {lang} (streaming): {len(ds)} samples")
                    except Exception as stream_e:
                        # Fallback to non-streaming if streaming fails
                        print(f"Streaming failed for {lang}, using regular load: {stream_e}")
                        ds = load_dataset(
                            "google/fleurs", 
                            lang, 
                            split=split, 
                            streaming=False,
                            trust_remote_code=True
                        )
                        # Apply limit after loading if not using streaming
                        if samples_per_lang is not None and len(ds) > samples_per_lang:
                            ds = ds.select(range(samples_per_lang))
                        print(f"Successfully loaded {lang} (regular): {len(ds)} samples")
                    
                    datasets_list.append(ds)
                except Exception as lang_e:
                    print(f"Warning: Failed to load {lang}: {lang_e}")
                    continue
            
            if not datasets_list:
                raise RuntimeError(f"Could not load any FLEURS language configs from {language_subset}")
            
            # Concatenate all language datasets
            self.dataset = concatenate_datasets(datasets_list)
            print(f"Loaded {len(self.dataset)} total samples from {len(datasets_list)} languages")
            
            # Apply final max_samples limit if we have more than requested
            # (This can happen if samples_per_lang * num_langs > max_samples)
            if max_samples is not None and len(self.dataset) > max_samples:
                print(f"Applying final limit: {len(self.dataset)} -> {max_samples} samples")
                self.dataset = self.dataset.select(range(max_samples))
        elif config_name == "all":
            # Load all languages (large download)
            print("Loading FLEURS 'all' config (this will download all 102 languages - large dataset!)")
            print("WARNING: This will download a very large dataset. Consider using language_subset instead.")
            # Try streaming first to minimize downloads
            try:
                ds_streaming = load_dataset(
                    "google/fleurs", 
                    config_name, 
                    split=split, 
                    streaming=True,
                    trust_remote_code=True
                )
                if max_samples is not None:
                    ds_streaming = ds_streaming.take(max_samples)
                ds_list = ds_streaming.to_list()
                from datasets import Dataset
                self.dataset = Dataset.from_list(ds_list)
                print(f"Loaded {len(self.dataset)} samples using streaming")
            except Exception as stream_e:
                print(f"Streaming failed, using regular load: {stream_e}")
                self.dataset = load_dataset(
                    "google/fleurs", 
                    config_name, 
                    split=split, 
                    streaming=False,
                    trust_remote_code=True
                )
                if max_samples is not None and len(self.dataset) > max_samples:
                    print(f"Limiting to {max_samples} samples (from {len(self.dataset)})")
                    self.dataset = self.dataset.select(range(max_samples))
        elif config_name is not None:
            # Load a specific language config
            print(f"Loading FLEURS config: {config_name}")
            # Try streaming first to minimize downloads
            try:
                ds_streaming = load_dataset(
                    "google/fleurs", 
                    config_name, 
                    split=split, 
                    streaming=True,
                    trust_remote_code=True
                )
                if max_samples is not None:
                    ds_streaming = ds_streaming.take(max_samples)
                ds_list = ds_streaming.to_list()
                from datasets import Dataset
                self.dataset = Dataset.from_list(ds_list)
                print(f"Loaded {len(self.dataset)} samples using streaming")
            except Exception as stream_e:
                print(f"Streaming failed, using regular load: {stream_e}")
                self.dataset = load_dataset(
                    "google/fleurs", 
                    config_name, 
                    split=split, 
                    streaming=False,
                    trust_remote_code=True
                )
                if max_samples is not None and len(self.dataset) > max_samples:
                    print(f"Limiting to {max_samples} samples (from {len(self.dataset)})")
                    self.dataset = self.dataset.select(range(max_samples))
        else:
            raise ValueError("Either config_name or language_subset must be provided")
        
        # Get unique languages and create a mapping to 0-indexed lang_ids
        # This is important when using language_subset, as lang_ids might not be consecutive
        unique_languages = sorted(list(set(self.dataset["language"])))
        self.lang_to_id = {lang: idx for idx, lang in enumerate(unique_languages)}
        self.lang_names = unique_languages
        self.num_classes = len(unique_languages)
        
        print(f"Found {self.num_classes} unique languages: {unique_languages}")
        
        # Remap lang_ids to be 0-indexed (0, 1, 2, ...)
        def remap_lang_id(example):
            lang = example["language"]
            example["lang_id"] = self.lang_to_id[lang]
            return example
        
        self.dataset = self.dataset.map(remap_lang_id)
        
        # Note: max_samples limit is already applied per-language before concatenation
        # This minimizes disk usage by downloading only what we need

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
        config_name="all",  # "all" for all languages, or specific language codes
        max_samples: Optional[int] = None,  # Limit samples per split (useful for testing)
        language_subset: Optional[list] = None,  # List of language codes to use
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
                        Or use specific language codes like "en_us", "hi_in", etc.
            max_samples: Maximum number of samples per split (None = use all). Useful for testing.
            language_subset: List of language codes to filter (e.g., ["en_us", "hi_in"]).
                           If provided, filters dataset to only these languages.
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
                max_samples=self.hparams.get("max_samples"),
                language_subset=self.hparams.get("language_subset"),
            )
            self.ds_val = FleursLanguageIdDataset(
                split="validation",
                target_sr=self.hparams.target_sr,
                max_audio_length=self.hparams.max_audio_length,
                config_name=self.hparams.config_name,
                max_samples=self.hparams.get("max_samples"),
                language_subset=self.hparams.get("language_subset"),
            )
            self.ds_test = FleursLanguageIdDataset(
                split="test",
                target_sr=self.hparams.target_sr,
                max_audio_length=self.hparams.max_audio_length,
                config_name=self.hparams.config_name,
                max_samples=self.hparams.get("max_samples"),
                language_subset=self.hparams.get("language_subset"),
            )
            
            # Get number of classes from training set
            # The dataset already has num_classes set after remapping
            self.num_classes = self.ds_train.num_classes
            
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

