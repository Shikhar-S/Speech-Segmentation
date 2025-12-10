"""EasyCall Dataset and DataModule

Usage:
    python -m src.data.easycall.common_datamodule \
        --easycall_root /path/to/EasyCall.zip \
        --data_dir /path/to/cache \
        --easycall_meta_csv /path/to/easycall_meta.csv \
"""

import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional, List
import zipfile
import tempfile
import epitran
from ipatok import tokenise as ipatok_tokenise


import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_easycall_clip(easycall_root: Path, item: dict, out: Path, sr: int) -> bool:
    """
    Extract and cache EasyCall audio clip.

    Supports both regular .wav files and .zip files containing audio.
    If the source is a zip file, extracts the specific audio file from it.

    Args:
        easycall_root: Path to EasyCall root directory
        item: Metadata item with 'path', 'zip_path', 'file', or 'zip_file' keys
        out: Output path for cached audio
        sr: Target sample rate

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if we're dealing with a zip file
        zip_path = item.get("zip_path") or item.get("zip_file")
        file_in_zip = item.get("file")

        if zip_path:
            # Extract from zip file
            if isinstance(zip_path, str):
                zip_path = Path(zip_path)
            if not zip_path.is_absolute():
                zip_path = easycall_root / zip_path

            if not zip_path.exists():
                logger.warning(f"Zip file not found: {zip_path}")
                return False

            # Extract the specific file from zip
            zip_entry = item.get("zip_entry")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Use zip_entry if available, otherwise search for the file
                file_to_extract = zip_entry
                if file_to_extract is None:
                    # Find the file in the zip (handle different path formats)
                    for name in zip_ref.namelist():
                        if (
                            name.endswith(file_in_zip)
                            or os.path.basename(name) == file_in_zip
                        ):
                            file_to_extract = name
                            break

                if file_to_extract is None or file_to_extract not in zip_ref.namelist():
                    logger.warning(f"File {file_in_zip} not found in zip {zip_path}")
                    return False

                # Extract to temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Extract the file (preserves directory structure if any)
                    zip_ref.extract(file_to_extract, tmp_dir)
                    # Construct full path to extracted file
                    extracted_path = os.path.join(tmp_dir, file_to_extract)

                    # Load and process audio
                    wav, s = torchaudio.load(extracted_path)
        else:
            # Regular file path
            source_path = item.get("path") or (easycall_root / item.get("file", ""))
            if isinstance(source_path, str):
                source_path = Path(source_path)
            if not source_path.is_absolute():
                source_path = easycall_root / source_path

            if not source_path.exists():
                logger.warning(f"Audio file not found: {source_path}")
                return False

            # Load and process audio
            wav, s = torchaudio.load(str(source_path))

        # Resample if necessary
        if s != sr:
            resampler = torchaudio.transforms.Resample(s, sr)
            wav = resampler(wav)
            s = sr

        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Save cached version
        out.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out), wav, sr)
        return True
    except Exception as e:
        logger.warning(f"speech extraction failed [{item.get('file', 'unknown')}]: {e}")
        return False


class EasyCallDataset(Dataset):
    """
    PyTorch dataset for EasyCall corpus.
    """

    def __init__(
        self,
        easycall_root: str,
        easycall_meta_csv: str,
        cache_path: str,
        tokenizer,
        target_sr: int = 16000,
        split: str = "train",
        max_speech_length: Optional[float] = None,  # in seconds
        use_epitran: bool = True,  # Convert text to phonemes using epitran
    ):
        """
        Args:
            easycall_root: Path to EasyCall root directory
            easycall_meta_csv: Path to easycall_meta.csv (speaker-level metadata with speaker, label, split, etc.)
            cache_path: Path to cache directory for storing extracted audio clips
            tokenizer: Tokenizer for converting phonemes to indices
            target_sr: Target sample rate
            max_speech_length: Maximum speech length in seconds (for truncation)
            split: Dataset split (e.g., "train", "test", "validation")
            use_epitran: Whether to convert text to phonemes using epitran (optional)
        """
        self.easycall_root = Path(easycall_root)
        self.cache_path = Path(cache_path)
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.split = split
        self.tokenizer = tokenizer
        self.use_epitran = use_epitran

        # Load speaker-level metadata from CSV
        df_meta = pd.read_csv(easycall_meta_csv)
        df_meta = df_meta[df_meta["label"].notna()]  # Filter missing labels
        self.speaker_to_label = dict(zip(df_meta["speaker"], df_meta["label"]))
        self.speaker_to_split = dict(zip(df_meta["speaker"], df_meta["split"]))
        self.speaker_to_sex = dict(zip(df_meta["speaker"], df_meta["sex"]))
        self.speaker_to_severity = dict(zip(df_meta["speaker"], df_meta["severity"]))

        self.split = split

        # Check if easycall_root is a zip file and extract if needed
        easycall_path = Path(self.easycall_root)
        if easycall_path.is_file() and easycall_path.suffix.lower() == ".zip":
            # Extract directly to cache_path (single folder)
            self.cache_path.mkdir(parents=True, exist_ok=True)

            # Check if cache directory already has wav files (already extracted)
            extracted_files = list(self.cache_path.glob("**/*.wav"))
            if len(extracted_files) == 0:
                logger.info(
                    f"Extracting zip file {easycall_path} to {self.cache_path}..."
                )
                try:
                    with zipfile.ZipFile(easycall_path, "r") as zip_ref:
                        zip_ref.extractall(self.cache_path)
                    logger.info(
                        f"Extraction complete. Extracted {len(list(self.cache_path.glob('**/*.wav')))} wav files."
                    )
                except Exception as e:
                    logger.error(f"Failed to extract zip file {easycall_path}: {e}")
                    raise
            else:
                logger.info(
                    f"Using existing extracted files in {self.cache_path} ({len(extracted_files)} wav files found)"
                )

            # Use cache_path as root (where files are extracted)
            self.easycall_root = self.cache_path

        # Scan EasyCall directory for audio files and create metadata
        self.metadata = []

        # Now easycall_root is always a directory (either original or extracted)
        # Scan directory for audio files
        for r, d, files in os.walk(self.easycall_root):
            for file in files:
                if file.endswith(".wav"):
                    speaker = file.split("_")[
                        0
                    ].strip()  # Extract speaker from filename

                    # Skip f04 (label unknown)
                    if speaker.lower() == "f04":
                        continue

                    # Get split from speaker mapping
                    speaker_split = self.speaker_to_split.get(speaker)
                    if not speaker_split:
                        continue

                    # Filter by split
                    if speaker_split != split:
                        continue

                    # Extract text from filename (everything after speaker_session)
                    parts = file.replace(".wav", "").split("_")
                    text = " ".join(parts[2:]) if len(parts) > 2 else ""

                    # Create metadata item with regular file path
                    item = {
                        "file": file,
                        "path": os.path.join(r, file),
                        "speaker_id": speaker,
                        "speaker": speaker,
                        "sex": self.speaker_to_sex.get(speaker),
                        "severity": self.speaker_to_severity.get(speaker),
                        "label": int(self.speaker_to_label[speaker]),
                        "split": speaker_split,
                        "text": text,
                    }
                    self.metadata.append(item)

        logger.info(f"Loaded {len(self.metadata)} items for split '{split}'")

        self.clip_paths_cache = {}

        # Initialize epitran if needed
        self.epitran_transliterator = None
        if self.use_epitran:
            try:
                import epitran

                # Use Italian for EasyCall (based on text examples)
                self.epitran_transliterator = epitran.Epitran("ita-Latn")
                logger.info("Initialized epitran transliterator for Italian")
            except ImportError:
                logger.warning(
                    "epitran not installed. Install with: pip install epitran"
                )
                self.use_epitran = False
            except Exception as e:
                logger.warning(f"Failed to initialize epitran: {e}")
                self.use_epitran = False

    def __len__(self):
        return len(self.metadata)

    def _construct_cache(self):
        """Construct all paths in the cache based on metadata."""
        for item in self.metadata:
            self._construct_clip_path(item)

    def _construct_clip_path(self, item) -> Path:
        """Get the path to the audio file (already extracted, just return the path)."""
        # Use file name as cache key
        cache_key = item.get("file", item.get("utt_id", f"item_{hash(str(item))}"))
        if cache_key in self.clip_paths_cache:
            return self.clip_paths_cache[cache_key]

        # Get path from metadata (already set during scanning)
        path = Path(item.get("path"))

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Cache the path
        self.clip_paths_cache[cache_key] = path
        return path

    def _text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phonemes using epitran and ipatok.
        Epitran converts text to IPA string, ipatok segments it into individual phonemes.

        Args:
            text: Input text string

        Returns:
            List of phoneme strings (IPA)
        """
        if self.use_epitran and self.epitran_transliterator:
            try:
                # Convert text to IPA phonemes using epitran
                ipa_string = self.epitran_transliterator.transliterate(text)

                # Use ipatok to segment IPA string into individual phonemes
                phonemes_list = ipatok_tokenise(ipa_string)

                if not phonemes_list:
                    logger.debug(
                        f"Epitran/ipatok returned empty phonemes for text '{text}', ipa_string: '{ipa_string}'"
                    )
                    return []

                # Join phonemes as space-separated text
                phonemes = " ".join(phonemes_list)
                return phonemes
            except Exception as e:
                logger.warning(
                    f"Epitran/ipatok conversion failed for text '{text}': {e}"
                )
                return []
        else:
            if not self.use_epitran:
                logger.debug(f"Epitran not enabled")
            if not self.epitran_transliterator:
                logger.debug(f"Epitran transliterator not initialized")

        return []

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - speech: Tensor of shape (T,)
                - speech_length: int, actual speech length
                - target: Tensor of phone indices
                - phones: String of IPA phones joined with spaces (if available)
                - text: String transcript
                - utt_id: String identifier
                - duration: Float, segment duration in seconds
                - speaker_id: String speaker identifier
                - label: int, speaker label
                - split: String split name
        """
        item = self.metadata[idx]

        # Load speech from cache
        speechpath = self._construct_clip_path(item)
        waveform, sr = torchaudio.load(str(speechpath))

        # Resample if necessary (should already be done, but double-check)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr

        # Ensure mono (should already be done, but double-check)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate if necessary
        if self.max_speech_length is not None:
            max_samples = int(self.max_speech_length * sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        # Process phones/phonemes
        phone_ipa: List[str] = []
        text = item.get("text", "")

        # Convert text to phonemes using epitran
        if text:
            if self.use_epitran:
                phone_ipa = self._text_to_phonemes(text)
                if not phone_ipa:
                    logger.warning(
                        f"Epitran returned empty phones for {item.get('file', 'unknown')}, text: '{text}'"
                    )
            else:
                # If epitran not enabled, log warning
                logger.warning(
                    f"No phones available for {item.get('file', 'unknown')}, text: {text}. Enable epitran with use_epitran=True"
                )
                phone_ipa = []
        else:
            logger.warning(f"No text available for {item.get('file', 'unknown')}")
            phone_ipa = []

        # Tokenize phones using tokenizer
        # phone_ipa is a space-separated string, split it back to list for tokenization
        phone_list = phone_ipa.split() if isinstance(phone_ipa, str) else phone_ipa
        target = self.tokenizer.tokens2ids(phone_list) if phone_list else []
        if phone_list and len(target) == 0:
            logger.warning(
                f"Tokenization returned empty target for {item.get('file', 'unknown')}, phones: {phone_ipa}"
            )

        return {
            "speech": waveform.squeeze(0),  # Shape: (T,)
            "speech_length": waveform.shape[1],
            "target": (
                torch.tensor(target, dtype=torch.long)
                if target
                else torch.tensor([], dtype=torch.long)
            ),
            "target_length": len(target),
            "phones": (
                phone_ipa
                if isinstance(phone_ipa, str)
                else " ".join(phone_ipa) if phone_ipa else ""
            ),  # Space-separated string of phones
            "text": text,
            "utt_id": item.get("file", item.get("utt_id", f"utt_{idx}")),
            "duration": item.get("duration", 0.0),
            "speaker_id": item.get("speaker_id", "unknown"),
            "label": item.get("label", -1),
            "split": self.split,
        }


def collate_fn(batch):
    """
    Custom collate function for batching variable-length sequences.
    No phone boundaries (unlike Buckeye).

    Returns:
        dict with keys:
            - speech: Tensor of shape (batch_size, max_speech_length)
            - speech_length: Tensor of shape (batch_size,), actual lengths
            - target: Tensor of shape (batch_size, max_target_length)
            - target_length: Tensor of shape (batch_size,), actual lengths
            - target_text: list[list[str]] phones (list of phone lists, matching Buckeye format)
            - utt_id: list[str]
            - split: list[str]
            - label: list[int] (speaker labels)
    """
    # Find max lengths
    max_speech_length = max(item["speech_length"] for item in batch)
    max_target_length = max(len(item["target"]) for item in batch) if batch else 0

    # Initialize tensors with -1 padding
    batch_size = len(batch)
    speech = torch.full((batch_size, max_speech_length), -1.0, dtype=torch.float32)
    speech_length = torch.zeros(batch_size, dtype=torch.long)
    phone_id = torch.full((batch_size, max_target_length), -1, dtype=torch.long)
    target_length = torch.zeros(batch_size, dtype=torch.long)

    # Fill tensors
    for i, item in enumerate(batch):
        speech_len = item["speech_length"]
        phone_len = len(item["target"])

        speech[i, :speech_len] = item["speech"]
        speech_length[i] = speech_len
        if phone_len > 0:
            phone_id[i, :phone_len] = item["target"]
        target_length[i] = phone_len

    return {
        "speech": speech,
        "speech_length": speech_length,
        "target": phone_id,
        "target_text": [item["phones"] for item in batch],
        "target_length": target_length,
        "utt_id": [item["utt_id"] for item in batch],
        "text": [item.get("text", "") for item in batch],
        "split": [item.get("split", "unknown") for item in batch],
        "label": [item.get("label", -1) for item in batch],
    }


class EasyCallDataModule(L.LightningDataModule):
    def __init__(
        self,
        easycall_root: str,
        local_cache_path: str,
        easycall_meta_csv: str,
        tokenizer=None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        max_speech_length: Optional[float] = None,
        use_epitran: bool = True,
    ):
        """
        Args:
            easycall_root: Path to EasyCall root directory
            local_cache_path: Path to cache directory for metadata and extracted data
            easycall_meta_csv: Path to easycall_meta.csv (speaker-level metadata)
            tokenizer: Tokenizer for converting phonemes to indices
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in dataloaders
            target_sr: Target sample rate
            max_speech_length: Maximum speech length in seconds (for truncation)
            use_epitran: Whether to convert text to phonemes using epitran (optional)
        """
        super().__init__()
        self.easycall_root = Path(easycall_root)
        self.local_cache_path = Path(local_cache_path)
        self.local_cache_path.mkdir(parents=True, exist_ok=True)
        self.easycall_meta_csv = easycall_meta_csv

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.use_epitran = use_epitran

    def setup(self, stage: Optional[str] = None):
        # Check if CSV file exists
        if not Path(self.easycall_meta_csv).exists():
            logger.warning(f"CSV metadata file not found at {self.easycall_meta_csv}")
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            return

        # Create datasets for each split
        # The dataset class scans the directory and filters by split internally
        self.train_dataset = EasyCallDataset(
            easycall_root=str(self.easycall_root),
            easycall_meta_csv=self.easycall_meta_csv,
            cache_path=str(self.local_cache_path),
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            split="train",
            use_epitran=self.use_epitran,
        )

        self.val_dataset = EasyCallDataset(
            easycall_root=str(self.easycall_root),
            easycall_meta_csv=self.easycall_meta_csv,
            cache_path=str(self.local_cache_path),
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            split="validation",
            use_epitran=self.use_epitran,
        )

        self.test_dataset = EasyCallDataset(
            easycall_root=str(self.easycall_root),
            easycall_meta_csv=self.easycall_meta_csv,
            cache_path=str(self.local_cache_path),
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            split="test",
            use_epitran=self.use_epitran,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError(
                "Train dataset not available. Check train_metadata.json exists."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset not available. Check val_metadata.json exists."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError(
                "Test dataset not available. Check test_metadata.json exists."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        datasets = []
        if self.train_dataset is not None:
            datasets.append(self.train_dataset)
        if self.val_dataset is not None:
            datasets.append(self.val_dataset)
        if self.test_dataset is not None:
            datasets.append(self.test_dataset)

        if not datasets:
            raise ValueError("No datasets available for prediction.")

        return DataLoader(
            ConcatDataset(datasets),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    """Example usage"""
    from src.data.easycall.dysarthria_severity_classification import EasyCallDataModule
    from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easycall_root",
        type=str,
        required=True,
        help="Path to EasyCall root directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory for cache (extracted audio clips)",
    )
    parser.add_argument(
        "--easycall_meta_csv",
        type=str,
        default="src/data/easycall/easycall_meta.csv",
        help="Path to easycall_meta.csv (speaker-level metadata)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--extract_all", action="store_true", help="Extract all audio files upfront"
    )
    parser.add_argument(
        "--extract_split",
        type=str,
        nargs="+",
        choices=["train", "valid", "test", "validation"],
        help="Extract specific split(s): train, valid, test (can specify multiple)",
    )
    parser.add_argument(
        "--show_examples", action="store_true", help="Show 5 examples from test_loader"
    )

    args = parser.parse_args()

    MODEL = "w2v2ph"
    if MODEL == "w2v2ph":
        tokenizer = build_wav2vec2phoneme_tokenizer(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )

    # Create dataloaders
    data_module = EasyCallDataModule(
        easycall_root=args.easycall_root,
        local_cache_path=args.data_dir,
        easycall_meta_csv=args.easycall_meta_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Print 5 examples if requested
    if args.show_examples:
        print("\n" + "=" * 60)
        print("Printing 5 examples from test_loader:")
        print("=" * 60)

        example_count = 0
        for batch_idx, batch in enumerate(test_loader):
            batch_size = len(batch["utt_id"])
            for i in range(batch_size):
                if example_count >= 5:
                    break

                example_dict = {
                    "utt_id": batch["utt_id"][i],
                    "text": batch.get("text", [""])[i] if "text" in batch else "",
                    "label": batch["label"][i],
                    "split": batch["split"][i],
                    "speech_length": batch["speech_length"][i].item(),
                    "target_length": batch["target_length"][i].item(),
                    "phones": batch["target_text"][i],
                    "target": (
                        batch["target"][i][: batch["target_length"][i]].tolist()
                        if batch["target_length"][i] > 0
                        else []
                    ),
                    "speech_shape": tuple(batch["speech"][i].shape),
                    "target_shape": tuple(batch["target"][i].shape),
                }
                print(f"\nExample {example_count + 1}:")
                print(example_dict)

                example_count += 1

            if example_count >= 5:
                break

        print("\n" + "=" * 60 + "\n")
