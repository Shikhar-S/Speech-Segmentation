"""UASpeech Dataset and DataLoader for forced alignment

Usage:
    # Option 1: Extract files first, then use datamodule
    python -m src.data.uaspeech.common_datamodule \
        --data_dir /data/user_data/eyeo2/data/UASpeech_noisereduce \
        --tgz_file_C /data/user_data/eyeo2/data/UASpeech_noisereduce_C.tgz \
        --tgz_file_D /data/user_data/eyeo2/data/UASpeech_noisereduce_FM.tgz \
        --extract_only
    
    python -m src.data.uaspeech.common_datamodule \
        --data_dir /data/user_data/eyeo2/data/UASpeech_noisereduce \
        --uaspeech_meta_csv src/data/uaspeech/uaspeech_meta.csv \
        --uaspeech_wordlist_csv src/data/uaspeech/uaspeech_wordlist.csv
    
    # Option 2: Let datamodule extract automatically (if files don't exist)
    python -m src.data.uaspeech.common_datamodule \
        --data_dir /data/user_data/eyeo2/data/UASpeech_noisereduce \
        --tgz_file_C /data/user_data/eyeo2/data/UASpeech_noisereduce_C.tgz \
        --tgz_file_D /data/user_data/eyeo2/data/UASpeech_noisereduce_FM.tgz \
        --uaspeech_meta_csv src/data/uaspeech/uaspeech_meta.csv \
        --uaspeech_wordlist_csv src/data/uaspeech/uaspeech_wordlist.csv

Note: 
    - If files are already extracted in data_dir, tgz_file_C and tgz_file_D are optional.
    - --uaspeech_meta_csv and --uaspeech_wordlist_csv have defaults and are optional.
    - Use --extract_only to only extract files without creating dataloaders.
    - Use --force_extract to re-extract even if files already exist.
"""

import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional, List
import tarfile
import tempfile
import epitran
from ipatok import tokenise as ipatok_tokenise


def extract_uaspeech_tgz_files(
    output_dir: str,
    tgz_file_C: str,
    tgz_file_D: str,
    force: bool = False,
):
    """
    Extract UASpeech tgz archive files to output directory.
    
    Args:
        output_dir: Directory to extract files to
        tgz_file_C: Path to first tgz archive file
        tgz_file_D: Path to second tgz archive file
        force: If True, extract even if files already exist
    
    Returns:
        Number of wav files extracted
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tgz_C = Path(tgz_file_C)
    tgz_D = Path(tgz_file_D)
    
    # Check if files already extracted
    extracted_files = list(output_path.glob("**/*.wav"))
    if len(extracted_files) > 0 and not force:
        logger.info(f"Found {len(extracted_files)} wav files in {output_path}. Skipping extraction.")
        logger.info("Set force=True to re-extract.")
        return len(extracted_files)
    
    # Extract first tgz file
    if not tgz_C.exists():
        raise FileNotFoundError(f"Tgz file not found: {tgz_C}")
    
    logger.info(f"Extracting {tgz_C} to {output_path}...")
    try:
        with tarfile.open(tgz_C, 'r:gz') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f"Extracting {tgz_C.name}"):
                tar_ref.extract(member, output_path)
        logger.info(f"Extraction complete for {tgz_C}")
    except Exception as e:
        logger.error(f"Failed to extract tgz file {tgz_C}: {e}")
        raise
    
    # Extract second tgz file
    if not tgz_D.exists():
        raise FileNotFoundError(f"Tgz file not found: {tgz_D}")
    
    logger.info(f"Extracting {tgz_D} to {output_path}...")
    try:
        with tarfile.open(tgz_D, 'r:gz') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f"Extracting {tgz_D.name}"):
                tar_ref.extract(member, output_path)
        logger.info(f"Extraction complete for {tgz_D}")
    except Exception as e:
        logger.error(f"Failed to extract tgz file {tgz_D}: {e}")
        raise
    
    # Count extracted files
    extracted_count = len(list(output_path.glob("**/*.wav")))
    logger.info(f"Extraction complete. Extracted {extracted_count} wav files to {output_path}")
    return extracted_count


import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_uaspeech_clip(
    uaspeech_root: Path,
    item: dict,
    out: Path,
    sr: int
) -> bool:
    """
    Extract and cache UASpeech audio clip.
    
    Supports both regular .wav files and .tgz files containing audio.
    If the source is a tgz file, extracts the specific audio file from it.
    
    Args:
        uaspeech_root: Path to UASpeech root directory
        item: Metadata item with 'path', 'tgz_path', 'file', or 'tgz_file' keys
        out: Output path for cached audio
        sr: Target sample rate
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if we're dealing with a tgz file
        tgz_path = item.get("tgz_path") or item.get("tgz_file")
        file_in_tgz = item.get("file")
        
        if tgz_path:
            # Extract from tgz file
            if isinstance(tgz_path, str):
                tgz_path = Path(tgz_path)
            if not tgz_path.is_absolute():
                tgz_path = uaspeech_root / tgz_path
            
            if not tgz_path.exists():
                logger.warning(f"Tgz file not found: {tgz_path}")
                return False
            
            # Extract the specific file from tgz
            tgz_entry = item.get("tgz_entry")
            with tarfile.open(tgz_path, 'r:gz') as tar_ref:
                # Use tgz_entry if available, otherwise search for the file
                file_to_extract = tgz_entry
                if file_to_extract is None:
                    # Find the file in the tgz (handle different path formats)
                    for name in tar_ref.getnames():
                        if name.endswith(file_in_tgz) or os.path.basename(name) == file_in_tgz:
                            file_to_extract = name
                            break
                
                if file_to_extract is None or file_to_extract not in tar_ref.getnames():
                    logger.warning(f"File {file_in_tgz} not found in tgz {tgz_path}")
                    return False
                
                # Extract to temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Extract the file (preserves directory structure if any)
                    tar_ref.extract(file_to_extract, tmp_dir)
                    # Construct full path to extracted file
                    extracted_path = os.path.join(tmp_dir, file_to_extract)
                    
                    # Load and process audio
                    wav, s = torchaudio.load(extracted_path)
        else:
            # Regular file path
            source_path = item.get("path") or (uaspeech_root / item.get("file", ""))
            if isinstance(source_path, str):
                source_path = Path(source_path)
            if not source_path.is_absolute():
                source_path = uaspeech_root / source_path
            
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
        logger.warning(
            f"speech extraction failed [{item.get('file', 'unknown')}]: {e}"
        )
        return False


class UASpeechDataset(Dataset):
    """
    PyTorch dataset for UASpeech corpus alignment evaluation.
    """

    def __init__(
        self,
        uaspeech_meta_csv: str,
        cache_path: str,
        tokenizer,
        tgz_file1: str,
        tgz_file2: str,
        uaspeech_wordlist_csv: str,
        target_sr: int = 16000,
        split: str = "train",
        max_speech_length: Optional[float] = None,  # in seconds
        use_epitran: bool = True,  # Convert text to phonemes using epitran
    ):
        """
        Args:
            uaspeech_meta_csv: Path to uaspeech_meta.csv (speaker-level metadata with speaker, label, split, etc.)
            cache_path: Path to cache directory for storing extracted audio clips
            tokenizer: Tokenizer for converting phonemes to indices
            tgz_file1: Path to first tgz archive file (will extract to cache_path)
            tgz_file2: Path to second tgz archive file (will extract to cache_path)
            uaspeech_wordlist_csv: Path to uaspeech_wordlist.csv (maps FILE_NAME to WORD)
            target_sr: Target sample rate
            max_speech_length: Maximum speech length in seconds (for truncation)
            split: Dataset split (e.g., "train", "test", "validation")
            use_epitran: Whether to convert text to phonemes using epitran (optional)
        """
        self.cache_path = Path(cache_path)
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.split = split
        self.tokenizer = tokenizer
        self.use_epitran = use_epitran
        self.tgz_file1 = Path(tgz_file1) if tgz_file1 else None
        self.tgz_file2 = Path(tgz_file2) if tgz_file2 else None

        # Load speaker-level metadata from CSV
        df_meta = pd.read_csv(uaspeech_meta_csv)
        df_meta = df_meta[df_meta['severity'].notna()]  # Filter missing severity
        # Use severity as label
        self.speaker_to_label = dict(zip(df_meta['speaker'], df_meta['severity']))
        self.speaker_to_split = dict(zip(df_meta['speaker'], df_meta['split']))
        self.speaker_to_sex = dict(zip(df_meta['speaker'], df_meta['sex']))
        self.speaker_to_severity = dict(zip(df_meta['speaker'], df_meta['severity']))
        
        # Load wordlist CSV and create mapping dictionary
        df_wordlist = pd.read_csv(uaspeech_wordlist_csv)
        self.word_file_dict = dict(zip(df_wordlist["FILE_NAME"], df_wordlist["WORD"]))
        
        self.split = split
        
        # Extract tgz files to cache_path using the extraction function (if tgz files provided)
        if self.tgz_file1 and self.tgz_file2:
            extract_uaspeech_tgz_files(
                output_dir=str(self.cache_path),
                tgz_file_C=str(self.tgz_file1),
                tgz_file_D=str(self.tgz_file2),
                force=False,
            )
        else:
            # Check if files already exist
            extracted_files = list(self.cache_path.glob("**/*.wav"))
            if len(extracted_files) == 0:
                raise ValueError(
                    "No extracted files found in data_dir and no tgz files provided. "
                    "Either provide --tgz_file_C and --tgz_file_D, or ensure files are already extracted."
                )
            logger.info(f"Using existing extracted files in {self.cache_path} ({len(extracted_files)} wav files found)")
        
        # Use cache_path as root (where files are extracted)
        # Files will be in subdirectories like audio/noisereduce/
        self.uaspeech_root = self.cache_path
        
        # Scan UASpeech directory for audio files and create metadata
        self.metadata = []
        
        # Now uaspeech_root is always a directory (either original or extracted)
        # Scan directory recursively for audio files (handles nested structure like audio/noisereduce/)
        for r, d, files in os.walk(self.uaspeech_root):
            for file in files:
                if file.endswith(".wav"):
                    # Extract speaker from filename (adjust based on UASpeech naming convention)
                    # Common format: speaker_session_word.wav or similar
                    speaker = file.split("_")[0].strip() if "_" in file else file.split(".")[0]
                    
                    # Get split from speaker mapping
                    speaker_split = self.speaker_to_split.get(speaker)
                    if not speaker_split:
                        continue
                    
                    # Filter by split
                    if speaker_split != split:
                        continue
                    
                    # Extract text from filename using map_word_to_file_name function
                    text = self._map_word_to_file_name(file, self.word_file_dict)
                    
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
                # Use English for UASpeech
                self.epitran_transliterator = epitran.Epitran('eng-Latn')
                logger.info("Initialized epitran transliterator for English")
            except ImportError:
                logger.warning("epitran not installed. Install with: pip install epitran")
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

    def _map_word_to_file_name(self, file_name: str, word_file_dict: dict) -> str:
        """
        Map UASpeech filename to word using wordlist dictionary.
        
        Args:
            file_name: Audio filename (e.g., "M01_B1_D3.wav")
            word_file_dict: Dictionary mapping FILE_NAME to WORD
            
        Returns:
            Word string corresponding to the filename
        """
        utt_id = file_name.split("_")[2].replace(".wav", "")
        if "UW" not in utt_id:
            return word_file_dict.get(utt_id, "")
        else:
            utt_id = file_name.split("_")[1] + "_" + file_name.split("_")[2].replace(".wav", "")
            return word_file_dict.get(utt_id, "")
    
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

    def _text_to_phonemes(self, text: str) -> str:
        """
        Convert text to phonemes using epitran and ipatok.
        Epitran converts text to IPA string, ipatok segments it into individual phonemes.
        
        Args:
            text: Input text string
            
        Returns:
            Space-separated string of phoneme strings (IPA)
        """
        if self.use_epitran and self.epitran_transliterator:
            try:
                # Convert text to IPA phonemes using epitran
                ipa_string = self.epitran_transliterator.transliterate(text)
                
                # Use ipatok to segment IPA string into individual phonemes
                phonemes_list = ipatok_tokenise(ipa_string)
                
                if not phonemes_list:
                    logger.debug(f"Epitran/ipatok returned empty phonemes for text '{text}', ipa_string: '{ipa_string}'")
                    return ""
                
                # Join phonemes as space-separated text
                phonemes = " ".join(phonemes_list)
                return phonemes
            except Exception as e:
                logger.warning(f"Epitran/ipatok conversion failed for text '{text}': {e}")
                return ""
        else:
            if not self.use_epitran:
                logger.debug(f"Epitran not enabled")
            if not self.epitran_transliterator:
                logger.debug(f"Epitran transliterator not initialized")
        
        return ""

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
        phone_ipa: str = ""
        text = item.get("text", "")
        
        # Convert text to phonemes using epitran
        if text:
            if self.use_epitran:
                phone_ipa = self._text_to_phonemes(text)
                if not phone_ipa:
                    logger.warning(f"Epitran returned empty phones for {item.get('file', 'unknown')}, text: '{text}'")
            else:
                # If epitran not enabled, log warning
                logger.warning(f"No phones available for {item.get('file', 'unknown')}, text: {text}. Enable epitran with use_epitran=True")
                phone_ipa = ""
        else:
            logger.warning(f"No text available for {item.get('file', 'unknown')}")
            phone_ipa = ""
        
        # Tokenize phones using tokenizer
        # phone_ipa is a space-separated string, split it back to list for tokenization
        phone_list = phone_ipa.split() if phone_ipa else []
        target = self.tokenizer.tokens2ids(phone_list) if phone_list else []
        if phone_list and len(target) == 0:
            logger.warning(f"Tokenization returned empty target for {item.get('file', 'unknown')}, phones: {phone_ipa}")

        return {
            "speech": waveform.squeeze(0),  # Shape: (T,)
            "speech_length": waveform.shape[1],
            "target": torch.tensor(target, dtype=torch.long) if target else torch.tensor([], dtype=torch.long),
            "target_length": len(target),
            "phones": phone_ipa,  # Space-separated string of phones
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
            - target_text: list[str] phones (list of phone strings)
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


class UASpeechDataModule(L.LightningDataModule):
    def __init__(
        self,
        local_cache_path: str,
        uaspeech_meta_csv: str,
        uaspeech_wordlist_csv: str,
        tokenizer=None,
        tgz_file1: str = None,
        tgz_file2: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        max_speech_length: Optional[float] = None,
        use_epitran: bool = True,
    ):
        """
        Args:
            local_cache_path: Path to cache directory for extracted data
            uaspeech_meta_csv: Path to uaspeech_meta.csv (speaker-level metadata)
            uaspeech_wordlist_csv: Path to uaspeech_wordlist.csv (maps FILE_NAME to WORD)
            tokenizer: Tokenizer for converting phonemes to indices
            tgz_file1: Path to first tgz archive file (will extract to local_cache_path)
            tgz_file2: Path to second tgz archive file (will extract to local_cache_path)
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in dataloaders
            target_sr: Target sample rate
            max_speech_length: Maximum speech length in seconds (for truncation)
            use_epitran: Whether to convert text to phonemes using epitran (optional)
        """
        super().__init__()
        self.local_cache_path = Path(local_cache_path)
        self.local_cache_path.mkdir(parents=True, exist_ok=True)
        self.uaspeech_meta_csv = uaspeech_meta_csv
        self.uaspeech_wordlist_csv = uaspeech_wordlist_csv
        self.tgz_file1 = tgz_file1
        self.tgz_file2 = tgz_file2
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.use_epitran = use_epitran

    def setup(self, stage: Optional[str] = None):
        # Check if CSV file exists
        if not Path(self.uaspeech_meta_csv).exists():
            logger.warning(f"CSV metadata file not found at {self.uaspeech_meta_csv}")
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            return

        # Create datasets for each split
        # The dataset class scans the directory and filters by split internally
        self.train_dataset = UASpeechDataset(
            uaspeech_meta_csv=self.uaspeech_meta_csv,
            cache_path=str(self.local_cache_path),
            tokenizer=self.tokenizer,
            tgz_file1=self.tgz_file1,
            tgz_file2=self.tgz_file2,
            uaspeech_wordlist_csv=self.uaspeech_wordlist_csv,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            split="train",
            use_epitran=self.use_epitran,
        )

        self.val_dataset = UASpeechDataset(
            uaspeech_meta_csv=self.uaspeech_meta_csv,
            cache_path=str(self.local_cache_path),
            tokenizer=self.tokenizer,
            tgz_file1=self.tgz_file1,
            tgz_file2=self.tgz_file2,
            uaspeech_wordlist_csv=self.uaspeech_wordlist_csv,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            split="validation",
            use_epitran=self.use_epitran,
        )

        self.test_dataset = UASpeechDataset(
            uaspeech_meta_csv=self.uaspeech_meta_csv,
            cache_path=str(self.local_cache_path),
            tokenizer=self.tokenizer,
            tgz_file1=self.tgz_file1,
            tgz_file2=self.tgz_file2,
            uaspeech_wordlist_csv=self.uaspeech_wordlist_csv,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            split="test",
            use_epitran=self.use_epitran,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not available. Check uaspeech_meta.csv exists.")
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
            raise ValueError("Validation dataset not available. Check uaspeech_meta.csv exists.")
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
            raise ValueError("Test dataset not available. Check uaspeech_meta.csv exists.")
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
    from src.data.uaspeech.common_datamodule import UASpeechDataModule
    from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory for cache (extracted audio clips)",
    )
    parser.add_argument(
        "--uaspeech_meta_csv",
        type=str,
        default="src/data/uaspeech/uaspeech_meta.csv",
        help="Path to uaspeech_meta.csv (speaker-level metadata)",
    )
    parser.add_argument(
        "--uaspeech_wordlist_csv",
        type=str,
        default="src/data/uaspeech/uaspeech_wordlist.csv",
        help="Path to uaspeech_wordlist.csv (maps FILE_NAME to WORD)",
    )
    parser.add_argument(
        "--tgz_file_C",
        type=str,
        default=None,
        help="Path to control tgz archive file",
    )
    parser.add_argument(
        "--tgz_file_D",
        type=str,
        default=None,
        help="Path to dys tgz archive file",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--show_examples", action="store_true", help="Show 5 examples from test_loader")
    parser.add_argument("--extract_only", action="store_true", help="Only extract tgz files, don't create dataloaders")
    parser.add_argument("--force_extract", action="store_true", help="Force extraction even if files already exist")

    args = parser.parse_args()
    
    # If extract_only flag is set, just extract and exit
    if args.extract_only:
        if not args.tgz_file_C or not args.tgz_file_D:
            raise ValueError("--tgz_file_C and --tgz_file_D are required for extraction")
        extract_uaspeech_tgz_files(
            output_dir=args.data_dir,
            tgz_file_C=args.tgz_file_C,
            tgz_file_D=args.tgz_file_D,
            force=args.force_extract,
        )
        print("Extraction complete!")
        exit(0)

    MODEL = "w2v2ph"
    if MODEL == "w2v2ph":
        tokenizer = build_wav2vec2phoneme_tokenizer(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )

    # Create dataloaders
    data_module = UASpeechDataModule(
        local_cache_path=args.data_dir,
        uaspeech_meta_csv=args.uaspeech_meta_csv,
        uaspeech_wordlist_csv=args.uaspeech_wordlist_csv,
        tokenizer=tokenizer,
        tgz_file1=args.tgz_file_C,
        tgz_file2=args.tgz_file_D,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Print 5 examples if requested
    if args.show_examples:
        print("\n" + "="*60)
        print("Printing 5 examples from test_loader:")
        print("="*60)
        
        example_count = 0
        for batch_idx, batch in enumerate(test_loader):
            batch_size = len(batch['utt_id'])
            for i in range(batch_size):
                if example_count >= 5:
                    break
                
                example_dict = {
                    "utt_id": batch['utt_id'][i],
                    "text": batch.get('text', [''])[i] if 'text' in batch else '',
                    "label": batch['label'][i],
                    "split": batch['split'][i],
                    "speech_length": batch['speech_length'][i].item(),
                    "target_length": batch['target_length'][i].item(),
                    "phones": batch['target_text'][i],
                    "target": batch['target'][i][:batch['target_length'][i]].tolist() if batch['target_length'][i] > 0 else [],
                    "speech_shape": tuple(batch['speech'][i].shape),
                    "target_shape": tuple(batch['target'][i].shape),
                }
                print(f"\nExample {example_count + 1}:")
                print(example_dict)
                
                example_count += 1
            
            if example_count >= 5:
                break
        
        print("\n" + "="*60 + "\n")
