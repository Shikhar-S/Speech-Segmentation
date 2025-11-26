"""TIMIT DataModule for forced alignment and phone recognition tasks.

Usage:
    python -m src.data.timit.common_datamodule \
           --timit_root /work/hdd/bbjs/shared/corpora/TIMIT/timit_nltk \
           --data_dir /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/timit_cache
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L

from src.core.ipa_utils import ARPABET_TO_IPA

logger = logging.getLogger(__name__)


class TimitDataset(Dataset):
    def __init__(
        self,
        timit_root: str,
        metadata_path: str,
        tokenizer,
        target_sr: int = 16000,
        mask_probability: float = 0.0,
        split: str = "train",
        max_speech_length: Optional[float] = None,  # in seconds
    ):
        """
        Args:
            timit_root: Path to TIMIT root
            metadata_path: Path to JSON metadata file created by timit_data_prep
            tokenizer: Tokenizer for converting phonemes to indices
            target_sr: Target sample rate (TIMIT is 16kHz; kept here for symmetry)
            mask_probability: Percentage of phones to mask with noise
            max_speech_length: Maximum speech length in seconds (for truncation)
            split: Dataset split name ("train", "val", "test")
        """
        self.timit_root = Path(timit_root)
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.split = split
        self.mask_probability = mask_probability
        self.tokenizer = tokenizer

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # For quick lookup if needed
        self._sr = target_sr

    def __len__(self):
        return len(self.metadata)

    def _load_waveform(self, segment_id: str) -> Tuple[torch.Tensor, int]:
        """
        segment_id is like 'dr1-fvmh0/sx206', so wav path is timit_root/segment_id.wav
        """
        wav_path = self.timit_root / f"{segment_id}.wav"
        waveform, sr = torchaudio.load(str(wav_path))

        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sr

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - speech: Tensor of shape (T,)
                - speech_length: int, number of samples
                - target: Tensor of phone indices
                - target_length: int
                - phone_pointstamps: List of (start, end) sample indices
                - phone_timestamps: List of (start, end) seconds (from metadata)
                - phones: List of IPA phones
                - text: String transcript
                - utt_id: segment id (e.g. 'dr1-fvmh0/sx206')
                - duration: Float duration in seconds
                - speaker_id: speaker identifier
                - split: split name
        """
        item = self.metadata[idx]

        segment_id = item["segment_id"]
        waveform, sr = self._load_waveform(segment_id)

        # Truncate if necessary
        if self.max_speech_length is not None:
            max_samples = int(self.max_speech_length * sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        # Convert timestamps (sec) -> pointstamps (samples)
        phone_pointstamps: List[Tuple[int, int]] = []
        phone_ipa: List[str] = []

        # metadata stores phone_timestamps in seconds
        for phone, (start_sec, end_sec) in zip(
            item["phones"], item["phone_timestamps"]
        ):
            start_idx = int(start_sec * self.target_sr)
            end_idx = int(end_sec * self.target_sr)

            phone_pointstamps.append((start_idx, end_idx))
            phone_ipa.append(ARPABET_TO_IPA.get(phone.lower(), phone.lower()))
            print(phone, " --> ", phone_ipa[-1])

        masked_duration = 0.0
        atleast_one = False
        if self.mask_probability > 0.0:
            for start_idx, end_idx in phone_pointstamps:
                if np.random.rand() < self.mask_probability and atleast_one:
                    waveform = waveform.clone()
                    seg_len = end_idx - start_idx
                    waveform[:, start_idx:end_idx] = torch.randn(1, seg_len)
                    masked_duration += seg_len / self.target_sr
                atleast_one = True

        target = self.tokenizer.tokens2ids(phone_ipa)
        assert len(target) != 0, f"No valid phones for {segment_id}."

        return {
            "speech": waveform.squeeze(0),  # Shape: (T,)
            "speech_length": waveform.shape[1],
            "target": torch.tensor(target, dtype=torch.long),
            "target_length": len(target),
            "phone_pointstamps": phone_pointstamps,
            "phone_timestamps": item["phone_timestamps"],
            "phones": phone_ipa,
            "text": item["text"],
            "utt_id": segment_id,
            "duration": item.get("duration", item["end_time"] - item["start_time"]),
            "speaker_id": item["speaker_id"],
            "split": self.split,
        }


def collate_fn(batch):
    """
    Custom collate function for batching variable-length sequences.

    Returns:
        dict with keys:
            - speech: (B, max_T)
            - speech_length: (B,)
            - target: (B, max_L)
            - target_length: (B,)
            - target_start: (B, max_L) in samples
            - target_end: (B, max_L) in samples
            - target_text: list[list[str]] phones
            - ground_truth_timestamps: list[list[(start_sec, end_sec)]]
            - utt_id: list[str]
            - split: list[str]
    """
    max_speech_length = max(item["speech_length"] for item in batch)
    max_target_length = max(len(item["target"]) for item in batch)

    batch_size = len(batch)
    speech = torch.full((batch_size, max_speech_length), -1.0, dtype=torch.float32)
    speech_length = torch.zeros(batch_size, dtype=torch.long)
    phone_id = torch.full((batch_size, max_target_length), -1, dtype=torch.long)
    target_length = torch.zeros(batch_size, dtype=torch.long)
    target_start = torch.full((batch_size, max_target_length), -1, dtype=torch.float32)
    target_end = torch.full((batch_size, max_target_length), -1, dtype=torch.float32)

    for i, item in enumerate(batch):
        speech_len = item["speech_length"]
        phone_len = len(item["target"])

        speech[i, :speech_len] = item["speech"]
        speech_length[i] = speech_len
        phone_id[i, :phone_len] = item["target"]
        target_length[i] = phone_len

        # pointstamps in samples
        pointstamps = item["phone_pointstamps"]
        for j, (start, end) in enumerate(pointstamps[:phone_len]):
            target_start[i, j] = start
            target_end[i, j] = end

    return {
        "speech": speech,
        "speech_length": speech_length,
        "target": phone_id,
        "target_text": [item["phones"] for item in batch],
        "target_length": target_length,
        "target_start": target_start,
        "target_end": target_end,
        "ground_truth_timestamps": [item["phone_timestamps"] for item in batch],
        "utt_id": [item["utt_id"] for item in batch],
        "split": [item.get("split", "unknown") for item in batch],
    }


class TimitDataModule(L.LightningDataModule):
    def __init__(
        self,
        timit_root: str,
        local_cache_path: str,
        tokenizer,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        mask_probability: float = 0.0,
        max_speech_length: Optional[float] = None,
    ):
        super().__init__()
        self.timit_root = timit_root
        self.local_cache_path = Path(local_cache_path)
        self.train_metadata = self.local_cache_path / "train_metadata.json"
        self.val_metadata = self.local_cache_path / "val_metadata.json"
        self.test_metadata = self.local_cache_path / "test_metadata.json"

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.mask_probability = mask_probability
        self.max_speech_length = max_speech_length

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = TimitDataset(
            timit_root=self.timit_root,
            metadata_path=str(self.train_metadata),
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            mask_probability=self.mask_probability,
            max_speech_length=self.max_speech_length,
            split="train",
        )
        self.val_dataset = TimitDataset(
            timit_root=self.timit_root,
            metadata_path=str(self.val_metadata),
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            mask_probability=self.mask_probability,
            max_speech_length=self.max_speech_length,
            split="val",
        )
        self.test_dataset = TimitDataset(
            timit_root=self.timit_root,
            metadata_path=str(self.test_metadata),
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            mask_probability=self.mask_probability,
            max_speech_length=self.max_speech_length,
            split="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    # Example standalone usage, mirroring the Buckeye script
    from tqdm import tqdm
    from src.model.powsm.token_id_converter import build_powsm_tokenizer
    from src.model.wav2vec2phoneme.builders import (
        build_wav2vec2phoneme_tokenizer,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with processed TIMIT data (metadata JSONs)",
    )
    parser.add_argument(
        "--timit_root",
        type=str,
        required=True,
        help="Path to TIMIT root",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    # MODEL = "w2v2ph"
    MODEL = "powsm"
    if MODEL == "powsm":
        tokenizer = build_powsm_tokenizer(
            work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
            hf_repo="espnet/powsm",
        )
    elif MODEL == "w2v2ph":
        tokenizer = build_wav2vec2phoneme_tokenizer(
            hf_repo="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
        )

    dm = TimitDataModule(
        timit_root=args.timit_root,
        local_cache_path=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Quick sanity check
    batch = next(iter(test_loader))
    print("Batch keys:", batch.keys())
    print(batch)
