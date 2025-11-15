"""
Buckeye Dataset and DataLoader for forced alignment

Usage:
    python -m src.data.buckeye.forced_alignment \
        --buckeye_root /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye \
        --data_dir /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/buckeye_cache \
        --batch_size 32 --num_workers 4
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging
import buckeye
import lightning as L

logger = logging.getLogger(__name__)

IPA_TO_ARPABET_EPITRAN = {
    "ɑ": "AA",
    "æ": "AE",
    "ʌ": "AH",  # unstressed “uh”; schwa is AX
    "ɔ": "AO",
    "aʊ": "AW",
    "aɪ": "AY",
    "b": "B",
    "tʃ": "CH",
    "d": "D",
    "ð": "DH",
    "ɛ": "EH",
    "ɚ": "AXR",  # r-colored schwa
    "ɝ": "ER",  # stressed r-colored vowel
    "eɪ": "EY",
    "f": "F",
    "ɡ": "G",
    "h": "HH",
    "ɪ": "IH",
    "i": "IY",
    "dʒ": "JH",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "oʊ": "OW",
    "ɔɪ": "OY",
    "p": "P",
    "ɹ": "R",
    "s": "S",
    "ʃ": "SH",
    "t": "T",
    "θ": "TH",
    "ʊ": "UH",
    "u": "UW",
    "v": "V",
    "w": "W",
    "j": "Y",
    "z": "Z",
    "ʒ": "ZH",
    "ə": "AX",
    "ɨ": "IX",
    "l̩": "EL",  # syllabic consonants
    "m̩": "EM",
    "n̩": "EN",
    "ŋ̩": "NX",
    "ɾ̃": "NX",
    "ɾ": "DX",
    "ʔ": "Q",
    "ə˞": "ER",
    "ɚ": "ER",
    "ɝ": "ER",
}
ARPABET_TO_IPA = {v.lower(): k for k, v in IPA_TO_ARPABET_EPITRAN.items()}


def extract_buckeye_clip(
    buckeye_root, item, t0: float, t1: float, out: Path, sr: int
) -> bool:
    try:
        # TODO(shikhar): check the track creation
        speaker = buckeye.Speaker.from_zip(
            Path(buckeye_root) / (item["speaker_id"] + ".zip"),
            load_wavs=True,
        )
        track = speaker.tracks[item["track_id"]]
        track.clip_wav(str(out), t0, t1)  # write original
        wav, s = torchaudio.load(str(out))
        if s != sr:
            wav = torchaudio.transforms.Resample(s, sr)(wav)
            torchaudio.save(str(out), wav, sr)
        return True
    except Exception as e:
        logger.warning(
            f"speech extraction failed [{track.name} {t0:.2f}-{t1:.2f}s]: {e}"
        )
        return False


class BuckeyeAlignmentDataset(Dataset):
    """
    PyTorch dataset for Buckeye corpus alignment evaluation.
    """

    def __init__(
        self,
        buckeye_root: str,
        metadata_path: str,
        model_tokenizer,
        target_sr: int = 16000,
        max_speech_length: Optional[float] = None,  # in seconds
    ):
        """
        Args:
            buckeye_root: Path to Buckeye root
            metadata_path: Path to JSON metadata file created by BuckeyeDataPreparator
            model_tokenizer: Tokenizer for converting phonemes to indices, dependent on model
            target_sr: Target sample rate
            max_speech_length: Maximum speech length in seconds (for truncation)
        """
        self.buckeye_root = buckeye_root
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.clip_paths_cache = {}
        self.tokenizer = model_tokenizer
        self.speechcache_root = Path(buckeye_root) / "audio"
        self.speechcache_root.mkdir(parents=True, exist_ok=True)
        # self._construct_cache()

    def __len__(self):
        return len(self.metadata)

    def _construct_cache(self):
        """Construct all paths in the cache based on metadata."""
        for item in self.metadata:
            self._construct_clip_path(item)

    def _construct_clip_path(self, item) -> Path:
        """Construct and cache the speech clip from metadata item and buckeye root."""
        if item["segment_id"] in self.clip_paths_cache:
            return self.clip_paths_cache[item["segment_id"]]
        path = Path(self.speechcache_root) / (item["segment_id"] + ".wav")
        extract_buckeye_clip(
            self.buckeye_root,
            item,
            item["start_time"],
            item["end_time"],
            path,
            self.target_sr,
        )
        self.clip_paths_cache[item["segment_id"]] = path
        return path

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - speech: Tensor of shape (1, T) or (T,)
                - speech_length: int, actual speech length
                - phone_ids: Tensor of phone indices
                - phone_timestamps: List of (start, end) tuples
                - text: String transcript
                - segment_id: String identifier
                - duration: Float, segment duration in seconds
        """
        item = self.metadata[idx]

        # Load speech
        speechpath = self._construct_clip_path(item)
        waveform, sr = torchaudio.load(str(speechpath))

        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Truncate if necessary
        if self.max_speech_length is not None:
            max_samples = int(self.max_speech_length * sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        # Convert phones to indices
        phones = [
            ARPABET_TO_IPA.get(p.lower(), p)
            for p in item["phones"]
            if p not in ("IVER", "VOCNOISE", "{B_TRANS}")
        ]
        phone_ids = self.tokenizer.tokens2ids(phones)
        phone_pointstamps = [
            (start * self.target_sr, end * self.target_sr)
            for start, end in item["phone_timestamps"]
        ]
        return {
            "speech": waveform.squeeze(0),  # Shape: (T,)
            "speech_length": waveform.shape[1],
            "phones": item["phones"],
            "phone_ids": torch.tensor(phone_ids, dtype=torch.long),
            "phone_pointstamps": phone_pointstamps,
            "phone_timestamps": item["phone_timestamps"],
            "text": item["text"],
            "segment_id": item["segment_id"],
            "duration": item["duration"],
            "speaker_id": item["speaker_id"],
        }


def collate_fn(batch):
    """
    Custom collate function for batching variable-length sequences.

    Returns:
        dict with keys:
            - speech: Tensor of shape (batch_size, max_speech_length)
            - speech_length: Tensor of shape (batch_size,), actual lengths
            - target: Tensor of shape (batch_size, max_target_length)
            - target_length: Tensor of shape (batch_size,), actual lengths
            - target_start: Tensor of shape (batch_size, max_target_length)
            - target_end: Tensor of shape (batch_size, max_target_length)
    """
    # Find max lengths
    max_speech_length = max(item["speech_length"] for item in batch)
    max_target_length = max(len(item["phone_ids"]) for item in batch)

    # Initialize tensors with -1 padding
    batch_size = len(batch)
    speech = torch.full((batch_size, max_speech_length), -1.0, dtype=torch.float32)
    speech_length = torch.zeros(batch_size, dtype=torch.long)
    phone_id = torch.full((batch_size, max_target_length), -1, dtype=torch.long)
    target_length = torch.zeros(batch_size, dtype=torch.long)
    target_start = torch.full(
        (batch_size, max_target_length), -1.0, dtype=torch.float32
    )
    target_end = torch.full((batch_size, max_target_length), -1.0, dtype=torch.float32)

    # Fill tensors
    for i, item in enumerate(batch):
        speech_len = item["speech_length"]
        phone_len = len(item["phone_ids"])

        speech[i, :speech_len] = item["speech"]
        speech_length[i] = speech_len
        phone_id[i, :phone_len] = item["phone_ids"]
        target_length[i] = phone_len

        # Extract phone start and end times in points
        timestamps = item["phone_pointstamps"]
        for j, (start, end) in enumerate(timestamps[:phone_len]):
            target_start[i, j] = start
            target_end[i, j] = end

    return {
        "speech": speech,
        "speech_length": speech_length,
        "target": phone_id,
        "target_length": target_length,
        "target_start": target_start,
        "target_end": target_end,
    }


class BuckeyeAlignment(L.LightningDataModule):
    def __init__(
        self,
        buckeye_root: str,
        train_metadata: str,
        val_metadata: str,
        test_metadata: str,
        model_tokenizer,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        max_speech_length: Optional[float] = None,
    ):
        super().__init__()
        self.buckeye_root = buckeye_root
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.test_metadata = test_metadata
        self.model_tokenizer = model_tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = BuckeyeAlignmentDataset(
            buckeye_root=self.buckeye_root,
            metadata_path=self.train_metadata,
            model_tokenizer=self.model_tokenizer,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
        )
        self.val_dataset = BuckeyeAlignmentDataset(
            buckeye_root=self.buckeye_root,
            metadata_path=self.val_metadata,
            model_tokenizer=self.model_tokenizer,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
        )
        self.test_dataset = BuckeyeAlignmentDataset(
            buckeye_root=self.buckeye_root,
            metadata_path=self.test_metadata,
            model_tokenizer=self.model_tokenizer,
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
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
    """Example usage"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with processed Buckeye data",
    )
    parser.add_argument(
        "--buckeye_root",
        type=str,
        required=True,
        help="Path to Buckeye root",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    # Paths to metadata files
    train_meta = Path(args.data_dir) / "train_metadata.json"
    val_meta = Path(args.data_dir) / "val_metadata.json"
    test_meta = Path(args.data_dir) / "test_metadata.json"

    from src.model.powsm.token_id_converter import build_powsm_tokenizer

    model_tokenizer = build_powsm_tokenizer(
        work_dir="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/powsm_cache",
        hf_repo="espnet/powsm",
    )

    # Create dataloaders
    data_module = BuckeyeAlignment(
        buckeye_root=args.buckeye_root,
        train_metadata=str(train_meta),
        val_metadata=str(val_meta),
        test_metadata=str(test_meta),
        model_tokenizer=model_tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Test loading a batch
    for batch in train_loader:
        for k, v in batch.items():
            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else len(v)}")
        break

    # Test evaluator
    print("===" * 20)
    print("Naive baseline - equal segmentation")
    from src.metrics.forced_alignment import AlignmentEvaluator

    evaluator = AlignmentEvaluator(tolerance_ms=20)
    test_batch = next(iter(test_loader))
    dummy_predictions = {}
    for i in range(len(test_batch["speech_length"])):
        n_phones = test_batch["target_length"][i].item()
        duration = test_batch["speech_length"][i].item() / 16000  # assuming 16kHz
        phone_duration = duration / n_phones
        boundaries = [
            (j * phone_duration, (j + 1) * phone_duration) for j in range(n_phones)
        ]
        dummy_predictions[f"segment_{i}"] = boundaries

    ground_truth = {}
    for i in range(len(test_batch["speech_length"])):
        n_phones = test_batch["target_length"][i].item()
        boundaries = [
            (
                test_batch["target_start"][i, j].item(),
                test_batch["target_end"][i, j].item(),
            )
            for j in range(n_phones)
            if test_batch["target_start"][i, j].item() >= 0
        ]
        ground_truth[f"segment_{i}"] = boundaries

    # Evaluate
    metrics = evaluator.evaluate_batch(dummy_predictions, ground_truth)
    print("\nEvaluation metrics:")
    evaluator.pretty_print(metrics, verbosity=2)
