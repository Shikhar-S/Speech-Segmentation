"""Buckeye Dataset and DataLoader for forced alignment
srun -A bbjs-dtai-gh --gpus=1 \
    --cpus-per-task=8 \
    --mem=32G --time=02:00:00 \
    --pty /bin/bash -c "cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench \
      && source setup_uv.sh .venv_dai \
    && python -m src.data.buckeye.common_datamodule \
    --buckeye_root /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye \
    --data_dir /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/buckeye_cache \
    --batch_size 32 --num_workers 4"
Usage:
    python -m src.data.buckeye.common_datamodule \
        --buckeye_root /work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_buckeye/buckeye \
        --data_dir /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/buckeye_cache \
        --batch_size 32 --num_workers 4
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging
import buckeye
import lightning as L
from src.core.ipa_utils import ARPABET_TO_IPA

logger = logging.getLogger(__name__)


def extract_buckeye_clip(
    buckeye_root, item, t0: float, t1: float, out: Path, sr: int
) -> bool:
    try:
        speaker = buckeye.Speaker.from_zip(
            Path(buckeye_root) / (item["speaker_id"] + ".zip"),
            load_wavs=True,
        )
        track = speaker.tracks[item["track_id"]]
        track.clip_wav(str(out), t0, t1)
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


class BuckeyeDataset(Dataset):
    """
    PyTorch dataset for Buckeye corpus alignment evaluation.
    """

    def __init__(
        self,
        buckeye_root: str,
        metadata_path: str,
        cache_path: str,
        tokenizer,
        target_sr: int = 16000,
        mask_probability: float = 0.0,
        split: str = "train",
        max_speech_length: Optional[float] = None,  # in seconds
    ):
        """
        Args:
            buckeye_root: Path to Buckeye root
            metadata_path: Path to JSON metadata file created by BuckeyeDataPreparator
            cache_path: Path to cache directory for storing extracted audio clips
            tokenizer: Tokenizer for converting phonemes to indices, dependent on model
            target_sr: Target sample rate
            mask_probability: Percentage of phones to mask with noise
            max_speech_length: Maximum speech length in seconds (for truncation)
            split: Dataset split (e.g., "train", "test", "validation")
        """
        self.buckeye_root = buckeye_root
        self.cache_path = cache_path
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.split = split
        self.mask_probability = mask_probability

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.clip_paths_cache = {}
        self.tokenizer = tokenizer
        self.speechcache_root = Path(cache_path) / "speech_clips"
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
        if path.exists():
            self.clip_paths_cache[item["segment_id"]] = path
            return path
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
                - target: Tensor of phone indices
                - phone_pointstamps: List of (start, end) tuples
                - text: String transcript
                - utt_id: String identifier
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

        phone_ipa, phone_pointstamps = [], []
        masked_phone_ipa = []
        atleast_one_unmasked = False
        for phone, (start, end) in zip(item["phones"], item["phone_timestamps"]):
            start_idx = int(start * self.target_sr)
            end_idx = int(end * self.target_sr)
            should_mask = (
                np.random.rand() < self.mask_probability and atleast_one_unmasked
            )
            if should_mask:
                # Replace the segment with noise
                waveform = waveform.clone()
                waveform[:, start_idx:end_idx] = torch.zeros(
                    end_idx - start_idx, dtype=waveform.dtype
                )
            atleast_one_unmasked = True
            phone_ipa.append(ARPABET_TO_IPA.get(phone.lower(), phone.lower()))
            masked_phone_ipa.append(phone_ipa[-1] if not should_mask else "[NOISE]")
            phone_pointstamps.append(
                (int(start * self.target_sr), int(end * self.target_sr))
            )
        target = self.tokenizer.tokens2ids(phone_ipa)
        assert (
            len(target) != 0
        ), f"No valid phones for {item['segment_id']} with transcript: {item['phones']}"
        return {
            "speech": waveform.squeeze(0),  # Shape: (T,)
            "speech_length": waveform.shape[1],
            "target": torch.tensor(target, dtype=torch.long),
            "target_length": len(target),
            "phone_pointstamps": phone_pointstamps,
            "phone_timestamps": item["phone_timestamps"],
            "phones": phone_ipa,
            "masked_phones": masked_phone_ipa,
            "text": item["text"],
            "utt_id": item["segment_id"],
            "duration": item["duration"],
            "speaker_id": item["speaker_id"],
            "split": self.split,
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
    max_target_length = max(len(item["target"]) for item in batch)

    # Initialize tensors with -1 padding
    batch_size = len(batch)
    speech = torch.full((batch_size, max_speech_length), -1.0, dtype=torch.float32)
    speech_length = torch.zeros(batch_size, dtype=torch.long)
    phone_id = torch.full((batch_size, max_target_length), -1, dtype=torch.long)
    target_length = torch.zeros(batch_size, dtype=torch.long)
    target_start = torch.full((batch_size, max_target_length), -1, dtype=torch.float32)
    target_end = torch.full((batch_size, max_target_length), -1, dtype=torch.float32)
    # NOTE(shikhar): init with -1 in target_end is good in downstream loss computation
    # since it helps ignore padding

    # Fill tensors
    for i, item in enumerate(batch):
        speech_len = item["speech_length"]
        phone_len = len(item["target"])

        speech[i, :speech_len] = item["speech"]
        speech_length[i] = speech_len
        phone_id[i, :phone_len] = item["target"]
        target_length[i] = phone_len

        # Extract phone start and end times in points
        pointstamps = item["phone_pointstamps"]
        for j, (start, end) in enumerate(pointstamps[:phone_len]):
            target_start[i, j] = start
            target_end[i, j] = end

    return {
        "speech": speech,
        "speech_length": speech_length,
        "target": phone_id,
        "target_text": [item["phones"] for item in batch],
        "masked_target_text": [item["masked_phones"] for item in batch],
        "target_length": target_length,
        "target_start": target_start,
        "target_end": target_end,
        "ground_truth_timestamps": [item["phone_timestamps"] for item in batch],
        "utt_id": [item["utt_id"] for item in batch],
        "split": [item.get("split", "unknown") for item in batch],
    }


class BuckeyeDataModule(L.LightningDataModule):
    def __init__(
        self,
        buckeye_root: str,
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
        self.buckeye_root = buckeye_root
        self.local_cache_path = local_cache_path
        self.train_metadata = Path(local_cache_path) / "train_metadata.json"
        self.val_metadata = Path(local_cache_path) / "val_metadata.json"
        self.test_metadata = Path(local_cache_path) / "test_metadata.json"
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.mask_probability = mask_probability
        self.max_speech_length = max_speech_length

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = BuckeyeDataset(
            buckeye_root=self.buckeye_root,
            metadata_path=self.train_metadata,
            cache_path=self.local_cache_path,
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            mask_probability=self.mask_probability,
            max_speech_length=self.max_speech_length,
            split="train",
        )
        self.val_dataset = BuckeyeDataset(
            buckeye_root=self.buckeye_root,
            metadata_path=self.val_metadata,
            cache_path=self.local_cache_path,
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            mask_probability=self.mask_probability,
            max_speech_length=self.max_speech_length,
            split="validation",
        )
        self.test_dataset = BuckeyeDataset(
            buckeye_root=self.buckeye_root,
            metadata_path=self.test_metadata,
            tokenizer=self.tokenizer,
            cache_path=self.local_cache_path,
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


def _naive_baseline_equal_segmentation(test_loader: DataLoader):
    """
    Naive baseline that divides the speech into equal segments for each target unit.
    """
    print("===" * 20)
    print("Naive baseline - equal segmentation")
    from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit
    from tqdm import tqdm

    evaluator = SegmentationEvaluator(tolerance_ms=20)
    naive_predictions = {}
    ground_truth = {}
    for batch_id, test_batch in tqdm(
        enumerate(test_loader), desc="Evaluating naive baseline", total=len(test_loader)
    ):
        # if batch_id > 2:
        #     break
        for i in tqdm(
            range(len(test_batch["speech_length"])),
            desc="Building predictions",
            leave=False,
        ):
            n_phones = test_batch["target_length"][i].item()
            duration = test_batch["speech_length"][i].item() / 16000  # assuming 16kHz
            phone_duration = duration / n_phones
            boundaries = [
                SegmentationUnit(
                    j * phone_duration,
                    (j + 1) * phone_duration,
                    test_batch["target_text"][i][j],
                )
                for j in range(n_phones)
            ]
            naive_predictions[f"segment_{batch_id}_{i}"] = boundaries
        for i in tqdm(
            range(len(test_batch["speech_length"])),
            desc="Building ground truth",
            leave=False,
        ):
            n_phones = test_batch["target_length"][i].item()
            boundaries = [
                SegmentationUnit(
                    test_batch["target_start"][i, j].item() / 16000,
                    test_batch["target_end"][i, j].item() / 16000,
                    test_batch["target_text"][i][j],
                )
                for j in range(n_phones)
                if test_batch["target_start"][i, j].item() >= 0
            ]
            ground_truth[f"segment_{batch_id}_{i}"] = boundaries

    # Evaluate
    metrics = evaluator.evaluate_batch(naive_predictions, ground_truth)
    print("\nEvaluation metrics:")
    evaluator.pretty_print(metrics, verbosity=2)
    LOG_METRICS = sorted(
        [
            "f1",
            "precision",
            "recall",
            "pbe_median",
            "start_err_median",
            "end_err_median",
            "dur_err_median",
            "pred_dur_mean",
            "gt_dur_mean",
        ]
    )
    print("==" * 20)
    for key in LOG_METRICS:
        print(key, end=",")
    print()
    for key in LOG_METRICS:
        print(evaluator._get_metric(metrics, key), end=",")
    print()


if __name__ == "__main__":
    """Example usage"""
    from src.data.buckeye.common_datamodule import BuckeyeDataModule
    from src.model.powsm.token_id_converter import build_powsm_tokenizer
    from src.model.wav2vec2phoneme.builders import build_wav2vec2phoneme_tokenizer

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

    # Create dataloaders
    data_module = BuckeyeDataModule(
        buckeye_root=args.buckeye_root,
        local_cache_path=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    _naive_baseline_equal_segmentation(test_loader)
