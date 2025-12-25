import io
import torch
import torchaudio
import lightning as L
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset, Audio as HFAudio
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _keep_not_too_short(example):
    return not example["too_short"]


def _resample_and_markshort(example):
    """Resamples from EDACC's native 32kHz to 16kHz and marks short clips."""
    resampler = torchaudio.transforms.Resample(32000, 16000)
    MIN_LENGTH = 8000  # 0.5 second at 16kHz

    audio = example["audio"]
    # Load from bytes to avoid backend-specific decoding issues
    wav, sr = torchaudio.load(io.BytesIO(example["audio"]["bytes"]))
    assert sr == 32000, "Expected original sample rate of 32000 Hz"

    # Standardize to mono if necessary
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = resampler(wav)

    buf = io.BytesIO()
    torchaudio.save(buf, wav, 16000, format="wav")

    audio["bytes"] = buf.getvalue()
    audio["sampling_rate"] = 16000
    example["audio"] = audio
    example["too_short"] = wav.shape[1] < MIN_LENGTH
    return example


def load_edacc_data(hf_repo: str, split: str, cache_dir: Optional[str] = None):
    ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
    ds = ds.cast_column("audio", HFAudio(decode=False))
    # Resampling occurs here and is cached by HF datasets
    ds = ds.map(_resample_and_markshort, desc=f"Preprocessing {split}")
    ds = ds.filter(_keep_not_too_short)
    return ds


class EdAccDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        l1_to_idx: Dict[str, int],
        split: str,
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
    ):
        self.hf_ds = hf_ds
        self.l1_to_idx = l1_to_idx
        self.split = split
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        speaker = str(sample["speaker"]).strip()
        utt_id = f"{speaker}_{self.split}_{idx}"

        # Target: Native Language (L1) classification
        l1_str = sample["l1"]
        label = self.l1_to_idx.get(l1_str, -1)

        waveform, sr = torchaudio.load(io.BytesIO(sample["audio"]["bytes"]))
        if self.max_duration_sec:
            max_samples = int(self.max_duration_sec * sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        return {
            "utt_id": utt_id,
            "speech": waveform.squeeze(0),  # (T,)
            "speech_length": waveform.shape[1],
            "lang_sym": "<eng>",  # Hardcoded for POWSM
            "target": label,
            "split": self.split,
            "speaker_id": speaker,
        }


def collate_fn(batch):
    max_speech_length = max(item["speech_length"] for item in batch)
    B = len(batch)

    speech = torch.full((B, max_speech_length), -1.0, dtype=torch.float32)
    speech_length = torch.zeros(B, dtype=torch.long)
    targets = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        s_len = item["speech_length"]
        speech[i, :s_len] = item["speech"]
        speech_length[i] = s_len
        targets[i] = item["target"]

    return {
        "utt_id": [item["utt_id"] for item in batch],
        "speech": speech,
        "speech_length": speech_length,
        "target": targets,
    }


class EdAccL1Classification(L.LightningDataModule):
    def __init__(
        self,
        hf_repo: str = "edinburghcstr/edacc",
        cache_dir: str = "exp/cache/edacc",
        target_sr: int = 16000,
        val_split_ratio: float = 0.2,  # 20% of HF validation split for dev
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        num_classes: int = 41,
        pin_memory: bool = False,
        max_duration_sec: Optional[float] = None,
    ):
        super().__init__()
        self.hf_repo = hf_repo
        self.cache_dir = Path(cache_dir)
        self.target_sr = target_sr
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.max_duration_sec = max_duration_sec
        self.pin_memory = pin_memory
        self.num_classes = num_classes

        self.l1_to_idx = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Download and resample (cached)
        for split in ["validation", "test"]:
            load_edacc_data(self.hf_repo, split, str(self.cache_dir))

    def setup(self, stage: Optional[str] = None):
        # Load datasets
        full_val_ds = load_edacc_data(self.hf_repo, "validation", str(self.cache_dir))
        test_ds = load_edacc_data(self.hf_repo, "test", str(self.cache_dir))

        # Build consistent label mapping from all seen L1s
        all_l1s = sorted(list(set(full_val_ds["l1"]) | set(test_ds["l1"])))
        self.l1_to_idx = {l1: i for i, l1 in enumerate(all_l1s)}
        assert self.num_classes == len(
            self.l1_to_idx
        ), f"Expected {self.num_classes} classes, but found {len(self.l1_to_idx)}"
        print(f"EDACC: Mapped {self.num_classes} L1 target classes.")

        # Split HF 'validation' into internal 'train' and 'dev' sets
        split_ds = full_val_ds.train_test_split(
            test_size=self.val_split_ratio, seed=self.seed
        )

        self.train_dataset = EdAccDataset(
            split_ds["train"],
            self.l1_to_idx,
            "train",
            max_duration_sec=self.max_duration_sec,
        )
        self.val_dataset = EdAccDataset(
            split_ds["test"],
            self.l1_to_idx,
            "val",
            max_duration_sec=self.max_duration_sec,
        )
        self.test_dataset = EdAccDataset(
            test_ds, self.l1_to_idx, "test", max_duration_sec=self.max_duration_sec
        )

    def _dl(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self._dl(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_dataset)

    def test_dataloader(self):
        return self._dl(self.test_dataset)


if __name__ == "__main__":
    # python -m src.data.edacc.l1_classification
    dm = EdAccL1Classification(batch_size=2, num_workers=0, max_duration_sec=20)
    dm.prepare_data()
    dm.setup()
    print(len(dm.train_dataloader()))
    print(len(dm.val_dataloader()))
    print(len(dm.test_dataloader()))

    for batch in dm.train_dataloader():
        print(batch)
        break
