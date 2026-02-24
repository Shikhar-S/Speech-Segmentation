"""TIMIT datamodule for phone recognition with optional Epitran-mixed targets.

This module is designed to work with XeusPR / PhoneRecognitionModel, producing
CTC-style batches with:
    - speech: (B, T)
    - speech_length: (B,)
    - text: (B, L)
    - text_length: (B,)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import epitran
import panphon
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio

from src.core.ipa_utils import ARPABET_TO_IPA


IGNORE_ID = -1


def _load_waveform(
    timit_root: Path, segment_id: str, target_sr: int
) -> Tuple[torch.Tensor, int]:
    """Load a TIMIT waveform, resample to target_sr and ensure mono."""
    wav_path = timit_root / f"{segment_id}.wav"
    waveform, sr = torchaudio.load(str(wav_path))

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0), sr  # (T,), sr


def build_timit_phone_sequence(item: Dict) -> List[str]:
    """Convert ARPABET phone sequence in metadata item to IPA phones."""
    phones_ipa: List[str] = []
    for ph in item["phones"]:
        phones_ipa.append(ARPABET_TO_IPA.get(ph.lower(), ph.lower()))
    return phones_ipa


def build_epitran_phone_sequence(
    epi: epitran.Epitran, ipa_segmenter: panphon.FeatureTable, text: str
) -> List[str]:
    """Convert orthographic text to a list of IPA phones using Epitran+panphon."""
    ipa_string = epi.transliterate(text or "")
    # ipa_segs returns a list of IPA symbols suitable for tokenization
    return [seg for seg in ipa_segmenter.ipa_segs(ipa_string) if seg]


def phones_to_token_ids(
    phones: List[str], vocab: Dict[str, int], unk_id: int
) -> List[int]:
    """Map a list of phone strings to token ids using the given vocabulary."""
    return [vocab.get(p, unk_id) for p in phones]


def choose_label_type(
    idx: int, epitran_mix_ratio: float, mix_mod_n: Optional[int]
) -> str:
    """Decide whether to use 'epitran' or 'timit' labels for a given index."""
    if epitran_mix_ratio <= 0.0 or mix_mod_n is None or mix_mod_n <= 0:
        return "timit"
    return "epitran" if (idx % mix_mod_n == 0) else "timit"


class TimitPRDataset(Dataset):
    """TIMIT dataset for phone recognition with optional Epitran-mixed targets."""

    def __init__(
        self,
        timit_root: str,
        metadata_path: str,
        vocab_file: str,
        split: str = "train",
        target_sr: int = 16000,
        max_speech_length: Optional[float] = None,
        epitran_mix_ratio: float = 0.0,
        epitran_lang: str = "eng-Latn",
    ) -> None:
        """
        Args:
            timit_root: Path to TIMIT root directory.
            metadata_path: Path to JSON metadata file created by timit_data_prep.
            vocab_file: Path to IPA vocabulary JSON (token -> id mapping).
            split: Dataset split name ("train", "val", "test").
            target_sr: Target sampling rate.
            max_speech_length: Optional maximum speech length in seconds.
            epitran_mix_ratio: Fraction of samples that should use Epitran labels
                (approximate; implemented via idx % N == 0).
            epitran_lang: Epitran language code, e.g., "eng-Latn".
        """
        super().__init__()
        self.timit_root = Path(timit_root)
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.epitran_mix_ratio = float(epitran_mix_ratio)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab: Dict[str, int] = json.load(f)
        self.unk_id = self.vocab.get("<unk>", IGNORE_ID)

        self.epi: Optional[epitran.Epitran] = None
        self.ipa_segmenter: Optional[panphon.FeatureTable] = None
        self.mix_mod_n: Optional[int] = None

        if self.epitran_mix_ratio > 0.0 and self.split == "train":
            # Approximate ratio via idx % N == 0
            self.mix_mod_n = max(1, int(round(1.0 / self.epitran_mix_ratio)))
            self.epi = epitran.Epitran(epitran_lang)
            self.ipa_segmenter = panphon.FeatureTable()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        item = self.metadata[idx]
        segment_id = item["segment_id"]

        speech, sr = _load_waveform(self.timit_root, segment_id, self.target_sr)

        if self.max_speech_length is not None:
            max_samples = int(self.max_speech_length * sr)
            if speech.shape[0] > max_samples:
                speech = speech[:max_samples]

        label_source = choose_label_type(idx, self.epitran_mix_ratio, self.mix_mod_n)

        if label_source == "epitran" and self.epi is not None and self.ipa_segmenter:
            phones = build_epitran_phone_sequence(
                self.epi, self.ipa_segmenter, item.get("text", "")
            )
        else:
            phones = build_timit_phone_sequence(item)
            label_source = "timit"

        token_ids = phones_to_token_ids(phones, self.vocab, self.unk_id)

        if len(token_ids) == 0:
            raise ValueError(f"No valid phone tokens for segment {segment_id}")

        return {
            "speech": speech.to(torch.float32),
            "speech_length": speech.shape[0],
            "text": torch.tensor(token_ids, dtype=torch.long),
            "text_length": len(token_ids),
            "utt_id": segment_id,
            "split": self.split,
            "label_source": label_source,
            # For compatibility with PhoneRecognitionModel / XeusPR, we can
            # optionally provide a language symbol.
            "lang_sym": "eng",
        }


def timit_pr_collate(batch: List[Dict]) -> Dict:
    """Collate function for TIMIT phone-recognition batches."""
    batch_size = len(batch)
    speech_lengths = torch.tensor([b["speech_length"] for b in batch], dtype=torch.long)
    max_T = int(speech_lengths.max().item())

    padded_speech = torch.zeros(batch_size, max_T, dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["speech"].shape[0]
        padded_speech[i, :T] = b["speech"]

    text_lengths = torch.tensor([b["text_length"] for b in batch], dtype=torch.long)
    max_L = int(text_lengths.max().item())
    padded_text = torch.full(
        (batch_size, max_L), IGNORE_ID, dtype=torch.long
    )
    for i, b in enumerate(batch):
        L_i = b["text"].shape[0]
        padded_text[i, :L_i] = b["text"]

    return {
        "speech": padded_speech,
        "speech_length": speech_lengths,
        "text": padded_text,
        "text_length": text_lengths,
        "utt_id": [b["utt_id"] for b in batch],
        "split": [b["split"] for b in batch],
        "label_source": [b["label_source"] for b in batch],
        "lang_sym": [b.get("lang_sym", "eng") for b in batch],
        "accent_sym": [b.get("accent_sym", "<unk>") for b in batch],
    }


class TimitPRDataModule(L.LightningDataModule):
    """LightningDataModule for TIMIT phone recognition with optional Epitran mix."""

    def __init__(
        self,
        timit_root: str,
        local_cache_path: str,
        vocab_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        max_speech_length: Optional[float] = None,
        epitran_mix_ratio: float = 0.0,
        epitran_lang: str = "eng-Latn",
    ) -> None:
        super().__init__()
        self.timit_root = timit_root
        self.local_cache_path = Path(local_cache_path)
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.max_speech_length = max_speech_length
        self.epitran_mix_ratio = float(epitran_mix_ratio)
        self.epitran_lang = epitran_lang

        self.train_metadata = self.local_cache_path / "train_metadata.json"
        self.val_metadata = self.local_cache_path / "val_metadata.json"
        self.test_metadata = self.local_cache_path / "test_metadata.json"

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = TimitPRDataset(
            timit_root=self.timit_root,
            metadata_path=str(self.train_metadata),
            vocab_file=self.vocab_file,
            split="train",
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            epitran_mix_ratio=self.epitran_mix_ratio,
            epitran_lang=self.epitran_lang,
        )
        # For validation / test, use ground-truth phones only
        self.val_dataset = TimitPRDataset(
            timit_root=self.timit_root,
            metadata_path=str(self.val_metadata),
            vocab_file=self.vocab_file,
            split="val",
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            epitran_mix_ratio=0.0,
            epitran_lang=self.epitran_lang,
        )
        self.test_dataset = TimitPRDataset(
            timit_root=self.timit_root,
            metadata_path=str(self.test_metadata),
            vocab_file=self.vocab_file,
            split="test",
            target_sr=self.target_sr,
            max_speech_length=self.max_speech_length,
            epitran_mix_ratio=0.0,
            epitran_lang=self.epitran_lang,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=timit_pr_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=timit_pr_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=timit_pr_collate,
        )

