"""EasyCall Dysarthria Severity Dataset

Usage:
    python -m src.data.easycall.dysarthria_severity \
        --hf_repo speech31/easycall-dysarthria \
        --cache_dir exp/cache/easycall \
        --easycall_meta_csv src/data/easycall/easycall_meta.csv
# NOTE(shikhar, eunjung): because of hf_repo now we can skip meta_csv?
"""

import io
from pathlib import Path
from typing import Optional, List

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L
from ipatok import tokenise as ipatok_tokenise
import epitran
from datasets import load_dataset, Audio as HFAudio
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _keep_not_too_short(example):
    return not example["too_short"]


def _resample_and_markshort(example):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    resampler = torchaudio.transforms.Resample(8000, 16000)
    MIN_LENGTH = 8000  # 0.5 second at 16kHz
    audio = example["audio"]
    wav, _ = torchaudio.load(io.BytesIO(example["audio"]["bytes"]))
    # wav = wav.to(device)
    wav = resampler(wav)  # .cpu()
    buf = io.BytesIO()
    torchaudio.save(buf, wav, 16000, format="wav")
    audio["bytes"] = buf.getvalue()
    audio["sampling_rate"] = 16000
    example["audio"] = audio
    example["too_short"] = wav.shape[1] < MIN_LENGTH
    return example


def load_easycall_data(
    hf_repo: str,
    split: str,
    cache_dir: Optional[str] = None,
):
    ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
    ds = ds.cast_column("audio", HFAudio(decode=False))  # no torchcodec
    ds = ds.map(_resample_and_markshort)
    ds = ds.filter(_keep_not_too_short)
    ds = ds.with_format(None)
    return ds


class EasyCallDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        hf_repo: str,
        cache_dir: str,
        easycall_meta_csv: str,
        tokenizer,
        target_sr: int = 16000,
        split: str = "train",
        max_duration_sec: Optional[float] = None,  # seconds
    ):
        """
        Args:
            hf_ds: HuggingFace Dataset for this split.
            easycall_meta_csv: Speaker-level CSV with columns:
                speaker, label, split, sex, severity, ...
            tokenizer: Object with `tokens2ids(List[str]) -> List[int]`.
            target_sr: Expected sample rate.
            split: "train" / "validation" / "test".
            max_duration_sec: Optional max duration in seconds.
        """
        self.hf_ds = hf_ds
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir
        self.target_sr = target_sr
        self.split = split
        self.max_duration_sec = max_duration_sec
        self.tokenizer = tokenizer

        df_meta = pd.read_csv(easycall_meta_csv)
        df_meta = df_meta[df_meta["label"].notna()]
        self.speaker_to_label = dict(zip(df_meta["speaker"], df_meta["label"]))
        self.speaker_to_split = dict(zip(df_meta["speaker"], df_meta["split"]))
        # self.speaker_to_sex = dict(zip(df_meta["speaker"], df_meta["sex"]))
        # self.speaker_to_severity = dict(zip(df_meta["speaker"], df_meta["severity"]))

        # FILTER
        self.indices = []
        for idx, ex in enumerate(self.hf_ds):
            speaker = str(ex["speaker"]).strip()
            if not speaker or speaker.lower() == "f04":
                continue
            if speaker not in self.speaker_to_label:
                continue
            if self.speaker_to_split.get(speaker) != self.split:
                continue
            self.indices.append(idx)

        log.info(
            f"EasyCallDataset split={self.split}: using {len(self.indices)} / {len(self.hf_ds)} examples"
        )
        self.epitran_transliterator = epitran.Epitran("ita-Latn")

    def __len__(self):
        return len(self.indices)
    
    def _cache_audio(self, waveform: torch.Tensor, sr: int, target_path: Path) -> None:
        if target_path.exists():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(target_path), waveform, sr)

    def _text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to a list of IPA phones using epitran + ipatok.
        Returns [] if text is empty.
        """
        if not text:
            return []
        ipa_string = self.epitran_transliterator.transliterate(text)
        try:
            phonemes_list = ipatok_tokenise(ipa_string)
        except Exception as e:
            # workaround for ipatok issues
            phonemes_list = ipa_string.split()
        if not phonemes_list:
            raise ValueError("Epitran/ipatok returned empty phoneme list")
        return phonemes_list

    def __getitem__(self, idx):
        ds_idx = self.indices[idx]
        sample = self.hf_ds[ds_idx]
        speaker = str(sample["speaker"]).strip()
        utt_id = f"{speaker}_{idx}"
        # unnused fields: sex, severity
        label = int(self.speaker_to_label[speaker])
        waveform, sr = torchaudio.load(io.BytesIO(sample["audio"]["bytes"]))
        if sr != self.target_sr:
            raise ValueError(
                f"Sample rate mismatch: expected {self.target_sr}, got {sr}"
            )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.max_duration_sec is not None:
            max_samples = int(self.max_duration_sec * sr)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
                
        target_path = Path(self.cache_dir) / "saved" / self.split / f"{utt_id}.wav"
        self._cache_audio(waveform, sr, target_path)

        text = str(sample["text"])
        phone_list = self._text_to_phonemes(text)
        phone_ids = self.tokenizer.tokens2ids(phone_list)

        return {
            "utt_id": utt_id,
            "audio_path": str(target_path),
            "split": self.split,
            "speech": waveform.squeeze(0),  # (T,)
            "speech_length": waveform.shape[1],
            "lang_sym": "<ita>",  # for powsm
            "target": label,
            "text": text,
            "phones": " ".join(phone_list),
            "phone_id": torch.tensor(phone_ids, dtype=torch.long),
            "phone_length": len(phone_ids),
            "speaker_id": speaker,
        }

    # These two functions are needed for dataset to be picklable which alllows
    # distributed inference using spawn
    def __getstate__(self):
        st = self.__dict__.copy()
        st["hf_ds"] = None  # HF Dataset breaks spawn pickling
        st["epitran_transliterator"] = None  # recreate cleanly
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if self.hf_ds is None:
            self.hf_ds = load_easycall_data(
                hf_repo=self.hf_repo,
                split=self.split,
                cache_dir=self.cache_dir,
            )
        if self.epitran_transliterator is None:
            self.epitran_transliterator = epitran.Epitran("ita-Latn")


def collate_fn(batch):
    if not batch:
        raise ValueError("Empty batch in collate_fn")

    max_speech_length = max(item["speech_length"] for item in batch)
    max_phone_length = max(item["phone_length"] for item in batch)

    B = len(batch)
    speech = torch.full((B, max_speech_length), -1.0, dtype=torch.float32)
    speech_length = torch.zeros(B, dtype=torch.long)
    phone_id = torch.full((B, max_phone_length), -1, dtype=torch.long)
    phone_length = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        s_len = item["speech_length"]
        plen = item["phone_length"]

        speech[i, :s_len] = item["speech"]
        speech_length[i] = s_len

        if plen > 0:
            phone_id[i, :plen] = item["phone_id"]
        phone_length[i] = plen

    return {
        "utt_id": [item["utt_id"] for item in batch],
        "split": [item.get("split", "unknown") for item in batch],
        "speech": speech,
        "speech_length": speech_length,
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "phone_id": phone_id,
        "phone_length": phone_length,
    }


class EasyCallDataModule(L.LightningDataModule):
    def __init__(
        self,
        hf_repo: str,
        cache_dir: str,
        easycall_meta_csv: str,
        tokenizer,
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
        num_classes: int = 4,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        predict_splits: Optional[List[str]] = None,  # Splits for predict_dataloader, default: all
    ):
        """
        Lightning DataModule for EasyCall.

        Uses HF dataset in cache_dir + speaker-level CSV to build splits.
        """
        super().__init__()
        self.hf_repo = hf_repo
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.easycall_meta_csv = easycall_meta_csv
        self.num_classes = num_classes

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.predict_splits = predict_splits or ["train", "validation", "test"]

        self._hf_train = None
        self._hf_val = None
        self._hf_test = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        for split in ["train", "validation", "test"]:
            # just download, variables set here are invisible
            load_easycall_data(
                hf_repo=self.hf_repo,
                split=split,
                cache_dir=str(self.cache_dir),
            )

    def _ds(self, split: str) -> EasyCallDataset:
        # reuse downloaded
        return EasyCallDataset(
            hf_ds=load_easycall_data(
                hf_repo=self.hf_repo,
                split=split,
                cache_dir=str(self.cache_dir),
            ),
            hf_repo=self.hf_repo,
            cache_dir=str(self.cache_dir),
            easycall_meta_csv=self.easycall_meta_csv,
            tokenizer=self.tokenizer,
            target_sr=self.target_sr,
            max_duration_sec=self.max_duration_sec,
            split=split,
        )

    def setup(self, stage: Optional[str] = None):
        if not Path(self.easycall_meta_csv).exists():
            log.warning(f"CSV metadata file not found at {self.easycall_meta_csv}")
            self.train_dataset = self.val_dataset = self.test_dataset = None
            return
        self.train_dataset = self._ds("train")
        self.val_dataset = self._ds("validation")
        self.test_dataset = self._ds("test")

    def _dl(self, dataset: Dataset, shuffle: bool):
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
        return self._dl(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        datasets = []
        if "train" in self.predict_splits and self.train_dataset:
            datasets.append(self.train_dataset)
        if "validation" in self.predict_splits and self.val_dataset:
            datasets.append(self.val_dataset)
        if "test" in self.predict_splits and self.test_dataset:
            datasets.append(self.test_dataset)
        if not datasets:
            datasets = [self.test_dataset]  # fallback to test
        return self._dl(ConcatDataset(datasets), shuffle=False)


if __name__ == "__main__":
    import argparse
    from src.core.ipa_utils import IPATokenizer
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--easycall_meta_csv", type=str, required=True)
    args = parser.parse_args()

    tokenizer = IPATokenizer()

    dm = EasyCallDataModule(
        hf_repo=args.hf_repo,
        cache_dir=args.cache_dir,
        easycall_meta_csv=args.easycall_meta_csv,
        tokenizer=tokenizer,
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        target_sr=16000,
        max_duration_sec=20.0,
        num_classes=4,
    )

    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        break