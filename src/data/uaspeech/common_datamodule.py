"""UASpeech Dysarthric Speech Dataset

Usage:
    python -m src.data.uaspeech.common_datamodule \
        --data_dir exp/download/uaspeech \
        --cache_dir exp/cache/uaspeech \
        --uaspeech_meta_csv src/data/uaspeech/uaspeech_meta.csv \
        --uaspeech_wordlist_csv src/data/uaspeech/uaspeech_wordlist.csv
"""

import argparse
import os
import tarfile
from pathlib import Path
from typing import Optional, List
import json

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L
from ipatok import tokenise as ipatok_tokenise
import epitran
from tqdm import tqdm
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def extract_uaspeech_tgz_files(
    output_dir: str, tgz_files: List[str], force: bool = False
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if list(output_path.glob("**/*.wav")) and not force:
        log.info(f"Skipping extraction - files exist in {output_path}")
        return

    for tgz_file in tgz_files:
        tgz_path = Path(tgz_file)
        log.info(f"Extracting {tgz_path}...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc=tgz_path.name):
                tar.extract(member, output_path)


class UASpeechDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        uaspeech_meta_csv: str,
        uaspeech_wordlist_csv: str,
        tokenizer,
        target_sr: int = 16000,
        split: str = "train",
        max_duration_sec: Optional[float] = None,
    ):
        # Always keep an absolute cache root so that paths stored in metadata are stable
        # across Hydra chdir and multiprocessing ("spawn") workers.
        self.cache_path = Path(cache_path).expanduser().resolve(strict=False)
        self.target_sr = target_sr
        self.split = split
        self.max_duration_sec = max_duration_sec
        self.tokenizer = tokenizer
        self.epitran_transliterator = epitran.Epitran("eng-Latn")

        self.metadata = self._build_metadata(
            uaspeech_meta_csv, uaspeech_wordlist_csv, split
        )
        log.info(f"UASpeechDataset split={split}: {len(self.metadata)} examples")

    def _should_keep_file(self, file: str) -> bool:
        """Filter to keep only M5 mic data + one explicit exception."""
        _MIC_KEEP = "M5"
        _EXTRA_UTT_IDS = {"F04_B2_C13_M7.wav"}

        if file in _EXTRA_UTT_IDS:
            return True
        suffix = f"_{_MIC_KEEP}.wav"
        return file.endswith(suffix)

    def _filter_metadata(self, metadata: list) -> list:
        """Apply mic filter (M5 + extra) to metadata list."""
        return [m for m in metadata if self._should_keep_file(str(m.get("file", "")))]

    def _build_metadata(self, meta_csv: str, wordlist_csv: str, split: str):
        json_path = self.cache_path / f"metadata_{split}.json"
        if json_path.exists():
            # only exists whn called in setup by all workers
            with open(json_path, "r") as f:
                metadata = json.load(f)
            # Backward-compat: older metadata may contain relative "path".
            # Normalize to absolute paths rooted at cache_path.
            fixed = []
            for m in metadata:
                rel_path = str(m.get("rel_path", "")).strip()
                path = str(m.get("path", "")).strip()
                if rel_path and (not path or not os.path.isabs(path)):
                    m["path"] = str((self.cache_path / rel_path).resolve(strict=False))
                fixed.append(m)
            return self._filter_metadata(fixed)

        df_meta = pd.read_csv(meta_csv)
        df_meta = df_meta[df_meta["severity"].notna()]
        speaker_to_label = dict(zip(df_meta["speaker"], df_meta["severity"]))
        speaker_to_split = dict(zip(df_meta["speaker"], df_meta["split"]))

        df_wordlist = pd.read_csv(wordlist_csv)
        word_dict = dict(zip(df_wordlist["FILE_NAME"], df_wordlist["WORD"]))

        metadata = []
        min_samples = int(0.5 * self.target_sr)

        for root, _, files in tqdm(
            os.walk(self.cache_path), desc="Scanning UASpeech", leave=False
        ):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                speaker = file.split("_")[0].strip()
                if (
                    speaker not in speaker_to_split
                    or speaker_to_split[speaker] != split
                ):
                    continue

                # Store absolute path to be robust to Hydra changing cwd and to worker cwd.
                path = str((Path(root) / file).resolve(strict=False))
                rel_path = os.path.relpath(path, self.cache_path)
                try:
                    if torchaudio.info(path).num_frames < min_samples:
                        continue
                except Exception:
                    continue

                utt_id = file.split("_")[2].replace(".wav", "")
                if "UW" in utt_id:
                    utt_id = file.split("_")[1] + "_" + utt_id

                metadata.append(
                    {
                        "file": file,
                        "rel_path": rel_path,
                        "path": path,
                        "speaker": speaker,
                        "label": int(speaker_to_label[speaker]),
                        "text": word_dict.get(utt_id, ""),
                    }
                )
        log.info(f"Total {len(metadata)} examples for split={split}")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return self._filter_metadata(metadata)

    def _text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to a list of IPA phones using epitran + ipatok.
        Returns [] if text is empty.
        """
        if not text:
            return []
        try:
            ipa_string = self.epitran_transliterator.transliterate(text)
            try:
                phonemes_list = ipatok_tokenise(ipa_string)
            except Exception as e:
                # workaround for ipatok issues
                phonemes_list = ipa_string.split()
            if not phonemes_list:
                raise ValueError("Epitran/ipatok returned empty phoneme list")
        except Exception as e:
            log.warning(f"Failed to convert text to phonemes: {text}. Error: {e}")
            phonemes_list = []
        return phonemes_list

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        waveform, sr = torchaudio.load(item["path"])
        assert sr == self.target_sr
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.max_duration_sec:
            max_samples = int(self.max_duration_sec * self.target_sr)
            waveform = waveform[:, :max_samples]
        phones = self._text_to_phonemes(item["text"])
        phone_ids = self.tokenizer.tokens2ids(phones)
        # NOTE(shikhar): Why are they empty!?!

        return {
            "utt_id": item["file"],
            "audio_path": item["path"],
            "split": self.split,
            "speech": waveform.squeeze(0),
            "speech_length": waveform.shape[1],
            "lang_sym": "<eng>",  # for powsm
            "target": item["label"],
            "text": item["text"],
            "phones": " ".join(phones),
            "phone_id": torch.tensor(phone_ids, dtype=torch.long),
            "phone_length": len(phone_ids),
            "speaker_id": item["speaker"],
        }


def collate_fn(batch):
    B = len(batch)
    max_speech = max(item["speech_length"] for item in batch)
    max_phone = max(item["phone_length"] for item in batch)

    speech = torch.full((B, max_speech), -1.0)
    speech_length = torch.zeros(B, dtype=torch.long)
    phone_id = torch.full((B, max_phone), -1, dtype=torch.long)
    phone_length = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        speech[i, : item["speech_length"]] = item["speech"]
        speech_length[i] = item["speech_length"]
        if item["phone_length"] > 0:
            phone_id[i, : item["phone_length"]] = item["phone_id"]
        phone_length[i] = item["phone_length"]

    return {
        "utt_id": [item["utt_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "speech": speech,
        "speech_length": speech_length,
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "phone_id": phone_id,
        "phone_length": phone_length,
    }


class UASpeechDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        uaspeech_meta_csv: str,
        uaspeech_wordlist_csv: str,
        tokenizer,
        num_classes: int,
        target_sr: int = 16000,
        max_duration_sec: Optional[float] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        predict_splits: Optional[List[str]] = None,  # Splits for predict_dataloader, default: all
    ):
        super().__init__()
        self.num_classes = num_classes
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.uaspeech_meta_csv = uaspeech_meta_csv
        self.uaspeech_wordlist_csv = uaspeech_wordlist_csv
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.predict_splits = predict_splits or ["train", "validation", "test"]

        self.tgz_files = [
            self.data_dir / "UASpeech_noisereduce_C.tgz",
            self.data_dir / "UASpeech_noisereduce_FM.tgz",
        ]

    def prepare_data(self):
        extract_dir = self.cache_dir / "extracted"
        extract_uaspeech_tgz_files(str(extract_dir), [str(f) for f in self.tgz_files])

        all_metadata = []
        for split in ["train", "validation", "test"]:
            ds = UASpeechDataset(
                str(extract_dir),
                self.uaspeech_meta_csv,
                self.uaspeech_wordlist_csv,
                self.tokenizer,
                self.target_sr,
                split,
            )
            all_metadata.extend(ds.metadata)

    def _ds(self, split: str):
        return UASpeechDataset(
            str(self.cache_dir / "extracted"),
            self.uaspeech_meta_csv,
            self.uaspeech_wordlist_csv,
            self.tokenizer,
            self.target_sr,
            split,
            self.max_duration_sec,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self._ds("train")
        self.val_dataset = self._ds("validation")
        self.test_dataset = self._ds("test")

    def _dl(self, dataset, shuffle):
        return DataLoader(
            dataset,
            self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self._dl(self.train_dataset, True)

    def val_dataloader(self):
        return self._dl(self.val_dataset, False)

    def test_dataloader(self):
        return self._dl(self.test_dataset, False)

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
        return self._dl(ConcatDataset(datasets), False)


if __name__ == "__main__":
    from src.core.ipa_utils import IPATokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--uaspeech_meta_csv", type=str, required=True)
    parser.add_argument("--uaspeech_wordlist_csv", type=str, required=True)
    args = parser.parse_args()

    dm = UASpeechDataModule(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        uaspeech_meta_csv=args.uaspeech_meta_csv,
        uaspeech_wordlist_csv=args.uaspeech_wordlist_csv,
        tokenizer=IPATokenizer(),
        batch_size=2,
        num_workers=1,
        num_classes=5,
    )

    dm.prepare_data()
    dm.setup()
    print("Length of train dataset:", len(dm.train_dataset))
    print("Length of val dataset:", len(dm.val_dataset))
    print("Length of test dataset:", len(dm.test_dataset))

    for batch in dm.train_dataloader():
        print(batch)
        break
