import os
import yaml
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# ================================================================
# 1. Character-level Tokenizer for IPA with tone marks
# ================================================================

class CharTokenizer:
    """
    Character-level tokenizer that supports tonal IPA strings.
    Each Unicode codepoint becomes one token.
    """
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]

        self.special_tokens = special_tokens
        self.char2id = {tok: i for i, tok in enumerate(special_tokens)}
        self.id2char = {i: tok for i, tok in enumerate(special_tokens)}

    def build_vocab(self, text_list):
        """Builds the vocab from all training texts."""
        for text in text_list:
            for ch in text:
                if ch not in self.char2id:
                    idx = len(self.char2id)
                    self.char2id[ch] = idx
                    self.id2char[idx] = ch

    def encode(self, text):
        """Encode a string into a list of IDs."""
        ids = []
        for ch in text:
            if ch in self.char2id:
                ids.append(self.char2id[ch])
            else:
                ids.append(self.char2id["<unk>"])
        return ids

    def decode(self, ids):
        """Decode a list of IDs into string."""
        return "".join(self.id2char.get(i, "<unk>") for i in ids)

    @property
    def pad_id(self):
        return self.char2id["<pad>"]


# ================================================================
# 2. Tusom Dataset
# ================================================================

class TusomDataset(Dataset):
    """
    Dataset for your Tusom tonal-IPA data.
    Returns audio + 3 text fields:
      - IPA_only        (no tones)
      - tone_only       (tones only)
      - IPA_with_tone   (combined IPA with tones)
    """

    def __init__(self, yaml_path, wav_dir, tokenizer=None, load_audio=True):
        """
        Args:
            yaml_path: path to YAML file listing items
            wav_dir: directory containing WAV files
            tokenizer: CharTokenizer instance
            load_audio: Whether to load audio in __getitem__
        """
        self.wav_dir = Path(wav_dir)
        self.yaml_path = Path(yaml_path)
        self.tokenizer = tokenizer
        self.load_audio = load_audio

        # Load YAML data
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        # Convert YAML dict to list of entries
        self.items = []
        for filename, fields in self.data.items():
            entry = {
                "filename": filename,
                "IPA_only": fields.get("ipa_only", ""),
                "tone_only": fields.get("tone_only", ""),
                "IPA_with_tone": fields.get("tone_letters", ""),  # tonal IPA output
            }
            self.items.append(entry)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav_path = self.wav_dir / item["filename"]

        # Load audio
        if self.load_audio:
            wav, sr = torchaudio.load(str(wav_path))
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)  # convert to mono
            wav = wav.squeeze(0)  # (T,)
            audio_length = wav.size(0)
        else:
            wav = None
            audio_length = 0

        # Text fields
        IPA_only = item["IPA_only"]
        tone_only = item["tone_only"]
        IPA_with_tone = item["IPA_with_tone"]

        # Tokenize the tonal IPA (model target)
        if self.tokenizer is not None:
            target_ids = torch.LongTensor(self.tokenizer.encode(IPA_with_tone))
        else:
            target_ids = torch.LongTensor([])  # If tokenizer not ready yet

        return {
            "segment_id": item["filename"],
            "audio": wav,
            "audio_length": audio_length,
            "IPA_only": IPA_only,
            "tone_only": tone_only,
            "IPA_with_tone": IPA_with_tone,
            "target_ids": target_ids,
            "target_length": len(target_ids),
        }


# ================================================================
# 3. Collate Function for Padding
# ================================================================

def tusom_collate_fn(batch, pad_id):
    # Pad audio
    max_audio_len = max(x["audio_length"] for x in batch)
    batch_audio = []
    batch_audio_len = []
    for x in batch:
        wav = x["audio"]
        pad = max_audio_len - wav.size(0)
        wav = torch.nn.functional.pad(wav, (0, pad), value=-1.0)
        batch_audio.append(wav)
        batch_audio_len.append(x["audio_length"])

    batch_audio = torch.stack(batch_audio, dim=0)       # (B, T)
    batch_audio_len = torch.LongTensor(batch_audio_len)  # (B,)

    # Pad target ids
    max_tgt_len = max(x["target_length"] for x in batch)
    batch_target = []
    batch_target_len = []
    for x in batch:
        tgt = x["target_ids"]
        pad = max_tgt_len - len(tgt)
        tgt = torch.nn.functional.pad(tgt, (0, pad), value=pad_id)
        batch_target.append(tgt)
        batch_target_len.append(x["target_length"])

    batch_target = torch.stack(batch_target, dim=0)     # (B, L)
    batch_target_len = torch.LongTensor(batch_target_len)

    # Raw text fields also included (no padding)
    IPA_only = [x["IPA_only"] for x in batch]
    tone_only = [x["tone_only"] for x in batch]
    IPA_with_tone = [x["IPA_with_tone"] for x in batch]

    return {
        "speech": batch_audio,
        "speech_length": batch_audio_len,
        "target": batch_target,
        "target_length": batch_target_len,
        "IPA_only": IPA_only,
        "tone_only": tone_only,
        "IPA_with_tone": IPA_with_tone,
        "segment_id": [x["segment_id"] for x in batch],
    }


# ================================================================
# 4. LightningDataModule
# ================================================================

class TusomDataModule(pl.LightningDataModule):
    def __init__(self, train_yaml, dev_yaml, test_yaml, wav_dir, batch_size=8, num_workers=4):
        super().__init__()
        self.train_yaml = train_yaml
        self.dev_yaml = dev_yaml
        self.test_yaml = test_yaml
        self.wav_dir = wav_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create tokenizer (built from TRAIN ONLY)
        self.tokenizer = CharTokenizer()

    def setup(self, stage=None):
        # ----- Step 1: Load training dataset w/o encoding yet -----
        raw_train = yaml.safe_load(open(self.train_yaml, "r", encoding="utf-8"))
        tonal_texts = [v.get("tone_letters", "") for v in raw_train.values()]
        self.tokenizer.build_vocab(tonal_texts)

        # ----- Step 2: Create datasets -----
        self.train_dataset = TusomDataset(
            yaml_path=self.train_yaml,
            wav_dir=self.wav_dir,
            tokenizer=self.tokenizer,
        )
        self.dev_dataset = TusomDataset(
            yaml_path=self.dev_yaml,
            wav_dir=self.wav_dir,
            tokenizer=self.tokenizer,
        )
        self.test_dataset = TusomDataset(
            yaml_path=self.test_yaml,
            wav_dir=self.wav_dir,
            tokenizer=self.tokenizer,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=lambda b: tusom_collate_fn(b, pad_id=self.tokenizer.pad_id),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda b: tusom_collate_fn(b, pad_id=self.tokenizer.pad_id),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda b: tusom_collate_fn(b, pad_id=self.tokenizer.pad_id),
        )


# ================================================================
# 5. Simple test
# ================================================================

if __name__ == "__main__":
    dm = TusomDataModule(
        train_yaml="/ocean/projects/cis250187p/kxu7/PhoneBench/tusom2021-main/data/train.yml",
        dev_yaml="/ocean/projects/cis250187p/kxu7/PhoneBench/tusom2021-main/data/dev.yml",
        test_yaml="/ocean/projects/cis250187p/kxu7/PhoneBench/tusom2021-main/data/test.yml",
        wav_dir="/ocean/projects/cis250187p/kxu7/PhoneBench/tusom2021-main/data/wav",
        batch_size=2,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    print("speech:", batch["speech"].shape)
    print("speech_length:", batch["speech_length"])
    print("target:", batch["target"].shape)
    print("target_length:", batch["target_length"])
    print("IPA_with_tone:", batch["IPA_with_tone"])