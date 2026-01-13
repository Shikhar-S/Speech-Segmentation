"""Datamodule to read kaldi style powsm datasets using scp indices."""

import torch
import torchaudio
import kaldiio
from torch.utils.data import Dataset
import lightning as L
import yaml
from typing import Optional, Dict, List
from tqdm import tqdm
from src.utils import RankedLogger
import json

log = RankedLogger(__name__, rank_zero_only=True)
# TODO(shikhar): Separate out tokenizer. Use char_tokenizer.


class KaldiDataset(Dataset):
    def __init__(
        self,
        wav_scp_file,
        text_file,
        lang_file,
        sampling_rate=16000,
        split="test",
        vocab_file: Optional[str] = None,
        task_set: Optional[List[str]] = None,
    ):
        self.sampling_rate = sampling_rate
        self.task_set = task_set  # set of tasks to filter on, e.g., ['pr', 'asr']
        self.wav_scp = self._load_wav_scp(wav_scp_file)
        self.text = self._load_text(text_file)
        self.key2lang = self._extract_language(lang_file)
        self.split = split
        self.max_duration_sec = 20

        # Load vocabulary for tokenization
        self.vocab = self._load_vocab(vocab_file) if vocab_file else None
        self.unk_id = -1 if not self.vocab else self.vocab.get("<unk>", -1)

        assert set(self.wav_scp.keys()).issubset(
            set(self.text.keys())
        ), "Extra key in wav.scp"
        self.keys = list(self.wav_scp.keys())
        assert all(
            k in self.key2lang for k in self.keys
        ), "Missing language tags for some keys"
        log.info(
            f"Loaded dataset: {len(self.key2lang)} lang keys, {len(self.keys)} samples"
        )
        log.info(f"Number of unique languages: {len(set(self.key2lang.values()))}")
        if vocab_file:
            log.info(
                f"Loaded vocabulary with {len(self.vocab)} tokens from {vocab_file}."
                f" Unk ID: {self.unk_id}"
            )

    def _keep_key(self, key: str) -> bool:
        """Check if a key should be kept based on task_set."""
        if self.task_set is None:
            return True
        return any(key.endswith(f"_{task}") for task in self.task_set)

    def _load_wav_scp(self, path):
        wav_scp = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key, wav_path = parts[0], parts[1]
                    if not self._keep_key(key):
                        continue
                    if not wav_path.startswith("/work"):
                        wav_path = f"/work/hdd/bbjs/shared/powsm/s2t1/{wav_path}"
                    wav_scp[key] = wav_path
        return wav_scp

    def _load_text(self, path):
        text_dict = {}
        with open(path) as f:
            for line in tqdm(f, desc="Reading text"):
                key, *remaining = line.strip().split()
                if not self._keep_key(key):
                    continue
                if len(remaining) >= 1:
                    # some examples have spaces in between
                    # we must retain full length of transcript
                    text_dict[key] = " ".join(remaining)
        return text_dict

    def _extract_language(self, path):
        key2lang = {}
        with open(path) as f:
            for line in tqdm(f, desc="Reading language"):
                key, tag = line.strip().split()[:2]
                if not self._keep_key(key):
                    continue
                if key.endswith("_pr") and "pr" not in (self.task_set or []):
                    # remove _pr suffix for some datasets in evalset index.
                    # but do not remove if we are using pr task in training.
                    # TODO(shikhar): Bad design, should be modified at source to make it generic.
                    key = key[:-3]
                key2lang[key] = tag.split("><")[0][1:].strip()
        return key2lang

    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Return a Dictionary that maps token string to token ID.
        vocab_file: Path to vocabulary file which is a json of token to id mapping.
        """
        with open(vocab_file) as f:
            vocab = json.load(f)
        return vocab

    def _tokenize_text(self, text: str) -> List[int]:
        """Return List of token IDs. Unknown tokens are replaced with ignore_id.
        text: Text string (typically phonetic transcription).
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not loaded. Provide vocab_file parameter.")
        tokens = [
            self.vocab.get(token.strip(), self.unk_id)
            for token in text.strip("/").split("//")
        ]
        return tokens

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        wav_path = self.wav_scp[key]
        transcription = self.text[key]

        if ".ark" in wav_path:
            sr, wav = kaldiio.load_mat(wav_path)
            waveform = torch.from_numpy(wav).float().unsqueeze(0)
        else:
            waveform, sr = torchaudio.load(wav_path)

        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        waveform = waveform.squeeze(0)  # (1, T) -> (T,)
        # Trim if longer than max duration
        waveform = waveform[: self.max_duration_sec * self.sampling_rate]
        wavlen = waveform.shape[-1]

        # Tokenize text if vocabulary is loaded
        text_tokens = self._tokenize_text(transcription) if self.vocab else None

        return {
            "key": key,
            "utt_id": key,
            "speech": waveform.to(torch.float32),
            "speech_length": wavlen,
            "sr": self.sampling_rate,
            "text_tokens": text_tokens,
            "wavpath": wav_path,
            # powsm lang sym. default is <unk> if missing in vocab
            "lang_sym": self.key2lang[key],
            "split": self.split,
            "metadata_idx": idx,
            "target": transcription,
            "text": transcription,
        }


class KaldiDataModule(L.LightningDataModule):
    def __init__(
        self,
        wav_scp_file: Dict[str, str],
        text_file: Dict[str, str],
        lang_file: Dict[str, str],
        sampling_rate=16000,
        batch_size=16,
        num_workers=4,
        task_set: Dict[str, List[str]] = None,
        vocab_file: Optional[str] = None,
    ):
        """
        For wav_scp_file, text_file, lang_file
        Dict[str, str] (mapping split name to file path) must be provided.
        """
        super().__init__()
        log.info(
            f"Initializing KaldiDataModule with {wav_scp_file}, {text_file}, {lang_file}"
        )
        self.wav_scp_file = wav_scp_file
        self.text_file = text_file
        self.lang_file = lang_file
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.task_set = task_set
        self.splits = list(wav_scp_file.keys())
        assert (
            set(self.splits) == set(text_file.keys()) == set(lang_file.keys())
        ), "Mismatch in splits across wav_scp, text, and lang files."
        self.ignore_id = -1  # for CTC loss padding

    def _ds(self, split):
        return KaldiDataset(
            self.wav_scp_file[split],
            self.text_file[split],
            self.lang_file[split],
            self.sampling_rate,
            split=split,
            vocab_file=self.vocab_file,
            task_set=self.task_set[split],
        )

    def setup(self, stage=None):
        for split in self.splits:
            setattr(self, f"{split}_dataset", self._ds(split=split))

    def _dl(self, split):
        return torch.utils.data.DataLoader(
            getattr(self, f"{split}_dataset"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return self._dl(split="train")

    def val_dataloader(self):
        return self._dl(split="dev1k")

    def test_dataloader(self):
        return self._dl(split="predict")

    def predict_dataloader(self):
        return self._dl(split="predict")

    def collate_fn(self, batch):
        keys = [item["key"] for item in batch]
        speeches = [item["speech"] for item in batch]
        speech_lengths = torch.tensor([item["speech_length"] for item in batch])
        texts = [item["text"] for item in batch]
        wavpaths = [item["wavpath"] for item in batch]
        languages = [item["lang_sym"] for item in batch]

        # Pad speeches to the max length in the batch
        max_speech_length = max(speech_lengths)
        max_speech_length = 20 * self.sampling_rate  # enforce max length
        padded_speeches = torch.zeros(len(batch), max_speech_length)
        for i, speech in enumerate(speeches):
            padded_speeches[i, : speech.shape[-1]] = speech

        # Handle text tokenization for CTC-based training
        text_data = {"text": texts}

        if batch[0].get("text_tokens") is not None:
            # Pad tokenized text to max length in batch
            text_tokens_list = [item["text_tokens"] for item in batch]
            max_text_length = max(len(tokens) for tokens in text_tokens_list)

            padded_texts = torch.full(
                (len(batch), max_text_length),
                self.ignore_id,
                dtype=torch.long,
            )
            text_lengths = torch.zeros(len(batch), dtype=torch.long)

            for i, tokens in enumerate(text_tokens_list):
                padded_texts[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                text_lengths[i] = len(tokens)

            text_data["text"] = padded_texts
            text_data["text_length"] = text_lengths

        return {
            "keys": keys,
            "speech": padded_speeches,
            "speech_length": speech_lengths,
            "text": text_data["text"],
            "text_length": text_data.get("text_length"),
            "wavpath": wavpaths,
            "lang_sym": languages,
        }


def build_kaldi_datamodule(
    dataset_name,
    dataset_config_path="configs/data/powsm_evalset_index.yaml",
    batch_size=16,
    num_workers=4,
    vocab_file: Optional[str] = None,
):
    with open(dataset_config_path) as f:
        config = yaml.safe_load(f)

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_config = config["datasets"][dataset_name]

    if "wav_scp" in ds_config:
        # we are missing the split level.
        # typically for eval datasets
        splits = ["predict"]
        ds_config = {splits[0]: ds_config}
    else:
        splits = list(ds_config.keys())
    log.info("splits:", splits)
    wav_scp_file, text_file, lang_file, task_set = {}, {}, {}, {}
    for split in splits:
        wav_scp_file[split] = ds_config[split]["wav_scp"]
        text_file[split] = ds_config[split]["text_phoneme"]
        lang_file[split] = ds_config[split]["language"]
        task_set[split] = ds_config[split].get("task_set", None)

    sampling_rate = config.get("sampling_rate", 16000)

    return KaldiDataModule(
        wav_scp_file=wav_scp_file,
        text_file=text_file,
        lang_file=lang_file,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        num_workers=num_workers,
        task_set=task_set,
        vocab_file=vocab_file,
    )


if __name__ == "__main__":
    # Test with: python -m src.data.kaldi_dataset
    # datamodule = build_kaldi_datamodule("doreco", batch_size=2, num_workers=1)
    # datamodule.setup()
    # print(len(datamodule.predict_dataloader().dataset))
    # for batch in datamodule.predict_dataloader().dataset:
    #     print(batch)
    #     break
    #######
    datamodule = build_kaldi_datamodule(
        dataset_name="pr_fixed",
        dataset_config_path="configs/data/ipapack_index.yaml",
        batch_size=2,
        num_workers=1,
        vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
    )
    datamodule.setup()
    for i in datamodule.train_dataloader().dataset:
        print(i["speech_length"])
        break
    # print(len(datamodule.predict_dataloader().dataset))
    # for batch in datamodule.predict_dataloader().dataset:
    #     print(batch)
    #     break
