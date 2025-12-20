"""Datamodule to read kaldi style powsm datasets using scp indices."""

import torch
import torchaudio
import kaldiio
from torch.utils.data import Dataset
import lightning as L
import yaml
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class KaldiDataset(Dataset):
    def __init__(self, wav_scp_file, text_file, lang_file, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.wav_scp = self._load_wav_scp(wav_scp_file)
        self.text = self._load_text(text_file)
        self.key2lang = self._extract_language(lang_file)

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

    def _load_wav_scp(self, path):
        wav_scp = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key, wav_path = parts[0], parts[1]
                    if not wav_path.startswith("/work"):
                        wav_path = f"/work/hdd/bbjs/shared/powsm/s2t1/{wav_path}"
                    wav_scp[key] = wav_path
        return wav_scp

    def _load_text(self, path):
        text_dict = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # some examples have spaces in between
                    # we must retain full length of transcript
                    text_dict[parts[0]] = " ".join(parts[1:])
        return text_dict

    def _extract_language(self, path):
        key2lang = {}
        with open(path) as f:
            for line in f:
                key, tag = line.strip().split()[:2]
                if key.endswith("_pr"):
                    # remove _pr suffix for some datasets.
                    # TODO(shikhar): Bad design, should be modified at source to make it generic.
                    key = key[:-3]
                key2lang[key] = tag.split("><")[0][1:].strip()
        return key2lang

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
        return {
            "key": key,
            "utt_id": key,
            "speech": waveform.to(torch.float32),
            "speech_length": waveform.shape[-1],
            "sr": self.sampling_rate,
            "wavpath": wav_path,
            # powsm lang sym. default is <unk> if missing in vocab
            "lang_sym": self.key2lang[key],
            "split": "test",
            "metadata_idx": idx,
            "target": transcription,
            "text": transcription,
        }


class KaldiDataModule(L.LightningDataModule):
    def __init__(
        self,
        wav_scp_file,
        text_file,
        lang_file,
        sampling_rate=16000,
        batch_size=16,
        num_workers=4,
    ):
        super().__init__()
        log.info(
            f"Initializing PowsmDataModule with {wav_scp_file}, {text_file}, {lang_file}"
        )
        self.wav_scp_file = wav_scp_file
        self.text_file = text_file
        self.lang_file = lang_file
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = KaldiDataset(
            self.wav_scp_file,
            self.text_file,
            self.lang_file,
            self.sampling_rate,
        )

    def train_dataloader(self):
        raise ValueError("This datamodule is not intended for training use.")

    def val_dataloader(self):
        raise ValueError("This datamodule is not intended for validation use.")

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        keys = [item["key"] for item in batch]
        speeches = [item["speech"] for item in batch]
        speech_lengths = torch.tensor([item["speech_length"] for item in batch])
        texts = [item["text"] for item in batch]
        wavpaths = [item["wavpath"] for item in batch]
        languages = [item["lang_sym"] for item in batch]

        # Pad speeches to the max length in the batch
        max_length = max(speech_lengths)
        padded_speeches = torch.zeros(len(batch), max_length)
        for i, speech in enumerate(speeches):
            padded_speeches[i, : speech.shape[-1]] = speech

        return {
            "keys": keys,
            "speech": padded_speeches,
            "speech_length": speech_lengths,
            "text": texts,
            "wavpath": wavpaths,
            "lang_sym": languages,
        }


def build_kaldi_datamodule(
    dataset_name,
    dataset_config_path="configs/data/powsm_evalset_index.yaml",
    sampling_rate=16000,
    batch_size=16,
    num_workers=4,
):
    with open(dataset_config_path) as f:
        config = yaml.safe_load(f)

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_config = config["datasets"][dataset_name]
    wav_scp_file = ds_config["wav_scp"]
    text_file = ds_config["text_phoneme"]
    lang_file = ds_config["language"]

    return KaldiDataModule(
        wav_scp_file=wav_scp_file,
        text_file=text_file,
        lang_file=lang_file,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # Test with: python -m src.data.kaldi_dataset
    datamodule = build_kaldi_datamodule("doreco", batch_size=2, num_workers=1)
    datamodule.setup()
    for batch in datamodule.predict_dataloader().dataset:
        print(batch)
        break
