import pyarrow.parquet as pq  # before torch
import os, io, torch
from typing import Dict, List, Optional, Tuple
import torchaudio
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import pandas as pd
import math


def pad_collate(batch):
    L = [b["audio"].shape[-1] for b in batch]
    M = max(L)
    A = [
        torch.nn.functional.pad(b["audio"], (0, M - b["audio"].shape[-1]))
        for b in batch
    ]
    return {
        "audio": torch.stack(A, 0),
        "lengths": torch.tensor(L),
        "sr": batch[0]["sr"],
        "pincode": [b["pincode"] for b in batch],
        "latitude": torch.tensor([b["latitude"] for b in batch]),
        "longitude": torch.tensor([b["longitude"] for b in batch]),
    }


class VaaniParquetDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        split: str,  # 'train', 'valid', or 'test'
        audio_root: str,
        target_sr: Optional[int] = 16000,
    ):
        self.audio_root = audio_root
        self.target_sr = target_sr
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(
            drop=True
        )
        self.audio_column = "audio"
        self.max_output_audio_length = 20  # seconds
        self._pf_cache = {}  # Cache ParquetFile objects

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        row = self.metadata.iloc[i]

        # Get parquet file (with caching)
        pq_path = os.path.join(self.audio_root, row["parquet_path"])
        if pq_path not in self._pf_cache:
            self._pf_cache[pq_path] = pq.ParquetFile(pq_path)
        pf = self._pf_cache[pq_path]

        # Read specific row
        tbl = pf.read_row_group(row["row_group"], columns=[self.audio_column])
        audio_data = tbl[self.audio_column][row["row_index"]].as_py()

        # Process audio
        b = audio_data["bytes"]
        wav, sr = torchaudio.load(io.BytesIO(b))
        if self.target_sr and sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            sr = self.target_sr

        # Trim
        max_len = sr * self.max_output_audio_length
        if wav.shape[-1] > max_len:
            wav = wav[:, :max_len]
        wav = wav.squeeze(0)  # (T,)

        # Convert lat/lon to radians
        latitude = (
            math.radians(row["latitude"]) if not math.isnan(row["latitude"]) else 0.0
        )
        longitude = (
            math.radians(row["longitude"]) if not math.isnan(row["longitude"]) else 0.0
        )

        return {
            "audio": wav,
            "lengths": wav.shape[-1],
            "sr": sr,
            "pincode": row["pincode"] if not pd.isna(row["pincode"]) else 0,
            "latitude": latitude,
            "longitude": longitude,
        }


class VaaniGeolocation(LightningDataModule):
    def __init__(
        self,
        data_dir,
        metadata_path,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        target_sr=16000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = self.ds_val = self.ds_test = None
        self.bs_dev = batch_size

    def setup(self, stage: Optional[str] = None):
        if self.trainer:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError("batch_size not divisible by world_size")
            self.bs_dev = self.hparams.batch_size // self.trainer.world_size

        if self.ds_train is None:
            self.ds_train = VaaniParquetDataset(
                self.hparams.metadata_path,
                "train",
                self.hparams.data_dir,
                self.hparams.target_sr,
            )
            self.ds_val = VaaniParquetDataset(
                self.hparams.metadata_path,
                "valid",
                self.hparams.data_dir,
                self.hparams.target_sr,
            )
            self.ds_test = VaaniParquetDataset(
                self.hparams.metadata_path,
                "test",
                self.hparams.data_dir,
                self.hparams.target_sr,
            )
            print(
                f"Dataset split into train: {len(self.ds_train)}, val: {len(self.ds_val)}, test: {len(self.ds_test)}"
            )

    def _dl(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.bs_dev,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=pad_collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dl(self.ds_train, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.ds_test, shuffle=False)


if __name__ == "__main__":
    dm = VaaniGeolocation(
        data_dir="/work/hdd/bbjs/shared/corpora/vaani_iisc",
        metadata_path="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/notebook/vaani_geolocation_metadata_small.withsplits.csv",  # Path to the generated metadata
        batch_size=2,
        num_workers=1,
        target_sr=16000,
    )
    dm.setup()
    for x in dm.train_dataloader():
        print(x["pincode"])
        print(x["audio"].shape)
        print(x["lengths"].shape)
        print(x["latitude"])
        print(x["longitude"])
        break
