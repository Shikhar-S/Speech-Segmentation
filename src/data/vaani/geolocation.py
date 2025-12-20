"""Vaani Geolocation Dataset and DataModule.

Usage (for naive baseline):
    python -m src.data.vaani.geolocation
"""

import pyarrow.parquet as pq  # before torch
import os, io, torch
from typing import Optional
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
import pandas as pd
import math


def pad_collate(batch):
    L = [b["speech"].shape[-1] for b in batch]
    M = max(L)
    A = [
        torch.nn.functional.pad(b["speech"], (0, M - b["speech"].shape[-1]))
        for b in batch
    ]
    return {
        "speech": torch.stack(A, 0),
        "speech_length": torch.tensor(L),
        "sr": batch[0]["sr"],
        "pincode": [b["pincode"] for b in batch],
        "target": torch.stack(
            [torch.tensor(b["target"], dtype=torch.float32) for b in batch]
        ),
        "split": [b.get("split", "none") for b in batch],
        "metadata_idx": [b["metadata_idx"] for b in batch],
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
            "speech": wav,
            "speech_length": wav.shape[-1],
            "sr": sr,
            "pincode": row["pincode"] if not pd.isna(row["pincode"]) else 0,
            # for powsm # NOTE(shikhar): maybe this can be better handled,
            # there is a self-reported (non-standard) language column
            "lang_sym": "<unk>",
            "split": row["split"],
            "metadata_idx": i,
            "target": [latitude, longitude],
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
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        if self.trainer:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError("batch_size not divisible by world_size")
            self.batch_size = self.hparams.batch_size // self.trainer.world_size

        if self.ds_train is None:
            self.ds_train = VaaniParquetDataset(
                self.hparams.metadata_path,
                "train",
                self.hparams.data_dir,
                self.hparams.target_sr,
            )
            self.ds_val = VaaniParquetDataset(
                self.hparams.metadata_path,
                "val",
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
            batch_size=self.batch_size,
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

    def predict_dataloader(self):
        return self._dl(
            ConcatDataset([self.ds_train, self.ds_val, self.ds_test]), shuffle=False
        )


def _naive_baseline():
    metadata_path = "exp/cache/vaani_geolocation/vaani_geolocation_metadata.train10.csv"
    import pandas as pd
    from tqdm import tqdm

    metadata = pd.read_csv(metadata_path)
    trainset = metadata[metadata["split"] == "train"]
    lat_av = trainset["latitude"].mean()
    long_av = trainset["longitude"].mean()
    print(f"Train set average latitude: {lat_av}")
    print(f"Train set average longitude: {long_av}")
    pred_lat = math.radians(lat_av) if not math.isnan(lat_av) else 0.0
    pred_long = math.radians(long_av) if not math.isnan(long_av) else 0.0
    pred_x = math.cos(pred_lat) * math.cos(pred_long)
    pred_y = math.cos(pred_lat) * math.sin(pred_long)
    pred_z = math.sin(pred_lat)
    pred_tensor = torch.tensor([[pred_x, pred_y, pred_z]])

    test_set = metadata[metadata["split"] == "test"]
    from src.recipe.common.geolocation_loss import GeolocationAngularLoss

    loss_fn = GeolocationAngularLoss()
    total_loss = 0.0
    count = 0
    for idx, row in tqdm(test_set.iterrows(), total=len(test_set)):
        true_lat = (
            math.radians(row["latitude"]) if not math.isnan(row["latitude"]) else 0.0
        )
        true_long = (
            math.radians(row["longitude"]) if not math.isnan(row["longitude"]) else 0.0
        )
        loss_val = loss_fn(
            prediction=pred_tensor,
            target=torch.tensor([[true_lat, true_long]]),
        )
        total_loss += loss_val.item()
        count += 1
    print(f"Total test loss: {total_loss}")
    print(f"Average test loss: {total_loss / count if count > 0 else 0}")


if __name__ == "__main__":
    # A naive baseline using average of the training set
    _naive_baseline()
