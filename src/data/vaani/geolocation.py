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
        "latitude": torch.tensor([b["latitude"] for b in batch]),
        "longitude": torch.tensor([b["longitude"] for b in batch]),
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
            "latitude": latitude,
            "longitude": longitude,
            "split": row["split"],
            "metadata_idx": i,
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

    def predict_dataloader(self):
        return self._dl(
            ConcatDataset([self.ds_train, self.ds_val, self.ds_test]), shuffle=False
        )


def _naive_baseline():
    metadata_path = "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/vaani_geolocation/data/tmp/vaani_geolocation_metadata.train10.csv"
    import pandas as pd

    metadata = pd.read_csv(metadata_path)
    trainset = metadata[metadata["split"] == "train"]
    lat_av = trainset["latitude"].mean()
    long_av = trainset["longitude"].mean()
    print(f"Train set average latitude: {lat_av}")
    print(f"Train set average longitude: {long_av}")

    test_set = metadata[metadata["split"] == "test"]
    from src.recipe.geolocation.model_module import GeolocationAngularLoss

    loss_fn = GeolocationAngularLoss()
    total_loss = 0.0
    count = 0
    for idx, row in test_set.iterrows():
        true_lat = (
            math.radians(row["latitude"]) if not math.isnan(row["latitude"]) else 0.0
        )
        true_long = (
            math.radians(row["longitude"]) if not math.isnan(row["longitude"]) else 0.0
        )
        pred_lat = math.radians(lat_av) if not math.isnan(lat_av) else 0.0
        pred_long = math.radians(long_av) if not math.isnan(long_av) else 0.0
        loss_val = loss_fn(
            pred_lat=torch.tensor([pred_lat]),
            pred_long=torch.tensor([pred_long]),
            true_lat=torch.tensor([true_lat]),
            true_long=torch.tensor([true_long]),
        )
        total_loss += loss_val.item()
        count += 1
    print(f"Total test loss: {total_loss}")
    print(f"Average test loss: {total_loss / count if count > 0 else 0}")


if __name__ == "__main__":
    # A naive baseline using average of the training set
    _naive_baseline()
    # from tqdm import tqdm

    # dm = VaaniGeolocation(
    #     data_dir="/work/hdd/bbjs/shared/corpora/vaani_iisc",
    #     metadata_path="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/vaani_geolocation/data/vaani_geolocation_metadata.train10.csv",  # Path to the generated metadata
    #     batch_size=2,
    #     num_workers=1,
    #     target_sr=16000,
    # )
    # # Average of dataset
    # dm.setup()
    # lat_av = 0
    # long_av = 0
    # count = 0
    # c = 0
    # for x in tqdm(dm.train_dataloader(), desc="Train batches", unit="batch"):
    #     lat_av += x["latitude"].sum().item()
    #     long_av += x["longitude"].sum().item()
    #     count += x["latitude"].numel()
    #     c += 1
    #     if c > 1000:
    #         break

    # lat_av /= count
    # long_av /= count
    # print(f"Train set average latitude (radians): {lat_av}")
    # print(f"Train set average longitude (radians): {long_av}")

    # calculate metrics based on this average
    # from src.recipe.geolocation.model_module import GeolocationAngularLoss

    # lat_av = 0.4340759042825375
    # long_av = 1.413005766156432
    # loss_fn = GeolocationAngularLoss()
    # total_loss = 0.0
    # count = 0
    # c_iter = 0
    # for x in tqdm(dm.val_dataloader(), desc="Validation batches", unit="batch"):
    #     loss_val = loss_fn(
    #         pred_lat=torch.full_like(x["latitude"], lat_av),
    #         pred_long=torch.full_like(x["longitude"], long_av),
    #         true_lat=x["latitude"],
    #         true_long=x["longitude"],
    #     )
    #     total_loss += loss_val.item()
    #     count += x["latitude"].numel()
    #     c_iter += 1
    #     if c_iter % 100:
    #         print(f"Processed {c_iter} validation batches")
    #         print(
    #             f"Current average validation loss: {total_loss / count if count > 0 else 0}"
    #         )
    #     # break
    #     # print(f"Validation loss: {loss_val.item()}")
    # print(f"Total validation loss: {total_loss}")
    # print(f"Average validation loss: {total_loss / count if count > 0 else 0}")
