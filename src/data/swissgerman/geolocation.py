"""Swiss-German Geolocation Datasets and DataModule.
https://github.com/gamba/swiss-geolocation/tree/master
for zipcode to lat/lon mapping.

Usage (for naive baseline):
    python -m src.data.swissgerman.geolocation
"""

import os, torch
from typing import Optional
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning import LightningDataModule
import pandas as pd
import math

from src.core.utils import resample_dataset


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
        "zipcode": [b["zipcode"] for b in batch],
        "target": torch.stack(
            [torch.tensor(b["target"], dtype=torch.float32) for b in batch]
        ),
        "split": [b.get("split", "none") for b in batch],
        "metadata_idx": [b["metadata_idx"] for b in batch],
    }


class SwissGermanDataset(Dataset):
    def __init__(
        self,
        metadata_df: str,
        data_dir: str,
        target_sr: Optional[int] = 16000,
    ):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.metadata = metadata_df
        self.path_key = "audio_path"
        self.max_output_audio_length = 20  # seconds
        self._pf_cache = {}  # Cache ParquetFile objects

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        row = self.metadata.iloc[i]
        full_audio_path = os.path.join(
            self.data_dir, f"resampled_{self.target_sr}Hz", row[self.path_key]
        )
        assert os.path.exists(
            full_audio_path
        ), f"Audio file not found: {full_audio_path}"
        wav, sr = torchaudio.load(full_audio_path)  # (1, T)
        # for mp3 files, decoding can result in overshooting beyond [-1, 1]
        wav = wav.clamp(-1.0, 1.0)
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
            "audio_path": full_audio_path,
            "speech_length": wav.shape[-1],
            "lang_sym": "<unk>",  # for powsm, swiss-german not in powsm list, <deu>
            "sr": sr,
            "zipcode": row["zipcode"] if not pd.isna(row["zipcode"]) else 0,
            "split": row["split"],
            "metadata_idx": i,
            "target": [latitude, longitude],
        }


class SwissGermanGeolocation(LightningDataModule):
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
        self.metadata_df = pd.read_csv(metadata_path)
        self.ds_train = self.ds_val = self.ds_test = None
        self.bs_dev = batch_size

    def prepare_data(self):
        """Prepare data by resampling audio."""
        tgt_dir = os.path.join(
            self.hparams.data_dir, f"resampled_{self.hparams.target_sr}Hz"
        )
        resample_dataset(
            metadata_df=pd.read_csv(self.hparams.metadata_path),
            path_key="audio_path",
            src_data_dir=self.hparams.data_dir,
            src_sr=44100,
            tgt_data_dir=tgt_dir,
            tgt_sr=self.hparams.target_sr,
            force_resample=False,
        )

    def setup(self, stage: Optional[str] = None):
        if self.trainer:
            if self.hparams.batch_size % self.trainer.world_size:
                raise RuntimeError("batch_size not divisible by world_size")
            self.bs_dev = self.hparams.batch_size // self.trainer.world_size

        if self.ds_train is None:
            self.ds_train = SwissGermanDataset(
                self.metadata_df[self.metadata_df["split"] == "train"].reset_index(
                    drop=True
                ),
                self.hparams.data_dir,
                self.hparams.target_sr,
            )
            self.ds_val = SwissGermanDataset(
                self.metadata_df[self.metadata_df["split"] == "valid"].reset_index(
                    drop=True
                ),
                self.hparams.data_dir,
                self.hparams.target_sr,
            )
            self.ds_test = SwissGermanDataset(
                self.metadata_df[self.metadata_df["split"] == "test"].reset_index(
                    drop=True
                ),
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
    metadata_path = (
        "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/cache/swissgerman/metadata.csv"
    )
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
    # from tqdm import tqdm

    # dm = SwissGermanGeolocation(
    #     data_dir="/work/hdd/bbjs/shared/corpora/swiss_german",
    #     metadata_path="/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/swissgerman_cache/metadata.csv",
    #     batch_size=2,
    #     num_workers=1,
    #     target_sr=16000,
    # )
    # dm.setup()
    # for batch in tqdm(dm.train_dataloader(), desc="Train"):
    #     pass
    # for batch in tqdm(dm.val_dataloader(), desc="Val"):
    #     pass
    # for batch in tqdm(dm.test_dataloader(), desc="Test"):
    #     pass
