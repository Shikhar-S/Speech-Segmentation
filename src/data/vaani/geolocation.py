"""Vaani Geolocation HF DataModule.

Usage:
    python -m src.data.vaani.geolocation --hf_repo shikhar7ssu/vaani140-geo \
    --cache_dir exp/cache/vaanihindi
"""

import io
import torch
import math
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, ConcatDataset
from lightning import LightningDataModule
from datasets import load_dataset
import torchaudio
from src.core.utils import download_hf_snapshot


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
        "utt_id": [b.get("utt_id", "none") for b in batch],
        "audio_path": [b.get("audio_path", "") for b in batch],
        "metadata_idx": [b.get("metadata_idx", -1) for b in batch],
    }


class VaaniHFDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, max_audio_length=20, target_sr=16000, cache_dir=None
    ):
        self.ds = hf_dataset
        self.max_audio_length = max_audio_length
        self.target_sr = target_sr
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.ds)

    def _cache_audio(self, waveform: torch.Tensor, sr: int, target_path: Path) -> None:
        if target_path.exists():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(target_path), waveform, sr)

    def __getitem__(self, i):
        item = self.ds[i]
        utt_id = f'{item["split"]}_{i}'
        wav, sr = torchaudio.load(io.BytesIO(item["audio"]))
        if self.target_sr and sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            sr = self.target_sr
        max_len = sr * self.max_audio_length
        if wav.shape[-1] > max_len:
            wav = wav[:, :max_len]
        target_path = Path(self.cache_dir) / f'saved_{item["split"]}' / f"{utt_id}.wav"
        self._cache_audio(wav, sr, target_path)
        lat = math.radians(item["latitude"])
        lon = math.radians(item["longitude"])
        wav = wav.squeeze(0)  # (T,)

        return {
            "speech": wav,
            "speech_length": wav.shape[-1],
            "sr": sr,
            "pincode": item["pincode"],
            "lang_sym": item["lang_sym"],
            "split": item["split"],
            "utt_id": utt_id,
            "target": [lat, lon],
            "audio_path": str(target_path),
            "metadata_idx": i,
        }


class VaaniGeolocation(LightningDataModule):
    def __init__(
        self,
        hf_repo: str,
        cache_dir: str,  # Local path to store/cache the dataset
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
        num_classes: int = 3,
        predict_splits: Optional[list] = None,  # Splits for predict_dataloader, default: all
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = self.ds_val = self.ds_test = None
        self.predict_splits = predict_splits or ["train", "val", "test"]

    def prepare_data(self):
        download_hf_snapshot(
            repo_id=self.hparams.hf_repo,
            work_dir=self.hparams.cache_dir,
            repo_type="dataset",
        )

    def _ds(self, ds):
        return VaaniHFDataset(
            ds,
            target_sr=self.hparams.target_sr,
            cache_dir=self.hparams.cache_dir,
        )

    def setup(self, stage: Optional[str] = None):
        full_ds = load_dataset(self.hparams.cache_dir, split="train")
        train_split = full_ds.filter(lambda x: x["split"] == "train")
        val_split = full_ds.filter(lambda x: x["split"] in ["val", "valid"])
        test_split = full_ds.filter(lambda x: x["split"] == "test")

        self.ds_train = self._ds(train_split)
        self.ds_val = self._ds(val_split)
        self.ds_test = self._ds(test_split)
        if stage == "fit" or stage is None:
            print(f"Train: {len(self.ds_train)}, Val: {len(self.ds_val)}")
        if stage == "test":
            print(f"Test: {len(self.ds_test)}")

    def _dl(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=pad_collate,
        )

    def train_dataloader(self):
        return self._dl(self.ds_train, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.ds_test, shuffle=False)

    def predict_dataloader(self):
        datasets = []
        if "train" in self.predict_splits and self.ds_train:
            datasets.append(self.ds_train)
        if "val" in self.predict_splits and self.ds_val:
            datasets.append(self.ds_val)
        if "test" in self.predict_splits and self.ds_test:
            datasets.append(self.ds_test)
        if not datasets:
            datasets = [self.ds_test]  # fallback to test
        return self._dl(ConcatDataset(datasets), shuffle=False)


def naive_baseline():
    from tqdm import tqdm
    import math
    import torch

    lat_min = math.inf
    lat_max = -math.inf
    long_min = math.inf
    long_max = -math.inf

    dm = VaaniGeolocation(
        hf_repo=args.hf_repo, cache_dir=args.cache_dir, batch_size=2, num_workers=1
    )
    dm.prepare_data()
    dm.setup()

    av_lat = 0
    av_long = 0
    count = 0

    # --- 1. Process Train Set ---
    for item in tqdm(dm.train_dataloader().dataset, "Scanning train set..."):
        av_lat += item["target"][0]
        av_long += item["target"][1]

        lat_min = min(lat_min, item["target"][0])
        lat_max = max(lat_max, item["target"][0])
        long_min = min(long_min, item["target"][1])
        long_max = max(long_max, item["target"][1])
        count += 1

    av_lat /= count
    av_long /= count

    # --- 2. Process Test Set ---
    from src.recipe.common.geolocation_loss import GeolocationAngularLoss

    pred_x = math.cos(av_lat) * math.cos(av_long)
    pred_y = math.cos(av_lat) * math.sin(av_long)
    pred_z = math.sin(av_lat)
    pred_tensor = torch.tensor([[pred_x, pred_y, pred_z]])
    loss_fn = GeolocationAngularLoss()

    total_loss = 0.0
    count = 0
    test_set = dm.test_dataloader().dataset

    for item in tqdm(test_set, total=len(test_set), desc="Scanning test set..."):
        target_tensor = torch.tensor([item["target"]])
        loss_val = loss_fn(
            prediction=pred_tensor,
            target=target_tensor,
        )
        total_loss += loss_val.item()
        count += 1

        lat_min = min(lat_min, item["target"][0])
        lat_max = max(lat_max, item["target"][0])
        long_min = min(long_min, item["target"][1])
        long_max = max(long_max, item["target"][1])

    # --- 3. Calculate Extents using Haversine ---
    def haversine_km(lat1, lon1, lat2, lon2):
        R_EARTH_KM = 6371.0
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R_EARTH_KM * c

    # Midpoint latitude for calculating the 'average' width
    mean_lat = 0.5 * (lat_min + lat_max)

    # North-South: Distance between min and max latitude (along the same longitude)
    ns_extent_km = haversine_km(lat_min, 0, lat_max, 0)

    # East-West: Distance between min and max longitude (at the mean latitude)
    ew_extent_km = haversine_km(mean_lat, long_min, mean_lat, long_max)

    print("-" * 30)
    print(f"North-South Extent: {ns_extent_km:.2f} km")
    print(f"East-West Extent:   {ew_extent_km:.2f} km")
    print(f"Total test loss:    {total_loss}")
    print(f"Average test loss:  {total_loss / count if count > 0 else 0}")
    print("-" * 30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    args = parser.parse_args()

    naive_baseline()
