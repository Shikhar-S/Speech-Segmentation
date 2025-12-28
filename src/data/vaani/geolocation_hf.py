"""Vaani Geolocation HF DataModule.

Usage:
    python -m src.data.vaani.geolocation_hf --hf_repo shikhar7ssu/vaani140-geo \
    --work_dir exp/cache/vaanihindi
"""

import io
import torch
import math
from typing import Optional
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from datasets import load_dataset
import torchaudio
from src.core.utils import download_hf_snapshot


def pad_collate(batch):
    # Determine max length in this batch
    L = [b["speech"].shape[-1] for b in batch]
    M = max(L)

    # Pad all audio to M
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


class VaaniHFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, max_audio_length=20, target_sr=16000):
        self.ds = hf_dataset
        self.max_audio_length = max_audio_length
        self.target_sr = target_sr

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        wav, sr = torchaudio.load(io.BytesIO(item["audio"]))
        if self.target_sr and sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            sr = self.target_sr
        max_len = sr * self.max_audio_length
        if wav.shape[-1] > max_len:
            wav = wav[:, :max_len]
        wav = wav.squeeze(0)  # (T,)
        # Convert lat/lon to radians
        lat = math.radians(item["latitude"])
        lon = math.radians(item["longitude"])

        return {
            "speech": wav,
            "speech_length": wav.shape[-1],
            "sr": sr,
            "pincode": item["pincode"],
            "lang_sym": item["lang_sym"],
            "split": item["split"],
            "metadata_idx": item["metadata_idx"],
            "utt_id": f'{item["split"]}_{item["metadata_idx"]}',
            "target": [lat, lon],
        }


class VaaniGeolocation(LightningDataModule):
    def __init__(
        self,
        hf_repo: str,
        work_dir: str,  # Local path to store/cache the dataset
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sr: int = 16000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = self.ds_val = self.ds_test = None

    def prepare_data(self):
        """
        Downloads the dataset.
        Lightning ensures this runs ONLY on rank 0 (main process) per node.
        Safe for distributed training.
        """
        download_hf_snapshot(
            repo_id=self.hparams.hf_repo,
            work_dir=self.hparams.work_dir,
            repo_type="dataset",
        )

    def setup(self, stage: Optional[str] = None):
        """
        Loads the dataset from the local work_dir.
        Runs on every GPU/worker.
        """
        if self.trainer and self.hparams.batch_size % self.trainer.world_size:
            pass

        # Load from the local snapshot directory
        # We explicitly tell HF to load this as a parquet dataset or generic folder
        # 'data_dir' argument often works best with 'parquet' builder if the snapshot is just files
        try:
            # Try loading as a standard dataset repository structure
            full_ds = load_dataset(self.hparams.work_dir, split="train")
        except ValueError:
            # Fallback: if it's just raw parquet files in a folder
            full_ds = load_dataset(
                "parquet", data_dir=self.hparams.work_dir, split="train"
            )

        # Filter based on the 'split' column string (as created in your upload script)
        # Using keep_in_memory=True or cache_file_name can help performance if RAM allows
        train_split = full_ds.filter(lambda x: x["split"] == "train")
        val_split = full_ds.filter(lambda x: x["split"] in ["val", "valid"])
        test_split = full_ds.filter(lambda x: x["split"] == "test")

        self.ds_train = VaaniHFDataset(train_split, target_sr=self.hparams.target_sr)
        self.ds_val = VaaniHFDataset(val_split, target_sr=self.hparams.target_sr)
        self.ds_test = VaaniHFDataset(test_split, target_sr=self.hparams.target_sr)

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
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dl(self.ds_train, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.ds_test, shuffle=False)


def naive_baseline():
    from tqdm import tqdm

    dm = VaaniGeolocation(
        hf_repo=args.hf_repo, work_dir=args.work_dir, batch_size=2, num_workers=1
    )
    dm.prepare_data()
    dm.setup()
    av_lat = 0
    av_long = 0
    count = 0
    for item in tqdm(dm.train_dataloader().dataset, "Calculating train set average..."):
        av_lat += item["target"][0]
        av_long += item["target"][1]
        count += 1
    av_lat /= count
    av_long /= count
    print(f"Average Latitude: {av_lat}, Average Longitude: {av_long}")
    from src.recipe.common.geolocation_loss import GeolocationAngularLoss

    pred_x = math.cos(av_lat) * math.cos(av_long)
    pred_y = math.cos(av_lat) * math.sin(av_long)
    pred_z = math.sin(av_lat)
    pred_tensor = torch.tensor([[pred_x, pred_y, pred_z]])
    loss_fn = GeolocationAngularLoss()
    total_loss = 0.0
    count = 0
    # test_set = dm.test_dataloader().dataset
    val_set = dm.val_dataloader().dataset
    for item in tqdm(val_set, total=len(val_set)):
        target_tensor = torch.tensor([item["target"]])
        loss_val = loss_fn(
            prediction=pred_tensor,
            target=target_tensor,
        )
        total_loss += loss_val.item()
        count += 1
    print(f"Total test loss: {total_loss}")
    print(f"Average test loss: {total_loss / count if count > 0 else 0}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    args = parser.parse_args()

    dm = VaaniGeolocation(
        hf_repo=args.hf_repo, work_dir=args.work_dir, batch_size=2, num_workers=1
    )
    dm.prepare_data()
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    print(batch)
    print(f"Loaded batch with {batch['speech'].shape[0]} samples.")
    del dm
    # Run naive baseline
    naive_baseline()
