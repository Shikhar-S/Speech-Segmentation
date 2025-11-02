import pyarrow.parquet as pq  # before torch
import os, io, glob, torch
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from tqdm import tqdm
import random
import pandas as pd
import re
import math

# TODO(shikhar): Balanced sampling based on pincodes
# TODO(shikhar): Use metadata from hf for fixing dataset creation
# TODO(shikhar): During pre-processing remove entries without pincode and store pincode to lat-long mapping


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
        parquet_files: List[str],
        audio_root: str,
        pincode_metadata_path: str,
        target_sr: Optional[int] = 16000,
    ):
        self.pincode_metadata_path = pincode_metadata_path
        self.parquet_files = parquet_files
        self.audio_root = audio_root
        self.target_sr = target_sr
        self.pfs = parquet_files
        self.pincode_to_latlong: Dict[int, Tuple[float, float]] = {}
        self.map = []
        self.cum = [0]
        self._build_index()
        self.N = self.cum[-1]
        assert self.N > 0, "No data found in parquet files."
        self.audio_column = "audio"
        self.output_column = "pincode"
        self.max_output_audio_length = 20  # seconds

    def _build_pincode_map(self):
        df = pd.read_csv(self.pincode_metadata_path)
        # print(df.head())
        skip_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building pincode map"):
            try:
                pincode, lat, lon = (
                    int(row["Pincode"]),
                    row["Latitude"],
                    row["Longitude"],
                )
                if pd.isna(lat) or pd.isna(lon):
                    skip_count += 1
                    continue
                lat, lon = str(lat).strip(), str(lon).strip()
                # use regex to extract all values before decimal and 4 digits after decimal
                lat_match = re.match(r"(\d+)\.(\d{4})", lat)
                lon_match = re.match(r"(\d+)\.(\d{4})", lon)
                if lat_match:
                    lat = f"{lat_match.group(1)}.{lat_match.group(2)}"
                if lon_match:
                    lon = f"{lon_match.group(1)}.{lon_match.group(2)}"
                latitude, longitude = float(lat), float(lon)
                self.pincode_to_latlong[pincode] = (latitude, longitude)
            except Exception as e:
                print(f"Skipping row due to error: {e}, row: {row}")
                continue
        print(f"Skipped {skip_count} entries due to invalid lat-long.")

    def _build_index(self):
        # Build index for pincode to lat-long mapping
        # Build index on parquet files
        self._build_pincode_map()
        for fi, pf in enumerate(tqdm(self.pfs, desc="Building index")):
            pf = pq.ParquetFile(pf)
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                self.map.append((fi, rg, n))
                self.cum.append(self.cum[-1] + n)

    def __len__(self):
        return self.N

    def _loc(self, i):
        if i < 0:
            i += self.N
        rg = bisect_right(self.cum, i) - 1
        return self.map[rg][0], self.map[rg][1], i - self.cum[rg]

    def __getitem__(self, i):
        flag = True
        while flag:
            fi, rg, ri = self._loc(i)
            tbl = pq.ParquetFile(self.pfs[fi]).read_row_group(rg)
            if (
                self.audio_column not in tbl.column_names
                or self.output_column not in tbl.column_names
            ):
                i = (i + 1) % self.N
                print(f"Skipping row {i} due to missing required columns.")
                continue
            flag = False
        row = {n: tbl[n][ri].as_py() for n in [self.audio_column, self.output_column]}
        b = row[self.audio_column]["bytes"]
        wav, sr = torchaudio.load(io.BytesIO(b))
        if self.target_sr and sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            sr = self.target_sr
        # trim
        max_len = sr * self.max_output_audio_length
        if wav.shape[-1] > max_len:
            wav = wav[:, :max_len]
        wav = wav.squeeze(0)  # (T,)

        ## output
        pincode = int(row[self.output_column])
        latitude, longitude = self.pincode_to_latlong.get(
            pincode, (0.0, 0.0)
        )  # convert to radians
        latitude = math.radians(latitude)
        longitude = math.radians(longitude)
        return {
            "audio": wav,
            "lengths": wav.shape[-1],
            "sr": sr,
            "pincode": pincode,
            "latitude": latitude,
            "longitude": longitude,
        }


class VaaniGeolocation(LightningDataModule):
    def __init__(
        self,
        data_dir,
        pincode_metadata_path,
        split=(0.8, 0.1, 0.1),
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
            root = self.hparams.data_dir
            files = sorted(
                glob.glob(os.path.join(root, "**", "train-*.parquet"), recursive=True)
            )
            if not files:
                raise FileNotFoundError(f"No shards under {root}")
            print(f"Found {len(files)} parquet files under {root}")

            ############################################################
            # TODO(shikhar): Use metadata for fixing sampling, and then remove this
            print("Sampling down to 100 files for quick testing...")
            # sample random 100 files deterministically using a fixed seed
            if len(files) > 100:
                rng = random.Random(42)
                files = rng.sample(files, 100)
                files.sort()  # keep a deterministic order after sampling
            ############################################################
            print("Creating VaaniParquetDataset...")
            ds = VaaniParquetDataset(
                files,
                audio_root=root,
                pincode_metadata_path=self.hparams.pincode_metadata_path,
                target_sr=self.hparams.target_sr,
            )
            self.ds_train, self.ds_val, self.ds_test = random_split(
                ds, self.hparams.split, generator=torch.Generator().manual_seed(42)
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
        data_dir="/work/hdd/bbjs/shared/corpora/vaani_iisc/Vaani/audio",
        pincode_metadata_path="/work/hdd/bbjs/shared/corpora/vaani_iisc/Vaani/pincode_metadata.csv",
        batch_size=2,
        num_workers=1,
        target_sr=16000,
    )
    dm.setup()
    for x in dm.train_dataloader():
        print(x["pincode"])
        print(x["audio"].shape)
        print(x["lengths"].shape)
    # x = next(iter(dm.train_dataloader()))
    # print("--" * 20)
    # print(x)
    # print("--" * 20)
    # print(x["audio"].shape, x["sr"], x["pincode"])

    # import pyarrow.parquet as pq
    # path = "/work/hdd/bbjs/shared/corpora/vaani_iisc/Vaani/audio/AndhraPradesh/Anantpur/train-00000-of-00052.parquet"
    # pf = pq.ParquetFile(path)
    # import torchaudio
    # import io
    # print("Schema:", pf.schema)
    # print("Number of row groups:", pf.num_row_groups)
    # for rg_idx in [0, 4]:
    #     table = pf.read_row_group(rg_idx)
    #     print(table.column_names, "column names")
    #     row = {n: table[n][0].as_py() for n in ["audio", "pincode"]}
    #     print(len(row), row.keys())
    #     audio = row["audio"]
    #     pincode = row["pincode"]
    #     print("Audio length (bytes):", len(audio))
    #     print(type(audio), audio.keys())
    #     b = audio["bytes"]
    #     wav, sr = torchaudio.load(io.BytesIO(b))
    #     print(wav.shape, sr)
    #     print("Pincode:", pincode)
