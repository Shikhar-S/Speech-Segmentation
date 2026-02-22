# python -m src.recipe.geolocation.local.vaanihindi_upload
import pandas as pd
import numpy as np
import torchaudio
import io
import os
import re
from datasets import Dataset, Features, Value

DOWNLOAD_DIR = "exp/download/vaani_hindibelt"
METADATA_CSV = f"{DOWNLOAD_DIR}/vaani_hindi_states_metadata_filtered.csv"  # The filtered metadata from the end of sampling
PINCODE_META = "/work/hdd/bbjs/shared/corpora/vaani_iisc/Vaani/pincode_metadata.csv"
REPO_ID = "shikhar7ssu/vaani140-geo"
TEMP_DIR = "exp/cache/vaanishards"
MAX_SHARD_SIZE = "1000MB"
NUM_WORKERS = 16


def parse_coord(x):
    s = str(x).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        val = float(m.group(0))
        if "S" in s or "W" in s:
            val = -val
        return val
    return np.nan


def load_and_process_audio(batch):
    audio_blobs = []
    for audiob_path in batch["audio_path"]:
        audiob_path = f"{DOWNLOAD_DIR}/{audiob_path}"
        wav, sr = torchaudio.load(io.BytesIO(open(audiob_path, "rb").read()))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        max_len = sr * 20  # 20 seconds max
        if wav.shape[-1] > max_len:
            wav = wav[:, :max_len]
        buf = io.BytesIO()
        torchaudio.save(buf, wav, sr, format="wav")
        audio_blobs.append(buf.getvalue())

    return {"audio": audio_blobs}


def create_splits(df):
    # Simple train/val/test split for each pincode
    tr_frac = 0.75
    val_frac = 0.1
    np.random.seed(42)
    pincodes = df["pincode"].unique()
    df["split"] = np.nan
    for pincode in pincodes:
        pdf = df[df["pincode"] == pincode]
        n = len(pdf)
        pdf = pdf.sample(frac=1, random_state=42)  # shuffle
        n_train = int(n * tr_frac)
        n_val = int(n * val_frac)
        df.loc[pdf.index[:n_train], "split"] = "train"
        df.loc[pdf.index[n_train : n_train + n_val], "split"] = "val"
        df.loc[pdf.index[n_train + n_val :], "split"] = "test"
    df.loc[df["split"].isna(), "split"] = "train"
    print("Split distribution:")
    print(df["split"].value_counts())
    return df


def main():
    print("Loading metadata and geolocations...")
    pm = pd.read_csv(
        PINCODE_META, usecols=["Pincode", "Latitude", "Longitude"], low_memory=False
    )
    pm["Pincode"] = pm["Pincode"].astype(str).str.extract(r"(\d{6})")[0]
    pm["Latitude"] = pm["Latitude"].apply(parse_coord)
    pm["Longitude"] = pm["Longitude"].apply(parse_coord)
    pin_map = (
        pm.dropna()
        .groupby("Pincode")[["Latitude", "Longitude"]]
        .mean()
        .to_dict("index")
    )

    df = pd.read_csv(METADATA_CSV)
    df["pinstr"] = df["pincode"].astype(str)
    coords = df["pinstr"].apply(
        lambda x: pin_map.get(x, {"Latitude": 0.0, "Longitude": 0.0})
    )
    df["latitude"] = coords.apply(lambda x: x["Latitude"])
    df["longitude"] = coords.apply(lambda x: x["Longitude"])

    df["metadata_idx"] = range(len(df))
    df = create_splits(df)
    df["lang_sym"] = "<unk>"  # for powsm

    ds = Dataset.from_pandas(df, preserve_index=False)
    print(f"Processing audio from WAVs with {NUM_WORKERS} workers...")
    ds = ds.map(
        load_and_process_audio,
        batched=True,
        batch_size=8,
        num_proc=NUM_WORKERS,
        remove_columns=["audio_path", "pinstr", "source_path"],  # clean up paths
    )
    features = Features(
        {
            "audio": Value("binary"),
            "pincode": Value("int32"),
            "latitude": Value("float32"),
            "longitude": Value("float32"),
            "split": Value("string"),
            "metadata_idx": Value("int32"),
            "lang_sym": Value("string"),
            "language": Value("string"),
            "gender": Value("string"),
            "state": Value("string"),
            "district": Value("string"),
        }
    )
    cols = [k for k in features.keys() if k in ds.column_names]
    ds = ds.select_columns(cols).cast(Features({k: features[k] for k in cols}))
    print(f"Saving to disk ({MAX_SHARD_SIZE} shards)...")
    ds.save_to_disk(TEMP_DIR, max_shard_size=MAX_SHARD_SIZE)
    print(f"Pushing to {REPO_ID}...")
    ds_sharded = Dataset.load_from_disk(TEMP_DIR)
    ds_sharded.push_to_hub(REPO_ID)
    print("Done.")


if __name__ == "__main__":
    main()
