import pandas as pd
import polars as pl
import os, random, concurrent.futures
from pathlib import Path
from tqdm import tqdm

SEED = 42
random.seed(SEED)

STATE_LIST = [
    "Chandigarh",
    "HimachalPradesh",
    "Delhi",
    "MadhyaPradesh",
    "Jharkhand",
    "Uttarakhand",
    "Bihar",
    "Chhattisgarh",
    "Haryana",
    "Rajasthan",
    "Punjab",
    "UttarPradesh",
]
districts_per_state = 4
files_per_district = 4
samples_per_file = 600
DOWNLOAD_DIR = "exp/download/vaani_hindibelt"
WAVS_DIR = f"{DOWNLOAD_DIR}/WAVS"
STATE_PATH_PATTERN = "/work/hdd/bbjs/shared/corpora/vaani_iisc/Vaani/audio/{state_name}"
PARQUET_FILE_PATTERN = "train-{fnum}-of-*.parquet"
os.makedirs(WAVS_DIR, exist_ok=True)
paths, queries = [], []
for statename in STATE_LIST:
    state_path = STATE_PATH_PATTERN.format(state_name=statename)
    all_districts = os.listdir(state_path)
    target_districts = random.sample(
        all_districts, k=min(districts_per_state, len(all_districts))
    )
    for districtname in target_districts:
        district_path = os.path.join(state_path, districtname)
        n_parquets = min(files_per_district, len(os.listdir(district_path)))
        for n in range(n_parquets):
            full_path = os.path.join(
                district_path, PARQUET_FILE_PATTERN.format(fnum=str(n).zfill(5))
            )
            q = (
                pl.scan_parquet(full_path)
                .select(
                    [
                        "language",
                        "pincode",
                        "gender",
                        "state",
                        "district",
                        pl.col("audio").struct.field("bytes").alias("audio_bytes"),
                    ]
                )
                .head(samples_per_file)
            )
            paths.append(full_path)
            queries.append(q)
print(f"Reading {len(queries)} files in parallel...")
results = pl.collect_all(queries)


def write_audio_file(args):
    path, data = args
    if data:
        with open(path, "wb") as f:
            f.write(data)


final_dfs = []
print(f"Writing audio to {WAVS_DIR}/...")

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    for src_path, df in tqdm(
        zip(paths, results), total=len(paths), desc="Processing files"
    ):
        stem = Path(src_path).parent.parent.name + "_" + Path(src_path).parent.name
        local_paths = [f"{WAVS_DIR}/{stem}_{i}.wav" for i in range(df.height)]
        executor.map(write_audio_file, zip(local_paths, df["audio_bytes"]))
        clean_df = df.with_columns(
            [
                pl.Series("audio_path", local_paths),
                pl.lit(src_path).alias("source_path"),
            ]
        ).drop("audio_bytes")
        final_dfs.append(clean_df)

if final_dfs:
    final_df = pl.concat(final_dfs)
    final_df.write_csv("vaani_hindi_states_metadata.csv")
    print(f"Done. Saved {len(final_df)} rows.")
else:
    print("No data found.")


################ FILTERING LOGIC ###########################
df = pd.read_csv("vaani_hindi_states_metadata.csv")
valid_pins, valid_utt = 0, 0
utt_per_lang = {}
nmales, nfemales = 0, 0
UTT_PER_PINCODE = 450
filtered_dfs = []
for _, gdf in df.groupby("pincode"):
    if gdf.shape[0] > UTT_PER_PINCODE:
        filtered_dfs.append(gdf)
        valid_pins += 1
        valid_utt += gdf.shape[0]
        nmales += gdf[gdf.gender == "Male"].shape[0]
        nfemales += gdf[gdf.gender == "Female"].shape[0]
        for l in gdf["language"].unique():
            utt_per_lang[l] = (
                utt_per_lang.get(l, 0) + gdf[gdf["language"] == l].shape[0]
            )
print(
    f"Valid pins:{valid_pins}, Valid utt:{valid_utt}, nfemales:{nfemales}, nmales:{nmales}"
)
print("Utt per lang:", utt_per_lang)
filtered_dfs = pd.concat(filtered_dfs)
filtered_dfs.to_csv("vaani_hindi_states_metadata_filtered.csv", index=False)
###########################################################
