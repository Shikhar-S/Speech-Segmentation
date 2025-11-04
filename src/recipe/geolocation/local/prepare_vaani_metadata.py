"""Prepare metadata for Vaani geolocation dataset.
Run with:

srun -p cpu -A bbjs-delta-cpu \
    --cpus-per-task=32 \
    --mem=64G \
    --time=10:00:00 \
    --job-name=vaani_geo \
    --output=logs/vaani_geo_%j.out \
    bash -c "source ~/.bashrc && conda activate pseld \
    && cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/src/recipe/geolocation/local \
    && python prepare_vaani_metadata.py"
"""

import os, random, itertools, io, re, math, pandas as pd, pyarrow.parquet as pq
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import soundfile as sf

PATH_PATTERN = "/work/hdd/bbjs/shared/corpora/vaani_iisc/Vaani/audio/**/*parquet"
PIN_META = "/work/hdd/bbjs/shared/corpora/vaani_iisc/Vaani/pincode_metadata.csv"
OUT_CSV = "vaani_geolocation_metadata.big.csv"
K = 1000  # Checkpoint every K items


# Build pincode map
def parse_coord(x):
    s = str(x).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        val = float(m.group(0))
        if "S" in s or "W" in s:
            val = -val
        return val
    return np.nan


pm = pd.read_csv(PIN_META, usecols=["Pincode", "Latitude", "Longitude"])
pm["Pincode"] = pm["Pincode"].astype(str).str.extract(r"(\d{6})")[0]
pm["Latitude"] = pm["Latitude"].apply(parse_coord)
pm["Longitude"] = pm["Longitude"].apply(parse_coord)
PINMAP = (
    pm.dropna().groupby("Pincode")[["Latitude", "Longitude"]].mean().to_dict("index")
)
print(f"Built pincode map with {len(PINMAP)} entries.")


def process_row(args):
    path, rg_idx = args
    pf = pq.ParquetFile(path)
    table = pf.read_row_group(rg_idx, columns=["pincode", "audio"])
    results = []
    try:
        # try catch because sometimes pincode is missing!
        pincode_col = table.column("pincode")
        audio_col = table.column("audio")

        rel_path = path.replace("/work/hdd/bbjs/shared/corpora/vaani_iisc/", "")

        for row_idx in range(table.num_rows):
            pincode_val = pincode_col[row_idx].as_py()
            pc6_match = re.search(r"\d{6}", str(pincode_val) or "")
            pc6 = pc6_match.group(0) if pc6_match else None

            coords = PINMAP.get(pc6, {})
            lat = coords.get("Latitude", math.nan)
            lon = coords.get("Longitude", math.nan)

            audio_entry = audio_col[row_idx].as_py()
            audio_bytes = (
                audio_entry.get("bytes")
                if isinstance(audio_entry, dict)
                else audio_entry
            ) or b""

            try:
                with sf.SoundFile(io.BytesIO(audio_bytes)) as sfh:
                    wlen = len(sfh) / sfh.samplerate
            except Exception:
                wlen = math.nan  # handle corrupted files safely

            results.append((rel_path, rg_idx, row_idx, pc6, lat, lon, wlen))
    except Exception as e:
        print(f"Error processing {path} row group {rg_idx}: {e}")
    return results


if __name__ == "__main__":
    # Collect all work items
    work_items = []
    paths = glob(PATH_PATTERN, recursive=True)
    paths = sorted(paths)
    # Sample few files for preparing data
    n = 20
    sampled_paths = [
        p
        for _, g in itertools.groupby(
            sorted(paths, key=os.path.dirname), key=os.path.dirname
        )
        for p in (lambda L: random.sample(L, min(n, len(L))))(list(g))
    ]
    print(len(sampled_paths), "files sampled for processing.")
    for path in tqdm(sampled_paths, desc="parquet files"):
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            work_items.append((path, rg_idx))

    # Process in parallel with checkpointing using as_completed
    columns = [
        "parquet_path",
        "row_group",
        "row_index",
        "pincode",
        "latitude",
        "longitude",
        "audio_length",
    ]

    buffer = []
    total_saved = 0
    header_written = False

    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = {executor.submit(process_row, item): item for item in work_items}

        # Process completed tasks with progress bar
        pbar = tqdm(as_completed(futures), total=len(futures))
        for future in pbar:
            result = future.result()
            buffer.extend(result)

            # Checkpoint when buffer reaches K items
            if len(buffer) >= K:
                df = pd.DataFrame(buffer, columns=columns)
                df.to_csv(
                    OUT_CSV,
                    mode="a" if header_written else "w",
                    header=not header_written,
                    index=False,
                )
                header_written = True
                total_saved += len(buffer)
                pbar.set_postfix({"saved": total_saved})
                buffer = []

    # Save any remaining items in buffer
    if buffer:
        df = pd.DataFrame(buffer, columns=columns)
        df.to_csv(
            OUT_CSV,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
        )
        total_saved += len(buffer)

    print(f"Wrote {OUT_CSV} with {total_saved} rows.")
