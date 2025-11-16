"""Prepare train/val/test splits for Vaani geolocation data.
Usage:
    python /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/src/recipe/geolocation/local/prepare_vaani_splits.py \
        --input-metadata /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/vaani_geolocation/data/vaani_geolocation_metadata.big.csv \
        --output-dir /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/vaani_geolocation/data/tmp \
        --max-pincodes 140
"""

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


TEST_FRAC = 0.025
VAL_FRAC = 0.025
MIN_TEST_PER_CLASS = 20
MIN_VAL_PER_CLASS = 20
SEED_TEST = 42
SEED_VAL = 43
TRAIN_SEED = 41
SUBFRACS = [0.10]
LABEL_COL = "pincode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create geolocation train/val/test splits with nested train subsets."
    )
    parser.add_argument(
        "--input-metadata",
        type=Path,
        required=True,
        help="Path to input CSV with columns including pincode, latitude, longitude, audio_length.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write split CSVs.",
    )
    parser.add_argument(
        "--max-pincodes",
        type=int,
        default=None,
        help="Maximum number of pincodes. If set and exceeded, "
        "pincodes are clustered and reduced to this many clusters.",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=2.0,
        help="Minimum audio length (seconds) to keep.",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=10.0,
        help="Maximum audio length (seconds) to keep.",
    )
    return parser.parse_args()


def load_and_filter_metadata(
    path: Path, min_len: float, max_len: float
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["latitude", "longitude", "audio_length"])
    mask = (df["audio_length"] > min_len) & (df["audio_length"] <= max_len)
    df = df[mask].copy()
    return df


def latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude in degrees to 3D unit sphere coordinates."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)


def limit_pincodes_by_clustering(
    df: pd.DataFrame, max_pincodes: int, random_state: int = 0
) -> pd.DataFrame:
    """Limit number of pincodes via KMeans clustering on (lat, lon) converted to 3D coords."""
    unique_pins = df[LABEL_COL].nunique()
    if max_pincodes is None or unique_pins <= max_pincodes:
        return df

    pin_stats = (
        df[[LABEL_COL, "latitude", "longitude"]]
        .groupby(LABEL_COL)
        .agg({"latitude": "mean", "longitude": "mean"})
    )
    n_clusters = min(max_pincodes, len(pin_stats))
    coords = latlon_to_unit_xyz(
        pin_stats["latitude"].values, pin_stats["longitude"].values
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(coords)

    pin_stats = pin_stats.assign(cluster=labels)
    centers = kmeans.cluster_centers_

    # Choose one representative pincode per cluster (closest to cluster center)
    chosen_pins = []
    for c in range(n_clusters):
        cluster_pins = pin_stats[pin_stats["cluster"] == c]
        cluster_coords = latlon_to_unit_xyz(
            cluster_pins["latitude"].values, cluster_pins["longitude"].values
        )
        center = centers[c][None, :]
        dists = np.linalg.norm(cluster_coords - center, axis=1)
        rep_pin = cluster_pins.index[np.argmin(dists)]
        chosen_pins.append(rep_pin)

    chosen_set = set(chosen_pins)
    return df[df[LABEL_COL].isin(chosen_set)].copy()


def sample_balanced(
    g: pd.DataFrame, frac: float, min_n: int, seed: int
) -> pd.DataFrame:
    """Sample by fraction but ensure at least min_n samples if available."""
    n = len(g)
    if n <= min_n:
        return g.sample(n=n // 3 if n > 2 else n, random_state=seed)
    k = min(n, max(min_n, ceil(frac * n)))
    return g.sample(n=k, random_state=seed)


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Test split
    test_idx = (
        df.groupby(LABEL_COL, group_keys=False)
        .apply(lambda g: sample_balanced(g, TEST_FRAC, MIN_TEST_PER_CLASS, SEED_TEST))
        .index
    )
    remaining_idx = df.index.difference(test_idx)

    # Validation split
    val_idx = (
        df.loc[remaining_idx]
        .groupby(LABEL_COL, group_keys=False)
        .apply(lambda g: sample_balanced(g, VAL_FRAC, MIN_VAL_PER_CLASS, SEED_VAL))
        .index
    )
    train_idx = remaining_idx.difference(val_idx)

    train_set = df.loc[train_idx].copy()
    val_set = df.loc[val_idx].copy()
    test_set = df.loc[test_idx].copy()
    return train_set, val_set, test_set


def nested_train_subsets(train_set: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Create nested train subsets for fractions in SUBFRACS."""
    shuffled_train = (
        train_set.groupby(LABEL_COL, group_keys=True)
        .apply(lambda g: g.sample(frac=1.0, random_state=TRAIN_SEED))
        .reset_index(level=0, drop=True)
    )

    def take_frac_nested(shuff: pd.DataFrame, frac: float) -> pd.DataFrame:
        sizes = shuff.groupby(LABEL_COL).size()
        k_per_class = (sizes * frac).astype(int).clip(lower=(1 if frac > 0 else 0))
        return shuff.groupby(LABEL_COL, group_keys=False).apply(
            lambda g: g.head(k_per_class.get(g.name, 0))
        )

    subsets = {}
    for f in SUBFRACS:
        subsets[int(f * 100)] = take_frac_nested(shuffled_train, f).copy()
    return subsets


def write_outputs(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
    output_dir: Path,
    base_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full split file
    final_df = pd.concat(
        [
            train_set.assign(split="train"),
            val_set.assign(split="val"),
            test_set.assign(split="test"),
        ]
    )
    full_path = output_dir / f"{base_name}.withsplits.csv"
    final_df.to_csv(full_path, index=False)
    print(f"Saved full splits: {full_path}  (total={len(final_df):,})")

    # Nested train subset files (each with its own train + shared val/test)
    val_out = val_set.assign(split="val")
    test_out = test_set.assign(split="test")
    subsets = nested_train_subsets(train_set)

    for pct, train_sub in subsets.items():
        train_sub = train_sub.assign(split="train")
        combined = pd.concat([train_sub, val_out, test_out], axis=0)
        out_path = output_dir / f"{base_name}.train{pct}.csv"
        combined.to_csv(out_path, index=False)
        print_duration_report(train_sub, val_set, test_set)


def main() -> None:
    args = parse_args()
    df = load_and_filter_metadata(args.input_metadata, args.min_length, args.max_length)
    print(f"Loaded {len(df):,} rows after filtering by length/NaNs.")
    print(f"Unique pincodes before limiting: {df[LABEL_COL].nunique()}")

    df = limit_pincodes_by_clustering(df, args.max_pincodes)
    print(f"Unique pincodes after limiting:  {df[LABEL_COL].nunique()}")

    train_set, val_set, test_set = make_splits(df)
    base_name = args.input_metadata.stem.replace(".big", "")
    write_outputs(train_set, val_set, test_set, args.output_dir, base_name)
    print_duration_report(train_set, val_set, test_set)


def print_duration_report(
    train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame
) -> None:
    # Quick duration report (hours)
    train_h = train_set["audio_length"].sum() / 3600
    val_h = val_set["audio_length"].sum() / 3600
    test_h = test_set["audio_length"].sum() / 3600

    n_train = len(train_set)
    n_val = len(val_set)
    n_test = len(test_set)

    train_av_elements_per_location = train_set.groupby("pincode").size().mean()
    val_av_elements_per_location = val_set.groupby("pincode").size().mean()
    test_av_elements_per_location = test_set.groupby("pincode").size().mean()
    print("==" * 20)
    print(
        f"Durations (hours) -> train: {train_h:.1f}, "
        f"val: {val_h:.1f}, test: {test_h:.1f}"
    )
    print(f"Counts -> train: {n_train:,}, val: {n_val:,}, test: {n_test:,}")
    print(
        f"Average elements per location -> train: {train_av_elements_per_location:.1f}, "
        f"val: {val_av_elements_per_location:.1f}, test: {test_av_elements_per_location:.1f}"
    )
    print("==" * 20)


if __name__ == "__main__":
    main()
