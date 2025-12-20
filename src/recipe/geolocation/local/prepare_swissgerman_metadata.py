"""Module for preparing Swiss German metadata for geolocation tasks.

Usage:
    python -m src.recipe.geolocation.local.prepare_swissgerman_metadata
"""

import pandas as pd
import os
from multiprocessing import Pool

data_root = "/work/hdd/bbjs/shared/corpora/swiss_german"
save_path = (
    "/work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/cache/swissgerman/metadata.csv"
)
SAMPLES_PER_ZIPCODE = 100


def read_df_with_location(
    metadata_path: str, split: str, column_map: dict, dataset_name: str
) -> pd.DataFrame:
    metadata = f"{metadata_path}/{split}.tsv"
    metadata_df = pd.read_csv(f"{data_root}/{metadata}", sep="\t")
    metadata_df.dropna(inplace=True, subset=["zipcode"])
    metadata_df["zipcode"] = metadata_df["zipcode"].astype(int)
    # Merge with location data
    location_df = pd.read_csv(f"{data_root}/post-codes.csv", skiprows=3)
    # take average of lat/lng for duplicate zip codes
    location_df = location_df.groupby("zip", as_index=False)[["lat", "lng"]].mean()
    metadata_df = metadata_df.merge(
        location_df, how="left", left_on="zipcode", right_on="zip"
    )
    # consistent and relevant columns
    metadata_df.rename(columns=column_map, inplace=True)
    metadata_df = metadata_df[list(column_map.values())]
    metadata_df["dataset"] = dataset_name
    metadata_df["audio_path"] = metadata_df.apply(
        lambda row: f"{data_root}/{dataset_name}/audio/{row['path']}", axis=1
    )
    return metadata_df


###### SDS200
sds_column_map = {
    "clip_path": "path",
    "duration": "duration",
    "zipcode": "zipcode",
    "lat": "latitude",
    "lng": "longitude",
    "zipcode": "zipcode",  # ensure zipcode column exists
}
sds_valid_df = read_df_with_location(
    "SDS200/metadata/splits", "valid", sds_column_map, "SDS200"
)
sds_train_df = read_df_with_location(
    "SDS200/metadata/splits", "train_clean", sds_column_map, "SDS200"
)
sds_test_df = read_df_with_location(
    "SDS200/metadata/splits", "test", sds_column_map, "SDS200"
)
###### STT4SG350v2.1
stt_column_map = {
    "path": "path",
    "duration": "duration",
    "zipcode": "zipcode",
    "lat": "latitude",
    "lng": "longitude",
    "zipcode": "zipcode",  # ensure zipcode column exists
}
stt_train_df = read_df_with_location(
    "STT4SG350v2.1/metadata/STT4SG-350 v2.1",
    "train_all",
    stt_column_map,
    "STT4SG350v2.1",
)
stt_valid_df = read_df_with_location(
    "STT4SG350v2.1/metadata/STT4SG-350 v2.1", "valid", stt_column_map, "STT4SG350v2.1"
)
stt_test_df = read_df_with_location(
    "STT4SG350v2.1/metadata/STT4SG-350 v2.1", "test", stt_column_map, "STT4SG350v2.1"
)

train_df = pd.concat([sds_train_df, stt_train_df], ignore_index=True)
valid_df = pd.concat([sds_valid_df, stt_valid_df], ignore_index=True)
test_df = pd.concat([sds_test_df, stt_test_df], ignore_index=True)

selected_zipcodes = test_df.value_counts("zipcode")
selected_zipcodes = selected_zipcodes[selected_zipcodes > 10]
print(f"Number of selected zipcodes: {len(selected_zipcodes)}")

train_df = train_df[train_df["zipcode"].isin(selected_zipcodes.index)]
valid_df = valid_df[valid_df["zipcode"].isin(selected_zipcodes.index)]
test_df = test_df[test_df["zipcode"].isin(selected_zipcodes.index)]
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


# Group by pincoce and select up to SAMPLES_PER_ZIPCODE
def limit_samples_per_zipcode(df, max_samples: int) -> pd.DataFrame:
    limited_df_list = []
    for zipcode, group in df.groupby("zipcode"):
        if len(group) > max_samples:
            limited_group = group.sample(n=max_samples, random_state=42)
        else:
            limited_group = group
        limited_df_list.append(limited_group)
    limited_df = pd.concat(limited_df_list, ignore_index=True)
    return limited_df


train_df = limit_samples_per_zipcode(train_df, max_samples=SAMPLES_PER_ZIPCODE)
valid_df = limit_samples_per_zipcode(valid_df, max_samples=SAMPLES_PER_ZIPCODE)
test_df = limit_samples_per_zipcode(test_df, max_samples=SAMPLES_PER_ZIPCODE)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


print(f"Train size: {len(train_df)}")
print(f"Valid size: {len(valid_df)}")
print(f"Test size: {len(test_df)}")

print(
    "Total hours in each split (train, valid, test):",
    train_df.duration.sum() / 3600,
    valid_df.duration.sum() / 3600,
    test_df.duration.sum() / 3600,
)

train_df["split"] = "train"
valid_df["split"] = "valid"
test_df["split"] = "test"
combined_metadata = pd.concat([train_df, valid_df, test_df], ignore_index=True)
print("Total data points before filtering:", len(combined_metadata))


#####################
# # drop paths with missing audio
# def exists(path):
#     return os.path.exists(path)


# with Pool() as p:
#     combined_metadata["audio_exists"] = p.map(exists, combined_metadata["audio_path"])
# print(
#     "Number of missing audio files:",
#     len(combined_metadata) - combined_metadata["audio_exists"].sum(),
# )
# combined_metadata = combined_metadata[combined_metadata["audio_exists"]]
# combined_metadata.drop(columns=["audio_exists"], inplace=True)
# combined_metadata.reset_index(drop=True, inplace=True)
#####################
print("Total data points after filtering:", len(combined_metadata))
print(
    "Maximum duration (seconds):",
    combined_metadata["duration"].max(),
    "Minimum duration (seconds):",
    combined_metadata["duration"].min(),
)

os.makedirs(os.path.dirname(save_path), exist_ok=True)
combined_metadata.to_csv(save_path, index=False)
