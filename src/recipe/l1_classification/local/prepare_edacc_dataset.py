import io, collections, torchaudio, os, random
from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio

CACHE_DIR = "/data/user_data/sbharad2/PhoneBench/exp/cache/edacc"
os.environ["HF_HOME"] = "/data/user_data/sbharad2/PhoneBench/exp/cache/hf"

L1_TO_ACCENT_CLUSTER = {
    # --- SOUTH_ASIAN (Indo-Aryan/Dravidian) ---
    "Hindi": "SOUTH_ASIAN",
    "Indian English": "SOUTH_ASIAN",
    "Urdu": "SOUTH_ASIAN",
    "Sinhalese": "SOUTH_ASIAN",
    # --- INNER_CIRCLE_ENGLISH (Non-Celtic L1) ---
    "English": "INNER_CIRCLE_ENGLISH",
    "Southern British English": "INNER_CIRCLE_ENGLISH",
    "Mainstream US English": "INNER_CIRCLE_ENGLISH",
    "South African English": "INNER_CIRCLE_ENGLISH",  # Grouped here by vowel similarity
    # --- CELTIC_ENGLISH (Distinct Rhotic/Prosodic L1) ---
    "Scottish English": "CELTIC_ENGLISH",
    "Irish English": "CELTIC_ENGLISH",
    # --- IBERO_ITALO_ROMANCE (Syllable-timed, 5-vowel systems) ---
    "Spanish": "ROMANCE",
    "Spanish (Mexican)": "ROMANCE",
    "Catalan": "ROMANCE",
    "Italian": "ROMANCE",
    "Portoguese": "ROMANCE",
    "Maltese": "ROMANCE",  # Not Semitic: phonetically closer to Italian influence
    # --- GALLO_ROMANCE (The French Outlier) ---
    "French": "GALLO_ROMANCE",
    # --- INSULAR_SOUTHEAST_ASIAN (Austronesian) ---
    "Indonesian": "INSULAR_SEA",
    "Bahasa": "INSULAR_SEA",
    "Filipino": "INSULAR_SEA",
    "Tagalog": "INSULAR_SEA",
    # --- MAINLAND_SOUTHEAST_ASIAN (Austroasiatic) ---
    "Vietnamese": "MAINLAND_SEA",
    # --- EAST_ASIAN (Sino-Tibetan/Japonic/Koreanic) ---
    "Mandarin": "EAST_ASIAN",
    "Japanese": "EAST_ASIAN",
    "Korean": "EAST_ASIAN",
    # --- AFRICAN_ENGLISH ---
    "Nigerian English": "AFRICAN_ENGLISH",
    "Kenyan English": "AFRICAN_ENGLISH",
    "Ghanain English": "AFRICAN_ENGLISH",
    # --- SLAVIC_BALKAN (Includes Romance-Slavic hybrids) ---
    "Russian": "SLAVIC_BALKAN",
    "Polish": "SLAVIC_BALKAN",
    "Bulgarian": "SLAVIC_BALKAN",
    "Macedonian": "SLAVIC_BALKAN",
    "Montenegrin": "SLAVIC_BALKAN",
    "Lithuanian": "SLAVIC_BALKAN",
    "Romanian": "SLAVIC_BALKAN",  # Slavic-adjacent
    # --- GERMANIC ---
    "German": "GERMANIC",
    "Dutch": "GERMANIC",
    "Icelandic": "GERMANIC",
    # --- AFROASIATIC_SEMITIC ---
    "Arabic": "AFROASIATIC_SEMITIC",
    "Hebrew": "AFROASIATIC_SEMITIC",
    # --- CARIBBEAN ---
    "Jamaican English": "CARIBBEAN",
}

N_CLUSTERS = len(set(L1_TO_ACCENT_CLUSTER.values()))
print(f"Defined {N_CLUSTERS} accent clusters.")


def get_conv_id(speaker_id):
    return speaker_id.split("-")[1] if "-" in speaker_id else speaker_id


def split_by_conversation_3way(dataset, test_ratio=0.4, val_ratio=0.15, seed=42):
    random.seed(seed)

    # 1. Map Conversations to L1s
    conv_to_l1s = collections.defaultdict(set)
    l1_to_convs = collections.defaultdict(list)

    # Use unique speakers to identify conversation contents
    spk_metadata = {}
    for row in dataset:
        spk_metadata[row["speaker"]] = row["accent_cluster"]

    for spk, l1 in spk_metadata.items():
        cid = get_conv_id(spk)
        conv_to_l1s[cid].add(l1)
        l1_to_convs[l1].append(cid)

    all_convs = sorted(list(conv_to_l1s.keys()))
    random.shuffle(all_convs)
    assigned_convs = {}  # cid -> split

    # 2. Coverage Pass: Guarantee representation where possible
    # Process rarest L1s first
    sorted_l1s = sorted(l1_to_convs.keys(), key=lambda x: len(set(l1_to_convs[x])))

    for l1 in sorted_l1s:
        convs_for_l1 = sorted(list(set(l1_to_convs[l1])))
        random.shuffle(convs_for_l1)

        represented = {assigned_convs[c] for c in convs_for_l1 if c in assigned_convs}
        # Priority order to ensure test is never empty, then train sets
        for split in ["train", "test", "validation"]:
            if split not in represented:
                available = [c for c in convs_for_l1 if c not in assigned_convs]
                if available:
                    assigned_convs[available[0]] = split
                    represented.add(split)

    # 3. Ratio Pass: Fill remaining to match target ratios
    remaining = [c for c in all_convs if c not in assigned_convs]
    random.shuffle(remaining)

    n_test = int(len(all_convs) * test_ratio)
    n_val = int(len(all_convs) * val_ratio)
    n_train = len(all_convs) - n_test - n_val

    curr_train = sum(1 for v in assigned_convs.values() if v == "train")
    curr_val = sum(1 for v in assigned_convs.values() if v == "validation")
    curr_test = sum(1 for v in assigned_convs.values() if v == "test")

    for c in remaining:
        if curr_train < n_train:
            assigned_convs[c] = "train"
            curr_train += 1
        elif curr_test < n_test:
            assigned_convs[c] = "test"
            curr_test += 1
        elif curr_val < n_val:
            assigned_convs[c] = "validation"
            curr_val += 1
    return assigned_convs


def preprocess(example):
    wav, sr = torchaudio.load(io.BytesIO(example["audio"]["bytes"]))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    buf = io.BytesIO()
    torchaudio.save(buf, wav, 16000, format="wav")
    accent_cluster = L1_TO_ACCENT_CLUSTER[example["l1"]]
    return {
        "audio": {
            "bytes": buf.getvalue(),
            "sampling_rate": 16000,
        },
        "duration_sec": wav.shape[1] / 16000,
        "too_short": wav.shape[1] < 8000,
        "accent_cluster": accent_cluster,
    }


if __name__ == "__main__":
    REPO_ID, TARGET_REPO = "edinburghcstr/edacc", "shikhar7ssu/edacc-l1cls"
    train_ds = load_dataset(REPO_ID, split="validation", cache_dir=CACHE_DIR)
    val_ds = load_dataset(REPO_ID, split="test", cache_dir=CACHE_DIR)
    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    val_ds = val_ds.cast_column("audio", Audio(decode=False))
    full_ds = concatenate_datasets([train_ds, val_ds])
    full_ds = full_ds.map(preprocess, desc="Processing")
    full_ds = full_ds.filter(lambda x: not x["too_short"]).remove_columns(["too_short"])

    # Perform 3-way split
    assignments = split_by_conversation_3way(full_ds)

    ds_train = full_ds.filter(
        lambda x: assignments[get_conv_id(x["speaker"])] == "train"
    )
    ds_val = full_ds.filter(
        lambda x: assignments[get_conv_id(x["speaker"])] == "validation"
    )
    ds_test = full_ds.filter(lambda x: assignments[get_conv_id(x["speaker"])] == "test")

    # Stats and Push
    DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test}).push_to_hub(
        TARGET_REPO
    )

    for name, ds in [("Train", ds_train), ("Val", ds_val), ("Test", ds_test)]:
        l1_to_convs = collections.defaultdict(set)
        for l1, spk in zip(ds["accent_cluster"], ds["speaker"]):
            l1_to_convs[l1].add(spk)
        conv_counts = [len(c) for c in l1_to_convs.values()]
        mean_convs = sum(conv_counts) / len(conv_counts) if conv_counts else 0
        std_convs = (
            (sum((c - mean_convs) ** 2 for c in conv_counts) / len(conv_counts)) ** 0.5
            if conv_counts
            else 0
        )
        print(
            f"{name}: {len(ds)} utts, {len(set(ds['l1']))} L1s, {len(set(ds['accent_cluster']))}"
            f" accent clusters, {len(set(s for s in ds['speaker']))} convs, "
            f"mean convs/L1: {mean_convs:.2f}, std convs/L1: {std_convs:.2f}"
        )
