# python -m src.recipe.langid.prepare_fleurs_lid
import torch
import torchaudio
from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
import io, os

CACHE_DIR = "/data/user_data/sbharad2/PhoneBench/exp/cache/fleurs"
os.environ["HF_HOME"] = "/data/user_data/sbharad2/PhoneBench/exp/cache/hf"

# Configuration
TARGET_REPO = "shikhar7ssu/fleurs24-lid"
CACHE_DIR = "exp/cache/fleurs"
TARGET_SR = 16000
MAX_SAMPLES_SEC = 20.0
LANG_SUBSET = [
    "as_in",
    "ast_es",
    "fa_ir",
    "fil_ph",
    "gu_in",
    "he_il",
    "hy_am",
    "ig_ng",
    "kam_ke",
    "kea_cv",
    "km_kh",
    "kn_in",
    "ckb_iq",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "luo_ke",
    "lv_lv",
    "ne_np",
    "nso_za",
    "oc_fr",
    "ps_af",
    "umb_ao",
    "wo_sn",
]
MAX_SAMPLES_PER_LANGUAGE = {"train": 4800, "test": 4800, "validation": 2400}
LANG_TO_ID = {}


def preprocess_function(example):
    """Resample, truncate, and add metadata tags."""
    wav, sr = torchaudio.load(io.BytesIO(example["audio"]["bytes"]))
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    max_len = int(TARGET_SR * MAX_SAMPLES_SEC)
    if wav.shape[-1] > max_len:
        wav = wav[:max_len]

    buf = io.BytesIO()
    torchaudio.save(buf, wav, 16000, format="wav")
    example["audio"]["bytes"] = buf.getvalue()
    example["audio"]["sampling_rate"] = TARGET_SR
    example["target"] = LANG_TO_ID[example["language"]]
    return example


def prepare_and_upload():
    final_dict = DatasetDict()

    for split in ["train", "validation", "test"]:
        print(f"Processing split: {split}...")
        ds_list = []
        samples_per_lang = MAX_SAMPLES_PER_LANGUAGE[split] // len(LANG_SUBSET)
        for lang in LANG_SUBSET:
            ds = load_dataset(
                "google/fleurs",
                data_dir=lang,
                split=split,
                revision="refs/convert/parquet",
                cache_dir=CACHE_DIR,
            )
            ds = ds.select(range(min(samples_per_lang, len(ds))))
            ds = ds.cast_column("audio", Audio(decode=False))
            langname = set(ds["language"]).pop()
            if langname not in LANG_TO_ID:
                LANG_TO_ID[langname] = len(LANG_TO_ID)
            ds_list.append(ds)

        merged_ds = concatenate_datasets(ds_list)
        processed_ds = merged_ds.map(
            preprocess_function, desc=f"Transforming {split}", num_proc=4
        )

        final_dict[split] = processed_ds

    print(f"Uploading to {TARGET_REPO}...")
    final_dict.push_to_hub(TARGET_REPO)


if __name__ == "__main__":
    prepare_and_upload()
