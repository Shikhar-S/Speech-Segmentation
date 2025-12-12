"""
Metadata preparation script for CMU + L2Arctic datasets for L1 classification.

This script scans CMU and L2Arctic raw data directories and generates a single
metadata.csv file with the following schema:
    - audio_path: relative path to audio file (from data_dir)
    - l1_label: L1 class ID (e.g., 'en', 'ko', 'zh', etc.)
    - split: 'train', 'val', or 'test'
    - speaker_id: speaker identifier
    - utt_id: utterance identifier

Data Download Instructions:
    1. L2-ARCTIC:
       - Download: https://drive.google.com/file/d/1ciCw_ttbw7a9r7d5DZzTJwoZq5rQB3TA/view
       - Command (gdown):
         $ mkdir -p {data_dir}/l2arctic_release_v5.0
         $ gdown https://drive.google.com/uc?id=1ciCw_ttbw7a9r7d5DZzTJwoZq5rQB3TA -O l2arctic_release_v5.0.zip
         $ unzip l2arctic_release_v5.0.zip -d {data_dir}/l2arctic_release_v5.0
         ```Since unzipping sometimes fails, use this snippet
            # (change l2arctic to l2arctic_release_v5.0 if you want)
            rm -rf l2arctic && mkdir l2arctic && \
            files=$(unzip -Z1 l2arctic_release_v5.0.zip) && \
            skipped="" && \
            for f in $files; do
                echo "Extracting $f"
                ok=0
                for i in $(seq 1 5); do
                    unzip -o l2arctic_release_v5.0.zip "$f" -d l2arctic/ >/dev/null 2>&1 && {
                        ok=1
                        break
                    }
                    echo "Retry $f ($i/5)..."
                    sleep 1
                done
                [ "$ok" -eq 0 ] && skipped="$skipped $f"
            done && \
            echo && echo "==== SKIPPED FILES ====" && echo "$skipped"
            # PS: You can re-run the above snippet with the skipped files to extract them.
            # Unzipping failed as the load on filesystem was large when I tried.
        ```
         $ cd {data_dir}/l2arctic_release_v5.0
         $ # Extract each speaker (zip files already contain speaker folder, e.g., ABA/)
         $ for file in *.zip; do echo "unzipping $file" && unzip -q "$file"; done
         $ rm *.zip
    
    2. CMU ARCTIC:
       - Selected speakers: BDL, SLT, CLB, RMS
       - Command:
         $ mkdir -p {data_dir}/cmu_arctic && cd {data_dir}/cmu_arctic
         $ wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.tar.bz2
         $ wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2
         $ wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_clb_arctic-0.95-release.tar.bz2
         $ wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_rms_arctic-0.95-release.tar.bz2
         $ for file in *.tar.bz2; do tar -xjf "$file"; done

Test Speakers (fixed, from L2-classification project):
    ['SKA', 'BWC', 'SVBI', 'HKK', 'NJS', 'HQTV', 'SLT']
    - 6 L2-ARCTIC speakers + 1 CMU speaker
    - Ensures no speaker overlap between train/test

Usage:
    python -m src.recipe.l1_classification.local.prepare_cmu_l2arctic_metadata \
        --l2arctic_root exp/download/cmu_l2arctic/l2arctic \
        --cmu_root exp/download/cmu_l2arctic/cmu \
        --output_csv exp/cache/cmu_l2arctic/metadata.csv \
        --data_dir exp/download/cmu_l2arctic \
        --val_speakers_per_l1 1  # (optional, default=1) stratified: 1 val speaker per L1
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List
from collections import Counter
import random
from tqdm import tqdm


# L2Arctic speaker to L1 mapping
# Based on L2Arctic corpus documentation
L2ARCTIC_SPEAKER_L1 = {
    "ABA": "ar",  # Arabic
    "SKA": "ar",  # Arabic
    "YBAA": "ar",  # Arabic
    "ZHAA": "ar",  # Arabic
    "BWC": "zh",  # Mandarin Chinese
    "LXC": "zh",  # Mandarin Chinese
    "NCC": "zh",  # Mandarin Chinese
    "TXHC": "zh",  # Mandarin Chinese
    "ASI": "hi",  # Hindi
    "RRBI": "hi",  # Hindi
    "SVBI": "hi",  # Hindi
    "TNI": "hi",  # Hindi
    "HJK": "ko",  # Korean
    "HKK": "ko",  # Korean
    "YDCK": "ko",  # Korean
    "YKWK": "ko",  # Korean
    "EBVS": "es",  # Spanish
    "ERMS": "es",  # Spanish
    "MBMPS": "es",  # Spanish
    "NJS": "es",  # Spanish
    "HQTV": "vi",  # Vietnamese
    "PNV": "vi",  # Vietnamese
    "THV": "vi",  # Vietnamese
    "TLV": "vi",  # Vietnamese
}

# CMU Arctic speakers are native English
# Selected speakers: BDL, SLT, CLB, RMS (SLT for test, others for train)
CMU_SPEAKERS = ["BDL", "SLT", "CLB", "RMS"]

# Fixed test speakers (from L2-classification project)
# Reference: https://github.com/...L2-classification/scripts/split_manifest.py
TEST_SPEAKERS = ["SKA", "BWC", "SVBI", "HKK", "NJS", "HQTV", "SLT"]


def _ensure_relative_to_data_dir(
    abs_path: Path, data_dir_path: Path, corpus_name: str
) -> Path:
    """
    Ensure the absolute file path lives under the declared data_dir.
    Raises ValueError with a descriptive message otherwise.
    """
    try:
        return abs_path.relative_to(data_dir_path)
    except ValueError as exc:
        raise ValueError(
            f"{corpus_name} file {abs_path} is not located under data_dir {data_dir_path}. "
            "Please make sure both corpora share the same data_dir root."
        ) from exc


def collect_l2arctic_files(l2arctic_root: str, data_dir: str) -> List[Dict[str, str]]:
    """
    Collect audio files from L2Arctic corpus.

    Args:
        l2arctic_root: Path to L2Arctic root directory
        data_dir: Common data directory for relative path calculation

    Returns:
        List of dicts with metadata
    """
    l2arctic_path = Path(l2arctic_root)
    data_dir_path = Path(data_dir).expanduser().resolve()
    samples = []

    for speaker_id, l1_label in tqdm(
        L2ARCTIC_SPEAKER_L1.items(), desc="Processing L2Arctic speakers"
    ):
        speaker_dir = l2arctic_path / speaker_id / "wav"
        if not speaker_dir.exists():
            print(f"Warning: Speaker directory not found: {speaker_dir}")
            continue

        for wav_file in speaker_dir.glob("*.wav"):
            # Calculate relative path from data_dir
            abs_path = wav_file.resolve()
            rel_path = _ensure_relative_to_data_dir(abs_path, data_dir_path, "L2Arctic")

            utt_id = wav_file.stem
            samples.append(
                {
                    "audio_path": str(rel_path),
                    "l1_label": l1_label,
                    "speaker_id": speaker_id,
                    "utt_id": utt_id,
                }
            )

    print(f"Collected {len(samples)} samples from L2Arctic")
    return samples


def collect_cmu_arctic_files(cmu_root: str, data_dir: str) -> List[Dict[str, str]]:
    """
    Collect audio files from CMU Arctic corpus.

    Args:
        cmu_root: Path to CMU Arctic root directory
        data_dir: Common data directory for relative path calculation

    Returns:
        List of dicts with metadata
    """
    cmu_path = Path(cmu_root)
    data_dir_path = Path(data_dir).expanduser().resolve()
    samples = []

    for speaker_id in tqdm(CMU_SPEAKERS, desc="Processing CMU speakers"):
        # CMU Arctic directory names are lowercase (e.g., cmu_us_bdl_arctic)
        speaker_dir = cmu_path / f"cmu_us_{speaker_id.lower()}_arctic" / "wav"
        if not speaker_dir.exists():
            print(f"Warning: CMU speaker directory not found: {speaker_dir}")
            continue

        for wav_file in speaker_dir.glob("*.wav"):
            # Calculate relative path from data_dir
            abs_path = wav_file.resolve()
            rel_path = _ensure_relative_to_data_dir(
                abs_path, data_dir_path, "CMU Arctic"
            )

            utt_id = wav_file.stem
            samples.append(
                {
                    "audio_path": str(rel_path),
                    "l1_label": "en",  # All CMU speakers are native English
                    "speaker_id": speaker_id,
                    "utt_id": utt_id,
                }
            )

    print(f"Collected {len(samples)} samples from CMU Arctic")
    return samples


def split_data(
    samples: List[Dict[str, str]],
    test_speakers: List[str] = None,
    val_speakers_per_l1: int = 1,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Split data into train/val/test sets by speaker with stratified L1 sampling.

    Uses fixed test speakers from L2-classification project to ensure consistency.
    Validation speakers are selected to ensure each L1 class is represented.

    Args:
        samples: List of sample dicts
        test_speakers: Fixed list of test speaker IDs (default: TEST_SPEAKERS constant)
        val_speakers_per_l1: Number of validation speakers per L1 class (default: 1)
        seed: Random seed

    Returns:
        List of samples with 'split' field added
    """
    if test_speakers is None:
        test_speakers = TEST_SPEAKERS

    random.seed(seed)

    # Normalize test speakers to uppercase for comparison
    test_speakers_upper = {s.upper() for s in test_speakers}

    # Group samples by speaker and get speaker -> L1 mapping
    speaker_samples = {}
    speaker_to_l1 = {}
    for sample in samples:
        spk = sample["speaker_id"]
        if spk not in speaker_samples:
            speaker_samples[spk] = []
            speaker_to_l1[spk] = sample["l1_label"]
        speaker_samples[spk].append(sample)

    # Separate test speakers and non-test speakers
    all_speakers = list(speaker_samples.keys())
    test_spk_found = []
    non_test_speakers = []

    for spk in all_speakers:
        spk_upper = spk.upper()
        if spk_upper in test_speakers_upper:
            test_spk_found.append(spk)
        else:
            non_test_speakers.append(spk)

    # Group non-test speakers by L1 for stratified validation selection
    l1_to_speakers = {}
    for spk in non_test_speakers:
        l1 = speaker_to_l1[spk]
        if l1 not in l1_to_speakers:
            l1_to_speakers[l1] = []
        l1_to_speakers[l1].append(spk)

    # Select validation speakers: stratified by L1 (1 speaker per L1 by default)
    val_speakers = []
    train_speakers = []

    for l1, spk_list in l1_to_speakers.items():
        random.shuffle(spk_list)
        n_val = min(val_speakers_per_l1, len(spk_list) - 1)  # Keep at least 1 for train
        n_val = max(0, n_val)  # Ensure non-negative

        val_speakers.extend(spk_list[:n_val])
        train_speakers.extend(spk_list[n_val:])

    # Convert to sets for O(1) lookup
    val_speakers_set = set(val_speakers)
    test_spk_found_set = set(test_spk_found)

    print(f"\nSplit statistics (stratified by L1):")
    print(f"  Train speakers: {len(train_speakers)}")
    print(f"  Val speakers: {len(val_speakers)} ({val_speakers_per_l1} per L1)")
    print(f"  Test speakers: {len(test_spk_found)} (fixed from L2-classification)")
    print(f"  Val speakers by L1:")
    for l1 in sorted(l1_to_speakers.keys()):
        val_spk_for_l1 = [s for s in val_speakers if speaker_to_l1[s] == l1]
        print(f"    {l1}: {val_spk_for_l1}")
    print(f"  Test speakers: {test_spk_found}")

    # Assign split to each sample
    result = []
    for sample in samples:
        spk = sample["speaker_id"]
        if spk in test_spk_found_set:
            sample["split"] = "test"
        elif spk in val_speakers_set:
            sample["split"] = "val"
        else:
            sample["split"] = "train"
        result.append(sample)

    # Print split statistics by samples
    split_counts = {"train": 0, "val": 0, "test": 0}
    label_by_split = {"train": Counter(), "val": Counter(), "test": Counter()}

    for sample in result:
        split = sample["split"]
        split_counts[split] += 1
        label_by_split[split][sample["l1_label"]] += 1

    print(f"\nSample statistics:")
    print(f"  Train samples: {split_counts['train']}")
    print(f"  Val samples: {split_counts['val']}")
    print(f"  Test samples: {split_counts['test']}")

    print(f"\nL1 label distribution:")
    for split in ["train", "val", "test"]:
        print(f"  {split.capitalize()}:")
        for label, count in sorted(label_by_split[split].items()):
            print(f"    {label}: {count}")

    return result


def write_metadata_csv(samples: List[Dict[str, str]], output_path: str):
    """
    Write samples to CSV file.

    Args:
        samples: List of sample dicts
        output_path: Path to output CSV file
    """
    fieldnames = ["audio_path", "l1_label", "split", "speaker_id", "utt_id"]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)

    print(f"\nMetadata CSV written to: {output_path}")
    print(f"Total samples: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata.csv for CMU + L2Arctic L1 classification"
    )
    parser.add_argument(
        "--l2arctic_root",
        type=str,
        required=True,
        help="Path to L2Arctic root directory",
    )
    parser.add_argument(
        "--cmu_root",
        type=str,
        required=True,
        help="Path to CMU Arctic root directory",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output metadata.csv",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Common data directory (for relative path calculation)",
    )
    parser.add_argument(
        "--test_speakers",
        type=str,
        nargs="+",
        default=None,
        help="List of test speaker IDs (default: fixed TEST_SPEAKERS from L2-classification)",
    )
    parser.add_argument(
        "--val_speakers_per_l1",
        type=int,
        default=1,
        help="Number of validation speakers per L1 class (default: 1, stratified)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )

    args = parser.parse_args()

    # Collect samples from both corpora
    print("Collecting L2Arctic samples...")
    l2arctic_samples = collect_l2arctic_files(args.l2arctic_root, args.data_dir)

    print("\nCollecting CMU Arctic samples...")
    cmu_samples = collect_cmu_arctic_files(args.cmu_root, args.data_dir)

    # Combine samples
    all_samples = l2arctic_samples + cmu_samples
    print(f"\nTotal samples collected: {len(all_samples)}")

    # Split data
    print(
        "\nSplitting data by speaker (stratified by L1, using fixed test speakers)..."
    )
    samples_with_split = split_data(
        all_samples,
        test_speakers=args.test_speakers,
        val_speakers_per_l1=args.val_speakers_per_l1,
        seed=args.seed,
    )

    # Write to CSV
    output_csv_path = Path(args.output_csv).expanduser()
    output_dir = output_csv_path.parent
    if str(output_dir) not in ("", "."):
        output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata_csv(samples_with_split, str(output_csv_path))

    # Print sample preview
    print("\n=== Sample preview (first 5 rows) ===")
    for i, sample in enumerate(samples_with_split[:5]):
        print(f"{i+1}. {sample}")


if __name__ == "__main__":
    main()
