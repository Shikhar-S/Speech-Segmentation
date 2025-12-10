"""
Metadata preparation script for speechocean762 dataset.

To download the corpus, clone this git repo, and remove .git folder: 
    https://github.com/jimbozhang/speechocean762

Usage:
    python -m src.recipe.l2_assessment.local.prepare_metadata_speechocean \
        --corpus_root /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/download/speechocean762 \
        --output_csv /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/cache/speechocean762/metadata.csv \
        --val_ratio 0.1 \
        --seed 42
"""

import os
import json
import argparse
import pandas as pd
import random
from tqdm import tqdm


def read_wav_scp(wav_scp_path):
    """
    Reads Kaldi wav.scp file.

    Expected format:
        utt_id  /absolute/or/relative/path/to/audio.wav

    Returns:
        dict: utt_id -> audio_path
    """
    mapping = {}
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, path = line.split(maxsplit=1)
            mapping[utt_id] = path
    return mapping


def extract_speaker_id(audio_rel_path):
    """
    WAVE/SPEAKER0001/000010011.WAV -> '0001'
    """
    speaker_dir = os.path.basename(os.path.dirname(audio_rel_path))
    assert speaker_dir.startswith("SPEAKER")
    return speaker_dir.replace("SPEAKER", "")


def create_val_split(df, val_ratio=0.1, seed=42, mode="speaker"):
    """
    Create val split from train split.

    Args:
        df: full metadata DataFrame
        val_ratio: fraction of train used as val
        seed: random seed
        mode:
            - 'utterance': random utterance-level split
            - 'speaker': speaker-disjoint split (recommended)
    """
    rng = random.Random(seed)

    train_df = df[df["split"] == "train"].copy()

    if mode == "utterance":
        indices = list(train_df.index)
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx = set(indices[:n_val])

        df.loc[val_idx, "split"] = "val"

    elif mode == "speaker":
        speakers = train_df["speaker_id"].unique().tolist()
        rng.shuffle(speakers)

        n_val_spk = max(1, int(len(speakers) * val_ratio))
        val_speakers = set(speakers[:n_val_spk])

        df.loc[
            (df["split"] == "train") & (df["speaker_id"].isin(val_speakers)),
            "split",
        ] = "val"

    else:
        raise ValueError(f"Unknown split mode: {mode}")

    return df


def main(args):
    corpus_root = os.path.abspath(args.corpus_root)

    # Load sentence-level scores
    scores_path = os.path.join(corpus_root, "resource/scores.json")
    with open(scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    rows = []

    for split in ("train", "test"):
        wav_scp = os.path.join(corpus_root, split, "wav.scp")
        utt2path = read_wav_scp(wav_scp)

        print(f"Processing {split} split: {len(utt2path)} utterances")

        for utt_id, audio_path in tqdm(utt2path.items()):

            speaker_id = extract_speaker_id(audio_path)
            score = scores[utt_id]

            rows.append(
                {
                    "audio_path": audio_path,  # relative audio path from corpus_root
                    "split": split,  # split: 'train' / 'val / 'test'
                    "speaker_id": speaker_id,  # speaker id
                    "utt_id": utt_id,  # utterance id
                    "text": score["text"],  # transcript text
                    "accuracy": score["accuracy"],  # sentence-level accuracy score
                    "completeness": score[
                        "completeness"
                    ],  # sentence-level completeness score
                    "fluency": score["fluency"],  # sentence-level fluency score
                    "prosodic": score["prosodic"],  # sentence-level prosodic score
                    "total": score["total"],  # sentence-level total score
                }
            )

    df = pd.DataFrame(rows)

    df = create_val_split(
        df,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mode=args.split_mode,
    )

    df = df.sort_values(["split", "speaker_id", "utt_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_root",
        type=str,
        required=True,
        help="Root directory of speechocean762 corpus",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="metadata.csv",
        help="Output metadata CSV path",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="speaker",
        choices=["utterance", "speaker"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
