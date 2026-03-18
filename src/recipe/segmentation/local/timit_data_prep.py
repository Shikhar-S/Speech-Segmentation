"""TIMIT Corpus Data Preparation for Forced Alignment

This script creates segment-level metadata (one segment = one TIMIT utterance)
for use with a forced-alignment data pipeline.

Usage:
    python -m src.recipe.segmentation.local.timit_data_prep \
        --timit_root /work/hdd/bbjs/shared/corpora/TIMIT/timit_nltk \
        --output_dir /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/cache/timit \
        --split_index /work/hdd/bbjs/shared/corpora/TIMIT/timit_nltk/split_index.txt

The script will create:
    - train_metadata.json / train_metadata.csv
    - val_metadata.json   / val_metadata.csv
    - test_metadata.json  / test_metadata.csv

Split creation logic depends on the task.
    There are pre-defined train and test split.
    (Dialect, speaker) disjoint split is also possible.
    Content disjoint split is also possible.

This script creates **speaker-disjoint** split sticking to the predefined 
train/test splits of TIMIT, while creating a (speaker-disjoint) validation 
set from the training speakers.
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.corpus.reader.timit import TimitCorpusReader
from nltk.data import FileSystemPathPointer

log = logging.getLogger("timit_prep")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SR = 16000  # TIMIT offsets are in 16 kHz samples


@dataclass
class TimitAlignmentSegment:
    segment_id: str
    speaker_id: str
    track_id: str
    start_time: float
    end_time: float
    text: str
    phones: List[str]
    phone_timestamps: List[Tuple[float, float]]


def load_all_segments(reader: TimitCorpusReader) -> List[TimitAlignmentSegment]:
    utterances = reader.utteranceids()
    log.info(f"Found {len(utterances)} utterances in TIMIT.")
    segments: List[TimitAlignmentSegment] = []
    for u in tqdm(utterances, desc="Processing utterances", unit="utt"):
        speaker_id = reader.spkrid(u)
        sent_id = reader.sentid(u)
        word_sents = reader.sents(u)
        if not word_sents:
            continue
        text = " ".join(word_sents[0])
        phone_times = reader.phone_times(u)
        if not phone_times:
            continue
        phones = []
        phone_timestamps = []
        for ph, start, end in phone_times:
            t0 = start / SR
            t1 = end / SR
            phones.append(ph)
            phone_timestamps.append((t0, t1))
        end_time = phone_timestamps[-1][1]
        segments.append(
            TimitAlignmentSegment(
                segment_id=u,
                speaker_id=speaker_id,
                track_id=sent_id,
                start_time=0.0,
                end_time=end_time,
                text=text,
                phones=phones,
                phone_timestamps=phone_timestamps,
            )
        )
    log.info(f"Constructed {len(segments)} segments.")
    return segments


def load_split_index(split_file: Path) -> Dict[str, List[str]]:
    """
    Load split_index.txt mapping TRAIN/TEST to segment_ids (speaker-folder/sentence).
    Returns: { "TRAIN": [...], "TEST": [...] }
    """
    mapping: Dict[str, List[str]] = {"TRAIN": [], "TEST": []}
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        split_name, full_path = parts
        # extract last two entries: e.g. dr1-mcpm0/sx384.txt
        rel = Path(full_path).parts[-2] + "/" + Path(full_path).stem
        mapping.setdefault(split_name, []).append(rel)
    log.info("Loaded split definitions: %s", {k: len(v) for k, v in mapping.items()})
    return mapping


def apply_predefined_split(
    segments: List[TimitAlignmentSegment],
    split_map: Dict[str, List[str]],
    val_ratio: float = 0.1,
) -> Dict[str, List[TimitAlignmentSegment]]:
    """
    Use official TRAIN vs TEST from split_map.
    From TRAIN speakers select val_ratio fraction of speakers for validation.
    Ensures speaker-disjoint for val/trn, and all TEST segments in test set.
    """
    test_ids = set(split_map.get("TEST", []))
    # filter segments in TEST set
    test_segs = [s for s in segments if s.segment_id in test_ids]
    # all others go into train pool
    trainpool = [s for s in segments if s.segment_id not in test_ids]
    # speaker disjoint split for val from trainpool
    spk2segs: Dict[str, List[TimitAlignmentSegment]] = {}
    for s in trainpool:
        spk2segs.setdefault(s.speaker_id, []).append(s)
    speakers = list(spk2segs.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(speakers)
    n_val = int(len(speakers) * val_ratio)
    val_spks = set(speakers[:n_val])
    val_segs = [s for s in trainpool if s.speaker_id in val_spks]
    trn_segs = [s for s in trainpool if s.speaker_id not in val_spks]
    log.info(
        "Final splits — train: %d, val: %d, test: %d",
        len(trn_segs),
        len(val_segs),
        len(test_segs),
    )
    return {"train": trn_segs, "val": val_segs, "test": test_segs}


def save_metadata(out_dir: Path, name: str, segs: List[TimitAlignmentSegment]) -> None:
    with open(out_dir / f"{name}_metadata.json", "w") as f:
        json.dump(
            [asdict(s) | {"duration": s.end_time - s.start_time} for s in segs],
            f,
            indent=2,
        )
    df = pd.DataFrame(
        [
            {
                "segment_id": s.segment_id,
                "track_id": s.track_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.end_time - s.start_time,
                "n_phones": len(s.phones),
                "n_words": len(s.text.split()),
                "speaker_id": s.speaker_id,
            }
            for s in segs
        ]
    )
    df.to_csv(out_dir / f"{name}_metadata.csv", index=False)
    log.info(f"Saved {name}: {len(segs)} segments")


def main():
    p = argparse.ArgumentParser(
        description="Prepare TIMIT corpus for forced alignment with official split"
    )
    p.add_argument(
        "--timit_root", required=True, type=Path, help="Path to TIMIT root directory"
    )
    p.add_argument(
        "--split_index", required=True, type=Path, help="Path to split_index.txt"
    )
    p.add_argument(
        "--output_dir", required=True, type=Path, help="Output directory for metadata"
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of train-speakers used for validation",
    )
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading TIMIT from: {args.timit_root}")
    reader = TimitCorpusReader(FileSystemPathPointer(str(args.timit_root)))
    segments = load_all_segments(reader)
    if not segments:
        log.error("No segments extracted. Check corpus path/structure.")
        return

    split_map = load_split_index(args.split_index)
    splits = apply_predefined_split(segments, split_map, val_ratio=args.val_ratio)
    for k, segs in splits.items():
        save_metadata(out, k, segs)

    durs = np.array([s.end_time - s.start_time for s in segments])
    log.info(
        "Durations (s): mean=%.2f std=%.2f min=%.2f max=%.2f total=%.2fh",
        durs.mean(),
        durs.std(),
        durs.min(),
        durs.max(),
        durs.sum() / 3600.0,
    )


if __name__ == "__main__":
    main()
