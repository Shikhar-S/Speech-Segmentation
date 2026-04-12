"""Convert LibriSpeech MFA alignment parquets + .flac audio to canonical HF segmentation format.

The HF parquets (`anyspeech/librispeech_MFA_alignments`) ship one file per
split (``train.clean.100``, ``dev.clean``, etc.) with columns
``[identifier, duration, phones, start, end]``. The split label is encoded in
the filename, not in a column. Audio lives at::

    LibriSpeech/<split-with-dashes>/<speaker>/<book>/<speaker>-<book>-<utt>.flac

with a per-book ``<speaker>-<book>.trans.txt`` mapping utt_id -> sentence.

Usage::

    python -m src.recipe.segment_recognize.local.librispeech_data_convert \\
        --alignments_dir exp/downloads/librispeech_mfa_alignments/data \\
        --speech_dir     /work/hdd/bbjs/shared/corpora/librispeech/LibriSpeech \\
        --output_dir     exp/downloads/librispeech-mfa-seg \\
        [--hf_repo       <user>/librispeech-mfa-aligned] \\
        [--splits        train.clean.100,dev.clean] \\
        [--limit         100]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import pyarrow.parquet as pq
from tqdm import tqdm

from src.recipe.segment_recognize.local._parquet_sharded import write_parquet_shards

log = logging.getLogger("librispeech_data_convert")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


SCHEMA = datasets.Features(
    {
        "utt_id": datasets.Value("string"),
        "audio": datasets.Audio(sampling_rate=16000),
        "text": datasets.Value("string"),
        "phones": datasets.Sequence(datasets.Value("string")),
        "phone_starts": datasets.Sequence(datasets.Value("float64")),
        "phone_ends": datasets.Sequence(datasets.Value("float64")),
        "language": datasets.Value("string"),
        "speaker_id": datasets.Value("string"),
        "duration": datasets.Value("float64"),
        "split": datasets.Value("string"),
    }
)

ALL_SPLITS = (
    "train.clean.100", "train.clean.360", "train.other.500",
    "dev.clean", "dev.other", "test.clean", "test.other",
)


def find_parquet(alignments_dir: Path, split_label: str) -> Optional[Path]:
    """Locate parquet for a split (filename starts with '<split>-')."""
    matches = sorted(alignments_dir.glob(f"{split_label}-*.parquet"))
    if not matches:
        log.warning("No parquet found for split '%s' in %s", split_label, alignments_dir)
        return None
    if len(matches) > 1:
        log.warning("Multiple parquets for '%s'; using first: %s", split_label, matches[0])
    return matches[0]


def load_trans(book_dir: Path) -> Dict[str, str]:
    """Read a LibriSpeech .trans.txt file -> {utt_id: text}."""
    trans: Dict[str, str] = {}
    for trans_path in book_dir.glob("*.trans.txt"):
        with open(trans_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    trans[parts[0]] = parts[1]
    return trans


def convert_split(
    parquet_path: Path,
    speech_dir: Path,
    split_label: str,
    limit: int,
) -> List[Dict]:
    """Build records for one LibriSpeech split."""
    dir_name = split_label.replace(".", "-")
    split_root = speech_dir / dir_name
    if not split_root.is_dir():
        log.error("Audio dir missing: %s", split_root)
        return []

    table = pq.read_table(parquet_path)
    n_rows = table.num_rows if limit <= 0 else min(limit, table.num_rows)
    log.info("Split '%s': %d rows in parquet (using %d)",
             split_label, table.num_rows, n_rows)

    rows = table.slice(0, n_rows).to_pylist()
    trans_cache: Dict[Path, Dict[str, str]] = {}
    records: List[Dict] = []
    n_bad = 0

    # Skip per-row audio existence check: we trust the parquet alignments
    # have corresponding audio. Per-row stat() against NFS is the dominant
    # cost (~50ms each at scale). Missing files would surface later when
    # the dataloader actually decodes them — acceptable failure mode.
    for row in tqdm(rows, desc=split_label, unit="utt"):
        utt_id = row["identifier"]
        parts = utt_id.split("-")
        if len(parts) != 3:
            log.warning("Unexpected identifier format: %s", utt_id)
            n_bad += 1
            continue
        speaker, book, _ = parts
        book_dir = split_root / speaker / book
        audio_path = book_dir / f"{utt_id}.flac"
        if book_dir not in trans_cache:
            trans_cache[book_dir] = load_trans(book_dir)
        text = trans_cache[book_dir].get(utt_id, "")

        records.append(
            {
                "utt_id": utt_id,
                "audio": str(audio_path),
                "text": text,
                "phones": list(row["phones"]),
                "phone_starts": list(row["start"]),
                "phone_ends": list(row["end"]),
                "language": "eng",
                "speaker_id": speaker,
                "duration": float(row["duration"]),
                "split": split_label,
            }
        )
    log.info("Split '%s': built %d records, skipped %d malformed",
             split_label, len(records), n_bad)
    return records


def save_dataset(
    splits_to_records: Dict[str, List[Dict]],
    output_dir: Path,
    num_workers: int,
) -> None:
    wrote = 0
    for split_name, recs in splits_to_records.items():
        wrote += write_parquet_shards(
            records=recs,
            features=SCHEMA,
            output_dir=output_dir,
            split_name=split_name,
            num_workers=num_workers,
        )
    if wrote == 0:
        log.error("No records produced; nothing written.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alignments_dir", required=True, type=Path)
    p.add_argument("--speech_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_workers", type=int, default=64,
                   help="Parallel processes for the embed+write step.")
    p.add_argument(
        "--splits",
        default=",".join(ALL_SPLITS),
        help="Comma-separated split labels (default: all 7).",
    )
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows per split (0 = all).")
    args = p.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    splits_to_records: Dict[str, List[Dict]] = {}
    for split_label in splits:
        parquet_path = find_parquet(args.alignments_dir, split_label)
        if parquet_path is None:
            continue
        splits_to_records[split_label] = convert_split(
            parquet_path, args.speech_dir, split_label, args.limit
        )
    save_dataset(splits_to_records, args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()
