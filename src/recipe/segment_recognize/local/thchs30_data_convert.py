"""Convert THCHS-30 alignment parquet + OpenSLR audio to canonical HF segmentation format.

The HF parquet (`anyspeech/THCHS-30-alignments`) has columns
``[id, phones, start, end, words, word_start, word_end]`` and 13,388 rows but
NO ``split`` column. Splits are recovered from the OpenSLR directory layout
(``data_thchs30/{train,dev,test}/<id>.wav``). Each wav has a sibling
``<id>.wav.trn`` whose first line is the Hanzi sentence we use as ``text``.

Usage::

    python -m src.recipe.segment_recognize.local.thchs30_data_convert \\
        --alignments_parquet exp/downloads/thchs30_alignments/data/train-00000-of-00001-c7710d0536782c3f.parquet \\
        --speech_dir         exp/downloads/thchs30_speech/data_thchs30 \\
        --output_dir         exp/downloads/thchs30-seg \\
        [--num_workers       64] \\
        [--limit             100]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import pyarrow.parquet as pq
from tqdm import tqdm

from src.recipe.segment_recognize.local._parquet_sharded import write_parquet_shards

log = logging.getLogger("thchs30_data_convert")
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

# Map OpenSLR source directory names -> canonical output split names.
# The existing segmentation datasets (changelinglab/buckeye-segment,
# changelinglab/timit-segment) use ``train/val/test``, so we rename
# ``dev`` -> ``val`` on output.
SRC_SPLITS = ("train", "dev", "test")
OUT_SPLIT_MAP = {"train": "train", "dev": "val", "test": "test"}
OUT_SPLITS = tuple(OUT_SPLIT_MAP[s] for s in SRC_SPLITS)


def index_audio(speech_dir: Path) -> Dict[str, Tuple[str, Path]]:
    """Map utterance id (e.g. 'A11_0') -> (out_split, absolute wav path)."""
    speech_dir = speech_dir.resolve()
    index: Dict[str, Tuple[str, Path]] = {}
    for src_split in SRC_SPLITS:
        split_dir = speech_dir / src_split
        if not split_dir.is_dir():
            log.warning("Split dir missing: %s", split_dir)
            continue
        out_split = OUT_SPLIT_MAP[src_split]
        for wav in split_dir.glob("*.wav"):
            index[wav.stem] = (out_split, wav)
        log.info(
            "Indexed %d wavs in src split '%s' -> out split '%s' under %s",
            sum(1 for s, _ in index.values() if s == out_split),
            src_split, out_split, split_dir,
        )
    return index


def read_transcript(wav_path: Path) -> str:
    """Return the Hanzi sentence for a THCHS-30 utterance.

    Layout: ``train/<id>.wav.trn`` is a 1-line file containing
    ``../data/<id>.wav.trn`` — a relative path to the real transcript.
    The canonical transcript at ``data/<id>.wav.trn`` has 3 lines:
    Hanzi sentence, pinyin, phoneme sequence. We want line 1 (Hanzi).
    """
    data_trn = wav_path.parent.parent / "data" / f"{wav_path.stem}.wav.trn"
    if not data_trn.is_file():
        return ""
    with open(data_trn, encoding="utf-8") as f:
        return f.readline().strip()


def build_records(
    parquet_path: Path,
    audio_index: Dict[str, Tuple[str, Path]],
    limit: int,
) -> Dict[str, List[Dict]]:
    """Iterate parquet rows, join audio + transcript, group by split."""
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows if limit <= 0 else min(limit, table.num_rows)
    log.info("Loaded %d rows from %s (using %d)", table.num_rows, parquet_path, n_rows)

    rows = table.slice(0, n_rows).to_pylist()
    by_split: Dict[str, List[Dict]] = {s: [] for s in OUT_SPLITS}
    n_missing = 0
    for row in tqdm(rows, desc="thchs30", unit="utt"):
        utt_id = row["id"]
        if utt_id not in audio_index:
            n_missing += 1
            continue
        split, wav_path = audio_index[utt_id]
        phones = list(row["phones"])
        starts = list(row["start"])
        ends = list(row["end"])
        by_split[split].append(
            {
                "utt_id": utt_id,
                "audio": str(wav_path),
                "text": read_transcript(wav_path),
                "phones": phones,
                "phone_starts": starts,
                "phone_ends": ends,
                "language": "cmn",
                "speaker_id": utt_id.split("_", 1)[0],
                "duration": float(ends[-1]) if ends else 0.0,
                "split": split,
            }
        )
    log.info(
        "Built records: train=%d val=%d test=%d (skipped %d missing audio)",
        len(by_split["train"]), len(by_split["val"]), len(by_split["test"]), n_missing,
    )
    return by_split


def save_dataset(
    by_split: Dict[str, List[Dict]],
    output_dir: Path,
    num_workers: int,
) -> None:
    """Write one set of parquet shards per split under ``output_dir/data/``.

    Uses the sharded writer helper so audio bytes are embedded in parallel.
    The resulting layout is loadable via ``datasets.load_dataset(output_dir)``.
    """
    wrote = 0
    for split_name, recs in by_split.items():
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
    p.add_argument("--alignments_parquet", required=True, type=Path)
    p.add_argument("--speech_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_workers", type=int, default=64,
                   help="Parallel processes for the embed+write step.")
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows to process (0 = all). For smoke testing.")
    args = p.parse_args()

    audio_index = index_audio(args.speech_dir)
    by_split = build_records(args.alignments_parquet, audio_index, args.limit)
    save_dataset(by_split, args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()
