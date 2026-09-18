"""Convert processed JSON metadata to the canonical HuggingFace segmentation dataset format.

Reads per-split metadata JSON files produced by ``timit_data_prep.py`` or
``buckeye_data_prep.py`` and writes a HuggingFace ``datasets.DatasetDict`` to disk
in the same schema as ``seg_data_convert.py``.

Usage (TIMIT-style — ``audio_root/{segment_id}.wav``)::

python scripts/data_prep/processed_data_convert.py \
    --metadata_dir exp/data/timit_meta \
    --audio_root   $TIMIT_ROOT/timit_nltk \
    --output_dir   exp/data/timit-segment \
    --hf_repo      <org>/<repo>   # then export SEG_REPO_TIMIT=<org>/<repo> \
    --language     eng

Usage (Buckeye-style — pre-extracted clips in ``clips_dir/{segment_id}.wav``)::

python scripts/data_prep/processed_data_convert.py \
    --metadata_dir exp/data/buckeye_meta \
    --clips_dir    exp/data/buckeye_meta/speech_clips \
    --output_dir   exp/data/buckeye-segment \
    --hf_repo      <org>/<repo>   # then export SEG_REPO_BUCKEYE=<org>/<repo> \
    --language     eng

Exactly one of ``--audio_root`` or ``--clips_dir`` must be supplied.

All ``*_metadata.json`` files found in ``--metadata_dir`` are processed; the
split name is inferred as ``stem.replace("_metadata", "")``.

Schema written (identical to ``seg_data_convert.py``)
-------------------------------------------------------
- ``utt_id``       : str
- ``audio``        : datasets.Audio  (path reference, decoded on-the-fly)
- ``text``         : str
- ``phones``       : List[str]
- ``phone_starts`` : List[float64]   (seconds)
- ``phone_ends``   : List[float64]   (seconds)
- ``language``     : str
- ``speaker_id``   : str
- ``duration``     : float64
- ``split``        : str
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import datasets
from datasets import DatasetDict

log = logging.getLogger("processed_data_convert")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _audio_path(
    segment_id: str,
    audio_root: Optional[Path],
    clips_dir: Optional[Path],
) -> str:
    """Return the absolute audio path for a segment.

    TIMIT-style (``audio_root``): ``audio_root/{segment_id}.wav``
    The segment_id may contain ``/`` (e.g. ``dr1-fvmh0/sx206``), which forms a
    sub-directory under ``audio_root``.

    Buckeye-style (``clips_dir``): ``clips_dir/{segment_id_safe}.wav``
    ``/`` in segment_id is replaced with ``-`` for the filename.
    """
    if audio_root is not None:
        return str(audio_root / f"{segment_id}.wav")
    seg_safe = segment_id.replace("/", "-")
    return str(clips_dir / f"{seg_safe}.wav")  # type: ignore[operator]


def convert(
    metadata_path: Path,
    audio_root: Optional[Path],
    clips_dir: Optional[Path],
    language: str,
    split: str,
) -> Optional[datasets.Dataset]:
    """Convert one metadata JSON to a HuggingFace Dataset.

    Returns the Dataset, or None if no records were produced.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    records: List[Dict] = []
    n_missing = 0

    for item in metadata:
        segment_id = item["segment_id"]
        audio = _audio_path(segment_id, audio_root, clips_dir)

        if not Path(audio).exists():
            log.warning("Audio not found for %s (%s); skipping.", segment_id, audio)
            n_missing += 1
            continue

        phone_timestamps = item["phone_timestamps"]
        phone_starts = [t[0] for t in phone_timestamps]
        phone_ends = [t[1] for t in phone_timestamps]

        duration = item.get("duration") or (item["end_time"] - item["start_time"])

        records.append(
            {
                "utt_id": segment_id,
                "audio": audio,
                "text": item.get("text", ""),
                "phones": item["phones"],
                "phone_starts": phone_starts,
                "phone_ends": phone_ends,
                "language": language,
                "speaker_id": item.get("speaker_id", ""),
                "duration": duration,
                "split": split,
            }
        )

    log.info(
        "Built %d records (%d skipped) for split '%s'.", len(records), n_missing, split
    )

    if not records:
        log.error("No records produced for split '%s'.", split)
        return None

    schema = datasets.Features(
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

    return datasets.Dataset.from_list(records, features=schema)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert processed JSON metadata to HuggingFace segmentation dataset format."
    )
    p.add_argument(
        "--metadata_dir",
        required=True,
        type=Path,
        help=(
            "Directory containing *_metadata.json files (e.g. train_metadata.json). "
            "Split name is inferred as the stem with '_metadata' removed."
        ),
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Root output directory; DatasetDict written directly here",
    )
    audio_group = p.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--audio_root",
        type=Path,
        default=None,
        help="Root dir for TIMIT-style audio: audio_root/{segment_id}.wav",
    )
    audio_group.add_argument(
        "--clips_dir",
        type=Path,
        default=None,
        help="Dir of pre-extracted WAV clips (Buckeye-style): clips_dir/{segment_id}.wav",
    )
    p.add_argument(
        "--language",
        default="",
        help="Language tag (no angle brackets) applied to all records (default: '')",
    )
    p.add_argument(
        "--hf_repo",
        default=None,
        help=(
            "Optional HuggingFace Hub repo name (e.g. 'username/dataset_name') to push the resulting DatasetDict to. "
            "If not supplied, the DatasetDict is only saved locally to --output_dir."
        ),
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Push as a public Hub repo (default: private; LDC-licensed data must stay private).",
    )
    args = p.parse_args()

    metadata_files = sorted(args.metadata_dir.glob("*_metadata.json"))
    if not metadata_files:
        log.error("No *_metadata.json files found in %s", args.metadata_dir)
        return

    splits: Dict[str, datasets.Dataset] = {}
    for meta_path in metadata_files:
        split_name = meta_path.stem.replace("_metadata", "")
        log.info("Processing split '%s' from %s", split_name, meta_path)
        ds = convert(
            metadata_path=meta_path,
            audio_root=args.audio_root,
            clips_dir=args.clips_dir,
            language=args.language,
            split=split_name,
        )
        if ds is not None:
            splits[split_name] = ds

    if not splits:
        log.error("No splits produced; nothing written.")
        return

    ddict = DatasetDict(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ddict.save_to_disk(str(args.output_dir))
    log.info(
        "Saved DatasetDict locally to %s (splits: %s).",
        args.output_dir,
        list(splits.keys()),
    )
    if args.hf_repo is not None:
        ddict.push_to_hub(args.hf_repo, private=not args.public)
        log.info("Pushed dataset to Hub repo %s", args.hf_repo)


if __name__ == "__main__":
    main()
