"""Convert Kaldi-style segmentation data to a HuggingFace dataset.

Reads wav.scp + TextGrid alignments (and optionally text / language files)
and writes a HuggingFace ``datasets.DatasetDict`` to disk in the canonical
segmentation schema used by this project.

Usage::

    python -m src.recipe.segmentation.local.seg_data_convert \\
        --wav_scp     test_timit/wav.scp \\
        --alignments  test_timit/alignments \\
        --output_dir  /path/to/output \\
        [--text       test_timit/text] \\
        [--language   test_timit/text.lang] \\
        [--tier_name  phones] \\
        [--split      test]

The script writes a ``DatasetDict`` to ``output_dir`` (HuggingFace
``save_to_disk`` format) which can be loaded with ``datasets.load_from_disk``.

Schema
------
- ``utt_id``       : str
- ``audio``        : datasets.Audio  (path reference, decoded on-the-fly)
- ``text``         : str             (word transcription, space-joined; "" if absent)
- ``phones``       : List[str]       (original labels including SIL/sp)
- ``phone_starts`` : List[float64]   (seconds)
- ``phone_ends``   : List[float64]   (seconds)
- ``language``     : str             (language tag stripped of angle brackets)
- ``speaker_id``   : str             (first "/" component of utt_id, or "")
- ``duration``     : float64         (last interval xmax, seconds)
- ``split``        : str             (dataset split name)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import tgt
import datasets
from datasets import DatasetDict

log = logging.getLogger("seg_data_convert")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --------------------------------------------------------------------------- #
# Parsing helpers                                                              #
# --------------------------------------------------------------------------- #


def parse_kaldi_file(path: Path) -> Dict[str, str]:
    """Parse a Kaldi file → {utt_id: value}."""
    result: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
        elif len(parts) == 1:
            result[parts[0]] = ""
    log.info("Parsed %d entries from %s", len(result), path)
    return result

def _strip_lang_tag(tag: str) -> str:
    """Strip surrounding angle brackets: ``<eng>`` → ``eng``."""
    return tag.strip("<>")


# --------------------------------------------------------------------------- #
# TextGrid parsing                                                             #
# --------------------------------------------------------------------------- #


def parse_textgrid(tg_path: Path, tier_name: str) -> List[Dict]:
    """Extract intervals from a TextGrid tier.

    Returns a list of dicts with keys ``start``, ``end``, ``label`` (all
    intervals, no filtering).  Raises ``ValueError`` if the tier is absent.
    """
    tg = tgt.io.read_textgrid(str(tg_path))
    tier_names = tg.get_tier_names()
    if tier_name not in tier_names:
        raise ValueError(
            f"Tier '{tier_name}' not found in {tg_path}. "
            f"Available tiers: {tier_names}"
        )
    tier = tg.get_tier_by_name(tier_name)
    return [
        {"start": iv.start_time, "end": iv.end_time, "label": iv.text}
        for iv in tier
    ]


# --------------------------------------------------------------------------- #
# Speaker ID inference                                                         #
# --------------------------------------------------------------------------- #


def infer_speaker_id(utt_id: str) -> str:
    """Return the component before the first '/' in utt_id, or ""."""
    if "/" in utt_id:
        return utt_id.split("/")[0]
    return ""


# --------------------------------------------------------------------------- #
# Main conversion                                                              #
# --------------------------------------------------------------------------- #


def convert(
    wav_scp: Path,
    alignments_dir: Path,
    output_dir: Path,
    text_file: Optional[Path],
    language_file: Optional[Path],
    tier_name: str,
    split: str,
) -> None:
    """Convert one dataset split and save as a HuggingFace dataset."""
    utt2wav = parse_kaldi_file(wav_scp)
    utt2text: Dict[str, str] = parse_kaldi_file(text_file) if text_file else {}
    utt2lang: Dict[str, str] = parse_kaldi_file(language_file) if language_file else {}

    records: List[Dict] = []
    n_missing = 0

    for utt_id, audio_path in utt2wav.items():
        utt_id_safe = utt_id.replace("/", "-")
        tg_path = alignments_dir / f"{utt_id_safe}.TextGrid"

        if not tg_path.exists():
            log.warning("TextGrid not found for %s (%s); skipping.", utt_id, tg_path)
            n_missing += 1
            continue

        try:
            intervals = parse_textgrid(tg_path, tier_name)
        except (ValueError, Exception) as exc:
            log.warning("Failed to parse %s: %s; skipping.", tg_path, exc)
            n_missing += 1
            continue

        if not intervals:
            log.warning("No intervals in %s; skipping.", tg_path)
            n_missing += 1
            continue

        phones = [iv["label"] for iv in intervals]
        phone_starts = [iv["start"] for iv in intervals]
        phone_ends = [iv["end"] for iv in intervals]
        duration = intervals[-1]["end"]

        raw_lang = utt2lang.get(utt_id, "")
        language = _strip_lang_tag(raw_lang)

        records.append(
            {
                "utt_id": utt_id,
                "audio": audio_path,
                "text": utt2text.get(utt_id, ""),
                "phones": phones,
                "phone_starts": phone_starts,
                "phone_ends": phone_ends,
                "language": language,
                "speaker_id": infer_speaker_id(utt_id),
                "duration": duration,
                "split": split,
            }
        )

    log.info(
        "Built %d records (%d skipped) for split '%s'.", len(records), n_missing, split
    )

    if not records:
        log.error("No records produced; nothing written.")
        return

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

    ds = datasets.Dataset.from_list(records, features=schema)

    output_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({split: ds}).save_to_disk(str(output_dir))
    log.info("Saved DatasetDict to %s (split '%s', %d rows).", output_dir, split, len(ds))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert Kaldi-style segmentation data to HuggingFace dataset format."
    )
    p.add_argument("--wav_scp", required=True, type=Path, help="Path to wav.scp")
    p.add_argument(
        "--alignments",
        required=True,
        type=Path,
        help="Directory containing <utt_id_safe>.TextGrid files",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Root output directory; DatasetDict written directly here",
    )
    p.add_argument(
        "--text",
        type=Path,
        default=None,
        help="Optional Kaldi text file (word transcriptions)",
    )
    p.add_argument(
        "--language",
        type=Path,
        default=None,
        help="Optional Kaldi text.lang file (language tags)",
    )
    p.add_argument(
        "--tier_name",
        default="phones",
        help="TextGrid tier name to extract (default: phones)",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Dataset split label (default: test)",
    )
    args = p.parse_args()

    convert(
        wav_scp=args.wav_scp,
        alignments_dir=args.alignments,
        output_dir=args.output_dir,
        text_file=args.text,
        language_file=args.language,
        tier_name=args.tier_name,
        split=args.split,
    )


if __name__ == "__main__":
    main()
