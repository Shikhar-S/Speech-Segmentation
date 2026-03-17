"""Convert VoxAngeles phone-level data to a HuggingFace segmentation dataset.

VoxAngeles TextGrids are stored inside per-language zip archives::

    {va_dir}/data/audited_aligned/{langcode}.zip
      └── {langcode}/{base_utt_id}.TextGrid

Kaldi utt_ids have a trailing sequential index (e.g. ``abk-002-000-0``) that is
stripped to obtain the TextGrid basename (``abk-002-000``).  Language is the
first hyphen-delimited component of the utt_id; speaker_id is the second.

Usage::

python -m src.recipe.forced_alignment.local.voxangeles_data_convert \
      --wav_scp   /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_voxangeles/wav.scp \
      --text      /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_voxangeles/text.good \
      --va_dir    /work/nvme/bbjs/sbharadwaj/powsm/voxangeles \
      --output_dir exp/voxangeles-seg \
      --hf_repo    changelinglab/voxangeles-segment

    python -m src.recipe.forced_alignment.local.voxangeles_data_convert \\
        --wav_scp   /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_voxangeles/wav.scp \\
        --text      /work/hdd/bbjs/shared/powsm/s2t1/dump/raw/test_voxangeles/text.good \\
        --va_dir    /work/nvme/bbjs/sbharadwaj/powsm/voxangeles \\
        --output_dir exp/voxangeles-seg \\
        [--tier_name phones] \\
        [--split     test]

Schema
------
- ``utt_id``       : str
- ``audio``        : datasets.Audio  (path reference, decoded on-the-fly)
- ``text``         : str             (word transcription; "" if absent)
- ``phones``       : List[str]
- ``phone_starts`` : List[float64]   (seconds)
- ``phone_ends``   : List[float64]   (seconds)
- ``language``     : str             (ISO 639-3 code from utt_id prefix)
- ``speaker_id``   : str             (second hyphen component of utt_id)
- ``duration``     : float64         (last interval xmax, seconds)
- ``split``        : str
"""

import argparse
import logging

from tqdm import tqdm
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import tgt
import datasets
from datasets import DatasetDict

log = logging.getLogger("voxangeles_data_convert")
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


_TIER_FALLBACKS = ["phone", "phones"]


def parse_textgrid(tg_path: Path, tier_name: str, encoding: str = "utf-8") -> List[Dict]:
    """Extract intervals from a TextGrid tier.

    Returns a list of dicts with keys ``start``, ``end``, ``label``.
    Tries ``tier_name`` first, then each name in ``_TIER_FALLBACKS``.
    Raises ``ValueError`` if none are found.
    """
    tg = tgt.io.read_textgrid(str(tg_path), encoding=encoding)
    tier_names = tg.get_tier_names()
    candidates = [tier_name] + [t for t in _TIER_FALLBACKS if t != tier_name]
    resolved = next((t for t in candidates if t in tier_names), None)
    if resolved is None:
        raise ValueError(
            f"None of {candidates} found in {tg_path}. "
            f"Available tiers: {tier_names}"
        )
    if resolved != tier_name:
        log.debug("Tier '%s' not found; using '%s' in %s.", tier_name, resolved, tg_path)
    tier = tg.get_tier_by_name(resolved)
    return [
        {"start": iv.start_time, "end": iv.end_time, "label": iv.text}
        for iv in tier
    ]


# --------------------------------------------------------------------------- #
# VoxAngeles-specific helpers                                                  #
# --------------------------------------------------------------------------- #


def _utt_base(utt_id: str) -> str:
    """Strip trailing sequential index: ``abk-002-000-0`` → ``abk-002-000``."""
    return utt_id.rsplit("-", 1)[0]


def _langcode(utt_id: str) -> str:
    """Return ISO 639-3 code from utt_id prefix: ``abk-002-000-0`` → ``abk``."""
    return utt_id.split("-")[0]


def _speaker_id(utt_id: str) -> str:
    """Return speaker component from utt_id: ``abk-002-000-0`` → ``002``."""
    parts = utt_id.split("-")
    return parts[1] if len(parts) > 1 else ""


_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


def _detect_encoding(data: bytes) -> str:
    """Return 'utf-16' if data starts with a UTF-16 BOM, else 'utf-8'."""
    return "utf-16" if data[:2] in _UTF16_BOMS else "utf-8"


def _read_textgrid_from_zip(
    zf: zipfile.ZipFile,
    entry_name: str,
    tier_name: str,
) -> List[Dict]:
    """Extract a TextGrid entry from an open ZipFile and parse it.

    Detects UTF-16 BOMs and passes the correct encoding to tgt.
    """
    data = zf.read(entry_name)
    encoding = _detect_encoding(data)
    with tempfile.NamedTemporaryFile(suffix=".TextGrid", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return parse_textgrid(Path(tmp_path), tier_name, encoding=encoding)
    finally:
        os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
# Main conversion                                                              #
# --------------------------------------------------------------------------- #


def convert(
    wav_scp: Path,
    va_dir: Path,
    output_dir: Path,
    text_file: Optional[Path],
    tier_name: str,
    split: str,
    hf_repo: Optional[str],
) -> None:
    """Convert VoxAngeles data and save as a HuggingFace DatasetDict."""
    utt2wav = parse_kaldi_file(wav_scp)
    wav_scp_dir = wav_scp.parent
    utt2wav = {
        uid: str((wav_scp_dir.parent.parent.parent / path).resolve()) if not Path(path).is_absolute() else path
        for uid, path in utt2wav.items()
    }
    utt2text: Dict[str, str] = parse_kaldi_file(text_file) if text_file else {}

    zip_dir = va_dir / "data" / "audited_aligned"
    zip_cache: Dict[str, zipfile.ZipFile] = {}

    records: List[Dict] = []
    n_missing = 0

    try:
        for utt_id, audio_path in tqdm(utt2wav.items(), desc="Converting", unit="utt"):
            lang = _langcode(utt_id)
            base = _utt_base(utt_id)
            entry_name = f"{lang}/{base}.TextGrid"
            zip_path = zip_dir / f"{lang}.zip"

            if lang not in zip_cache:
                if not zip_path.exists():
                    log.warning("Zip not found: %s; skipping %s.", zip_path, utt_id)
                    n_missing += 1
                    continue
                zip_cache[lang] = zipfile.ZipFile(zip_path, "r")

            zf = zip_cache[lang]
            if entry_name not in zf.namelist():
                log.warning(
                    "Entry '%s' not in %s; skipping %s.", entry_name, zip_path, utt_id
                )
                n_missing += 1
                continue

            try:
                intervals = _read_textgrid_from_zip(zf, entry_name, tier_name)
            except (ValueError, Exception) as exc:
                log.warning("Failed to parse %s from %s: %s; skipping.", entry_name, zip_path, exc)
                n_missing += 1
                continue

            if not intervals:
                log.warning("No intervals in %s; skipping.", entry_name)
                n_missing += 1
                continue

            phones = [iv["label"] for iv in intervals]
            phone_starts = [iv["start"] for iv in intervals]
            phone_ends = [iv["end"] for iv in intervals]
            duration = intervals[-1]["end"]

            records.append(
                {
                    "utt_id": utt_id,
                    "audio": audio_path,
                    "text": utt2text.get(utt_id, ""),
                    "phones": phones,
                    "phone_starts": phone_starts,
                    "phone_ends": phone_ends,
                    "language": lang,
                    "speaker_id": _speaker_id(utt_id),
                    "duration": duration,
                    "split": split,
                }
            )
    finally:
        for zf in zip_cache.values():
            zf.close()

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
    ddict=DatasetDict({split: ds})
    ddict.save_to_disk(str(output_dir))
    log.info(
        "Saved DatasetDict locally to %s.",
        output_dir,
    )
    if hf_repo is not None:
        ddict.push_to_hub(hf_repo)
        log.info("Pushed dataset to Hub repo %s", hf_repo)
    log.info("Saved DatasetDict to %s (split '%s', %d rows).", output_dir, split, len(ds))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert VoxAngeles phone data to HuggingFace dataset format."
    )
    p.add_argument("--wav_scp", required=True, type=Path, help="Path to Kaldi wav.scp")
    p.add_argument(
        "--va_dir",
        required=True,
        type=Path,
        help="Root of the VoxAngeles repository",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory for DatasetDict.save_to_disk",
    )
    p.add_argument(
        "--text",
        type=Path,
        default=None,
        help="Optional Kaldi text file for word transcriptions (e.g. text.good)",
    )
    p.add_argument(
        "--tier_name",
        default="phones",
        help="TextGrid tier to extract (default: phones)",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Dataset split label (default: test)",
    )
    p.add_argument(
        "--hf_repo",
        default=None,
        help=(
            "Optional HuggingFace Hub repo name (e.g. 'username/dataset_name') to push the resulting DatasetDict to. "
            "If not supplied, the DatasetDict is only saved locally to --output_dir."
        ),
    )
    args = p.parse_args()

    convert(
        wav_scp=args.wav_scp,
        va_dir=args.va_dir,
        output_dir=args.output_dir,
        text_file=args.text,
        tier_name=args.tier_name,
        split=args.split,
        hf_repo=args.hf_repo,
    )


if __name__ == "__main__":
    main()
