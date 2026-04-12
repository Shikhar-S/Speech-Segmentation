"""Convert charsiu cv_ali TextGrid bundles + CommonVoice audio to canonical HF segmentation format.

cv_ali ships per-language tarballs ``<lang>_output.tar.gz`` containing
``<lang>_output/<bucket>/<clip>.TextGrid`` (Praat format with `phones` tier).
We stream them with ``tarfile`` (no extraction). Splits and transcripts come
from the CommonVoice TSVs at
``<cv_dir>/<lang>/{train,dev,test}.tsv``. cv_ali was built from an older
CommonVoice release, so the join is inner — clips missing from either side
are silently skipped and counted in the per-language summary log.

Audio path: ``<cv_dir>/<lang>/clips/<row.path>.mp3``

Usage::

    python -m src.recipe.segment_recognize.local.cv_data_convert \\
        --alignments_dir exp/downloads/cv_ali_alignments \\
        --cv_dir         /work/hdd/bbjs/shared/corpora/commonvoice/cv-corpus-13.0-2023-03-09 \\
        --output_dir     exp/downloads/cv-aligned-seg \\
        [--languages     ba,be,ca,de,en,es,fr,it,rw,sw] \\
        [--splits        train,dev,test] \\
        [--hf_repo       <user>/cv-aligned] \\
        [--limit         100]
"""

import argparse
import csv
import logging
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

# CommonVoice TSVs occasionally include very long fields; raise the csv
# default 128 KB cap so DictReader can parse them.
csv.field_size_limit(sys.maxsize)

import datasets
from datasets import DatasetDict
from tqdm import tqdm

log = logging.getLogger("cv_data_convert")
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

ALL_LANGS = ("ba", "be", "ca", "de", "en", "es", "fr", "it", "rw", "sw")
DEFAULT_SPLITS = ("train", "dev", "test")


# Regex parser tailored to the cv_ali long-format Praat TextGrid layout:
#     name = "phones"
#     ...
#     intervals [N]:
#         xmin = <float>
#         xmax = <float>
#         text = "<label>"
# Bypassing tgt + tempfile cuts ~50ms/file down to ~0.5ms/file.
_PHONES_TIER_RE = re.compile(rb'name\s*=\s*"phones"')
_INTERVAL_RE = re.compile(
    rb'xmin\s*=\s*([\d.]+)\s+xmax\s*=\s*([\d.]+)\s+text\s*=\s*"([^"]*)"'
)


def parse_textgrid_bytes(data: bytes) -> List[Dict]:
    """Extract the `phones` tier intervals from raw TextGrid bytes.

    Empty-label intervals (silence padding) are dropped to match tgt's
    behavior in voxangeles_data_convert.py.
    """
    m = _PHONES_TIER_RE.search(data)
    if m is None:
        return []
    intervals: List[Dict] = []
    # The phones tier always comes after the words tier in cv_ali; scan
    # from the tier marker to the end of the file.
    for iv in _INTERVAL_RE.finditer(data, m.end()):
        label = iv.group(3).decode("utf-8", errors="replace")
        if not label.strip():
            continue
        intervals.append(
            {
                "start": float(iv.group(1)),
                "end": float(iv.group(2)),
                "label": label,
            }
        )
    return intervals


def build_clip_lookup(
    cv_lang_dir: Path,
    splits: List[str],
    limit: int,
) -> Dict[str, Dict]:
    """Read CV TSVs once -> {clip_stem: {split, client_id, sentence}}.

    The first split listed wins for clips appearing in multiple TSVs (e.g.
    validated.tsv vs train.tsv) — order ``splits`` accordingly.
    """
    lookup: Dict[str, Dict] = {}
    for split in splits:
        tsv_path = cv_lang_dir / f"{split}.tsv"
        if not tsv_path.is_file():
            log.warning("Missing TSV: %s", tsv_path)
            continue
        n_added = 0
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if limit > 0 and n_added >= limit:
                    break
                clip_name = row.get("path", "")
                if not clip_name:
                    continue
                stem = Path(clip_name).stem
                if stem in lookup:
                    continue
                lookup[stem] = {
                    "split": split,
                    "client_id": row.get("client_id", ""),
                    "sentence": row.get("sentence", ""),
                    "clip_name": clip_name,
                }
                n_added += 1
        log.info("  %s.tsv: added %d new clips (total lookup size %d)",
                 split, n_added, len(lookup))
    return lookup


def convert_language(
    lang: str,
    alignments_dir: Path,
    cv_dir: Path,
    splits: List[str],
    limit: int,
) -> Dict[str, List[Dict]]:
    """Build records for one language, grouped by split.

    Single sequential pass over the tarball to avoid the gzip-reseek penalty:
    we precompute a clip_id -> CV row map, then walk the tar once and emit a
    record whenever a TextGrid matches.
    """
    tar_path = alignments_dir / f"{lang}_output.tar.gz"
    if not tar_path.is_file():
        log.warning("Missing tarball: %s", tar_path)
        return {}
    cv_lang_dir = cv_dir / lang
    if not cv_lang_dir.is_dir():
        log.warning("Missing CV dir: %s", cv_lang_dir)
        return {}
    clips_dir = cv_lang_dir / "clips"

    log.info("[%s] Building CV clip lookup...", lang)
    lookup = build_clip_lookup(cv_lang_dir, splits, limit)
    if not lookup:
        return {}

    by_split: Dict[str, List[Dict]] = {s: [] for s in splits}
    n_no_audio = n_bad_tg = n_seen = 0

    log.info("[%s] Walking tarball %s", lang, tar_path)
    with tarfile.open(tar_path, "r|gz") as tar:  # streaming mode
        for member in tqdm(tar, desc=lang, unit="tg"):
            if not member.isfile() or not member.name.endswith(".TextGrid"):
                continue
            n_seen += 1
            stem = Path(member.name).stem
            cv_row = lookup.get(stem)
            if cv_row is None:
                continue
            audio_path = clips_dir / cv_row["clip_name"]
            if not audio_path.is_file():
                n_no_audio += 1
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            intervals = parse_textgrid_bytes(fobj.read())
            if not intervals:
                n_bad_tg += 1
                continue
            ends = [iv["end"] for iv in intervals]
            by_split[cv_row["split"]].append(
                {
                    "utt_id": stem,
                    "audio": str(audio_path),
                    "text": cv_row["sentence"],
                    "phones": [iv["label"] for iv in intervals],
                    "phone_starts": [iv["start"] for iv in intervals],
                    "phone_ends": ends,
                    "language": lang,
                    "speaker_id": cv_row["client_id"],
                    "duration": float(ends[-1]) if ends else 0.0,
                    "split": cv_row["split"],
                }
            )
    log.info(
        "[%s] tar walked: %d TextGrids; kept per-split %s; skipped(no_audio=%d, bad_tg=%d)",
        lang, n_seen, {s: len(r) for s, r in by_split.items()}, n_no_audio, n_bad_tg,
    )
    return by_split


def merge(
    accum: Dict[str, List[Dict]],
    new: Dict[str, List[Dict]],
) -> None:
    for split, recs in new.items():
        accum.setdefault(split, []).extend(recs)


def save_dataset(
    splits_to_records: Dict[str, List[Dict]],
    output_dir: Path,
    hf_repo: Optional[str],
) -> None:
    splits = {
        s: datasets.Dataset.from_list(recs, features=SCHEMA)
        for s, recs in splits_to_records.items() if recs
    }
    if not splits:
        log.error("No records produced; nothing written.")
        return
    ddict = DatasetDict(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    ddict.save_to_disk(str(output_dir))
    log.info("Saved DatasetDict to %s (splits: %s)", output_dir, list(splits))
    if hf_repo is not None:
        ddict.push_to_hub(hf_repo)
        log.info("Pushed to Hub repo %s", hf_repo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alignments_dir", required=True, type=Path)
    p.add_argument("--cv_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--languages", default=",".join(ALL_LANGS),
                   help="Comma-separated language codes.")
    p.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                   help="Comma-separated CV split names.")
    p.add_argument("--hf_repo", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows per (lang, split) combination (0 = all).")
    args = p.parse_args()

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    accum: Dict[str, List[Dict]] = {}
    for lang in langs:
        merge(accum, convert_language(
            lang, args.alignments_dir, args.cv_dir, splits, args.limit,
        ))

    save_dataset(accum, args.output_dir, args.hf_repo)


if __name__ == "__main__":
    main()
