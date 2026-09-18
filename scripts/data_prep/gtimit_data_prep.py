"""Global TIMIT (accented English + Thai) → HuggingFace segmentation dataset.

Walks one Global TIMIT sub-corpus and builds a HuggingFace ``DatasetDict``
matching the schema consumed by ``src/data/segmentation/segmentation_dataset.py``.

All utterances are placed in a single ``test`` split (these corpora are
evaluation-only). Speaker identity is preserved as ``<accent>_<rawSpeaker>``
(e.g. ``L2_SP01``) because L1 and L2 sub-corpora reuse the same raw speaker
codes (``SP01``–``SP30``) for different speakers.

Sub-corpora handled:
    L1ENGsimple, L2ENGsimple   (under global_timit_learner_simple_eng/)
    L1ENGtreebank, L2ENGtreebank (under global_timit_learner_tbnk_eng/)
    THA                         (under global_timit_tha/)

Phone files come in two formats:
    "start end label"  (most subsets)
    "label start end"  (L2ENGsimple)
The parser auto-detects by inspecting the first token of line 1.

Usage:
    python scripts/data_prep/gtimit_data_prep.py \\
        --subset         L2ENGsimple \\
        --downloads_root $GTIMIT_ROOT   # holds global_timit_learner_simple_eng/, global_timit_learner_tbnk_eng/, global_timit_tha/
        --output_dir     exp/data/global-timit-l2eng-simple-segment

Then push the prepared sub-corpora with ``scripts/data_prep/push_gtimit_to_hub.py``.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
from datasets import DatasetDict
from tqdm import tqdm

log = logging.getLogger("accented_timit_prep")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# corpus_subdir is relative to --downloads_root; audio_subdir/seg_subdir
# are relative to corpus_subdir.
SUBSET_CONFIG: Dict[str, Dict[str, str]] = {
    "L1ENGsimple": {
        "corpus_subdir": "global_timit_learner_simple_eng/data/L1ENGsimple",
        "audio_subdir": "flac",
        "seg_subdir": "segmentation",
        "language": "eng",
        "accent": "L1",
    },
    "L2ENGsimple": {
        "corpus_subdir": "global_timit_learner_simple_eng/data/L2ENGsimple",
        "audio_subdir": "flac",
        "seg_subdir": "segmentation",
        "language": "eng",
        "accent": "L2",
    },
    "L1ENGtreebank": {
        "corpus_subdir": "global_timit_learner_tbnk_eng/data/L1ENGtreebank",
        "audio_subdir": "flac",
        "seg_subdir": "segmentation",
        "language": "eng",
        "accent": "L1",
    },
    "L2ENGtreebank": {
        "corpus_subdir": "global_timit_learner_tbnk_eng/data/L2ENGtreebank",
        "audio_subdir": "flac",
        "seg_subdir": "segmentation",
        "language": "eng",
        "accent": "L2",
    },
    "THA": {
        "corpus_subdir": "global_timit_tha/data",
        "audio_subdir": "audio",
        "seg_subdir": "segmentation",
        "language": "tha",
        "accent": "L1",
    },
}

_SILENCE_LABELS = {"sil", "sp", "<sil>", ""}


@dataclass
class GlobalTimitSegment:
    utt_id: str
    speaker_id: str
    audio_path: str
    text: str
    phones: List[str]
    phone_starts: List[float]
    phone_ends: List[float]
    duration: float
    language: str
    tones: Optional[List[str]] = None  # relevant for Thai


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_interval_file(path: Path) -> List[Tuple[float, float, str]]:
    """Parse a `.phones` or `.words` file → list of (start, end, label).

    Auto-detects "start end label" vs "label start end" by inspecting the
    first non-blank line. Whitespace-tolerant (handles tabs and multiple
    spaces). Skips blank lines.
    """
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    first = lines[0].split()
    label_first = not _is_float(first[0])
    out: List[Tuple[float, float, str]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        if label_first:
            label, start, end = parts[0], parts[1], parts[2]
        else:
            start, end, label = parts[0], parts[1], parts[2]
        out.append((float(start), float(end), label))
    return out


def assign_phone_tones(
    phone_intervals: List[Tuple[float, float, str]],
    tone_intervals: List[Tuple[float, float, str]],
) -> List[str]:
    """Label each phone with the tone of the span covering its midpoint.

    THA tone spans are exact unions of consecutive phone intervals (verified
    across the corpus: every phone nests entirely within one span, and ``sil``
    phones align with ``sil`` tone spans), so midpoint lookup is unambiguous.
    Note that this is validated for GTIMIT-THA dataset only.

    Returns one tone label per phone, parallel to ``phone_intervals``.
    """
    tones: List[str] = []
    for start, end, _ in phone_intervals:
        mid = (start + end) / 2
        label = next(
            (lab for ts, te, lab in tone_intervals if ts <= mid <= te), None
        )
        if label is None:
            raise ValueError(f"no tone span covers phone at {start}-{end}")
        tones.append(label)
    return tones


def words_text(words_path: Path) -> str:
    """Build a whitespace-joined transcript, dropping silence tokens."""
    if not words_path.exists():
        return ""
    intervals = parse_interval_file(words_path)
    toks = [
        lab for _, _, lab in intervals if lab.lower() not in _SILENCE_LABELS
    ]
    return " ".join(toks)


def load_subset(subset: str, downloads_root: Path) -> List[GlobalTimitSegment]:
    cfg = SUBSET_CONFIG[subset]
    corpus_dir = downloads_root / cfg["corpus_subdir"]
    audio_dir = corpus_dir / cfg["audio_subdir"]
    seg_dir = corpus_dir / cfg["seg_subdir"]
    accent = cfg["accent"]
    language = cfg["language"]

    phone_files = sorted(seg_dir.glob("*.phones"))
    log.info(
        "Subset %s: %d phone files under %s", subset, len(phone_files), seg_dir
    )

    segments: List[GlobalTimitSegment] = []
    n_missing_audio = 0
    n_empty_phones = 0
    for pf in tqdm(phone_files, desc=f"Loading {subset}", unit="utt"):
        utt_id = pf.stem
        audio_path = audio_dir / f"{utt_id}.flac"
        if not audio_path.exists():
            n_missing_audio += 1
            log.warning("Audio missing for %s; skipping.", utt_id)
            continue
        intervals = parse_interval_file(pf)
        if not intervals:
            n_empty_phones += 1
            continue
        phones = [lab for _, _, lab in intervals]
        starts = [s for s, _, _ in intervals]
        ends = [e for _, e, _ in intervals]
        text = words_text(seg_dir / f"{utt_id}.words")
        raw_speaker = utt_id.split("_")[0]
        tones_path = seg_dir / f"{utt_id}.tones"
        tones = (
            assign_phone_tones(intervals, parse_interval_file(tones_path))
            if tones_path.exists()
            else None
        )
        segments.append(
            GlobalTimitSegment(
                utt_id=f"{accent}_{utt_id}",
                speaker_id=f"{accent}_{raw_speaker}",
                audio_path=str(audio_path.resolve()),
                text=text,
                phones=phones,
                phone_starts=starts,
                phone_ends=ends,
                duration=ends[-1],
                language=language,
                tones=tones,
            )
        )
    log.info(
        "Built %d segments (%d missing-audio, %d empty-phones).",
        len(segments),
        n_missing_audio,
        n_empty_phones,
    )
    return segments


def build_dataset(
    segments: List[GlobalTimitSegment], split: str
) -> datasets.Dataset:
    has_tones = segments[0].tones is not None
    features = {
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
    if has_tones:
        features["tones"] = datasets.Sequence(datasets.Value("string"))
    schema = datasets.Features(features)
    records = []
    for s in segments:
        rec = {
            "utt_id": s.utt_id,
            "audio": s.audio_path,
            "text": s.text,
            "phones": s.phones,
            "phone_starts": s.phone_starts,
            "phone_ends": s.phone_ends,
            "language": s.language,
            "speaker_id": s.speaker_id,
            "duration": s.duration,
            "split": split,
        }
        if has_tones:
            rec["tones"] = s.tones
        records.append(rec)
    return datasets.Dataset.from_list(records, features=schema)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare a Global TIMIT sub-corpus as a HuggingFace segmentation dataset."
    )
    p.add_argument(
        "--subset",
        required=True,
        choices=sorted(SUBSET_CONFIG.keys()),
        help="Sub-corpus to convert.",
    )
    p.add_argument(
        "--downloads_root",
        required=True,
        type=Path,
        help="Root directory containing the unzipped Global TIMIT corpora.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory for DatasetDict.save_to_disk.",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Split label written into each row and used as DatasetDict key.",
    )
    args = p.parse_args()

    segments = load_subset(args.subset, args.downloads_root)
    if not segments:
        log.error("No segments produced; nothing written.")
        return

    ds = build_dataset(segments, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ddict = DatasetDict({args.split: ds})
    ddict.save_to_disk(str(args.output_dir))
    log.info(
        "Saved %d rows to %s under split '%s'.",
        len(ds),
        args.output_dir,
        args.split,
    )


if __name__ == "__main__":
    main()
