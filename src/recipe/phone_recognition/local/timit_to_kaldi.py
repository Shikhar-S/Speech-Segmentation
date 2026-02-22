"""Convert TIMIT metadata JSONs (from timit_data_prep) to Kaldi format: wav.scp, text.ctc, text.

Reads train_metadata.json, val_metadata.json, test_metadata.json from a metadata directory,
maps ARPABET phones to IPA, and writes per-split Kaldi files under out_dir/<split>_timit/.

Usage:
    python -m src.recipe.phone_recognition.local.timit_to_kaldi \
        --timit_root /path/to/TIMIT/timit_nltk \
        --metadata_dir /path/to/exp/cache/timit \
        --out_dir /path/to/exp/cache/timit

Then use in config with dataset_config_path and train_split=train_timit, dev_splits=[val_timit].
"""

import argparse
import json
import os
import sys
from pathlib import Path

from src.core.ipa_utils import ARPABET_TO_IPA, arpabet_to_ipa


SPLITS = ("train", "val", "test")


def convert_metadata_to_kaldi(
    metadata_path: Path,
    timit_root: Path,
    out_dir: Path,
    lang_tag: str = "<eng>",
):
    """Write wav.scp, text.ctc, text for one split from a metadata JSON."""
    with open(metadata_path) as f:
        segments = json.load(f)

    mapped_counts = {}
    fallback_counts = {}

    wav_lines = []
    text_ctc_lines = []
    text_lang_lines = []

    for item in segments:
        segment_id = item["segment_id"]
        phones = item.get("phones", [])
        if not phones:
            continue

        wav_path = os.path.normpath(str(timit_root / f"{segment_id}.wav"))
        wav_lines.append(f"{segment_id} {wav_path}")

        ipa_list = arpabet_to_ipa(phones)
        for p in phones:
            key = p.lower()
            if key in ARPABET_TO_IPA:
                mapped_counts[key] = mapped_counts.get(key, 0) + 1
            else:
                fallback_counts[key] = fallback_counts.get(key, 0) + 1

        ctc_str = "/" + "//".join(ipa_list) + "/"
        text_ctc_lines.append(f"{segment_id} {ctc_str}")
        text_lang_lines.append(f"{segment_id} {lang_tag}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wav.scp", "w") as f:
        f.write("\n".join(wav_lines) + "\n")
    with open(out_dir / "text.ctc", "w") as f:
        f.write("\n".join(text_ctc_lines) + "\n")
    with open(out_dir / "text", "w") as f:
        f.write("\n".join(text_lang_lines) + "\n")

    total_mapped = sum(mapped_counts.values())
    total_fallback = sum(fallback_counts.values())
    total = total_mapped + total_fallback
    coverage_pct = (100.0 * total_mapped / total) if total else 100.0

    return {
        "split": metadata_path.stem.replace("_metadata", ""),
        "num_utterances": len(wav_lines),
        "total_phones": total,
        "mapped_count": total_mapped,
        "fallback_count": total_fallback,
        "coverage_pct": coverage_pct,
        "mapped_symbols": mapped_counts,
        "fallback_symbols": fallback_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIMIT metadata JSONs to Kaldi format (wav.scp, text.ctc, text)."
    )
    parser.add_argument(
        "--timit_root",
        required=True,
        type=Path,
        help="Path to TIMIT root (NLTK-style: segment_id.wav lives here).",
    )
    parser.add_argument(
        "--metadata_dir",
        required=True,
        type=Path,
        help="Directory containing train_metadata.json, val_metadata.json, test_metadata.json.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Base output directory; creates <out_dir>/train_timit, <out_dir>/val_timit, <out_dir>/test_timit.",
    )
    parser.add_argument(
        "--lang_tag",
        default="<eng>",
        help="Language tag for each utterance (default: <eng>).",
    )
    args = parser.parse_args()

    timit_root = args.timit_root.resolve()
    metadata_dir = args.metadata_dir.resolve()
    out_base = args.out_dir.resolve()

    if not timit_root.is_dir():
        print(f"ERROR: timit_root is not a directory: {timit_root}", file=sys.stderr)
        sys.exit(1)
    if not metadata_dir.is_dir():
        print(f"ERROR: metadata_dir is not a directory: {metadata_dir}", file=sys.stderr)
        sys.exit(1)

    for split in SPLITS:
        meta_file = metadata_dir / f"{split}_metadata.json"
        if not meta_file.exists():
            print(f"Skip {split}: {meta_file} not found", file=sys.stderr)
            continue
        split_out = out_base / f"{split}_timit"
        stats = convert_metadata_to_kaldi(
            meta_file, timit_root, split_out, lang_tag=args.lang_tag
        )
        print(
            f"Wrote {stats['num_utterances']} utterances to {split_out} "
            f"(ARPABET→IPA coverage: {stats['mapped_count']}/{stats['total_phones']} "
            f"({stats['coverage_pct']:.1f}%) mapped)",
            file=sys.stderr,
        )
        if stats["fallback_symbols"]:
            print(
                f"  Unmapped symbols: {sorted(stats['fallback_symbols'].keys())}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
