"""Convert JSONL (file_name, id, ARPABET phones) to Kaldi format: wav.scp, text.ctc, language file.
Phoneme transcript format: id /ipa1//ipa2//ipa3//.../
Also writes ARPABET→IPA coverage report (mapped vs fallback counts).

Usage:
    python -m src.recipe.phone_recognition.local.huperjsonl2kaldi \
        --jsonl /work/nvme/bbjs/sbharadwaj/powsm/HuPER/HuPER-clean100-proxyphones/train/metadata.jsonl \
        --base_data_path /work/nvme/bbjs/sbharadwaj/powsm/HuPER/HuPER-clean100-proxyphones/train \
        --out_dir exp/data/huper_librispeech \
        --lang_tag '<eng>'
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

from src.core.ipa_utils import ARPABET_TO_IPA, arpabet_to_ipa


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to Kaldi format (wav.scp, text.ctc, text)."
    )
    parser.add_argument("--jsonl", required=True, help="Path to JSONL index file.")
    parser.add_argument(
        "--base_data_path", required=True, help="Prefix for file_name to get wav path."
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write wav.scp, text.ctc, text, coverage report.",
    )
    parser.add_argument(
        "--lang_tag",
        default="<eng>",
        help="Language tag for each utterance (default: <eng>).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Coverage: count per ARPABET symbol whether it was mapped or fallback
    mapped_counts = {}
    fallback_counts = {}

    wav_lines = []
    text_ctc_lines = []
    text_lang_lines = []

    with open(args.jsonl) as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            file_name = obj["file_name"]
            utt_id = f'{obj["id"]}_pr'
            phones = obj.get("phones", [])
            if not phones:
                continue

            # Wav path
            wav_path = os.path.normpath(os.path.join(args.base_data_path, file_name))
            wav_lines.append(f"{utt_id} {wav_path}")

            # ARPABET → IPA (using ipa_utils) and coverage tracking
            ipa_list = arpabet_to_ipa(phones)
            for p in phones:
                key = p.lower()
                if key in ARPABET_TO_IPA:
                    mapped_counts[key] = mapped_counts.get(key, 0) + 1
                else:
                    fallback_counts[key] = fallback_counts.get(key, 0) + 1

            # Format: /ipa1//ipa2//ipa3//.../
            ctc_str = "/" + "//".join(ipa_list) + "/"
            text_ctc_lines.append(f"{utt_id} {ctc_str}")
            text_lang_lines.append(
                f"{utt_id} {args.lang_tag}<pr><notimestamps> {ctc_str}"
            )

    # Write Kaldi files
    with open(os.path.join(args.out_dir, "wav.scp"), "w") as f:
        f.write("\n".join(wav_lines) + "\n")
    with open(os.path.join(args.out_dir, "text.ctc"), "w") as f:
        f.write("\n".join(text_ctc_lines) + "\n")
    with open(os.path.join(args.out_dir, "text"), "w") as f:
        f.write("\n".join(text_lang_lines) + "\n")

    # Coverage report
    total_mapped = sum(mapped_counts.values())
    total_fallback = sum(fallback_counts.values())
    total = total_mapped + total_fallback
    coverage_pct = (100.0 * total_mapped / total) if total else 100.0

    report = {
        "total_phones": total,
        "mapped_count": total_mapped,
        "fallback_count": total_fallback,
        "coverage_pct": round(coverage_pct, 2),
        "mapped_symbols": dict(sorted(mapped_counts.items(), key=lambda x: -x[1])),
        "fallback_symbols": dict(sorted(fallback_counts.items(), key=lambda x: -x[1])),
        "unmapped_list": sorted(fallback_counts.keys()),
    }

    report_path = os.path.join(args.out_dir, "arpabet_coverage.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Also print summary to stderr
    print(f"Wrote {len(wav_lines)} utterances to {args.out_dir}", file=sys.stderr)
    print(
        f"ARPABET→IPA coverage: {total_mapped}/{total} ({coverage_pct:.1f}%) mapped",
        file=sys.stderr,
    )
    if fallback_counts:
        print(
            f"Unmapped (fallback) symbols: {sorted(fallback_counts.keys())}",
            file=sys.stderr,
        )
        print(
            f"Fallback counts: {dict(sorted(fallback_counts.items(), key=lambda x: -x[1]))}",
            file=sys.stderr,
        )
    print(f"Coverage report: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
