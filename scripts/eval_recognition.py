"""Compute PER and PFER from distributed_inference JSONL output.

Reads one or more JSONL shard files where each line carries a phonvec
prediction list (``[{start, end, label}, ...]``) plus a passthrough block
with the reference IPA transcript in ``text``. Scores via
:func:`phone_metrics.phone_error_rates`.

Usage:
    python scripts/eval_recognition.py "exp/runs/.../timit_phonvec.*.jsonl"
    python scripts/eval_recognition.py shard0.jsonl shard1.jsonl \
        --out-csv exp/runs/.../timit_metrics.csv
"""

import argparse
import csv
import glob
import json

from phone_metrics import (
    Utterance,
    canonical_ipa,
    phone_error_rates,
    tokenize_ipa,
)
from phone_metrics.timit import Seg

from src.core.ipa_utils import IPA_SILENCE_LABELS

RESERVED_PRED_LABELS = frozenset({"<blank>", "<sos>", "<eos>", "<unk>"})


def _pred_phone_labels(units) -> list[str]:
    """Filter unit list to phone strings, dropping vocab reserved tokens."""
    out: list[str] = []
    for u in units:
        if "label" not in u:
            continue
        lab = str(u["label"])
        if lab in RESERVED_PRED_LABELS:
            continue
        out.append(lab)
    return out


def parse_pred_labels(pred, head=None) -> list[str]:
    """Return the label sequence from a prediction payload.

    Accepts either:
      - flat ``[{start, end, label}, ...]`` (phonvec wire format)
      - head-keyed dict ``{head: {utt_id: [{start, end, label}, ...]}}``
        (segment_recognize multi-head).
    Drops reserved vocab tokens (``<blank>/<sos>/<eos>/<unk>``).
    Skips error rows.
    """
    if isinstance(pred, dict) and "error" in pred:
        return []
    if isinstance(pred, list):
        return _pred_phone_labels(pred)
    if isinstance(pred, dict):
        assert (
            head is not None or len(pred) == 1
        ), f"Multiple heads {list(pred)}; pass --head."
        for head_name, head_output in pred.items():
            if head is not None and head_name != head:
                continue
            (units,) = head_output.values()
            return _pred_phone_labels(units)
    return []


def parse_ref_labels(text) -> list[str]:
    """Tokenize a reference IPA transcript into phone labels."""
    if not text:
        return []
    s = str(text).strip()
    if " " in s:
        return s.split()
    return tokenize_ipa(canonical_ipa(s))


def load_shards(
    files, head=None
) -> tuple[list[Utterance], list[list[str]], int]:
    """Read JSONL shards and build (utterances, predictions, n_errors).

    Each line in a shard is ``{<idx>: {"pred": [...], "passthrough": {...}}}``
    as written by :mod:`src.core.distributed_inference`.
    """
    utts: list[Utterance] = []
    preds: list[list[str]] = []
    n_errors = 0
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec:
                    continue
                _, payload = next(iter(rec.items()))
                pred_labels = parse_pred_labels(payload.get("pred"), head=head)
                passthrough = payload.get("passthrough", {})
                phones = passthrough.get("phones")
                if phones:
                    # Segmentation datasets (e.g. the phonvec oracle run):
                    # GT phones come through the ``phones`` passthrough.
                    ref_labels = [
                        str(p) for p in phones if p not in IPA_SILENCE_LABELS
                    ]
                    pred_labels = [p for p in pred_labels if p != "_"]
                else:
                    # this is for prism recognition datasets
                    ref_labels = parse_ref_labels(passthrough.get("text"))
                if not ref_labels:
                    n_errors += 1
                    continue
                segs = [Seg(0.0, 0.0, lab, lab) for lab in ref_labels]
                utt = Utterance(
                    audio_path=str(
                        passthrough.get("wavpath")
                        or passthrough.get("key")
                        or passthrough.get("utt_id", "")
                    ),
                    language=str(
                        passthrough.get("lang_sym")
                        or passthrough.get("language")
                        or "und"
                    ),
                    split=str(passthrough.get("split", "test")),
                    segments=segs,
                )
                utts.append(utt)
                preds.append(pred_labels)
    return utts, preds, n_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PER / PFER scoring from distributed_inference JSONL."
    )
    parser.add_argument(
        "files", nargs="+", help="JSONL shard files or glob patterns."
    )
    parser.add_argument(
        "--out-csv", metavar="PATH", help="Write CSV metrics to this file."
    )
    parser.add_argument(
        "--head",
        default=None,
        help="For multi-head pred dicts (segment_recognize), select a single head name (e.g. 'ctc').",
    )
    args = parser.parse_args()

    files = [f for p in args.files for f in (sorted(glob.glob(p)) or [p])]
    utts, preds, n_errors = load_shards(files, head=args.head)
    print(
        f"Loaded {len(utts)} utterances from {len(files)} file(s) "
        f"({n_errors} skipped/error)."
    )
    if not utts:
        return

    per_result = phone_error_rates(utts, preds, label="ipa")
    pfer_result = phone_error_rates(utts, preds, label="ipa", pfer=True)
    results = {
        "n_utts": len(utts),
        "per": per_result.per,
        "pfer": pfer_result.pfer,
        "macro_lang_per": per_result.macro_language_per,
    }
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results.keys())
            writer.writerow(results.values())


if __name__ == "__main__":
    main()
