#!/usr/bin/env python
"""Compute segmentation metrics from distributed_inference JSONL output.

Usage:
    python -m scripts.eval_segmentation "exp/runs/.../seg_pxeus_timit.*.jsonl"
    python -m scripts.eval_segmentation shard0.jsonl shard1.jsonl --tolerance-ms 20
    python -m scripts.eval_segmentation "exp/runs/.../seg_pxeus_timit.*.jsonl" --forced
"""

import argparse
import csv
import glob
import json

from tqdm import tqdm

from src.core.ipa_utils import IPA_SILENCE_LABELS
from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)


def parse_groundtruth(passthrough, forced, strip_outer=False):
    ts = passthrough.get("phone_timestamps") or passthrough.get(
        "ground_truth_timestamps"
    )
    phones = passthrough.get("phones")
    if strip_outer and phones:
        ts = list(ts)
        phones = list(phones)
        while phones and phones[0] in IPA_SILENCE_LABELS:
            ts, phones = ts[1:], phones[1:]
        while phones and phones[-1] in IPA_SILENCE_LABELS:
            ts, phones = ts[:-1], phones[:-1]
    label_src = phones if forced else None
    return [
        SegmentationUnit(
            start=s, end=e, label=(label_src[i] if label_src else 0)
        )
        for i, (s, e) in enumerate(ts)
    ]


def parse_predictions(pred, head=None):
    """Parse pred payload into a list of SegmentationUnit.

    Accepts either:
      - flat list ``[{start, end, label}, ...]`` (phonvec)
      - head-keyed dict ``{head: {utt_id: [{start, end, label}, ...]}}``
        (segment_recognize multi-head)
    """
    if isinstance(pred, dict) and "error" in pred:
        return []
    if isinstance(pred, list):
        return [
            SegmentationUnit(
                start=u["start"], end=u["end"], label=u.get("label", 0)
            )
            for u in pred
            if "start" in u
        ]
    results = []
    for head_name, head_output in pred.items():
        if head is not None and head_name != head:
            continue
        utt_keys = list(head_output.keys())
        assert (
            len(utt_keys) == 1
        ), f"Expected exactly one utterance in head output dict, got {len(utt_keys)}: {utt_keys}"
        utt_preds = head_output[utt_keys[0]]
        results = [
            SegmentationUnit(
                start=unit["start"], end=unit["end"], label=unit.get("label", 0)
            )
            for unit in utt_preds
        ]
    return results


def load_utterances_from_shards(files, forced, strip_outer=False):
    predictions, ground_truth, symbols_dict = {}, {}, {}
    n_errors = 0
    for filepath in tqdm(files, desc="Loading shards"):
        with open(filepath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for idx, data in json.loads(line).items():
                    if idx == "__error__":
                        continue
                    passthrough = data.get("passthrough", {})
                    if passthrough.get("split") != "test":
                        continue
                    utt_id = passthrough.get("utt_id", str(idx))
                    pred = data["pred"]
                    if isinstance(pred, dict) and "error" in pred:
                        n_errors += 1
                    predictions[utt_id] = parse_predictions(pred)
                    ground_truth[utt_id] = parse_groundtruth(
                        passthrough, forced, strip_outer=strip_outer
                    )
                    if forced:
                        symbols_dict[utt_id] = passthrough["phones"]
    return predictions, ground_truth, symbols_dict, n_errors


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation from JSONL output"
    )
    parser.add_argument(
        "files", nargs="+", help="JSONL shard files or glob patterns"
    )
    parser.add_argument("--tolerance-ms", type=int, default=20)
    parser.add_argument(
        "--forced",
        action="store_true",
        help="Also emit per-phone boundary error stats",
    )
    parser.add_argument(
        "--out-csv", metavar="PATH", help="Write CSV metrics to this file"
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "lenient"],
        default="lenient",
        help="Strict means one GT corresponds to one predicted boundary.",
    )
    parser.add_argument(
        "--strip-outer-silences",
        action="store_true",
        default=False,
        help="Drop leading/trailing silence segments (h#, pau, ʔ̞) from GT.",
    )
    args = parser.parse_args()

    files = [f for p in args.files for f in (sorted(glob.glob(p)) or [p])]
    predictions, ground_truth, symbols_dict, n_errors = load_utterances_from_shards(
        files, args.forced, strip_outer=args.strip_outer_silences
    )
    print(f"Loaded {len(predictions)} utterances from {len(files)} file(s)"
          f" ({n_errors} skipped/error, scored as 0)")

    evaluator = SegmentationEvaluator(
        tolerance_ms=args.tolerance_ms, forced=args.forced, match_mode=args.mode
    )
    results = evaluator.evaluate_batch(
        predictions, ground_truth, symbols_dict=symbols_dict or None
    )
    evaluator.pretty_print(results)
    if args.out_csv:
        flat = {k: v for k, v in results.items() if k != "symbol_errors"}
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(flat.keys())
            writer.writerow(flat.values())


if __name__ == "__main__":
    main()
