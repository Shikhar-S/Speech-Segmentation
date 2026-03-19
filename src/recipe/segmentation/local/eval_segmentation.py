#!/usr/bin/env python
"""Compute segmentation metrics from distributed_inference JSONL output.

Usage:
    python -m src.recipe.segmentation.local.eval_segmentation "exp/runs/.../seg_pxeus_timit.*.jsonl"
    python -m src.recipe.segmentation.local.eval_segmentation shard0.jsonl shard1.jsonl --tolerance-ms 20
    python -m src.recipe.segmentation.local.eval_segmentation "exp/runs/.../seg_pxeus_timit.*.jsonl" --forced
"""

import argparse
import csv
import glob
import json

from tqdm import tqdm

from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit

def parse_groundtruth(passthrough, forced):
    ts = passthrough.get("phone_timestamps") or passthrough.get("ground_truth_timestamps")
    phones = passthrough.get("phones") if forced else None
    return [SegmentationUnit(start=s, end=e, label=(phones[i] if phones else 0)) for i, (s, e) in enumerate(ts)]


def parse_predictions(pred_list):
    return [SegmentationUnit(start=p["start"], end=p["end"], label=p["label"]) for p in pred_list]


def load_utterances_from_shards(files, forced):
    predictions, ground_truth, symbols_dict = {}, {}, {}
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
                    predictions[utt_id] = parse_predictions(data["pred"])
                    ground_truth[utt_id] = parse_groundtruth(passthrough, forced)
                    if forced:
                        symbols_dict[utt_id] = passthrough["phones"]
    return predictions, ground_truth, symbols_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation from JSONL output")
    parser.add_argument("files", nargs="+", help="JSONL shard files or glob patterns")
    parser.add_argument("--tolerance-ms", type=int, default=20)
    parser.add_argument("--forced", action="store_true", help="Also emit per-phone boundary error stats")
    parser.add_argument("--out-csv", metavar="PATH", help="Write CSV metrics to this file")
    args = parser.parse_args()

    files= [f for p in args.files for f in (sorted(glob.glob(p)) or [p])]
    predictions, ground_truth, symbols_dict = load_utterances_from_shards(files, args.forced)
    print(f"Loaded {len(predictions)} utterances from {len(files)} file(s)")

    evaluator = SegmentationEvaluator(tolerance_ms=args.tolerance_ms, forced=args.forced)
    results = evaluator.evaluate_batch(predictions, ground_truth, symbols_dict=symbols_dict or None)
    evaluator.pretty_print(results)
    if args.out_csv:
        flat = {k: v for k, v in results.items() if k != "symbol_errors"}
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(flat.keys())
            writer.writerow(flat.values())


if __name__ == "__main__":
    main()
