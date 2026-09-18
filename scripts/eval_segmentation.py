#!/usr/bin/env python
"""Compute segmentation metrics from distributed_inference JSONL output.

Usage:
    python -m scripts.eval_segmentation "exp/runs/.../seg_pxeus_timit.*.jsonl"
    python -m scripts.eval_segmentation shard0.jsonl shard1.jsonl --tolerance-ms 20
    python -m scripts.eval_segmentation "exp/runs/.../seg.*.jsonl" --mode strict

For PER/PFER scoring, call ``phone_metrics.phone_error_rates`` directly.
"""

import argparse
import csv
import glob
import json

import numpy as np
from phone_metrics import PrecisionRecallMetric
from tqdm import tqdm

from src.core.ipa_utils import IPA_SILENCE_LABELS
from src.metrics import evaluate_boundaries, units_to_boundary_times
from src.metrics.types import SegmentationUnit


def per_utt_stats(predictions, ground_truth, *, tolerance_ms, mode):
    """Cache per-utterance sufficient stats for bootstrap.

    Returns an int64 array of shape (N_utt, 4) with columns
    ``[n_gt, n_pred, p_contrib, r_contrib]``, where the *_contrib columns
    already fold the strict + lenient-dup contributions to match
    ``PrecisionRecallMetric.compute()``'s aggregate behavior. Aggregating
    these sums and feeding them through ``get_metrics`` is bit-identical to
    rerunning ``compute()`` on the same utterance set.
    """
    metric = PrecisionRecallMetric(tolerance=tolerance_ms / 1000.0, mode=mode)
    rows = []
    for utt_id, gts in ground_truth.items():
        if not gts:
            continue
        gt_b = units_to_boundary_times(gts)
        if gt_b.size == 0:
            continue
        # Empty/missing prediction -> full recall miss (get_counts returns 0).
        pred_b = units_to_boundary_times(predictions.get(utt_id))
        p_strict, p_dup = metric.get_counts(gt_b, pred_b)
        r_strict, r_dup = metric.get_counts(pred_b, gt_b)
        p_contrib = p_strict + p_dup if mode == "lenient" else p_strict
        r_contrib = r_strict + r_dup if mode == "lenient" else r_strict
        rows.append([len(gt_b), len(pred_b), p_contrib, r_contrib])
    return np.asarray(rows, dtype=np.int64)


def _metrics_from_sums(sums, *, tolerance_ms, mode):
    """Derive the metric dict from aggregate (n_gt, n_pred, p, r) sums."""
    metric = PrecisionRecallMetric(tolerance=tolerance_ms / 1000.0, mode=mode)
    n_gt, n_pred, p_count, r_count = (int(x) for x in sums)
    return metric.get_metrics(p_count, r_count, n_pred, n_gt)


def bootstrap_metrics(predictions, ground_truth, *, n, tolerance_ms, mode):
    """Percentile-based 95% CIs via cached per-utterance sufficient stats.

    O(n_utts) per iteration after a single O(n_utts) match-cost precompute.
    Bit-identical to the naive bootstrap on the same resample (both paths
    feed the same integers into ``PrecisionRecallMetric.get_metrics``).
    """
    stats = per_utt_stats(
        predictions,
        ground_truth,
        tolerance_ms=tolerance_ms,
        mode=mode,
    )
    if stats.size == 0:
        return {}
    rng = np.random.default_rng(0)
    n_utts = stats.shape[0]
    pooled: dict[str, list[float]] = {}
    for _ in tqdm(range(n), desc=f"Bootstrap (n={n})"):
        idx = rng.integers(0, n_utts, size=n_utts)
        sums = stats[idx].sum(axis=0)
        for k, v in _metrics_from_sums(
            sums, tolerance_ms=tolerance_ms, mode=mode
        ).items():
            pooled.setdefault(k, []).append(float(v))
    return {
        k: tuple(float(p) for p in np.percentile(vs, [2.5, 97.5]))
        for k, vs in pooled.items()
    }


def parse_groundtruth(passthrough, strip_outer=False):
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
    return [SegmentationUnit(start=s, end=e, label=0) for s, e in ts]


def parse_predictions(pred, head=None):
    """Parse a pred payload into a list of SegmentationUnit.

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
    assert (
        head is not None or len(pred) == 1
    ), f"Multiple heads {list(pred)}; pass --head."
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


def load_utterances_from_shards(files, strip_outer=False, head=None):
    predictions, ground_truth = {}, {}
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
                    predictions[utt_id] = parse_predictions(pred, head=head)
                    ground_truth[utt_id] = parse_groundtruth(
                        passthrough, strip_outer=strip_outer
                    )
    return predictions, ground_truth, n_errors


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation from JSONL output"
    )
    parser.add_argument(
        "files", nargs="+", help="JSONL shard files or glob patterns"
    )
    parser.add_argument("--tolerance-ms", type=int, default=20)
    parser.add_argument(
        "--out-csv", metavar="PATH", help="Write CSV metrics to this file"
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "lenient"],
        default="strict",
        help="Strict means one GT corresponds to one predicted boundary.",
    )
    parser.add_argument(
        "--strip-outer-silences",
        action="store_true",
        default=False,
        help="Drop leading/trailing silence segments (h#, pau, ʔ̞) from GT.",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="For multi-head pred dicts (segment_recognize), select a single head name (e.g. 'bce', 'count_ctc').",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help="If set, compute 95%% percentile CIs via N utt-level bootstrap resamples (default: None, no CIs).",
    )
    args = parser.parse_args()

    files = [f for p in args.files for f in (sorted(glob.glob(p)) or [p])]
    predictions, ground_truth, n_errors = load_utterances_from_shards(
        files,
        strip_outer=args.strip_outer_silences,
        head=args.head,
    )
    print(
        f"Loaded {len(predictions)} utterances from {len(files)} file(s)"
        f" ({n_errors} skipped/error, scored as 0)"
    )

    results = evaluate_boundaries(
        predictions,
        ground_truth,
        tolerance_ms=args.tolerance_ms,
        mode=args.mode,
    )
    cis = (
        bootstrap_metrics(
            predictions,
            ground_truth,
            n=args.bootstrap,
            tolerance_ms=args.tolerance_ms,
            mode=args.mode,
        )
        if args.bootstrap
        else {}
    )
    for k, v in results.items():
        ci = f"  (95% CI: [{cis[k][0]:.4f}, {cis[k][1]:.4f}])" if cis else ""
        print(f"{k}: {v:.4f}{ci}")
    if args.out_csv:
        header = list(results) + [
            f"{k}_ci_{b}" for k in cis for b in ("lo", "hi")
        ]
        row = list(results.values()) + [v for k in cis for v in cis[k]]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)


if __name__ == "__main__":
    main()
