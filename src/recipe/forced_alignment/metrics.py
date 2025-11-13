from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForcedAlignmentData:
    """Represents a single aligned unit (e.g., phone)."""

    start: int | float
    end: int | float
    label: str | int


class AlignmentEvaluator:
    """Evaluates CTC-based forced alignment against ground truth."""

    def __init__(self, tolerance_ms: int = 20):
        self.tolerance_sec = tolerance_ms / 1000.0

    # ------------------------ core single-segment eval ------------------------ #

    def evaluate_boundaries(
        self,
        predicted: List[ForcedAlignmentData],
        ground_truth: List[ForcedAlignmentData],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate predicted phone boundaries against ground truth.

        Args:
            predicted: list of AlignmentResult for predicted boundaries.
            ground_truth: list of AlignmentResult for ground truth boundaries.
            symbols: optional list of symbol labels (usually phones) aligned
                     1-to-1 with ground_truth/predicted. If None, symbol-wise
                     analysis is skipped.
        """
        n = min(len(predicted), len(ground_truth))
        if n == 0:
            return {}

        predicted = predicted[:n]
        ground_truth = ground_truth[:n]
        if symbols:
            symbols = symbols[:n]

        # Compute errors for each phoneme
        metrics = np.array(
            [
                self._compute_metrics(p.start, p.end, g.start, g.end)
                for p, g in zip(predicted, ground_truth)
            ]
        )
        start_err, end_err, pbe, dur_err, gt_dur, pred_dur = metrics.T

        # Count correct predictions
        correct = np.sum(
            (start_err <= self.tolerance_sec) & (end_err <= self.tolerance_sec)
        )
        precision = recall = f1 = correct / n

        # Build results dictionary
        percentiles = [5, 25, 50, 75, 95, 99]
        results = {"n": n, "f1": f1, "precision": precision, "recall": recall}

        # Add statistics for each error type (convert to ms)
        error_types = [
            ("start_err", start_err),
            ("end_err", end_err),
            ("pbe", pbe),
            ("boundary_err", np.concatenate([start_err, end_err])),
            ("dur_err", dur_err),
            ("gt_dur", gt_dur),
            ("pred_dur", pred_dur),
        ]
        for name, data in error_types:
            results.update(self._compute_stats(data * 1000, name, percentiles))

        # Add symbol-wise analysis if symbols provided
        if symbols:
            results["symbol_errors"] = self._analyze_by_symbol(
                symbols, start_err, end_err, pbe, dur_err
            )

        return results

    def _compute_metrics(self, ps, pe, gs, ge):
        """Compute metrics for a single phoneme (in seconds)."""
        start_err = abs(ps - gs)
        end_err = abs(pe - ge)
        return (
            start_err,
            end_err,
            0.5 * (start_err + end_err),  # pbe
            abs((pe - ps) - (ge - gs)),  # dur_err
            ge - gs,  # gt_dur
            pe - ps,  # pred_dur
        )

    def _compute_stats(self, data, prefix, percentiles):
        """Compute statistics for a data array (assumed in ms)."""
        stats = {
            f"{prefix}_mean": np.mean(data),
            f"{prefix}_std": np.std(data),
            f"{prefix}_median": np.median(data),
        }
        stats.update({f"{prefix}_p{p}": np.percentile(data, p) for p in percentiles})
        return stats

    def _analyze_by_symbol(self, symbols, start_err, end_err, pbe, dur_err):
        """Analyze errors grouped by phoneme symbol.

        Errors are still in seconds at this point; we convert to ms here.
        """
        symbol_data = defaultdict(
            lambda: {"start": [], "end": [], "pbe": [], "dur": []}
        )

        for sym, se, ee, pb, de in zip(symbols, start_err, end_err, pbe, dur_err):
            symbol_data[sym]["start"].append(se * 1000)
            symbol_data[sym]["end"].append(ee * 1000)
            symbol_data[sym]["pbe"].append(pb * 1000)
            symbol_data[sym]["dur"].append(de * 1000)

        return {
            sym: {
                "count": len(data["pbe"]),
                "pbe_mean": np.mean(data["pbe"]),
                "pbe_std": np.std(data["pbe"]),
                "start_mean": np.mean(data["start"]),
                "end_mean": np.mean(data["end"]),
                "dur_mean": np.mean(data["dur"]),
            }
            for sym, data in symbol_data.items()
        }

    # ------------ helpers for printing aggregated vs per-segment ------------ #

    def _get_metric(self, results: Dict, key: str, default: float = 0.0) -> float:
        """Get metric, falling back to mean_<key> for batch results."""
        if key in results and results[key] is not None:
            return results[key]
        mean_key = f"mean_{key}"
        if mean_key in results and results[mean_key] is not None:
            return results[mean_key]
        return default

    def _get_percentile_value(
        self, results: Dict, metric: str, p: int, default: float = 0.0
    ) -> float:
        """Get percentile metric, falling back to mean_<metric>_pX for batch."""
        key = f"{metric}_p{p}"
        if key in results:
            return results[key]
        mean_key = f"mean_{key}"
        if mean_key in results:
            return results[mean_key]
        return default

    # ------------------------ pretty-printing logic ------------------------- #

    def pretty_print(self, results: Dict, verbosity: int = 1) -> None:
        """Print results as ASCII table."""
        if not results:
            print("No results")
            return

        print("\nALIGNMENT EVALUATION RESULTS")
        print("=" * 50)

        verbosity_map = {
            0: self._print_minimal,
            1: self._print_standard,
            2: self._print_detailed,
        }
        verbosity_map.get(verbosity, self._print_full)(results)

    def _print_table(self, rows, col_widths=None):
        """Print a simple ASCII table."""
        if not rows:
            return

        if col_widths is None:
            col_widths = [
                max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))
            ]

        for row in rows:
            print(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))

    def _print_minimal(self, results):
        """Minimal output."""
        f1 = self._get_metric(results, "f1", 0.0)
        be_mean = self._get_metric(results, "boundary_err_mean", 0.0)
        be_std = self._get_metric(results, "boundary_err_std", 0.0)
        dur_mean = self._get_metric(results, "dur_err_mean", 0.0)
        dur_std = self._get_metric(results, "dur_err_std", 0.0)

        self._print_table(
            [
                ["Metric", "Value"],
                ["-" * 20, "-" * 30],
                ["F1 Score", f"{f1:.3f}"],
                ["Boundary Error (ms)", f"{be_mean:.2f} +/- {be_std:.2f}"],
                ["Duration Error (ms)", f"{dur_mean:.2f} +/- {dur_std:.2f}"],
            ]
        )

    def _print_standard(self, results):
        """Standard output."""
        f1 = self._get_metric(results, "f1", 0.0)
        start_mean = self._get_metric(results, "start_err_mean", 0.0)
        start_std = self._get_metric(results, "start_err_std", 0.0)
        end_mean = self._get_metric(results, "end_err_mean", 0.0)
        end_std = self._get_metric(results, "end_err_std", 0.0)
        pbe_mean = self._get_metric(results, "pbe_mean", 0.0)
        pbe_std = self._get_metric(results, "pbe_std", 0.0)
        dur_mean = self._get_metric(results, "dur_err_mean", 0.0)
        dur_std = self._get_metric(results, "dur_err_std", 0.0)

        samples = results.get("n", results.get("total_samples", 0))
        segments = results.get("total_segments", None)

        rows = [
            ["Metric", "Value"],
            ["-" * 20, "-" * 30],
            ["Samples", samples],
        ]
        if segments is not None:
            rows.append(["Segments", segments])

        rows.extend(
            [
                ["F1/Precision/Recall", f"{f1:.3f}"],
                [
                    "Start Error (ms)",
                    f"{start_mean:.2f} +/- {start_std:.2f}",
                ],
                [
                    "End Error (ms)",
                    f"{end_mean:.2f} +/- {end_std:.2f}",
                ],
                [
                    "PBE (ms)",
                    f"{pbe_mean:.2f} +/- {pbe_std:.2f}",
                ],
                [
                    "Duration Error (ms)",
                    f"{dur_mean:.2f} +/- {dur_std:.2f}",
                ],
            ]
        )

        self._print_table(rows)

    def _print_detailed(self, results):
        """Detailed output with medians and durations."""
        self._print_standard(results)

        pbe_med = self._get_metric(results, "pbe_median", 0.0)
        start_med = self._get_metric(results, "start_err_median", 0.0)
        end_med = self._get_metric(results, "end_err_median", 0.0)
        dur_med = self._get_metric(results, "dur_err_median", 0.0)

        gt_mean = self._get_metric(results, "gt_dur_mean", 0.0)
        gt_std = self._get_metric(results, "gt_dur_std", 0.0)
        pred_mean = self._get_metric(results, "pred_dur_mean", 0.0)
        pred_std = self._get_metric(results, "pred_dur_std", 0.0)

        print("\nMedians:")
        self._print_table(
            [
                ["Metric", "Median (ms)"],
                ["-" * 20, "-" * 15],
                ["PBE", f"{pbe_med:.2f}"],
                ["Start Error", f"{start_med:.2f}"],
                ["End Error", f"{end_med:.2f}"],
                ["Duration Error", f"{dur_med:.2f}"],
            ]
        )

        print("\nDurations:")
        self._print_table(
            [
                ["Type", "Mean +/- Std (ms)"],
                ["-" * 20, "-" * 25],
                ["Ground Truth", f"{gt_mean:.2f} +/- {gt_std:.2f}"],
                ["Predicted", f"{pred_mean:.2f} +/- {pred_std:.2f}"],
            ]
        )

        if "symbol_errors" in results:
            self._print_symbol_errors(results["symbol_errors"])

    def _print_full(self, results):
        """Full output with percentiles."""
        self._print_detailed(results)

        print("\nPercentiles:")
        percentiles = [5, 25, 50, 75, 95, 99]
        rows = [
            ["Metric"] + [f"P{p}" for p in percentiles],
            ["-" * 15] + ["-" * 8] * len(percentiles),
        ]
        for metric in ["pbe", "start_err", "end_err", "dur_err"]:
            rows.append(
                [metric.replace("_", " ").title()]
                + [
                    f"{self._get_percentile_value(results, metric, p, 0.0):.1f}"
                    for p in percentiles
                ]
            )
        self._print_table(rows)

    def _print_symbol_errors(self, symbol_errors):
        """Print symbol-wise error analysis."""
        if not symbol_errors:
            return

        print("\nPer-Symbol Analysis (sorted by PBE):")
        sorted_symbols = sorted(symbol_errors.items(), key=lambda x: x[1]["pbe_mean"])
        rows = [
            [
                "Symbol",
                "Count",
                "PBE Mean",
                "PBE Std",
                "Start",
                "End",
                "Duration",
            ],
            [
                "-" * 10,
                "-" * 8,
                "-" * 10,
                "-" * 10,
                "-" * 8,
                "-" * 8,
                "-" * 10,
            ],
        ]
        for sym, stats in sorted_symbols[:20]:
            rows.append(
                [
                    str(sym)[:8],
                    stats["count"],
                    f"{stats['pbe_mean']:.1f}",
                    f"{stats['pbe_std']:.1f}",
                    f"{stats['start_mean']:.1f}",
                    f"{stats['end_mean']:.1f}",
                    f"{stats['dur_mean']:.1f}",
                ]
            )
        if len(sorted_symbols) > 20:
            rows.append(["...", "...", "...", "...", "...", "...", "..."])
        self._print_table(rows)

    # -------------------------- batch evaluation ---------------------------- #

    def evaluate_batch(
        self,
        predictions: Dict[str, List[ForcedAlignmentData]],
        ground_truth: Dict[str, List[ForcedAlignmentData]],
        symbols_dict: Optional[Dict[str, List[str]]] = None,
    ) -> Dict:
        """Evaluate batch of predictions.

        Args:
            predictions: Dict mapping segment IDs to lists of AlignmentResult
                         for predicted boundaries.
            ground_truth: Dict mapping segment IDs to lists of AlignmentResult
                          for ground truth boundaries.
            symbols_dict: Optional dict mapping segment IDs to symbol lists.
        Returns:
            Dict containing aggregated metrics for the batch.
        """
        all_results = []

        for seg_id in ground_truth:
            if seg_id not in predictions:
                continue

            preds = predictions[seg_id]
            gts = ground_truth[seg_id]
            syms = symbols_dict.get(seg_id) if symbols_dict else None

            res = self.evaluate_boundaries(preds, gts, syms)
            all_results.append(res)

        if not all_results:
            return {}

        # Aggregate results
        metric_names = [k for k in all_results[0] if k not in ("symbol_errors", "n")]
        aggregated = {
            "total_segments": len(all_results),
            "total_samples": sum(r["n"] for r in all_results),
            **{
                f"mean_{metric}": np.mean([r[metric] for r in all_results])
                for metric in metric_names
            },
        }

        # Merge symbol errors if present
        if "symbol_errors" in all_results[0]:
            merged_symbols = defaultdict(
                lambda: {
                    "count": 0,
                    "pbe_all": [],
                    "start_all": [],
                    "end_all": [],
                    "dur_all": [],
                }
            )
            for r in all_results:
                for sym, stats in r.get("symbol_errors", {}).items():
                    merged_symbols[sym]["count"] += stats["count"]
                    merged_symbols[sym]["pbe_all"].append(stats["pbe_mean"])
                    merged_symbols[sym]["start_all"].append(stats["start_mean"])
                    merged_symbols[sym]["end_all"].append(stats["end_mean"])
                    merged_symbols[sym]["dur_all"].append(stats["dur_mean"])

            aggregated["symbol_errors"] = {
                sym: {
                    "count": data["count"],
                    "pbe_mean": np.mean(data["pbe_all"]),
                    "pbe_std": np.std(data["pbe_all"]),
                    "start_mean": np.mean(data["start_all"]),
                    "end_mean": np.mean(data["end_all"]),
                    "dur_mean": np.mean(data["dur_all"]),
                }
                for sym, data in merged_symbols.items()
            }

        return aggregated


if __name__ == "__main__":
    # Example usage of AlignmentEvaluator
    evaluator = AlignmentEvaluator(tolerance_ms=20)

    # Example 1: Single segment evaluation
    print("=" * 60)
    print("Example 1: Single Segment Evaluation")
    print("=" * 60)

    ground_truth = [
        ForcedAlignmentData(0.0, 0.1, "AH"),
        ForcedAlignmentData(0.1, 0.2, "T"),
        ForcedAlignmentData(0.2, 0.3, "AH"),
        ForcedAlignmentData(0.3, 0.4, "K"),
    ]
    predicted = [
        ForcedAlignmentData(0.01, 0.11, "AH"),
        ForcedAlignmentData(0.11, 0.21, "T"),
        ForcedAlignmentData(0.21, 0.31, "AH"),
        ForcedAlignmentData(0.31, 0.41, "K"),
    ]
    symbols = ["AH", "T", "AH", "K"]

    results = evaluator.evaluate_boundaries(predicted, ground_truth, symbols)
    evaluator.pretty_print(results, verbosity=1)

    # Example 2: Batch evaluation
    print("\n" + "=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)

    batch_pred = {
        "segment_001": [
            ForcedAlignmentData(0.01, 0.11, "AH"),
            ForcedAlignmentData(0.11, 0.21, "T"),
            ForcedAlignmentData(0.21, 0.31, "AH"),
        ],
        "segment_002": [
            ForcedAlignmentData(0.02, 0.12, "K"),
            ForcedAlignmentData(0.12, 0.22, "AH"),
        ],
        "segment_003": [
            ForcedAlignmentData(0.0, 0.1, "T"),
            ForcedAlignmentData(0.1, 0.2, "AH"),
            ForcedAlignmentData(0.2, 0.3, "K"),
            ForcedAlignmentData(0.3, 0.4, "AH"),
        ],
    }
    batch_gt = {
        "segment_001": [
            ForcedAlignmentData(0.0, 0.1, "AH"),
            ForcedAlignmentData(0.1, 0.2, "T"),
            ForcedAlignmentData(0.2, 0.3, "AH"),
        ],
        "segment_002": [
            ForcedAlignmentData(0.0, 0.1, "K"),
            ForcedAlignmentData(0.1, 0.2, "AH"),
        ],
        "segment_003": [
            ForcedAlignmentData(0.0, 0.1, "T"),
            ForcedAlignmentData(0.1, 0.2, "AH"),
            ForcedAlignmentData(0.2, 0.3, "K"),
            ForcedAlignmentData(0.3, 0.4, "AH"),
        ],
    }
    batch_symbols = {
        "segment_001": ["AH", "T", "AH"],
        "segment_002": ["K", "AH"],
        "segment_003": ["T", "AH", "K", "AH"],
    }

    batch_results = evaluator.evaluate_batch(batch_pred, batch_gt, batch_symbols)
    evaluator.pretty_print(batch_results, verbosity=2)

    # Example 3: Different verbosity levels
    print("\n" + "=" * 60)
    print("Example 3: Minimal Verbosity")
    print("=" * 60)
    evaluator.pretty_print(results, verbosity=0)

    print("\n" + "=" * 60)
    print("Example 4: Full Verbosity")
    print("=" * 60)
    evaluator.pretty_print(results, verbosity=3)
