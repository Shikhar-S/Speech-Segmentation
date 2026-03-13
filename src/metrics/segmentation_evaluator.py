import csv
import io
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class SegmentationUnit:
    """Represents a single aligned unit (e.g., phone)."""

    start: int | float
    end: int | float
    label: str | int


class SegmentationEvaluator:
    """Evaluates CTC-based forced alignment against ground truth."""

    def __init__(self, tolerance_ms: int = 20):
        self.tolerance_sec = tolerance_ms / 1000.0

    def evaluate_boundaries(
        self,
        predicted: List[SegmentationUnit],
        ground_truth: List[SegmentationUnit],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate predicted phone boundaries against ground truth
            for a single utterance.

        Args:
            predicted: list of SegmentationUnit for predicted boundaries.
            ground_truth: list of SegmentationUnit for ground truth boundaries.
            symbols: optional list of symbol labels aligned 1-to-1 with
                ground_truth/predicted. If None, symbol-wise analysis is
                skipped.
        """
        assert len(predicted) == len(ground_truth), (
            "Predicted and ground truth lists must be of the same length "
            f"for single utterance evaluation, but got {len(predicted)} and "
            f"{len(ground_truth)}."
        )
        assert len(ground_truth) > 0, "Ground truth list is empty."
        n = len(predicted)
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
        """Compute metrics for a single phone (in seconds)."""
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

        Errors are still in seconds at this point.
        """
        symbol_data = defaultdict(
            lambda: {"start": [], "end": [], "pbe": [], "dur": []}
        )

        # group by symbol
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

    def _get_metric(self, results: Dict, key: str, default: float = 0.0) -> float:
        """Get metric, falling back to mean_<key> for batch results."""
        if key in results and results[key] is not None:
            return results[key]
        mean_key = f"mean_{key}"
        if mean_key in results and results[mean_key] is not None:
            return results[mean_key]
        return default

    def pretty_print(self, results: Dict) -> None:
        """Print results as a rich table followed by a CSV dump."""
        if not results:
            print("No results")
            return

        console = Console()

        g = self._get_metric
        samples = results.get("n", results.get("total_samples", 0))
        segments = results.get("total_segments", None)

        # --- rich table ---
        table = Table(title="Alignment Evaluation Results", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        count_row = f"{samples} samples"
        if segments is not None:
            count_row += f", {segments} segments"
        table.add_row("Count", count_row)
        table.add_section()
        table.add_row("F1", f"{g(results, 'f1'):.3f}")
        table.add_row("Precision", f"{g(results, 'precision'):.3f}")
        table.add_row("Recall", f"{g(results, 'recall'):.3f}")
        table.add_section()
        for label, prefix in [
            ("Start Error (ms)", "start_err"),
            ("End Error (ms)", "end_err"),
            ("Phone Boundary Error (ms)", "pbe"),
            ("Duration Error (ms)", "dur_err"),
            ("GT Duration (ms)", "gt_dur"),
            ("Pred Duration (ms)", "pred_dur"),
        ]:
            mean = g(results, f"{prefix}_mean")
            std = g(results, f"{prefix}_std")
            table.add_row(label, f"{mean:.2f} ± {std:.2f}")

        if "symbol_errors" in results:
            sym_table = Table(title="Per-Symbol PBE (top 20)", show_lines=True)
            for col in (
                "Symbol",
                "Count",
                "PBE Mean",
                "PBE Std",
                "Start",
                "End",
                "Dur",
            ):
                sym_table.add_column(col, justify="right")
            for sym, s in sorted(
                results["symbol_errors"].items(), key=lambda x: x[1]["pbe_mean"]
            )[:20]:
                sym_table.add_row(
                    str(sym)[:8],
                    str(s["count"]),
                    f"{s['pbe_mean']:.1f}",
                    f"{s['pbe_std']:.1f}",
                    f"{s['start_mean']:.1f}",
                    f"{s['end_mean']:.1f}",
                    f"{s['dur_mean']:.1f}",
                )

        console.print(table)
        if "symbol_errors" in results:
            console.print(sym_table)

        # --- CSV dump ---
        flat = {k: v for k, v in results.items() if k != "symbol_errors"}
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(flat.keys())
        writer.writerow(flat.values())
        console.print(buf.getvalue())

    # -------------------------- batch evaluation ---------------------------- #

    def evaluate_batch(
        self,
        predictions: Dict[str, List[SegmentationUnit]],
        ground_truth: Dict[str, List[SegmentationUnit]],
        symbols_dict: Optional[Dict[str, List[str]]] = None,
        skip_symbols: Optional[set] = None,
    ) -> Dict:
        """Evaluate batch of predictions.

        Args:
            predictions: Dict mapping segment IDs to lists of ForceAlignedUnit
                         for predicted boundaries.
            ground_truth: Dict mapping segment IDs to lists of ForceAlignedUnit
                          for ground truth boundaries.
            symbols_dict: Optional dict mapping segment IDs to symbol lists.
            skip_symbols: Optional set of symbols to skip in evaluation.
                This can be the unk symbol for the model.
        Returns:
            Dict containing aggregated metrics for the batch.
        """
        all_results = []

        for seg_id in ground_truth:
            if seg_id not in predictions:
                log.warning(f"Segment ID {seg_id} missing in predictions; skipping.")
                continue

            preds = predictions[seg_id]
            gts = ground_truth[seg_id]
            syms = symbols_dict.get(seg_id) if symbols_dict else None

            if skip_symbols:
                preds_, gts_ = [], []
                for p, g in zip(preds, gts):
                    if g.label not in skip_symbols and p.label not in skip_symbols:
                        preds_.append(p)
                        gts_.append(g)
                preds = preds_
                gts = gts_
                if syms:
                    syms = [s for s in syms if s not in skip_symbols]

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
    # python -m src.metrics.segmentation_evaluator
    evaluator = SegmentationEvaluator(tolerance_ms=20)

    # Example 1: Single segment evaluation
    print("=" * 60)
    print("Example 1: Single Segment Evaluation")
    print("=" * 60)

    ground_truth = [
        SegmentationUnit(0.0, 0.1, "AH"),
        SegmentationUnit(0.1, 0.2, "T"),
        SegmentationUnit(0.2, 0.3, "AH"),
        SegmentationUnit(0.3, 0.4, "K"),
    ]
    predicted = [
        SegmentationUnit(0.01, 0.11, "AH"),
        SegmentationUnit(0.11, 0.21, "T"),
        SegmentationUnit(0.21, 0.31, "AH"),
        SegmentationUnit(0.31, 0.41, "K"),
    ]
    symbols = ["AH", "T", "AH", "K"]

    results = evaluator.evaluate_boundaries(predicted, ground_truth, symbols)
    evaluator.pretty_print(results)

    # Example 2: Batch evaluation
    print("\n" + "=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)

    batch_pred = {
        "segment_001": [
            SegmentationUnit(0.01, 0.11, "AH"),
            SegmentationUnit(0.11, 0.21, "T"),
            SegmentationUnit(0.21, 0.31, "AH"),
        ],
        "segment_002": [
            SegmentationUnit(0.02, 0.12, "K"),
            SegmentationUnit(0.12, 0.22, "AH"),
        ],
        "segment_003": [
            SegmentationUnit(0.0, 0.1, "T"),
            SegmentationUnit(0.1, 0.2, "AH"),
            SegmentationUnit(0.2, 0.3, "K"),
            SegmentationUnit(0.3, 0.4, "AH"),
        ],
    }
    batch_gt = {
        "segment_001": [
            SegmentationUnit(0.0, 0.1, "AH"),
            SegmentationUnit(0.1, 0.2, "T"),
            SegmentationUnit(0.2, 0.3, "AH"),
        ],
        "segment_002": [
            SegmentationUnit(0.0, 0.1, "K"),
            SegmentationUnit(0.1, 0.2, "AH"),
        ],
        "segment_003": [
            SegmentationUnit(0.0, 0.1, "T"),
            SegmentationUnit(0.1, 0.2, "AH"),
            SegmentationUnit(0.2, 0.3, "K"),
            SegmentationUnit(0.3, 0.4, "AH"),
        ],
    }
    batch_symbols = {
        "segment_001": ["AH", "T", "AH"],
        "segment_002": ["K", "AH"],
        "segment_003": ["T", "AH", "K", "AH"],
    }

    batch_results = evaluator.evaluate_batch(batch_pred, batch_gt, batch_symbols)
    evaluator.pretty_print(batch_results)
