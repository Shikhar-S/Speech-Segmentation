import argparse
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


# NOTE(shikhar): This still does not do the equidistant splitting
# between two close enough boundaries that R-Value paper recommends.
# https://d1wqtxts1xzle7.cloudfront.net/35865703/IS09_r_value-libre.pdf?1418037042=&response-content-disposition=inline%3B+filename%3DAn_improved_speech_segmentation_quality.pdf&Expires=1773463828&Signature=V8NdeaZ721135Z2F9y85-CLT31h9w~CTFEyPVFQeOYeXBMYorTRdSVdHe~DTxI2~Zb-mob33cAn8OjcpU86jHHUAQHp0A0KcLXadreo1AXEVpfiWpafiT11h~pIfsxqZvEMlQUCNKJVB9lSqFFFv~mUJ-i0msHUoZf9I9q6-THfeDSHpBu8OeF91lV~uO0k69OeKVt49QTrMcmXtaDbaVJEO9NOQxHWyBJmdusvb9dphh~oof039vvbPJ4x0ySV-mizVtk8uOj7ARPXqtPgKNUgCb6ooK7fCWgTnErJBfob2IA2FwB2BL~5eOvTZaaKdT7HR4QRu2MVwaPTNa6ewOA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
# but 1) with a low-tolerance setting that kind of splitting will have low impact
# 2) results from this should be comparable to Jian's earlier papers


@dataclass
class SegmentationUnit:
    """Represents a single aligned unit (e.g., phone)."""

    start: int | float
    end: int | float
    label: str | int


class SegmentationEvaluator:
    """Evaluates CTC-based forced alignment against ground truth.

    Args:
        tolerance_ms: Boundary tolerance in milliseconds (default 20).
        forced: If True, operate in forced mode: requires paired segments and
            computes per-phone error statistics in addition to boundary P/R/F1/Rval.
            If False (default), operate in free mode: accepts any segment counts
            and returns only boundary-level metrics.
    """

    def __init__(self, tolerance_ms: int = 20, forced: bool = False):
        self.tolerance_sec = tolerance_ms / 1000.0
        self.forced = forced

    def evaluate_boundaries(
        self,
        predicted: List[SegmentationUnit],
        ground_truth: List[SegmentationUnit],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate predicted phone boundaries against ground truth
            for a single utterance.

        In free mode (forced=False): accepts lists of any length and returns
        boundary P/R/F1/Rval only. In forced mode (forced=True): zips
        predicted/GT pairs and additionally returns per-phone error statistics.

        Args:
            predicted: list of SegmentationUnit for predicted boundaries.
            ground_truth: list of SegmentationUnit for ground truth boundaries.
            symbols: optional list of symbol labels aligned 1-to-1 with
                ground_truth/predicted. Only used in forced mode.
        """
        if not predicted or not ground_truth:
            return {}
        counts = self._get_boundary_counts(predicted, ground_truth)
        precision, recall, f1, rval = self._get_boundary_metrics(
            counts["precision_counter"],
            counts["recall_counter"],
            counts["pred_counter"],
            counts["gt_counter"],
        )
        results = {
            "n_pred": len(predicted),
            "n_gt": len(ground_truth),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rval": rval,
        }
        if not self.forced:
            return results
        return self._add_forced_alignment_metrics(results, predicted, ground_truth, symbols)

    def _get_boundary_counts(
        self,
        predicted: List[SegmentationUnit],
        ground_truth: List[SegmentationUnit],
    ) -> Dict[str, int]:
        """Return raw boundary match counts for free-mode aggregation."""
        pred_times = self._extract_boundary_times(predicted)
        gt_times = self._extract_boundary_times(ground_truth)

        precision_counter = sum(
            np.abs(gt_times - t).min() <= self.tolerance_sec for t in pred_times
        )
        recall_counter = sum(
            np.abs(pred_times - t).min() <= self.tolerance_sec for t in gt_times
        )

        return {
            "precision_counter": int(precision_counter),
            "recall_counter": int(recall_counter),
            "pred_counter": int(len(pred_times)),
            "gt_counter": int(len(gt_times)),
        }

    def _add_forced_alignment_metrics(
        self,
        results: Dict[str, float],
        predicted: List[SegmentationUnit],
        ground_truth: List[SegmentationUnit],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Add forced-mode metrics to existing results."""
        # print('length of predicted and ground_truth', len(predicted), len(ground_truth))
        n = min(len(predicted), len(ground_truth))
        metrics = np.array(
            [
                self._compute_metrics(p.start, p.end, g.start, g.end)
                for p, g in zip(predicted, ground_truth)
            ]
        )
        # # print first 5 times
        # print("First 5 predicted vs GT times:")
        # for i in range(min(5, len(metrics))):
        #     ps, pe, gs, ge = predicted[i].start, predicted[i].end, ground_truth[i].start, ground_truth[i].end
        #     print(f"Pred: ({ps:.3f}, {pe:.3f}), GT: ({gs:.3f}, {ge:.3f})")
            
        
        # print('-=-' * 20)
        # print('Largest predicted and GT time for largest 5 pbe')
        # pbe = metrics[:, 2]
        # largest_indices = np.argsort(pbe)[-5:]
        # for idx in largest_indices:
        #     ps, pe, gs, ge = predicted[idx].start, predicted[idx].end, ground_truth[idx].start, ground_truth[idx].end
        #     print(f"PBE: {pbe[idx]:.3f} sec - Pred: ({ps:.3f}, {pe:.3f}), GT: ({gs:.3f}, {ge:.3f})")  

        start_err, end_err, pbe, dur_err, gt_dur, pred_dur = metrics.T

        percentiles = [5, 50, 95]
        results.update({'n': n})
        
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

        if symbols:
            results["symbol_errors"] = self._analyze_by_symbol(
                symbols, start_err, end_err, pbe, dur_err
            )

        return results

    # ------------------------------------------------------------------ #
    # Boundary helpers                                                   #
    # ------------------------------------------------------------------ #

    def _extract_boundary_times(self, units: List[SegmentationUnit]) -> np.ndarray:
        """Extract all unique boundary times (N starts + final end) from units."""
        times = [u.start for u in units] + [units[-1].end]
        return np.unique(times)

    def _get_boundary_metrics(
        self,
        precision_counter: float,
        recall_counter: float,
        pred_counter: int,
        gt_counter: int,
    ):
        """Compute precision, recall, F1, and R-value from boundary match counts.

        R-value formula adapted from UnsupSeg (github.com/felixkreuk/UnsupSeg),
        also used in charsiu_eval.py.
        """
        eps = 1e-7
        precision = precision_counter / (pred_counter + eps)
        recall = recall_counter / (gt_counter + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        os = recall / (precision + eps) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os**2)
        r2 = (-os + recall - 1) / np.sqrt(2)
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return precision, recall, f1, rval

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
        samples = results.get("n", results.get("total_samples", results.get("n_gt", 0)))
        segments = results.get("total_segments", None)

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
        table.add_row("R-value", f"{g(results, 'rval'):.3f}")

        has_per_phone = (
            "start_err_mean" in results
            or "mean_start_err" in results
            or "mean_start_err_mean" in results
        )
        if has_per_phone:
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

        micro_precision_counter = 0
        micro_recall_counter = 0
        micro_pred_counter = 0
        micro_gt_counter = 0

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
                # NOTE: syms is filtered independently of the paired preds/gts
                # filtering above, which may cause misalignment if labels diverge.
                if syms:
                    syms = [s for s in syms if s not in skip_symbols]

            res = self.evaluate_boundaries(preds, gts, syms)
            if not res:
                continue
            all_results.append(res)

            counts = self._get_boundary_counts(preds, gts)
            micro_precision_counter += counts["precision_counter"]
            micro_recall_counter += counts["recall_counter"]
            micro_pred_counter += counts["pred_counter"]
            micro_gt_counter += counts["gt_counter"]

        if not all_results:
            return {}

        boundary_metrics = {"f1", "precision", "recall", "rval"}
        count_keys = {"n", "n_pred", "n_gt"}
        metric_names = [
            k
            for k in all_results[0]
            if k not in ("symbol_errors", *count_keys, *boundary_metrics)
        ]
        aggregated = {
            "total_segments": len(all_results),
            "total_samples": sum(r.get("n", r.get("n_gt", 0)) for r in all_results),
            **{
                f"mean_{metric}": np.mean([r[metric] for r in all_results])
                for metric in metric_names
            },
        }

        precision, recall, f1, rval = self._get_boundary_metrics(
            micro_precision_counter,
            micro_recall_counter,
            micro_pred_counter,
            micro_gt_counter,
        )
        aggregated.update(
            {
                "precision_counter": micro_precision_counter,
                "recall_counter": micro_recall_counter,
                "pred_counter": micro_pred_counter,
                "gt_counter": micro_gt_counter,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "rval": rval,
            }
        )

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

            # NOTE: Unweighted mean-of-means — each utterance's symbol
            # mean has equal weight regardless of instance count.
            # Weighting by count would give more accurate aggregates.
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
    # Example usage of SegmentationEvaluator
    # python -m src.metrics.segmentation_evaluator
    # python -m src.metrics.segmentation_evaluator --forced
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forced",
        action="store_true",
        help="Run in forced mode (requires equal segment counts, adds per-phone stats)",
    )
    args = parser.parse_args()
    evaluator = SegmentationEvaluator(tolerance_ms=20, forced=args.forced)

    mode_label = "Forced" if args.forced else "Free"

    print("=" * 60)
    print(f"Example 1: Single Segment Evaluation ({mode_label} mode)")
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

    results = evaluator.evaluate_boundaries(
        predicted, ground_truth, symbols if args.forced else None
    )
    evaluator.pretty_print(results)

    print("\n" + "=" * 60)
    print(f"Example 2: Batch Evaluation ({mode_label} mode)")
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

    batch_results = evaluator.evaluate_batch(
        batch_pred, batch_gt, batch_symbols if args.forced else None
    )
    evaluator.pretty_print(batch_results)
