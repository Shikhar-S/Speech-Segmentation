"""Correctness tests for scripts.eval_segmentation.bootstrap_metrics.

The fast path caches per-utterance sufficient stats then aggregates them per
resample. These tests assert the fast path is bit-identical to the naive
"rebuild dicts and re-run PrecisionRecallMetric.compute()" path on the same
resample, and that the cached point-estimate matches evaluate_boundaries on
the full dataset.
"""

import numpy as np
import pytest

from scripts.eval_segmentation import (
    bootstrap_metrics,
    per_utt_stats,
    _metrics_from_sums,
)
from src.metrics import evaluate_boundaries
from src.metrics.types import SegmentationUnit


def _u(start, end, label="x"):
    return SegmentationUnit(start=start, end=end, label=label)


@pytest.fixture
def small_dataset():
    """Five hand-crafted utterances with varying alignment quality."""
    preds = {
        "u1": [_u(0.0, 0.1), _u(0.1, 0.25), _u(0.25, 0.4)],
        "u2": [_u(0.0, 0.2), _u(0.2, 0.5)],
        "u3": [_u(0.0, 0.15), _u(0.15, 0.3), _u(0.3, 0.45), _u(0.45, 0.6)],
        "u4": [_u(0.0, 0.5)],
        "u5": [_u(0.0, 0.12), _u(0.12, 0.24), _u(0.24, 0.5)],
    }
    gt = {
        "u1": [_u(0.0, 0.105), _u(0.105, 0.26), _u(0.26, 0.4)],
        "u2": [_u(0.0, 0.18), _u(0.18, 0.4), _u(0.4, 0.5)],
        "u3": [_u(0.0, 0.14), _u(0.14, 0.32), _u(0.32, 0.6)],
        "u4": [_u(0.0, 0.25), _u(0.25, 0.5)],
        "u5": [_u(0.0, 0.10), _u(0.10, 0.30), _u(0.30, 0.5)],
    }
    return preds, gt


@pytest.mark.parametrize("mode", ["lenient", "strict"])
def test_per_utt_stats_aggregate_matches_compute(small_dataset, mode):
    """Sum of per-utt stats fed to get_metrics == evaluate_boundaries output."""
    preds, gt = small_dataset
    stats = per_utt_stats(preds, gt, tolerance_ms=20, mode=mode)
    sums = stats.sum(axis=0)
    fast = _metrics_from_sums(sums, tolerance_ms=20, mode=mode)
    slow = evaluate_boundaries(preds, gt, tolerance_ms=20, mode=mode)
    for k in slow:
        assert fast[k] == pytest.approx(slow[k], abs=1e-12), (
            f"{mode}/{k}: fast={fast[k]} slow={slow[k]}"
        )


@pytest.mark.parametrize("mode", ["lenient", "strict"])
def test_resample_iter_matches_naive(small_dataset, mode):
    """For a fixed resample, fast metrics == naive metrics bit-for-bit."""
    preds, gt = small_dataset
    utt_ids = [u for u in gt if preds.get(u)]
    rng = np.random.default_rng(42)
    stats = per_utt_stats(preds, gt, tolerance_ms=20, mode=mode)
    for _ in range(20):
        idx = rng.integers(0, len(utt_ids), size=len(utt_ids))
        # Fast path: sum cached stats.
        fast = _metrics_from_sums(
            stats[idx].sum(axis=0), tolerance_ms=20, mode=mode,
        )
        # Naive path: rebuild dicts and recompute from scratch.
        rp = {i: preds[utt_ids[j]] for i, j in enumerate(idx)}
        rg = {i: gt[utt_ids[j]] for i, j in enumerate(idx)}
        naive = evaluate_boundaries(rp, rg, tolerance_ms=20, mode=mode)
        for k in naive:
            assert fast[k] == pytest.approx(naive[k], abs=1e-12), (
                f"{mode}/{k}: fast={fast[k]} naive={naive[k]} idx={idx}"
            )


def test_bootstrap_point_within_ci(small_dataset):
    """Bootstrap CI should bracket the point estimate (most of the time)."""
    preds, gt = small_dataset
    point = evaluate_boundaries(preds, gt, tolerance_ms=20, mode="lenient")
    cis = bootstrap_metrics(
        preds, gt, n=200, tolerance_ms=20, mode="lenient",
    )
    # Point estimate should sit inside or very near the CI for non-degenerate
    # metrics. With only 5 utts the CI can be wide; allow slack at boundaries.
    for k in ("precision", "recall", "f1", "rval"):
        lo, hi = cis[k]
        assert lo <= point[k] <= hi or (point[k] == 0 and lo == 0), (
            f"{k}: point={point[k]} not in [{lo}, {hi}]"
        )


def test_bootstrap_shape_and_keys(small_dataset):
    """bootstrap_metrics returns a (lo, hi) tuple per metric key."""
    preds, gt = small_dataset
    cis = bootstrap_metrics(
        preds, gt, n=50, tolerance_ms=20, mode="lenient",
    )
    expected = {"precision", "recall", "f1", "rval", "over_seg"}
    assert set(cis) == expected
    for k, v in cis.items():
        assert isinstance(v, tuple) and len(v) == 2
        assert v[0] <= v[1]


def test_bootstrap_empty_inputs():
    """Empty inputs return an empty dict, no crash."""
    cis = bootstrap_metrics(
        {}, {}, n=10, tolerance_ms=20, mode="lenient",
    )
    assert cis == {}
