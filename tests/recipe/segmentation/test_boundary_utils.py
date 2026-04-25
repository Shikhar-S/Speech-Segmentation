"""Unit tests for src.recipe.segmentation.boundary_utils."""

import pytest
import torch

from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.recipe.common.boundary_utils import (
    argmax_to_boundaries,
    boundaries_to_units,
    boundary_rval_metrics,
    evaluate_boundaries,
    phone_starts_to_gt_units,
)


# -- argmax_to_boundaries ---------------------------------------


def test_argmax_to_boundaries_no_blank():
    """FA case: boundary at frame 0 and every phone transition."""
    preds = [1, 1, 1, 2, 2, 3, 3, 3]
    flags = argmax_to_boundaries(preds, len(preds), blank_id=None)
    assert flags == [
        True, False, False, True, False, True, False, False,
    ]


def test_argmax_to_boundaries_with_blank():
    """CTC/ASG case: blanks skipped, transitions between non-blanks."""
    preds = [0, 0, 1, 1, 0, 2, 2, 0, 1, 1]
    flags = argmax_to_boundaries(preds, len(preds), blank_id=0)
    assert flags == [
        False, False, True, False, False, True, False, False,
        True, False,
    ]


def test_argmax_to_boundaries_empty():
    """Zero-length input returns an empty flag list."""
    flags = argmax_to_boundaries([], 0, blank_id=None)
    assert flags == []


def test_argmax_to_boundaries_all_blank():
    """All-blank CTC frames produce no onsets."""
    preds = [0, 0, 0, 0]
    flags = argmax_to_boundaries(preds, len(preds), blank_id=0)
    assert flags == [False, False, False, False]


# -- boundaries_to_units ----------------------------------------


def test_boundaries_to_units_basic():
    """Flags at indices 2 and 4 produce two units: [0, 2) and [2, 4)."""
    flags = [False, False, True, False, True, False]
    units = boundaries_to_units(
        flags, valid_len=6, points_by_frames=320.0, sampling_rate=16000,
    )
    assert len(units) == 2
    assert units[0].start == 0.0
    assert units[0].end == pytest.approx(2 * 320.0 / 16000)
    assert units[1].start == pytest.approx(2 * 320.0 / 16000)
    assert units[1].end == pytest.approx(4 * 320.0 / 16000)


def test_boundaries_to_units_no_boundaries():
    """Without any True flag after index 0, no units are emitted."""
    flags = [False, False, False, False]
    units = boundaries_to_units(
        flags, valid_len=4, points_by_frames=320.0, sampling_rate=16000,
    )
    assert units == []


def test_boundaries_to_units_flag_at_zero_ignored():
    """``True`` at index 0 is not a splitter; only indices >= 1 emit units."""
    flags = [True, False, False, True, False]
    units = boundaries_to_units(
        flags, valid_len=5, points_by_frames=320.0, sampling_rate=16000,
    )
    assert len(units) == 1
    assert units[0].start == 0.0
    assert units[0].end == pytest.approx(3 * 320.0 / 16000)


def test_boundaries_to_units_asserts_positive_len():
    """valid_len == 0 triggers the assertion guard."""
    with pytest.raises(AssertionError):
        boundaries_to_units([], valid_len=0, points_by_frames=320.0, sampling_rate=16000)


def test_boundaries_to_units_label_is_none():
    """Returned units have no label (default ``None``)."""
    flags = [False, True, False]
    units = boundaries_to_units(flags, 3, 320.0, 16000)
    assert all(u.label is None for u in units)


# -- phone_starts_to_gt_units -----------------------------------


def test_phone_starts_to_gt_units_basic():
    """Consecutive starts produce segments [start_i, start_{i+1})."""
    phone_start_idx = torch.tensor([[0, 3, 6, 9]])
    phone_length = torch.tensor([4])
    feature_lens = torch.tensor([12])
    gt = phone_starts_to_gt_units(
        phone_start_idx, phone_length, feature_lens,
        points_by_frames=320.0, sampling_rate=16000,
    )
    assert set(gt.keys()) == {"0"}
    units = gt["0"]
    assert len(units) == 4
    expected_edges = [(0, 3), (3, 6), (6, 9), (9, 12)]
    for u, (s, e) in zip(units, expected_edges):
        assert u.start == pytest.approx(s * 320.0 / 16000)
        assert u.end == pytest.approx(e * 320.0 / 16000)


def test_phone_starts_to_gt_units_multi_utt():
    """Per-utt phone_length honored; last segment closes at feature_lens[b]."""
    phone_start_idx = torch.tensor([
        [0, 2, 5, 0],  # only 3 valid phones
        [0, 4, 0, 0],  # only 2 valid phones
    ])
    phone_length = torch.tensor([3, 2])
    feature_lens = torch.tensor([8, 6])
    gt = phone_starts_to_gt_units(
        phone_start_idx, phone_length, feature_lens, 320.0, 16000,
    )
    assert len(gt["0"]) == 3
    assert len(gt["1"]) == 2
    assert gt["0"][-1].end == pytest.approx(8 * 320.0 / 16000)
    assert gt["1"][-1].end == pytest.approx(6 * 320.0 / 16000)


def test_phone_starts_to_gt_units_label_is_none():
    """GT units carry no label (default ``None``)."""
    phone_start_idx = torch.tensor([[0, 4]])
    phone_length = torch.tensor([2])
    feature_lens = torch.tensor([8])
    gt = phone_starts_to_gt_units(phone_start_idx, phone_length, feature_lens, 320.0, 16000)
    assert all(u.label is None for u in gt["0"])


# -- evaluate_boundaries ----------------------------------------


def test_evaluate_boundaries_keys():
    """Returns exactly the requested metrics."""
    ev = SegmentationEvaluator(tolerance_ms=20)
    preds = {"0": [SegmentationUnit(0.0, 0.5), SegmentationUnit(0.5, 1.0)]}
    gt = {"0": [SegmentationUnit(0.0, 0.5), SegmentationUnit(0.5, 1.0)]}
    metrics = evaluate_boundaries(ev, preds, gt)
    assert set(metrics.keys()) == {"precision", "recall", "f1", "rval"}


def test_evaluate_boundaries_perfect_match_scores_one():
    """Identical preds and GT yield precision=recall=f1=1.0."""
    ev = SegmentationEvaluator(tolerance_ms=20)
    units = [SegmentationUnit(0.0, 0.5), SegmentationUnit(0.5, 1.0)]
    preds = {"0": list(units)}
    gt = {"0": list(units)}
    metrics = evaluate_boundaries(ev, preds, gt)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_evaluate_boundaries_custom_metrics():
    """``metrics`` kwarg selects which scalars are extracted."""
    ev = SegmentationEvaluator(tolerance_ms=20)
    preds = {"0": [SegmentationUnit(0.0, 0.5)]}
    gt = {"0": [SegmentationUnit(0.0, 0.5)]}
    out = evaluate_boundaries(ev, preds, gt, metrics=("precision", "recall"))
    assert set(out.keys()) == {"precision", "recall"}


# -- boundary_rval_metrics --------------------------------------


def _rval_batch(B=2, T=20, N=4, stride=3):
    """Minimal batch with phone_start_idx + phone_length for rval eval."""
    return {
        "phone_start_idx": torch.stack(
            [torch.arange(N) * stride] * B,
        ),
        "phone_length": torch.full((B,), N, dtype=torch.long),
    }


def test_boundary_rval_metrics_with_gt():
    """Returns the standard P/R/F1/rval dict when supervision is present."""
    ev = SegmentationEvaluator(tolerance_ms=20)
    B, T, C = 2, 20, 8
    logits = torch.randn(B, T, C)
    feature_lens = torch.full((B,), T, dtype=torch.long)
    metrics = boundary_rval_metrics(
        logits, feature_lens, _rval_batch(B=B, T=T),
        ev, points_by_frames=320.0, sampling_rate=16000,
    )
    assert set(metrics.keys()) == {"precision", "recall", "f1", "rval"}


def test_boundary_rval_metrics_asserts_missing_supervision():
    """Raises when ``phone_start_idx`` is absent (new precondition)."""
    ev = SegmentationEvaluator(tolerance_ms=20)
    logits = torch.randn(2, 20, 8)
    feature_lens = torch.full((2,), 20, dtype=torch.long)
    with pytest.raises(AssertionError):
        boundary_rval_metrics(
            logits, feature_lens, {"phone_length": torch.tensor([4, 4])},
            ev, 320.0, 16000,
        )


def test_boundary_rval_metrics_blank_id_routes():
    """The ``blank_id`` kwarg is forwarded into ``argmax_to_boundaries``."""
    ev = SegmentationEvaluator(tolerance_ms=20)
    B, T, C = 2, 20, 8
    logits = torch.randn(B, T, C)
    feature_lens = torch.full((B,), T, dtype=torch.long)
    batch = _rval_batch(B=B, T=T)
    # blank_id variants should all return the standard metric dict.
    for bid in (None, 0):
        metrics = boundary_rval_metrics(
            logits, feature_lens, batch, ev, 320.0, 16000, blank_id=bid,
        )
        assert set(metrics.keys()) == {"precision", "recall", "f1", "rval"}
