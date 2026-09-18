"""Project-wide metrics layer.

Re-exports the :mod:`phone_metrics` library primitives (the canonical
implementation of boundary P/R/F1/R-value and PER/PFER), and adds two thin
adapters so xeuspr callers do not have to massage data shapes at every
callsite:

* :func:`evaluate_boundaries` accepts the existing xeuspr
  ``{utt_id: [SegmentationUnit]}`` dicts and returns the metric dict.
* :func:`unit_to_seg` / :func:`seg_to_unit` cross the wire-format vs
  ``phone_metrics.Seg`` boundary.
"""

from typing import Iterable, List, Sequence

import numpy as np
from phone_metrics import (
    OracleAccuracy,
    PhoneErrorRates,
    PrecisionRecallMetric,
    RecognitionCounts,
    Utterance,
    boundary_secs,
    canonical_ipa,
    load_timit,
    load_voxangeles,
    oracle_phone_accuracy,
    phone_error_rates,
)
from phone_metrics.timit import SILENCE, Seg

from src.metrics.types import SegmentationUnit

__all__ = [
    "OracleAccuracy",
    "PhoneErrorRates",
    "PrecisionRecallMetric",
    "RecognitionCounts",
    "SILENCE",
    "Seg",
    "SegmentationUnit",
    "Utterance",
    "boundary_secs",
    "canonical_ipa",
    "evaluate_boundaries",
    "load_timit",
    "load_voxangeles",
    "oracle_phone_accuracy",
    "phone_error_rates",
    "seg_to_unit",
    "unit_to_seg",
    "units_to_boundary_times",
]


def unit_to_seg(unit: SegmentationUnit) -> Seg:
    """Convert a wire-format :class:`SegmentationUnit` to ``phone_metrics.Seg``.

    The single ``label`` is placed in both ``raw_label`` and ``ipa_label``
    so downstream label/feature edit-distance computations work without
    further normalization.
    """
    label = "" if unit.label is None else str(unit.label)
    return Seg(
        start=float(unit.start),
        end=float(unit.end),
        raw_label=label,
        ipa_label=label,
    )


def seg_to_unit(seg: Seg) -> SegmentationUnit:
    """Convert a ``phone_metrics.Seg`` to a wire-format :class:`SegmentationUnit`."""
    label = seg.ipa_label if seg.ipa_label is not None else seg.raw_label
    return SegmentationUnit(start=seg.start, end=seg.end, label=label)


def units_to_boundary_times(units: Sequence[SegmentationUnit]) -> np.ndarray:
    """Extract unique boundary times (N starts + final end) as a numpy array.

    Every segment start is a boundary, plus the final end. Returns
    ``np.array([])`` for empty inputs.
    """
    if not units:
        return np.array([], dtype=np.float64)
    times = [float(u.start) for u in units] + [float(units[-1].end)]
    return np.unique(times)


def evaluate_boundaries(
    preds_dict: dict[str, List[SegmentationUnit]],
    gt_dict: dict[str, List[SegmentationUnit]],
    *,
    tolerance_ms: int = 20,
    mode: str = "lenient",
) -> dict[str, float]:
    """Aggregate boundary P/R/F1/R-value across utterances.

    Wraps :class:`phone_metrics.PrecisionRecallMetric`. Constructs a fresh
    metric per call (the upstream class has no ``reset``).

    Args:
        preds_dict: Predicted segments keyed by utterance id.
        gt_dict: Ground-truth segments keyed by utterance id.
        tolerance_ms: Boundary tolerance in milliseconds (converted to
            seconds for :class:`phone_metrics.PrecisionRecallMetric`).
        mode: Either ``"lenient"`` (independent nearest-neighbour) or
            ``"strict"`` (greedy one-to-one).

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``rval``, ``over_seg``.
        Empty dict if no utterance had both non-empty predictions and GT.
    """
    metric = PrecisionRecallMetric(tolerance=tolerance_ms / 1000.0, mode=mode)
    seen = 0
    for utt_id, gts in gt_dict.items():
        if not gts:
            continue
        gt_bounds = units_to_boundary_times(gts)
        if gt_bounds.size == 0:
            continue
        pred_bounds = units_to_boundary_times(preds_dict.get(utt_id))
        metric.update(gt_bounds, pred_bounds)
        seen += 1
    if seen == 0:
        return {}
    return metric.compute()


def utterances_from_units(
    units_dict: dict[str, List[SegmentationUnit]],
    *,
    language: str = "eng",
    split: str = "test",
) -> list[Utterance]:
    """Build :class:`phone_metrics.Utterance` list from a label dict.

    For PER/PFER computation we need ``Utterance`` references. The
    ``audio_path`` field carries the utterance id (the upstream library
    uses it only as an identifier in error reports).
    """
    result: list[Utterance] = []
    for utt_id, units in units_dict.items():
        segs = [unit_to_seg(u) for u in units]
        result.append(
            Utterance(
                audio_path=utt_id,
                language=language,
                split=split,
                segments=segs,
            )
        )
    return result
