"""Boundary extraction utilities shared across recipes.

* :func:`argmax_to_boundaries` — per-frame prediction → boolean onset flags.
* :func:`boundaries_to_units` — boolean flags → :class:`SegmentationUnit` list.
* :func:`frame_label_to_units` — per-frame labels → :class:`SegmentationUnit` list.
* :func:`redistribute_blank_segments` — merge CTC blank segments into neighbours.
* :func:`target_boundaries_to_gt_units` — frame indices → ground-truth dict.
* :func:`phone_starts_to_gt_units` — phone-start indices → ground-truth dict.
* :func:`boundary_rval_metrics` — end-to-end convenience wrapper from
  argmax logits to the rval metric dict via :func:`src.metrics.evaluate_boundaries`.

The evaluator is no longer constructed here — :func:`src.metrics.evaluate_boundaries`
wraps :class:`phone_metrics.PrecisionRecallMetric` and is stateless.
"""

from collections.abc import Mapping
from typing import Any, List, Optional

import torch

from src.metrics import evaluate_boundaries
from src.metrics.types import SegmentationUnit


def argmax_to_boundaries(
    preds: List[int],
    valid_len: int,
    blank_id: Optional[int] = None,
) -> List[bool]:
    """Return per-frame phone-onset flags.

    Args:
        preds: Per-frame class id, length >= ``valid_len``.
        valid_len: Number of valid (non-padded) frames.
        blank_id: CTC blank / ASG repeat index. ``None`` for forced
            alignment, where every phone change is a boundary.

    Returns:
        Boolean list of length ``valid_len`` marking phone onsets.
    """
    flags = [False] * valid_len
    if valid_len == 0:
        return flags

    if blank_id is None:
        flags[0] = True
        for i in range(1, valid_len):
            if preds[i] != preds[i - 1]:
                flags[i] = True
        return flags

    prev_phone: Optional[int] = None
    for i in range(valid_len):
        p = preds[i]
        if p == blank_id:
            continue
        if p != prev_phone:
            flags[i] = True
            prev_phone = p
    return flags


def boundaries_to_units(
    boundary_flags: List[bool],
    valid_len: int,
    points_by_frames: float,
    sampling_rate: int,
) -> List[SegmentationUnit]:
    """Convert per-frame boundary flags to ``SegmentationUnit`` segments.

    Args:
        boundary_flags: Length ``valid_len``; ``True`` marks a phone onset.
        valid_len: Number of valid (non-padded) frames.
        points_by_frames: Audio points per frame (#points/#frames).
        sampling_rate: Audio sampling rate in Hz.

    Returns:
        ``List[SegmentationUnit]`` with ``label=None`` (boundary-only).
    """
    units: List[SegmentationUnit] = []
    start = 0
    assert (
        valid_len > 0
    ), "valid_len should be > 0 to create at least one segment"
    for i in range(1, valid_len):
        if boundary_flags[i]:
            units.append(
                SegmentationUnit(
                    start=start * points_by_frames / sampling_rate,
                    end=i * points_by_frames / sampling_rate,
                )
            )
            start = i
    return units


def frame_label_to_units(
    frame_labels: List[int],
    valid_len: int,
    points_by_frames: float,
    sampling_rate: int,
    token_list: Optional[List[str]] = None,
) -> List[SegmentationUnit]:
    """Convert per-frame class labels to ``SegmentationUnit`` segments.

    Args:
        frame_labels: Length ``valid_len``; class id per frame.
        valid_len: Number of valid (non-padded) frames.
        points_by_frames: Audio points per frame.
        sampling_rate: Audio sampling rate in Hz.
        token_list: Optional id → string mapping for the ``label`` field.

    Returns:
        ``List[SegmentationUnit]`` with class labels.
    """
    units: List[SegmentationUnit] = []
    assert (
        valid_len > 0
    ), "valid_len should be > 0 to create at least one segment"
    start = 0
    current_label = frame_labels[0]
    for i in range(1, valid_len):
        if frame_labels[i] != current_label:
            units.append(
                SegmentationUnit(
                    start=start * points_by_frames / sampling_rate,
                    end=i * points_by_frames / sampling_rate,
                    label=(
                        current_label
                        if token_list is None
                        else token_list[current_label]
                    ),
                )
            )
            start = i
            current_label = frame_labels[i]
    units.append(
        SegmentationUnit(
            start=start * points_by_frames / sampling_rate,
            end=valid_len * points_by_frames / sampling_rate,
            label=(
                current_label
                if token_list is None
                else token_list[current_label]
            ),
        )
    )
    return units


def redistribute_blank_segments(
    units: List[SegmentationUnit],
    blank_label: Any,
    frac: float = 0.5,  # frac=0.5 is best
) -> List[SegmentationUnit]:
    """Absorb CTC blank segments into neighbouring phones.

    Forced-alignment frame labels carry a CTC blank run between every pair of
    phone spikes. This function merges each blank segment into its neighbours,
    placing the resulting boundary at fraction ``frac`` through the blank gap
    (where frac: ``0`` = previous phone's offset, ``1`` = next phone's onset, ``0.5`` = midpoint)

    Args:
        units: Segments from :func:`frame_label_to_units` (may contain blanks).
        blank_label: The label marking blank segments.
        frac: Position of the boundary within each inter-phone blank gap.

    Returns:
        Blank-free ``List[SegmentationUnit]``, one segment per phone.
    """
    res: List[SegmentationUnit] = []
    pending_gap: Optional[tuple] = None
    for u in units:
        if u.label == blank_label:
            pending_gap = (
                (u.start, u.end)
                if pending_gap is None
                else (pending_gap[0], u.end)
            )
            continue
        if pending_gap is not None:
            gs, ge = pending_gap
            if res:
                boundary = gs + frac * (ge - gs)
                res[-1].end = boundary
                u.start = boundary
            else:
                u.start = gs  # leading blank -> first phone
            pending_gap = None
        res.append(u)
    if pending_gap is not None and res:
        res[-1].end = pending_gap[1]  # trailing blank -> last phone
    return res


NON_PHONE_LABELS = frozenset({"<blank>", "<sos>", "<eos>", "<unk>", "_"})


def strip_outer_brackets(
    units: List[SegmentationUnit],
    silence_labels: frozenset = NON_PHONE_LABELS,
    end_t: Optional[float] = None,
    eps: float = 1e-6,
) -> List[SegmentationUnit]:
    """Drop leading/trailing outer-edge units from a predicted segmentation.

    Trained heads tile the whole utterance ``[0, end_t]`` we strip the end-most
    boundaries to ensure consistent evaluation. Remove ``start ≈ 0`` leading, 
    ``end ≈ end_t`` trailing. Always keeps at least one unit.
    """
    if not units:
        return units
    units = list(units)

    def lead(u: SegmentationUnit) -> bool:
        return u.label in silence_labels or (
            u.label is None and abs(u.start) < eps
        )

    def trail(u: SegmentationUnit) -> bool:
        # NOTE(shikhar): this relies on the fact that for the ctc and
        # fce the last label is one of the silence labels. Hacky.
        return u.label in silence_labels or (
            u.label is None and end_t is not None and abs(u.end - end_t) < eps
        )

    while len(units) > 1 and lead(units[0]):
        units.pop(0)
    while len(units) > 1 and trail(units[-1]):
        units.pop()
    return units


def target_boundaries_to_gt_units(
    target_start_idx: torch.Tensor,
    target_end_idx: torch.Tensor,
    target_length: torch.Tensor,
    feature_lens: torch.Tensor,
    points_by_frames: float,
    sampling_rate: int,
    utt_id: List[str],
) -> dict[str, List[SegmentationUnit]]:
    """Build per-utterance ground-truth segments from target start/end indices."""
    gt: dict[str, List[SegmentationUnit]] = {}
    B = target_start_idx.shape[0]
    for b in range(B):
        n = int(target_length[b])
        starts = target_start_idx[b, :n].tolist()
        ends = target_end_idx[b, :n].tolist()
        gt[utt_id[b]] = [
            SegmentationUnit(
                start=starts[i] * points_by_frames / sampling_rate,
                end=ends[i] * points_by_frames / sampling_rate,
            )
            for i in range(n)
        ]
    return gt


def phone_starts_to_gt_units(
    target_start_idx: torch.Tensor,
    target_length: torch.Tensor,
    feature_lens: torch.Tensor,
    points_by_frames: float,
    sampling_rate: int,
) -> dict[str, List[SegmentationUnit]]:
    """Build per-utterance ground-truth segments from phone-start indices."""
    gt: dict[str, List[SegmentationUnit]] = {}
    B = target_start_idx.shape[0]
    for b in range(B):
        n = int(target_length[b])
        vlen = int(feature_lens[b])
        starts = target_start_idx[b, :n].tolist()
        gt[str(b)] = [
            SegmentationUnit(
                start=starts[i] * points_by_frames / sampling_rate,
                end=(starts[i + 1] if i + 1 < n else vlen)
                * points_by_frames
                / sampling_rate,
            )
            for i in range(n)
        ]
    return gt


@torch.no_grad()
def boundary_rval_metrics(
    logits: torch.Tensor,
    feature_lens: torch.Tensor,
    batch: Mapping[str, Any],
    points_by_frames: float,
    sampling_rate: int,
    blank_id: Optional[int] = None,
    *,
    tolerance_ms: int = 20,
    mode: str = "lenient",
) -> dict[str, float]:
    """Compute P/R/F1/R-val from argmax phone predictions and ground truth."""
    assert (
        "target_start_idx" in batch and "target_length" in batch
    ), "target_start_idx and target_length are required for boundary evaluation"
    gt_dict = phone_starts_to_gt_units(
        batch["target_start_idx"],
        batch["target_length"],
        feature_lens,
        points_by_frames,
        sampling_rate,
    )

    B = logits.shape[0]
    preds_dict: dict[str, List[SegmentationUnit]] = {}
    for b in range(B):
        vlen = int(feature_lens[b])
        preds = logits[b, :vlen].argmax(dim=-1).tolist()
        flags = argmax_to_boundaries(preds, vlen, blank_id)
        preds_dict[str(b)] = boundaries_to_units(
            flags,
            vlen,
            points_by_frames,
            sampling_rate,
        )
    return evaluate_boundaries(
        preds_dict, gt_dict, tolerance_ms=tolerance_ms, mode=mode
    )
