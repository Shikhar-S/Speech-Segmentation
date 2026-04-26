"""Boundary extraction utilities shared across recipes.

* ``argmax_to_boundaries``: per-frame prediction to boolean flags marking onsets.
* ``boundaries_to_units``: boolean flags to ``SegmentationUnit`` list.
* ``phone_starts_to_gt_units``: per-utterance phone-start indices to
  ground-truth ``SegmentationUnit`` dict.
* ``evaluate_boundaries``: run ``SegmentationEvaluator`` and extract
  the standard ``(precision, recall, f1, rval)`` scalars.
* ``boundary_rval_metrics``: end-to-end convenience wrapper from
  argmax logits to the rval-metric dict.

# TODO(shikhar): optimize, Vectorized utils
"""

from collections.abc import Mapping
from typing import Any, List, Optional

import torch

from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)


def argmax_to_boundaries(
    preds: List[int],
    valid_len: int,
    blank_id: Optional[int] = None,
) -> List[bool]:
    """Returns a list where element[i] is True if ith frame is a phone onset.

    Args:
        preds: Per-frame class id, length should be >= ``valid_len``.
        valid_len: Number of valid (non-padded) frames.
        blank_id: CTC blank / ASG repeat index.
        For FA, its ``None`` and every phone change is a boundary.

    Returns:
        Boolean list of length ``valid_len``; ``True`` marks a phone onset.
    """
    flags = [False] * valid_len
    if valid_len == 0:
        return flags

    # Case for Forced Alignment
    if blank_id is None:
        flags[0] = True  # why?
        for i in range(1, valid_len):
            if preds[i] != preds[i - 1]:
                flags[i] = True
        return flags

    # This is the logic for CTC/ASG (first prediction marks the onset)
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
        ``List[SegmentationUnit]`` with ``label=None`` (boundary-only; no phone
        class info).
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
    """Convert per-frame labels to ``SegmentationUnit`` segments.

    Args:
        frame_labels: Length ``valid_len``; class id for each frame.
        valid_len: Number of valid (non-padded) frames.
        points_by_frames: Audio points per frame (#points/#frames).
        sampling_rate: Audio sampling rate in Hz.
        token_list: Optional list mapping class ids to strings; 
            if provided, will be used to populate the ``label`` field of the output units, 
            else the raw class id will be used.
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
            #TODO(shikhar): remove entries with label = blank id
            units.append(
                SegmentationUnit(
                    start=start * points_by_frames / sampling_rate,
                    end=i * points_by_frames / sampling_rate,
                    label=current_label if token_list is None else token_list[current_label],
                )
            )
            start = i
            current_label = frame_labels[i]
    # add last
    units.append(
        SegmentationUnit(
            start=start * points_by_frames / sampling_rate,
            end=valid_len * points_by_frames / sampling_rate,
            label=current_label if token_list is None else token_list[current_label],
        )
    )
    return units

def target_boundaries_to_gt_units(
    target_start_idx: torch.Tensor,
    target_end_idx: torch.Tensor,
    target_length: torch.Tensor,
    feature_lens: torch.Tensor,
    points_by_frames: float,
    sampling_rate: int,
) -> dict[str, List[SegmentationUnit]]:
    """Build per-utterance ground-truth segments from target starts and ends.

    Args:
        target_start_idx: ``(B, N_max)`` frame-space target-start indices.
        target_end_idx: ``(B, N_max)`` frame-space target-end indices.
        target_length: ``(B,)`` number of phones per utterance.
        feature_lens: ``(B,)`` valid frame counts (used to close the last
            segment of each utterance).
        points_by_frames: Audio points per frame.
        sampling_rate: Audio sampling rate in Hz.

    Returns:
        ``{str(b): [SegmentationUnit, ...]}`` keyed by batch index.
    """
    gt: dict[str, List[SegmentationUnit]] = {}
    B = target_start_idx.shape[0]
    for b in range(B):
        n = int(target_length[b])
        vlen = int(feature_lens[b])
        starts = target_start_idx[b, :n].tolist()
        ends = target_end_idx[b, :n].tolist()
        gt[str(b)] = [
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
    """Build per-utterance ground-truth segments from phone-start indices.

    Args:
        target_start_idx: ``(B, N_max)`` frame-space phone-start indices.
        target_length: ``(B,)`` number of phones per utterance.
        feature_lens: ``(B,)`` valid frame counts (used to close the last
            segment of each utterance).
        points_by_frames: Audio points per frame.
        sampling_rate: Audio sampling rate in Hz.

    Returns:
        ``{str(b): [SegmentationUnit, ...]}`` keyed by batch index.
    """
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


### Evaluation utilities


def evaluate_boundaries(
    evaluator: SegmentationEvaluator,
    preds_dict: dict[str, List[SegmentationUnit]],
    gt_dict: dict[str, List[SegmentationUnit]],
    metrics: Optional[tuple[str]] = ("precision", "recall", "f1", "rval"),
) -> dict[str, float]:
    """Run evaluator and extract the standard P/R/F1/rval scalar metrics."""
    results = evaluator.evaluate_batch(preds_dict, gt_dict)
    return {k: evaluator._get_metric(results, k, 0.0) for k in metrics}


@torch.no_grad()
def boundary_rval_metrics(
    logits: torch.Tensor,
    feature_lens: torch.Tensor,
    batch: Mapping[str, Any],
    evaluator: SegmentationEvaluator,
    points_by_frames: float,
    sampling_rate: int,
    blank_id: Optional[int] = None,
) -> dict[str, float]:
    """Computes P/R/F1/rval from argmax phone predictions and ground truth by calling other utils here."""
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
    return evaluate_boundaries(evaluator, preds_dict, gt_dict)
