"""Inference-side dataclass for segment boundaries.

Kept in-repo (rather than re-using :class:`phone_metrics.Seg`) because the
JSONL inference shards produced by :mod:`src.core.distributed_inference`
encode each segment as ``{"start", "end", "label"}`` and ten-plus inference
backends construct units positionally as ``SegmentationUnit(start, end, label)``.
``phone_metrics.Seg`` has both ``raw_label`` and ``ipa_label``; bridging that
to the wire format would touch every backend without adding signal.

Conversion to :class:`phone_metrics.Seg` happens at the eval boundary in
:func:`src.metrics.evaluate_boundaries` and :func:`src.metrics.unit_to_seg`.
"""

from dataclasses import dataclass


@dataclass
class SegmentationUnit:
    """Single aligned span (phone, word, etc.) on the time axis.

    Attributes:
        start: Segment start in seconds (or sample index for raw outputs).
        end: Segment end in seconds (or sample index for raw outputs).
        label: Optional phone/word label. ``None`` for boundary-only
            outputs (no class info).
    """

    start: int | float
    end: int | float
    label: str | int | None = None
