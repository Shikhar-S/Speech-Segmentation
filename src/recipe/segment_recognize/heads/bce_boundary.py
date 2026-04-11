"""BCE boundary detection head."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.recipe.segmentation.inference import (
    _boundary_flags_to_units,
)
from src.recipe.segmentation.segmentation_loss import BoundaryLoss
from src.recipe.segment_recognize.heads.base import TaskHead


class BCEBoundaryHead(TaskHead):
    """Binary boundary detection head.

    Owns a linear projection that maps encoder features to per-frame
    boundary logits, plus a ``BoundaryLoss`` criterion and a
    ``SegmentationEvaluator`` for P/R/F1/R-value metrics.

    Attributes:
        boundary_head: Linear projection ``(D) -> (1)``.
        criterion: Masked BCE loss on boundary frames.
        evaluator: Boundary-level metric evaluator.
        effective_pbf: Audio samples per encoder frame.
        audio_sr: Audio sample rate in Hz.
    """

    log_name = "bce"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        encoder_dim: int,
        pos_weight: float = 1.0,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
        tolerance_ms: int = 20,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
        self.boundary_head = nn.Linear(encoder_dim, 1)
        self.criterion = BoundaryLoss(pos_weight=pos_weight)
        self.evaluator = SegmentationEvaluator(
            tolerance_ms=tolerance_ms,
        )
        self.effective_pbf = effective_pbf
        self.audio_sr = audio_sr

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> dict[str, Any]:
        """Compute BCE boundary loss.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``target_start_idx`` and
                ``text_length``.

        Returns:
            Dict with ``loss`` and ``boundary_logits``.
        """
        logits = self.boundary_head(features).squeeze(-1)
        out = self.criterion(
            logits,
            feature_lens,
            batch["target_start_idx"],
            batch["text_length"],
        )
        out["boundary_logits"] = logits
        return out

    @torch.no_grad()
    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Boundary P/R/F1/R-value via ``SegmentationEvaluator``."""
        boundary_logits = output["boundary_logits"]
        target_start_idx = batch["target_start_idx"]
        target_len = batch["text_length"]
        pbf, sr = self.effective_pbf, self.audio_sr

        preds_dict: dict[str, list[SegmentationUnit]] = {}
        gt_dict: dict[str, list[SegmentationUnit]] = {}
        for b in range(boundary_logits.shape[0]):
            vlen = int(feature_lens[b])
            flags = (boundary_logits[b, :vlen] > 0).tolist()
            preds_dict[str(b)] = _boundary_flags_to_units(
                flags, vlen, pbf, sr,
            )
            n = int(target_len[b])
            starts = target_start_idx[b, :n].tolist()
            gt_dict[str(b)] = [
                SegmentationUnit(
                    start=starts[i] * pbf / sr,
                    end=(
                        starts[i + 1] if i + 1 < n else vlen
                    )
                    * pbf
                    / sr,
                    label=0,
                )
                for i in range(n)
            ]

        results = self.evaluator.evaluate_batch(
            preds_dict, gt_dict,
        )
        return {
            k: self.evaluator._get_metric(results, k, 0.0)
            for k in ("precision", "recall", "f1", "rval")
        }
