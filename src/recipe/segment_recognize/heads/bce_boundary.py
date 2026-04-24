"""BCE boundary detection head."""

from collections.abc import Mapping
from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.recipe.segmentation.boundary_utils import (
    boundaries_to_units,
    evaluate_boundaries,
    phone_starts_to_gt_units,
)
from src.recipe.segmentation.segmentation_loss import BoundaryLoss
from src.recipe.segment_recognize.heads.base import TaskHead


class BCEBoundaryHead(TaskHead):
    """Binary boundary detection head.

    Boundary head is a linear projection to map encoder features 
    into per-frame binary boundary logits.

    Attributes:
        boundary_head: Linear projection ``(D) -> (1)``.
        criterion: Masked BCE loss on boundary frames.
        evaluator: Boundary-level metric evaluator (shared).
        effective_pbf: Audio samples per encoder frame.
        audio_sr: Audio sample rate in Hz.
    """

    log_name = "bce"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        encoder_dim: int,
        evaluator: SegmentationEvaluator,
        pos_weight: float = 1.0,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
        self.boundary_head = nn.Linear(encoder_dim, 1)
        self.criterion = BoundaryLoss(pos_weight=pos_weight)
        self.evaluator = evaluator
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
                ``target_length``.

        Returns:
            Dict with ``loss`` and ``boundary_logits``.
        """
        logits = self.boundary_head(features).squeeze(-1)
        out = self.criterion(
            logits,
            feature_lens,
            batch["target_start_idx"],
            batch["target_length"],
        )
        out["boundary_logits"] = logits
        return out

    def _preds_dict(
        self,
        boundary_logits: torch.Tensor,
        feature_lens: torch.Tensor,
        boundary_threshold: float = 0.0,
    ) -> dict[str, List[SegmentationUnit]]:
        pbf, sr = self.effective_pbf, self.audio_sr
        out: dict[str, List[SegmentationUnit]] = {}
        for b in range(boundary_logits.shape[0]):
            vlen = int(feature_lens[b])
            flags = (boundary_logits[b, :vlen] > boundary_threshold).tolist()
            out[str(b)] = boundaries_to_units(flags, vlen, pbf, sr)
        return out

    @torch.no_grad()
    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Boundary P/R/F1/R-value via the shared evaluator."""
        preds_dict = self._preds_dict(output["boundary_logits"], feature_lens)
        gt_dict = phone_starts_to_gt_units(
            batch["target_start_idx"],
            batch["target_length"],
            feature_lens,
            self.effective_pbf,
            self.audio_sr,
        )
        return evaluate_boundaries(self.evaluator, preds_dict, gt_dict)

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        boundary_threshold: float = 0.0,
        **ctx: Any,
    ) -> List[Dict[str, Any]]:
        """Decode boundary logits into per-utterance segmentation dicts."""
        logits = self.boundary_head(features)
        preds_dict = self._preds_dict(logits, feature_lens, boundary_threshold)
        return [{"utterance_id": str(i), "pred_units": units} for i, units in preds_dict.items()]
