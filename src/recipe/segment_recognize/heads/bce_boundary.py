"""BCE boundary detection head.

This applies binary cross-entropy on per-frame boundary logits.
During decoding it applies a threshold to get boundary flags, then
converts those into segmentation units.
"""

from collections.abc import Mapping
from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.metrics import evaluate_boundaries
from src.metrics.types import SegmentationUnit
from src.recipe.common.boundary_utils import (
    boundaries_to_units,
    strip_outer_brackets,
    target_boundaries_to_gt_units,
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
        weight: float = 1.0,
        boundary_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
        self.boundary_head = nn.Linear(encoder_dim, 1)
        self.criterion = BoundaryLoss(pos_weight=pos_weight)
        self.effective_pbf = effective_pbf
        self.audio_sr = audio_sr
        self.boundary_threshold = boundary_threshold

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
        out["boundary_logits"] = logits.detach()
        return out

    def _process_predictions(
        self,
        boundary_logits: torch.Tensor,
        feature_lens: torch.Tensor,
        utt_id: List[str],
        boundary_threshold: float = 0.5,
    ) -> dict[str, List[SegmentationUnit]]:
        out: dict[str, List[SegmentationUnit]] = {}
        boundary_prob = torch.sigmoid(boundary_logits)  # B, T
        bs = boundary_logits.shape[0]
        for b in range(bs):
            vlen = int(feature_lens[b])
            flags = (boundary_prob[b, :vlen] > boundary_threshold).tolist()
            out[utt_id[b]] = strip_outer_brackets(
                boundaries_to_units(
                    flags, vlen, self.effective_pbf, self.audio_sr
                )
            )
        return out

    @torch.no_grad()
    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Boundary P/R/F1/R-value via phone_metrics.PrecisionRecallMetric."""
        preds_dict = self._process_predictions(
            output["boundary_logits"], feature_lens, batch["utt_id"]
        )
        gt_dict = target_boundaries_to_gt_units(
            batch["target_start_idx"],
            batch["target_end_idx"],
            batch["target_length"],
            feature_lens,
            self.effective_pbf,
            self.audio_sr,
            batch["utt_id"],
        )
        return evaluate_boundaries(preds_dict, gt_dict)

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> List[Dict[str, Any]]:
        """Decode boundary logits into per-utterance segmentation dicts."""
        logits = self.boundary_head(features).squeeze(-1).detach()
        preds_dict = self._process_predictions(
            logits, feature_lens, batch["utt_id"], self.boundary_threshold
        )
        return preds_dict
