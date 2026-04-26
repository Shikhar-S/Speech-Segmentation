"""Frame-wise cross-entropy segmentation head."""

from collections.abc import Mapping
from typing import Any, List

import torch

from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit
from src.recipe.common.boundary_utils import evaluate_boundaries, target_boundaries_to_gt_units
from src.recipe.segmentation.segmentation_loss import SegmentationLoss
from src.recipe.segment_recognize.heads.base import TaskHead
from src.recipe.common.boundary_utils import frame_label_to_units


class FCESegmentationHead(TaskHead):
    """Frame cross-entropy head.
    
    Attributes:
        criterion: Frame-wise alignment loss.
        evaluator: Shared boundary-level metric evaluator.
        effective_pbf: Audio samples per encoder frame.
        audio_sr: Audio sample rate in Hz.
    """

    log_name = "framece"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        evaluator: SegmentationEvaluator,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
        self.criterion = SegmentationLoss()
        self.evaluator = evaluator
        self.effective_pbf = effective_pbf
        self.audio_sr = audio_sr

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        net: torch.nn.Module,
        **ctx: Any,
    ) -> dict[str, Any]:
        """Compute frame-wise cross-entropy loss.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``target``, ``target_start_idx``,
                ``target_end_idx``, and ``target_length``.
            net: Encoder module whose ``ctc.ctc_lo`` provides the
                phone-level projection.

        Returns:
            Dict with ``loss``, ``accuracy``, and ``logits``.
        """
        logits = net.ctc.ctc_lo(features)
        out = self.criterion(
            logits,
            feature_lens,
            batch["target"],
            batch["target_start_idx"],
            batch["target_end_idx"],
            batch["target_length"],
        )
        out["logits"] = logits
        return out
    
    def _process_predictions(
        self,
        logits: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> dict[str, List[SegmentationUnit]]:
        """Convert frame-level logits into per-utterance segmentation units."""
        B = logits.shape[0]
        preds_dict = {}
        for b in range(B):
            frame_labels = logits[b].argmax(dim=-1)[:int(feature_lens[b])].tolist()
            preds_dict[str(b)] = frame_label_to_units(frame_labels, feature_lens[b], self.effective_pbf, self.audio_sr)
        return preds_dict

    @torch.no_grad()
    def _rval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Run boundary rval when supervision is present, else {}."""
        if "target_start_idx" not in batch:
            return {}
        gt_dict = target_boundaries_to_gt_units(
            batch["target_start_idx"],
            batch["target_end_idx"],
            batch["target_length"],
            feature_lens,
            self.effective_pbf,
            self.audio_sr,
        )
        pred_dict = self._process_predictions(output["logits"], feature_lens)
        print(pred_dict)
        return evaluate_boundaries(self.evaluator, pred_dict, gt_dict)

    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Frame-level accuracy + boundary rval metrics."""
        metrics: dict[str, float] = {}
        acc = output.get("accuracy")
        if acc is not None:
            metrics["frame_accuracy"] = acc
        metrics.update(self._rval_metrics(output, feature_lens, batch))
        return metrics

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        net: torch.nn.Module,
        **ctx: Any,
    ) -> None:
        """Run greedy decode strategy by using argmax per-frame."""
        results = self._process_predictions(net.ctc.ctc_lo(features), feature_lens)
        return results
