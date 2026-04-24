"""Frame-wise cross-entropy segmentation head."""

from collections.abc import Mapping
from typing import Any

import torch

from src.metrics.segmentation_evaluator import SegmentationEvaluator
from src.recipe.segmentation.boundary_utils import boundary_rval_metrics
from src.recipe.segmentation.segmentation_loss import SegmentationLoss
from src.recipe.segment_recognize.heads.base import TaskHead


class FASegmentationHead(TaskHead):
    """Frame-level cross-entropy head.

    No owned parameters -- uses ``net.ctc.ctc_lo`` (passed via
    context) to project encoder features into CTC logits, then
    applies ``SegmentationLoss`` to predict which target phone
    corresponds to each frame.

    Attributes:
        criterion: Frame-wise alignment loss.
        evaluator: Shared boundary-level metric evaluator.
        effective_pbf: Audio samples per encoder frame.
        audio_sr: Audio sample rate in Hz.
    """

    log_name = "fa"
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
        return boundary_rval_metrics(
            output["logits"], feature_lens, batch,
            self.evaluator, self.effective_pbf, self.audio_sr,
            blank_id=None,
        )

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
            metrics["fa_accuracy"] = acc
        metrics.update(self._rval_metrics(output, feature_lens, batch))
        return metrics

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> None:
        """FA head does not produce standalone decode output.

        Returns ``None`` so ``SegmentRecognizeModel.predict_step`` skips it.
        Use ``src.recipe.segmentation.inference.SegmentationInference`` to
        get frame-level alignments.
        """
        return None
