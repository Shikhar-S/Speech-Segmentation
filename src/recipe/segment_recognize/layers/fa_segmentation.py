"""Frame-wise forced-alignment segmentation loss module."""

from collections.abc import Mapping
from typing import Any

import torch

from src.recipe.segmentation.segmentation_loss import SegmentationLoss
from src.recipe.segment_recognize.layers.base import LossModule


class FASegmentationLoss(LossModule):
    """Frame-level forced-alignment loss.

    No owned parameters -- uses ``net.ctc.ctc_lo`` (passed via
    context) to project encoder features into CTC logits, then
    applies ``SegmentationLoss`` to predict which target phone
    corresponds to each frame.

    Attributes:
        criterion: Frame-wise alignment loss.
    """

    log_name = "fa"

    def __init__(self, weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(weight)
        self.criterion = SegmentationLoss()

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        net: torch.nn.Module,
        **ctx: Any,
    ) -> dict[str, Any]:
        """Compute forced-alignment loss.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``text``, ``target_start_idx``,
                ``target_end_idx``, and ``text_length``.
            net: Encoder module whose ``ctc.ctc_lo`` provides the
                phone-level projection.

        Returns:
            Dict with ``loss`` and ``accuracy``.
        """
        logits = net.ctc.ctc_lo(features)
        return self.criterion(
            logits,
            feature_lens,
            batch["text"],
            batch["target_start_idx"],
            batch["target_end_idx"],
            batch["text_length"],
        )

    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Return frame-level accuracy from forward output."""
        acc = output.get("accuracy")
        if acc is not None:
            return {"fa_accuracy": acc}
        return {}
