"""ASG phone-recognition head."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from src.recipe.segment_recognize.heads.base import TaskHead
from src.recipe.segment_recognize.losses.asg import (
    AutoSegmentationCriterion,
)


class ASGRecognitionHead(TaskHead):
    """ASG phone-recognition head with learnable transitions.

    Owns a linear projection from encoder dim to label space and
    an ``AutoSegmentationCriterion`` instance.  Unlike the CTC
    head, this head does **not** delegate to the encoder — it is
    fully self-contained.

    Args:
        encoder_dim: Encoder output dimensionality (injected).
        num_labels: Vocabulary size including the repeat token.
        repeat_idx: Index used for the repeat token.
        use_transitions: Learn label-to-label transition scores.
        use_double_scores: Use float64 for DP precision.
        weight: Loss weight in the multi-task sum.
    """

    log_name = "asg"

    def __init__(
        self,
        encoder_dim: int,
        num_labels: int,
        repeat_idx: int = 0,
        use_transitions: bool = True,
        use_double_scores: bool = False,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
        self.proj = nn.Linear(encoder_dim, num_labels)
        self.criterion = AutoSegmentationCriterion(
            num_labels=num_labels,
            repeat_idx=repeat_idx,
            use_transitions=use_transitions,
            use_double_scores=use_double_scores,
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> dict[str, Any]:
        """Compute ASG loss.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``text`` and ``text_length``.

        Returns:
            Dict with ``loss`` (scalar).
        """
        logits = self.proj(features)
        loss_per_utt = self.criterion(
            logits,
            batch["text"],
            feature_lens,
            batch["text_length"],
        )
        return {"loss": loss_per_utt.mean()}
