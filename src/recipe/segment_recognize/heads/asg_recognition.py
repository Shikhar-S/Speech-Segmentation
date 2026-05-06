"""ASG phone-recognition head.
#TODO(shikhar): Check this head later!
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.metrics.segmentation_evaluator import SegmentationEvaluator
from src.recipe.common.boundary_utils import (
    boundaries_to_units,
    boundary_rval_metrics,
)
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
        evaluator: Shared boundary-level metric evaluator (injected).
        num_labels: Vocabulary size including the repeat token.
        repeat_idx: Index used for the repeat token.
        use_transitions: Learn label-to-label transition scores.
        use_double_scores: Use float64 for DP precision.
        effective_pbf: Audio samples per encoder frame.
        audio_sr: Audio sample rate in Hz.
        weight: Loss weight in the multi-task sum.
    """

    log_name = "asg"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        encoder_dim: int,
        evaluator: SegmentationEvaluator,
        num_labels: int,
        repeat_idx: int = 0,
        use_transitions: bool = True,
        use_double_scores: bool = False,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
        self.repeat_idx = repeat_idx
        self.proj = nn.Linear(encoder_dim, num_labels)
        self.criterion = AutoSegmentationCriterion(
            num_labels=num_labels,
            repeat_idx=repeat_idx,
            use_transitions=use_transitions,
            use_double_scores=use_double_scores,
        )
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
        """Compute ASG loss.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``target`` and ``target_length``.

        Returns:
            Dict with ``loss`` and ``logits``.
        """
        logits = self.proj(features)
        loss_per_utt = self.criterion(
            logits,
            batch["target"],
            feature_lens,
            batch["target_length"],
        )
        return {"loss": loss_per_utt.mean(), "logits": logits}

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
            output["logits"],
            feature_lens,
            batch,
            self.evaluator,
            self.effective_pbf,
            self.audio_sr,
            blank_id=self.repeat_idx,
        )

    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Boundary rval metrics from argmax phone predictions."""
        return self._rval_metrics(output, feature_lens, batch)

    def _argmax_to_boundaries(self, preds: List[int]) -> List[bool]:
        """ASG-specific phone-onset flags from per-frame argmax.
        Args:
            preds: Per-frame argmax class ids.
        Returns:
            Boolean list of the same length; ``True`` marks a phone onset.
            Example:
            For repeat_idx=0, [3, 3, 0, 0, 5, 5, 0] -> [T,F,T,T,T,F,T]
        """
        flags = [False] * len(preds)
        prev_phone: Optional[int] = None
        for i, p in enumerate(preds):
            if prev_phone is None:
                # first frame is always a boundary
                flags[i] = True
                prev_phone = p
                continue
            if p != prev_phone or p == self.repeat_idx:
                flags[i] = True
                prev_phone = p
        return flags

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Greedy ASG decode + per-frame boundaries.
        Returns ``{utt_id: boundaries}`` per utterance.
        """
        y_hat = torch.argmax(self.proj(features), dim=-1)
        pbf, sr = self.effective_pbf, self.audio_sr
        out: Dict[str, List[Dict[str, Any]]] = {}
        for b in range(y_hat.size(0)):
            vlen = int(feature_lens[b])
            preds = y_hat[b, :vlen].tolist()
            flags = self._argmax_to_boundaries(preds)
            out[batch["utt_id"][b]] = boundaries_to_units(flags, vlen, pbf, sr)
        return out
