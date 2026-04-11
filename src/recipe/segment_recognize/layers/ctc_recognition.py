"""CTC phone-recognition loss module."""

from collections.abc import Mapping
from typing import Any

import torch

from src.recipe.segment_recognize.layers.base import LossModule


class CTCRecognitionLoss(LossModule):
    """CTC phone-recognition loss.

    Owns no parameters -- delegates entirely to
    ``net._calc_ctc_loss`` (passed via context).

    The forward output includes the raw stats dict returned by the
    encoder's CTC module (e.g. ``loss_ctc``, ``cer_ctc``), which
    ``eval_metrics`` extracts as scalars for logging.
    """

    log_name = "pr"

    def __init__(self, weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(weight)

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        *,
        net: torch.nn.Module,
        **ctx: Any,
    ) -> dict[str, Any]:
        """Compute CTC loss via the encoder's CTC module.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``text`` and ``text_length``.
            net: Encoder module exposing ``_calc_ctc_loss``.

        Returns:
            Dict with ``loss`` and CTC stats.
        """
        loss, stats = net._calc_ctc_loss(
            features,
            feature_lens,
            batch["text"],
            batch["text_length"],
            lang_sym=batch.get("lang_sym"),
        )
        out: dict[str, Any] = {"loss": loss}
        if stats:
            out.update(stats)
        return out

    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Extract scalar CTC stats (``loss_ctc``, ``cer_ctc``)."""
        metrics: dict[str, float] = {}
        for k, v in output.items():
            if k == "loss":
                continue
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif isinstance(v, torch.Tensor) and v.ndim == 0:
                metrics[k] = v.item()
        return metrics
