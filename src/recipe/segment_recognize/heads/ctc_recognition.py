"""CTC phone-recognition head."""

from collections.abc import Mapping
from typing import Any, Dict, List

import torch

from src.metrics.segmentation_evaluator import SegmentationEvaluator
from src.recipe.common.greedy_ctc_strategy import (
    ctc_collapse_vectorized,
)
from src.recipe.common.boundary_utils import (
    argmax_to_boundaries,
    boundaries_to_units,
    boundary_rval_metrics,
)
from src.recipe.segment_recognize.heads.base import TaskHead
# NOTE(shikhar): This head only works for pxeus and xeus nets for now.

class CTCRecognitionHead(TaskHead):
    """CTC phone-recognition head.

    Delegates to ``net._calc_ctc_loss``.

    The forward output includes the raw stats dict returned by the
    encoder's CTC module (e.g. ``loss_ctc``, ``cer_ctc``), which
    ``eval_metrics`` extracts as scalars for logging.  Additionally
    stores CTC logits for boundary-based rval evaluation.

    Attributes:
        evaluator: Shared boundary-level metric evaluator.
        effective_pbf: Audio samples per encoder frame.
        audio_sr: Audio sample rate in Hz.
    """

    log_name = "ctc"
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
        """Compute CTC loss via the encoder's CTC module.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``target`` and ``target_length``.
            net: Encoder module exposing ``_calc_ctc_loss``.

        Returns:
            Dict with ``loss``, CTC stats, and ``logits``.
        """
        assert hasattr(net, "_calc_ctc_loss"), "CTCRecognitionHead requires net._calc_ctc_loss"
        loss, stats, logits = net._calc_ctc_loss(
            features,
            feature_lens,
            batch["target"],
            batch["target_length"],
            lang_sym=batch.get("lang_sym"),
            return_logits=True,
        )
        out: dict[str, Any] = {
            "loss": loss,
            "logits": logits,
        }
        if stats:
            out.update(stats)
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
            blank_id=0,
        )

    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """CTC scalar stats + boundary rval metrics."""
        metrics: dict[str, float] = {}
        for k, v in output.items():
            if k in ("loss", "logits"):
                continue
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif isinstance(v, torch.Tensor) and v.ndim == 0:
                metrics[k] = v.item()
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
    ) -> List[Dict[str, Any]]:
        """Greedy CTC decode + per-frame boundaries.
        #TODO(shikhar): add 2 pass approach for onset/offset decoding
        #TODO(shikhar): refactor this and eval metrics to a shared CTC decoding fn
        # 1. Use greedy ctc decoding to get raw phone predictions
        # 2. Use forced alignment strategy with (1)

        Returns one dict per utterance with phone recognition output
        (``phone_ids``, ``target``, ``transcript``) and segmentation
        ``boundaries`` derived from argmax phone-change frames.
        """
        y_hat = torch.argmax(net.ctc.ctc_lo(features), dim=-1)
        blank_id = net.get_blank_id()
        collapsed = ctc_collapse_vectorized(y_hat, blank_id)
        token_list = net.token_list
        pbf, sr = self.effective_pbf, self.audio_sr
        out: List[Dict[str, Any]] = []
        for b, ids in enumerate(collapsed):
            vlen = int(feature_lens[b])
            preds = y_hat[b, :vlen].tolist()
            flags = argmax_to_boundaries(preds, vlen, blank_id=blank_id)
            out.append({
                "phone_ids": ids,
                "target": [token_list[i] for i in ids],
                "transcript": "/".join(token_list[i] for i in ids),
                "boundaries": boundaries_to_units(flags, vlen, pbf, sr),
            })
        return out
