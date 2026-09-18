"""CTC phone-recognition head."""

from collections.abc import Mapping
from typing import Any, Dict, List

import torch

from src.metrics import evaluate_boundaries
from src.recipe.common.error_calculator import ErrorCalculator
from src.recipe.common.decode_align_strategy import DecodeAlignStrategy
from src.recipe.common.forced_alignment_strategy import ForcedAlignmentInference
from src.recipe.common.greedy_ctc_strategy import GreedyCTCInference
from src.recipe.common.boundary_utils import (
    frame_label_to_units,
    redistribute_blank_segments,
    strip_outer_brackets,
    target_boundaries_to_gt_units,
)
from src.recipe.segment_recognize.heads.base import TaskHead


class CTCRecognitionHead(TaskHead):
    """CTC phone-recognition head.

    Attributes:
        effective_pbf: Audio samples per encoder frame.
        audio_sr: Audio sample rate in Hz.
    """

    log_name = "ctc"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
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
        assert hasattr(
            net, "_calc_ctc_loss"
        ), "CTCRecognitionHead requires net._calc_ctc_loss"
        # Lazily build the recognition error calculator (needs the net's vocab);
        # cached as a plain object so it is not registered as a submodule.
        if getattr(self, "_err_calc", None) is None:
            self._err_calc = ErrorCalculator(
                list(net.token_list), net.get_blank_id(), log_phone_metrics=True
            )
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
            "logits": logits.detach(),
        }
        if stats:
            out.update(stats)
        return out

    # Helpers
    @torch.no_grad()
    def _decode_with_alignment(
        self,
        logits: torch.Tensor,
        feature_lens: torch.Tensor,
        blank_id: int,
        token_list: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """2-pass decode via DecodeAlignStrategy: greedy CTC then forced alignment.

        Args:
            logits: ``(B, T, C)`` unnormalised frame logits.
            feature_lens: ``(B,)`` valid frame counts.
            blank_id: CTC blank token index.
            token_list: Vocabulary list; defaults to ``str(id)`` when ``None``.

        Returns:
            List of ``{"ids": List[int], "flags": List[bool]}`` per utterance.
        """
        if token_list is None:
            # hack: does not matter for segmentation
            token_list = [str(i) for i in range(logits.shape[-1])]
        strategy = DecodeAlignStrategy(
            decode_strategy=GreedyCTCInference(
                token_list=token_list, blank_id=blank_id
            ),
            align_strategy=ForcedAlignmentInference(blank_idx=blank_id),
        )
        results = strategy(
            net=None,
            speech=None,
            speech_lengths=None,
            logits=logits,
            feature_lens=feature_lens,
        )

        # postprocess
        processed_results = [{"labels": r["aligned_labels"]} for r in results]
        # convert into segmentation units
        pbf, sr = self.effective_pbf, self.audio_sr
        blank_label = token_list[blank_id]
        for res in processed_results:
            labels = res["labels"].tolist()
            units = frame_label_to_units(
                labels, len(labels), pbf, sr, token_list
            )
            # remove long blank runs, then drop outer-silence brackets
            res["boundaries"] = strip_outer_brackets(
                redistribute_blank_segments(units, blank_label)
            )
        return processed_results

    @torch.no_grad()
    def _rval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """2-pass boundary rval when supervision is present, else {}."""
        if "target_start_idx" not in batch:
            return {}
        decoded = self._decode_with_alignment(
            output["logits"], feature_lens, blank_id=0
        )
        preds_dict = {}
        for b, res in enumerate(decoded):
            preds_dict[batch["utt_id"][b]] = res["boundaries"]
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
    def _per_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Recognition PER/PFER/CER via greedy CTC decode (fractions, 0-1).

        Validation-only: panphon-based PER is too costly to run every train
        step (``eval_metrics`` is invoked on both). The head is in eval mode
        only during validation, so ``self.training`` gates it.
        """
        if self.training:
            return {}
        if "target" not in batch or getattr(self, "_err_calc", None) is None:
            return {}
        ys_hat = output["logits"].argmax(-1)  # (B, T) frame token ids
        # Mask padded frames to blank so CTC collapse never emits their tokens.
        frame_idx = torch.arange(ys_hat.size(1), device=ys_hat.device)
        pad = frame_idx.unsqueeze(0) >= feature_lens.unsqueeze(1)
        ys_hat = ys_hat.masked_fill(pad, self._err_calc.blank_id)
        m = self._err_calc(ys_hat, batch["target"], batch["target_length"])
        # ErrorCalculator reports percentages; normalize to match eval_recognition.
        return {k: m[k] / 100.0 for k in ("per", "pfer", "cer") if k in m}

    # helpers, end

    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """CTC scalar stats + boundary rval + recognition PER metrics."""
        metrics: dict[str, float] = {}
        for k, v in output.items():
            if k in ("loss", "logits"):
                continue
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif isinstance(v, torch.Tensor) and v.ndim == 0:
                metrics[k] = v.item()
        metrics.update(self._rval_metrics(output, feature_lens, batch))
        metrics.update(self._per_metrics(output, feature_lens, batch))
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
        """2-pass CTC decode: greedy phone recognition + forced-alignment boundaries.

        Returns one dict per utterance with ``phone_ids``, ``target``,
        ``transcript``, and segmentation ``boundaries``.
        """
        token_list = net.token_list
        blank_id = net.get_blank_id()
        decoded = self._decode_with_alignment(
            net.ctc.ctc_lo(features), feature_lens, blank_id, token_list
        )
        out = {}
        for b, res in enumerate(decoded):
            out[batch["utt_id"][b]] = res["boundaries"]
        return out
