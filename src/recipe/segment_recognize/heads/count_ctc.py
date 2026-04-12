"""CountCTC boundary detection head.

Replaces every phone in the transcript with a fixed, user-defined
token sequence (the ``substitution`` string), then applies standard
CTC loss over a 3-or-fewer-class vocab built dynamically from those
tokens.  CTC's peaky alignment lets the model discover phone-level
structure without time-aligned supervision.

The head has no built-in semantics for "boundary" vs. "interior" --
it only does literal token substitution and CTC.  Optionally, the
caller can designate one of the substitution tokens as the
``boundary_token``; when set, ``eval_metrics`` decodes argmax frames
of that class as predicted phone onsets and reports P/R/F1/R-value
against ground-truth boundaries (when available).
"""

from collections.abc import Mapping
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from src.metrics.segmentation_evaluator import (
    SegmentationEvaluator,
    SegmentationUnit,
)
from src.recipe.segmentation.inference import (
    _boundary_flags_to_units,
)
from src.recipe.segment_recognize.heads.base import TaskHead
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_BLANK_ID = 0


def _parse_substitution(
    substitution: str,
) -> Tuple[dict[str, int], List[int]]:
    """Parse a space-separated substitution string into vocab + id sequence.

    Args:
        substitution: e.g. ``"1 0"`` or ``"1"`` or ``"a b c"``.

    Returns:
        vocab: ``{token: class_id}`` with ids assigned in order of
            first appearance, starting at 1 (0 is reserved for blank).
        per_phone_seq: List of class ids in the order tokens appear in
            ``substitution``.  Length equals the number of tokens.

    Raises:
        ValueError: If ``substitution`` is empty or only whitespace.
    """
    tokens = substitution.split()
    if not tokens:
        raise ValueError(
            "CountCTCHead: `substitution` must be a non-empty "
            "space-separated string."
        )
    vocab: dict[str, int] = {}
    next_id = 1  # 0 reserved for blank
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
    per_phone_seq = [vocab[tok] for tok in tokens]
    return vocab, per_phone_seq


def _build_targets(
    text_length: torch.Tensor,
    per_phone_seq: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build flat CTC targets by replacing each phone with ``per_phone_seq``.

    Args:
        text_length: ``(B,)`` number of phones per utterance.
        per_phone_seq: Class ids that replace each phone (length L).

    Returns:
        targets_flat: ``(sum(L * N_i),)`` 1-D long tensor.
        target_lengths: ``(B,)`` long tensor of ``L * text_length``.
    """
    L = len(per_phone_seq)
    target_lengths = L * text_length
    pat = torch.tensor(per_phone_seq, dtype=torch.long)
    pieces = [pat.repeat(int(n)) for n in text_length.tolist()]
    targets_flat = (
        torch.cat(pieces)
        if pieces
        else torch.empty(0, dtype=torch.long)
    )
    return targets_flat, target_lengths


def _ctc_collapse(preds: List[int]) -> List[int]:
    """Standard CTC collapse: drop consecutive duplicates, then blanks."""
    collapsed: List[int] = []
    prev: Optional[int] = None
    for p in preds:
        if p != prev:
            prev = p
            if p != _BLANK_ID:
                collapsed.append(p)
    return collapsed


class CountCTCHead(TaskHead):
    """CountCTC head: substitution-based CTC loss for unaligned boundary supervision.

    Builds a private 3-or-fewer-class vocabulary from a user-supplied
    ``substitution`` string and trains a CTC head over that vocab.
    Useful for joint training with ``BCEBoundaryHead`` to leverage
    transcription-only data alongside time-aligned segmentation data.

    Attributes:
        substitution: The space-separated token string driving the
            substitution + vocab.
        vocab: ``{token: class_id}`` derived from ``substitution``.
        per_phone_seq: Class ids substituted in for each phone.
        num_classes: Vocab size including the CTC blank (index 0).
        boundary_token: Optional token marking phone onsets for eval.
        boundary_class_id: Class id of ``boundary_token`` (or ``None``).
        proj: ``Linear(encoder_dim, num_classes)`` projection.
        ctc_loss: ``nn.CTCLoss`` instance with ``blank=0``.
        evaluator: ``SegmentationEvaluator`` for boundary metrics.
    """

    log_name = "count_ctc"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        encoder_dim: int,
        substitution: str = "1 0",
        boundary_token: Optional[str] = None,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
        tolerance_ms: int = 20,
        weight: float = 1.0,
        zero_infinity: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)

        self.substitution = substitution
        self.boundary_token = boundary_token
        self.vocab, self.per_phone_seq = _parse_substitution(substitution)
        self.num_classes = 1 + len(self.vocab)

        if boundary_token is not None:
            if boundary_token not in self.vocab:
                raise ValueError(
                    f"CountCTCHead: boundary_token={boundary_token!r} "
                    f"is not in substitution vocab {sorted(self.vocab)}."
                )
            self.boundary_class_id: Optional[int] = self.vocab[boundary_token]
        else:
            self.boundary_class_id = None

        self.proj = nn.Linear(encoder_dim, self.num_classes)
        self.ctc_loss = nn.CTCLoss(
            blank=_BLANK_ID,
            reduction="none",
            zero_infinity=zero_infinity,
        )
        self.evaluator = SegmentationEvaluator(
            tolerance_ms=tolerance_ms,
        )
        self.effective_pbf = effective_pbf
        self.audio_sr = audio_sr

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> dict[str, Any]:
        """Compute CountCTC loss.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``text_length``.

        Returns:
            Dict with ``loss`` and ``logits`` (the latter for eval).
        """
        text_length = batch["text_length"]
        logits = self.proj(features)  # (B, T, C)

        L = len(self.per_phone_seq)
        required = L * text_length
        valid = required <= feature_lens

        if not valid.any():
            log.warning(
                "CountCTCHead: every utterance violates L*N <= T "
                "(L=%d); returning zero loss.",
                L,
            )
            return {
                "loss": torch.zeros(
                    (), device=features.device, requires_grad=True,
                ),
                "logits": logits,
            }

        if not valid.all():
            n_skip = int((~valid).sum().item())
            log.warning(
                "CountCTCHead: skipping %d/%d utterances where L*N > T.",
                n_skip,
                int(valid.numel()),
            )

        logits_v = logits[valid]
        feat_lens_v = feature_lens[valid]
        text_len_v = text_length[valid]

        targets_flat, target_lens = _build_targets(
            text_len_v.cpu(), self.per_phone_seq,
        )
        targets_flat = targets_flat.to(features.device)
        target_lens = target_lens.to(features.device)

        # CTCLoss expects (T, B, C) log-probs.
        log_probs = logits_v.log_softmax(dim=-1).transpose(0, 1)

        loss_per_utt = self.ctc_loss(
            log_probs,
            targets_flat,
            feat_lens_v,
            target_lens,
        )
        loss = loss_per_utt.sum() / logits_v.size(0)

        return {"loss": loss, "logits": logits}

    def _decode_boundaries(
        self,
        logits: torch.Tensor,
        valid_len: int,
    ) -> Tuple[List[bool], int]:
        """Argmax-decode logits and extract boundary flags + count.

        Args:
            logits: ``(T, C)`` raw logits for one utterance.
            valid_len: Number of valid (non-padded) frames.

        Returns:
            is_boundary: Boolean flags marking frames whose argmax is
                ``boundary_class_id``.
            phone_count: Number of boundary tokens after CTC collapse.
        """
        boundary_id = self.boundary_class_id
        assert boundary_id is not None, (
            "_decode_boundaries called without a boundary_token configured."
        )
        preds = logits[:valid_len].argmax(dim=-1).tolist()
        collapsed = _ctc_collapse(preds)
        phone_count = sum(1 for c in collapsed if c == boundary_id)
        is_boundary = [p == boundary_id for p in preds]
        return is_boundary, phone_count

    @torch.no_grad()
    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Count + boundary metrics.

        Always reports ``count_accuracy`` and ``count_mae``.  When
        ``target_start_idx`` is present in the batch, additionally
        computes ``precision``, ``recall``, ``f1`` and ``rval`` via
        ``SegmentationEvaluator``.  Returns an empty dict when no
        ``boundary_token`` is configured.
        """
        if self.boundary_class_id is None:
            return {}

        logits = output["logits"]
        text_length = batch["text_length"]
        pbf, sr = self.effective_pbf, self.audio_sr
        B = logits.shape[0]
        has_gt = "target_start_idx" in batch

        preds_dict: dict[str, list[SegmentationUnit]] = {}
        gt_dict: dict[str, list[SegmentationUnit]] = {}
        count_correct = 0
        count_abs_err_sum = 0

        for b in range(B):
            vlen = int(feature_lens[b])
            is_boundary, pred_count = self._decode_boundaries(
                logits[b], vlen,
            )
            gt_count = int(text_length[b])
            count_abs_err_sum += abs(pred_count - gt_count)
            if pred_count == gt_count:
                count_correct += 1

            preds_dict[str(b)] = _boundary_flags_to_units(
                is_boundary, vlen, pbf, sr,
            )

            if has_gt:
                n = int(text_length[b])
                starts = batch["target_start_idx"][b, :n].tolist()
                gt_dict[str(b)] = [
                    SegmentationUnit(
                        start=starts[i] * pbf / sr,
                        end=(
                            starts[i + 1] if i + 1 < n else vlen
                        )
                        * pbf
                        / sr,
                        label=0,
                    )
                    for i in range(n)
                ]

        metrics: dict[str, float] = {
            "count_accuracy": count_correct / B,
            "count_mae": count_abs_err_sum / B,
        }
        if has_gt:
            results = self.evaluator.evaluate_batch(
                preds_dict, gt_dict,
            )
            for k in ("precision", "recall", "f1", "rval"):
                metrics[k] = self.evaluator._get_metric(
                    results, k, 0.0,
                )
        return metrics
