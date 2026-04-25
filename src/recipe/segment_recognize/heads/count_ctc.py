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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.metrics.segmentation_evaluator import SegmentationEvaluator
from src.recipe.common.greedy_ctc_strategy import (
    ctc_collapse_vectorized,
)
from src.recipe.common.boundary_utils import (
    boundaries_to_units,
    evaluate_boundaries,
    phone_starts_to_gt_units,
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
    target_length: torch.Tensor,
    per_phone_seq: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build flat CTC targets by replacing each phone with ``per_phone_seq``.

    Args:
        target_length: ``(B,)`` number of phones per utterance.
        per_phone_seq: Class ids that replace each phone (length L).

    Returns:
        targets_flat: ``(sum(L * N_i),)`` 1-D long tensor.
        target_lengths: ``(B,)`` long tensor of ``L * target_length``.
    """
    L = len(per_phone_seq)
    target_lengths = L * target_length
    pat = torch.tensor(per_phone_seq, dtype=torch.long)
    pieces = [pat.repeat(int(n)) for n in target_length.tolist()]
    targets_flat = (
        torch.cat(pieces)
        if pieces
        else torch.empty(0, dtype=torch.long)
    )
    return targets_flat, target_lengths


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
        evaluator: Shared ``SegmentationEvaluator`` for boundary metrics.
    """

    log_name = "count_ctc"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        encoder_dim: int,
        evaluator: SegmentationEvaluator,
        substitution: str = "1 0",
        boundary_token: Optional[str] = None,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
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
        """Compute CountCTC loss.

        Args:
            features: ``(B, T, D)`` encoder output.
            feature_lens: ``(B,)`` valid frame counts.
            batch: Must contain ``target_length``.

        Returns:
            Dict with ``loss`` and ``logits`` (the latter for eval).
        """
        target_length = batch["target_length"]
        logits = self.proj(features)  # (B, T, C)

        L = len(self.per_phone_seq)
        required = L * target_length
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
        phone_len_v = target_length[valid]

        targets_flat, target_lens = _build_targets(
            phone_len_v.cpu(), self.per_phone_seq,
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

    def _boundary_flags(
        self,
        logits_b: torch.Tensor,
        valid_len: int,
    ) -> List[bool]:
        """Per-frame flags: ``True`` where argmax equals ``boundary_class_id``."""
        preds = logits_b[:valid_len].argmax(dim=-1).tolist()
        return [p == self.boundary_class_id for p in preds]

    @torch.no_grad()
    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Count + boundary metrics.

        Always reports ``count_accuracy`` and ``count_mae``.  When
        ``phone_start_idx`` is present in the batch, additionally
        computes ``precision``, ``recall``, ``f1`` and ``rval`` via
        the shared evaluator.  Returns an empty dict when no
        ``boundary_token`` is configured.
        """
        bid = self.boundary_class_id
        if bid is None:
            return {}

        logits = output["logits"]
        target_length = batch["target_length"]
        pbf, sr = self.effective_pbf, self.audio_sr
        B = logits.shape[0]

        argmax = logits.argmax(dim=-1)
        collapsed = ctc_collapse_vectorized(argmax, blank_id=_BLANK_ID)
        pred_counts = [sum(1 for c in ids if c == bid) for ids in collapsed]
        gt_counts = target_length.tolist()
        count_abs_err_sum = sum(
            abs(p - int(g)) for p, g in zip(pred_counts, gt_counts)
        )
        count_correct = sum(
            1 for p, g in zip(pred_counts, gt_counts) if p == int(g)
        )

        metrics: dict[str, float] = {
            "count_accuracy": count_correct / B,
            "count_mae": count_abs_err_sum / B,
        }

        if "phone_start_idx" not in batch:
            return metrics

        preds_dict: dict[str, list] = {}
        for b in range(B):
            vlen = int(feature_lens[b])
            flags = self._boundary_flags(logits[b], vlen)
            preds_dict[str(b)] = boundaries_to_units(flags, vlen, pbf, sr)
        gt_dict = phone_starts_to_gt_units(
            batch["phone_start_idx"], target_length, feature_lens, pbf, sr,
        )
        metrics.update(evaluate_boundaries(self.evaluator, preds_dict, gt_dict))
        return metrics

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> List[Dict[str, Any]]:
        """Decode argmax boundaries into per-utterance segmentation dicts."""
        if self.boundary_class_id is None:
            raise NotImplementedError(
                "CountCTCHead.decode requires a boundary_token."
            )
        logits = self.proj(features)
        pbf, sr = self.effective_pbf, self.audio_sr
        out: List[Dict[str, Any]] = []
        for b in range(features.shape[0]):
            vlen = int(feature_lens[b])
            flags = self._boundary_flags(logits[b], vlen)
            out.append({"boundaries": boundaries_to_units(flags, vlen, pbf, sr)})
        return out
