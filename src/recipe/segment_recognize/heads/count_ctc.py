"""CountCTC boundary detection head.

Vocab: [<blank>, <boundary>]
All non-blank ids in target are collapsed to the boundary class.
Eg. [2, 5, 1 ] -> [1, 1, 1]
In this setup the boundary is supposed to mark onset of phones.
For offset, two cases:
1. Last 1 in a contiguous run of 1s marks the offset
2. First 1 in the next contiguous run marks the offset (this assumes no silence)
Current code and evals are setup with case (1)

TODO(shikhar): The key in results dicts will collide among batches. Fix this across all heads.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.metrics.segmentation_evaluator import SegmentationEvaluator, SegmentationUnit
from src.recipe.common.greedy_ctc_strategy import (
    ctc_collapse_vectorized,
)
from src.recipe.common.boundary_utils import (
    target_boundaries_to_gt_units,
    evaluate_boundaries,
)
from src.recipe.segment_recognize.heads.base import TaskHead
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_countctc_target(
    target_length: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    """Build CountCTC target sequences from target lengths.

    Args:
        target_length: (B,) tensor of counts of phones per utterance.
        blank_id: ID to use for blank symbol (default: 0).
    Returns:
        targets: (B, max_target_length) tensor of target IDs. 
            For element i the number of 1 is equal to the target_length[i].
    """
    B = target_length.size(0)
    max_target_length = target_length.max().item()
    targets = torch.arange(max_target_length, device=target_length.device).expand(B, max_target_length)
    targets = (targets < target_length.unsqueeze(1)) # (B, max_target_length) with 1s where index < target_length
    targets = torch.where(targets, 1, blank_id)
    return targets
    

class CountCTCHead(TaskHead):
    """CountCTC head: CTC loss for unaligned boundary supervision.

    Attributes:
        proj: ``Linear(encoder_dim, 2)`` projection.
        ctc_loss: ``nn.CTCLoss`` instance with ``blank=0``.
        evaluator: Shared ``SegmentationEvaluator`` for boundary metrics.
    
    NOTE(shikhar): Possible architectures:
    A) Full IPA vocab for initial CTC --> projection into 2 classes for countctc
    B) Separate head for countctc with vocabsize 2
    C) Common shared head for countctc and boundary bce
    This implementation is for B
    """

    log_name = "count_ctc"
    prog_bar_keys = frozenset({"rval"})

    def __init__(
        self,
        encoder_dim: int,
        evaluator: SegmentationEvaluator,
        effective_pbf: float = 640.0,
        audio_sr: int = 16000,
        weight: float = 1.0,
        zero_infinity: bool = True,
        blank_id: int=0,
        projection_module: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight)
        self.nclasses = 2
        self.proj = nn.Linear(encoder_dim, self.nclasses) if projection_module is None else projection_module
        self.blank_id=blank_id
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
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
            Dict with ``loss`` and ``logits``.
        """
        target_length = batch["target_length"] # count of phones = count of onsets for countctc
        logits = self.proj(features)  # (B, T, 2)
        target_ = build_countctc_target(target_length, self.blank_id)
        # CTCLoss expects (T, B, C) log-probs.
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss_per_utt = self.ctc_loss(
            log_probs,
            target_,
            feature_lens,
            target_length,
        )
        loss = loss_per_utt.sum() / logits.size(0)
        return {"loss": loss, "logits": logits.detach()}

    @torch.no_grad()
    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Compute metrics.
        Boundary metrics: via the shared evaluator
        CountCTC specific metric: mean absolute error
        """
        # count ctc metric
        logits = output["logits"]
        target_length = batch["target_length"]
        B = logits.shape[0]
        preds = logits.argmax(dim=-1) # (B, T)
        preds = ctc_collapse_vectorized(preds, feature_lens, self.blank_id) # (B, max_target_length)
        preds = (preds == 1).sum(dim=1) # count of predicted boundaries
        count_abs_err_sum = (preds - target_length).abs().sum().item()
        metrics: dict[str, float] = {
            "mae": count_abs_err_sum / B,
        }
        
        # boundary metrics
        if "target_start_idx" not in batch or "target_end_idx" not in batch:
            return metrics
        
        preds_dict = self._process_predictions(logits, feature_lens, batch['utt_id'])
        gt_dict = target_boundaries_to_gt_units(
            batch["target_start_idx"], 
            batch['target_end_idx'],
            target_length,
            feature_lens,
            self.effective_pbf, 
            self.audio_sr
        )
        metrics.update(evaluate_boundaries(self.evaluator, preds_dict, gt_dict))
        return metrics
    
    def _process_predictions(
        self,
        logits: torch.Tensor,
        feature_lens: torch.Tensor,
        utt_id: List[str],
    ) -> dict[str, List[SegmentationUnit]]:
        """Convert argmax logits to predicted segmentation based on 
        approach (1), first and last spikes in a contiguous run are the 
        boundaries.
        """
        predid = logits.argmax(dim=-1)
        predid_leftshift = torch.cat([predid[:, 1:], torch.tensor([[self.blank_id]], device=predid.device)], dim=1)
        predid_rightshift = torch.cat([torch.tensor([[self.blank_id]], device=predid.device), predid[:, :-1]], dim=1)
        onsets = (predid != predid_rightshift)
        offsets = (predid != predid_leftshift) 
        out: dict[str, List[SegmentationUnit]] = {}
        for b in range(logits.shape[0]):
            vlen = int(feature_lens[b])
            segmentation_units = []
            start, end = None, None
            for i in range(vlen):
                if onsets[b, i]:
                    start=i
                if offsets[b, i]:
                    end=i
                    segmentation_units.append(SegmentationUnit(
                        start=start * self.effective_pbf / self.audio_sr,
                        end=end * self.effective_pbf / self.audio_sr,
                    ))
            out[utt_id[b]] = segmentation_units
        return out

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> List[Dict[str, Any]]:
        logits = self.proj(features).detach()
        results = self._process_predictions(logits, feature_lens, batch['utt_id'])
        return results
