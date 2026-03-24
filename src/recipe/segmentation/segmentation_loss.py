import torch
import torch.nn as nn
from typing import Dict, Optional


class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        logit_lens: torch.Tensor,
        targets: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        t_lens: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Compute the forced alignment loss.
        Args:
            logits: (B, S, C) - Logits from the model
            logit_lens: (B,) - Valid lengths of logits
            targets: (B, T) - Target class indices
            t_start: (B, T) - Start frame indices for each target
            t_end: (B, T) - End frame indices for each target
            t_lens: (B,) - Valid lengths of targets
        Returns:
            Dict with keys:
                "loss": Scalar loss tensor
                "accuracy": Scalar accuracy value
        """
        B, S, C = logits.shape
        T = targets.shape[1]
        # Frame indices (S) and Target indices (T)
        idx = torch.arange(S, device=logits.device).view(1, S, 1)
        t_idx = torch.arange(T, device=logits.device).view(1, 1, T)
        # mask[b, s, t] = True if target[t] should be predicted for frame s
        mask = (idx >= t_start.unsqueeze(1)) & (idx <= t_end.unsqueeze(1))
        mask &= (targets != self.ignore_index).unsqueeze(1)

        mask &= idx < logit_lens.view(B, 1, 1)
        if t_lens is not None:
            mask &= t_idx < t_lens.view(B, 1, 1)

        # valid_frames[b, s] = True if frame s is covered by at least 1 target
        valid_frames = mask.any(dim=2)
        norm_mask = mask.float() / mask.sum(dim=2, keepdim=True).clamp_min(1.0)

        # Gather specific target log-probs for every frame
        safe_tgt = (
            targets.masked_fill(targets == self.ignore_index, 0)
            # .clamp_min(0)
            .unsqueeze(1).expand(-1, S, -1)
        )  # [B,S,T]
        log_p = logits.log_softmax(dim=-1)
        log_p = log_p.gather(2, index=safe_tgt)  # retain only relevant classes
        # [B, S, C] -> [B, S, T]
        loss = -(log_p * norm_mask).sum() / valid_frames.sum().clamp_min(1)

        # Correct if prediction matches ANY valid target at that frame
        preds = logits.argmax(dim=-1, keepdim=True)  # (B, S, 1)
        is_correct = ((preds == safe_tgt) & mask).any(dim=2).float()
        acc = is_correct.sum() / valid_frames.sum().clamp_min(1)

        return {"loss": loss, "accuracy": acc.item()}


class BoundaryLoss(nn.Module):
    """Masked BCE loss that predicts phone boundary frames.

    A boundary label of 1 is placed at the start frame of every phone
    (equivalent to all phone-to-phone transitions plus the utterance start).
    All padding frames are excluded from the loss.

    Args:
        pos_weight: Weight for the positive (boundary) class. Increase to
            penalise missed boundaries more heavily (default: 1.0).
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([pos_weight]),
        )

    def forward(
        self,
        boundary_logits: torch.Tensor,
        logit_lens: torch.Tensor,
        target_start_idx: torch.Tensor,
        t_lens: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Compute masked BCE boundary loss.

        Args:
            boundary_logits: (B, S) – raw per-frame boundary scores.
            logit_lens: (B,) – valid frame counts (padding is ignored).
            target_start_idx: (B, T) – start frame index of each phone.
            t_lens: (B,) – number of valid phones per utterance.

        Returns:
            Dict with key "loss": scalar loss tensor.
        """
        B, S = boundary_logits.shape
        T = target_start_idx.shape[1]

        t_idx = torch.arange(T, device=boundary_logits.device).unsqueeze(0)
        valid_t = t_idx < (t_lens.unsqueeze(1) if t_lens is not None else T)
        in_range = (target_start_idx >= 0) & (target_start_idx < S)
        scatter_mask = (valid_t & in_range).float()

        labels = torch.zeros(B, S, device=boundary_logits.device)
        labels.scatter_add_(1, target_start_idx.clamp(0, S - 1), scatter_mask)
        labels.clamp_(max=1.0)

        s_idx = torch.arange(S, device=boundary_logits.device).unsqueeze(0)
        frame_mask = s_idx < logit_lens.unsqueeze(1)

        loss = (self.bce(boundary_logits, labels) * frame_mask).sum() / frame_mask.sum().clamp_min(1)
        return {"loss": loss}
