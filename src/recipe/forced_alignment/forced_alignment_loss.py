import torch
import torch.nn as nn
from typing import Dict, Optional


class ForcedAlignmentLoss(nn.Module):
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
