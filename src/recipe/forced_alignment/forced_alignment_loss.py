"""Forced alignment loss.
Simple cross-entropy loss repurposed for forced alignment based on time-aligned labels.
"""

import torch
from torch import nn


class ForcedAlignmentLoss(nn.Module):
    """Forced alignment loss module.

    Args:
        ignore_index: index to ignore
    """

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        # Initialize the criterion with reduction="none" to get per-frame loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_start: torch.Tensor,
        target_end: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forced alignment loss.

        Args:
            logits: (Batch, Length, Class)
            targets: (Batch, TargetLength)
            target_start: (Batch, TargetLength): target_start[b,i] points to the start index of targets[b,i] in logits
            target_end: (Batch, TargetLength): target_end[b,i] points to the end index of targets[b,i] in logits.
                                              This implementation assumes the end index is *inclusive*.
            input_lengths: (Batch,): each target_start[b,i] and target_end[b,i] <= input_lenghths[b]

        Returns:
            torch.Tensor: Loss value (scalar).
        """
        bs, ilen, vocab_sz = logits.size()

        # 1. Create frame-level alignment mask (B, I, T)
        # (1, I, 1)
        frame_indices = (
            torch.arange(ilen, device=logits.device, dtype=target_start.dtype)
            .unsqueeze(0)
            .unsqueeze(2)
        )

        # (B, 1, T)
        target_start_exp = target_start.unsqueeze(1)
        target_end_exp = target_end.unsqueeze(1)
        target_pad_mask = (targets != self.ignore_index).unsqueeze(1)

        # (B, I, T) - True if frame `i` belongs to target `t`
        alignment_mask = (
            (frame_indices >= target_start_exp)
            & (frame_indices <= target_end_exp)
            & target_pad_mask
        )

        # 2. Create frame-level target tensor (B, I)
        # Find which target index `t` each frame `i` belongs to
        # (B, I)
        target_t_indices = torch.argmax(alignment_mask.int(), dim=2)

        # (B, I)
        expanded_targets = torch.gather(targets, 1, target_t_indices)

        # Mask frames that don't belong to any target
        has_alignment = alignment_mask.any(dim=2)
        expanded_targets.masked_fill_(~has_alignment, self.ignore_index)

        # 3. Compute loss
        # (B*I, V)
        logits_flat = logits.reshape(-1, vocab_sz)
        # (B*I,)
        targets_flat = expanded_targets.reshape(-1)

        # (B*I,) - Loss is 0 for frames where targets_flat == ignore_index
        loss_per_frame = self.criterion(logits_flat, targets_flat)

        # 4. Create input length mask (B, I) -> (B*I,)
        input_len_mask_flat = (
            torch.arange(ilen, device=logits.device).unsqueeze(0)
            < input_lengths.unsqueeze(1)
        ).reshape(-1)

        # 5. Mask loss for padding frames and sum
        loss_per_frame.masked_fill_(~input_len_mask_flat, 0.0)
        total_loss = loss_per_frame.sum()

        # 6. Average over valid frames
        # Valid frames are those *not* ignored by criterion AND *within* input_lengths
        valid_frame_mask = (targets_flat != self.ignore_index) & input_len_mask_flat
        num_valid_frames = valid_frame_mask.sum().float()

        return total_loss / (num_valid_frames + 1e-8)
