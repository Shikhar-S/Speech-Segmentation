"""Simple representation probing head that combines attention based pooling
with an MLP for classification/regression tasks.

"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.heads.base_head import BaseHead, TaskType
from src.model.common.utils import get_kv_pooling_mask


class AttentionMLPHead(BaseHead):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        task_type: str = "classification",
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension (from pooled encoder output).
            output_dim: Output dimension (number of classes for classification).
            hidden_dim: Hidden dimension for MLP. If None, uses input_dim // 2.
            dropout: Dropout rate.
            task_type: Task type ("classification" or "regression").
        """
        task_type_enum = TaskType(task_type)
        super().__init__(task_type=task_type_enum, output_dim=output_dim)
        self.input_dim = input_dim
        self.query_vector = nn.Parameter(torch.randn(1, 1, self.input_dim))
        self.attentive_pooling = nn.MultiheadAttention(
            embed_dim=self.input_dim, num_heads=1
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or (input_dim // 2)

        self.f = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: Encoder output tensor of shape (batch, seq_len, input_dim).
            encoder_out_lengths: Lengths of encoder outputs (batch,).

        Returns:
            Logits tensor of shape (batch, output_dim).
        """
        b, t, d = encoder_out.size()
        h = F.normalize(encoder_out, dim=-1, eps=1e-8)
        key_mask = get_kv_pooling_mask(encoder_out_lengths)
        q = F.normalize(self.query_vector, dim=-1, eps=1e-8).expand(
            1, b, -1
        )  # (1, B, D)
        h = self.attentive_pooling(
            query=q,  # (1, B, D)
            key=h.transpose(0, 1),
            value=h.transpose(0, 1),
            key_padding_mask=key_mask,
        )[0].squeeze(0)
        h = F.normalize(h, dim=-1, eps=1e-8)
        return self.f(h)  # (batch, output_dim)
