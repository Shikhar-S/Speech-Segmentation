"""Transformer-based head for downstream tasks.

This module provides a Transformer-based classification/regression head
with positional encoding and various pooling strategies.

References:
- PyTorch TransformerEncoder: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- ESPnet positional encoding: https://github.com/espnet/espnet
  (espnet/nets/pytorch_backend/transformer/embedding.py - Apache-2.0 License)
- PyTorch Transformer Classifier example:
  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.heads.base_head import BaseHead, TaskType
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PositionalEncodingType(str, Enum):
    """Type of positional encoding."""

    SINUSOIDAL = "sinusoidal"
    LEARNABLE = "learnable"
    NONE = "none"


class TransformerPoolingType(str, Enum):
    """Pooling strategy for Transformer output."""

    CLS = "cls"  # Use [CLS] token (prepended)
    MEAN = "mean"  # Mean pooling over valid timesteps
    ATTENTION = "attention"  # Attention-weighted pooling


@dataclass
class TransformerHeadConfig:
    """Configuration for TransformerHead.

    Attributes:
        input_dim: Input feature dimension from encoder.
        d_model: Model dimension (will project input if different).
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Dimension of feedforward network.
        output_dim: Output dimension (num_classes for classification).
        dropout: Dropout rate.
        positional_encoding: Type of positional encoding.
        pooling: Pooling strategy for sequence aggregation.
        max_seq_len: Maximum sequence length for positional encoding.
        task_type: Task type (classification or regression).
    """

    input_dim: int
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 1024
    output_dim: int = 2
    dropout: float = 0.1
    positional_encoding: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL
    pooling: TransformerPoolingType = TransformerPoolingType.MEAN
    max_seq_len: int = 5000
    task_type: TaskType = TaskType.CLASSIFICATION


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Reference:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - ESPnet implementation: espnet/nets/pytorch_backend/transformer/embedding.py
      (Apache-2.0 License)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        """Initialize sinusoidal positional encoding.

        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        # Reference: ESPnet PositionalEncoding implementation
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved/loaded)
        # Shape: (1, max_len, d_model) for batch broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding.

    Uses a learnable embedding table for positions.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        """Initialize learnable positional encoding.

        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

        # Initialize with small values
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pe(positions).unsqueeze(0)
        return self.dropout(x)


class TransformerAttentionPooling(nn.Module):
    """Attention-based pooling for Transformer outputs.

    Uses a learnable query to attend over the sequence.

    Reference:
    - SpeechBrain attention pooling pattern
      (speechbrain/nnet/pooling.py - Apache-2.0 License)
    """

    def __init__(self, d_model: int) -> None:
        """Initialize attention pooling.

        Args:
            d_model: Model dimension.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention-weighted pooling.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Boolean mask of shape (batch, seq_len) where True indicates
                valid positions.

        Returns:
            Pooled tensor of shape (batch, d_model).
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            # Mask invalid positions
            scores = scores.masked_fill(~mask, float("-inf"))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Handle edge case where all positions are masked
        weights = weights.masked_fill(torch.isnan(weights), 0.0)

        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, d_model)

        return pooled


class TransformerHead(BaseHead):
    """Transformer-based head for classification/regression tasks.

    This head processes encoder outputs through Transformer encoder layers
    with positional encoding and applies a pooling strategy to obtain a
    fixed-size representation for classification or regression.

    Reference:
    - PyTorch TransformerEncoder with src_key_padding_mask
      (https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
    - PyTorch Transformer Classifier example from Context7
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
        pooling: str = "mean",
        max_seq_len: int = 5000,
        task_type: str = "classification",
    ) -> None:
        """Initialize the TransformerHead.

        Args:
            input_dim: Input feature dimension from encoder.
            output_dim: Output dimension (number of classes for classification).
            d_model: Model dimension for Transformer.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout rate.
            positional_encoding: Type of positional encoding
                ("sinusoidal", "learnable", or "none").
            pooling: Pooling strategy ("cls", "mean", or "attention").
            max_seq_len: Maximum sequence length for positional encoding.
            task_type: Task type ("classification" or "regression").
        """
        # Convert string task_type to enum
        task_type_enum = TaskType(task_type)
        if task_type_enum == TaskType.ORDINAL_REGRESSION:
            log.info(
                f"Using ORDINAL_REGRESSION task type: adjusting output_dim to {output_dim - 1}"
            )
            output_dim = output_dim - 1  # Adjust output dim for ordinal regression
        super().__init__(task_type=task_type_enum, output_dim=output_dim)

        self.input_dim = input_dim
        self.d_model = d_model
        self.pooling_type = TransformerPoolingType(pooling.lower())
        self.pos_enc_type = PositionalEncodingType(positional_encoding.lower())

        # Input projection if dimensions don't match
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # CLS token for cls pooling
        if self.pooling_type == TransformerPoolingType.CLS:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0, std=0.02)

        # Positional encoding
        # Reference: ESPnet positional encoding patterns
        if self.pos_enc_type == PositionalEncodingType.SINUSOIDAL:
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=d_model,
                dropout=dropout,
                max_len=max_seq_len,
            )
        elif self.pos_enc_type == PositionalEncodingType.LEARNABLE:
            self.pos_encoder = LearnablePositionalEncoding(
                d_model=d_model,
                dropout=dropout,
                max_len=max_seq_len,
            )
        else:
            self.pos_encoder = nn.Dropout(dropout)

        # Transformer encoder
        # Reference: PyTorch TransformerEncoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Pooling layer
        if self.pooling_type == TransformerPoolingType.ATTENTION:
            self.pooling = TransformerAttentionPooling(d_model)
        else:
            self.pooling = None

        # Classifier/regressor
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, output_dim)

    def _create_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Create padding mask for Transformer.

        Args:
            lengths: Length tensor of shape (batch,).
            max_len: Maximum sequence length.

        Returns:
            Boolean mask of shape (batch, max_len) where True indicates
            padding positions (to be masked out in attention).
        """
        # Reference: PyTorch Transformer src_key_padding_mask convention
        # True = padding position (will be masked)
        batch_size = lengths.size(0)
        mask = (
            torch.arange(max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = mask >= lengths.unsqueeze(1)
        return mask

    def _pool_mean(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling over valid timesteps.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Padding mask of shape (batch, seq_len) where True indicates
                padding positions.

        Returns:
            Pooled tensor of shape (batch, d_model).
        """
        # Invert mask: True = valid positions
        valid_mask = ~mask

        # Expand mask for broadcasting
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()

        # Compute mean over valid positions
        sum_out = (x * valid_mask_expanded).sum(dim=1)
        lengths = valid_mask.sum(dim=1, keepdim=True).float()

        # Avoid division by zero
        lengths = lengths.clamp(min=1.0)
        pooled = sum_out / lengths

        return pooled

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Transform encoder outputs into logits.

        Args:
            encoder_out: Encoder output tensor of shape (batch, time, hidden_dim).
            encoder_out_lengths: Length tensor of shape (batch,) indicating the
                valid length for each sample in the batch.

        Returns:
            Logits tensor of shape (batch, output_dim).
        """
        batch_size = encoder_out.size(0)
        max_len = encoder_out.size(1)

        # Project input to d_model dimension
        x = self.input_proj(encoder_out)  # (batch, time, d_model)

        # Handle CLS token pooling
        if self.pooling_type == TransformerPoolingType.CLS:
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + time, d_model)
            max_len = max_len + 1
            # Adjust lengths for CLS token
            adjusted_lengths = encoder_out_lengths + 1
        else:
            adjusted_lengths = encoder_out_lengths

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask for Transformer
        # Reference: PyTorch TransformerEncoder src_key_padding_mask
        # True = position to be masked (padding)
        padding_mask = self._create_padding_mask(adjusted_lengths, max_len)

        # Pass through Transformer encoder
        # Note: src_key_padding_mask expects True for positions to mask
        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Apply pooling
        if self.pooling_type == TransformerPoolingType.CLS:
            # Use the CLS token output (first position)
            pooled = encoded[:, 0, :]
        elif self.pooling_type == TransformerPoolingType.MEAN:
            pooled = self._pool_mean(encoded, padding_mask)
        elif self.pooling_type == TransformerPoolingType.ATTENTION:
            # Invert mask for attention pooling (True = valid)
            valid_mask = ~padding_mask
            pooled = self.pooling(encoded, valid_mask)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Apply layer norm and dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        # Classify
        logits = self.classifier(pooled)

        return logits
