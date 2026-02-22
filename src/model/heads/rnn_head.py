"""RNN-based head for downstream tasks.

This module provides an RNN-based (GRU/LSTM) classification/regression head
with various pooling strategies.

References:
- SpeechBrain RNN modules: https://github.com/speechbrain/speechbrain
  (speechbrain/nnet/RNN.py - Apache-2.0 License)
- PyTorch pack_padded_sequence usage:
  https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.model.heads.base_head import BaseHead, TaskType
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PoolingType(str, Enum):
    """Pooling strategy for sequence-to-vector transformation."""

    LAST = "last"  # Use last valid hidden state
    MEAN = "mean"  # Mean pooling over valid timesteps
    ATTENTION = "attention"  # Attention-weighted pooling


@dataclass
class RNNHeadConfig:
    """Configuration for RNNHead.

    Attributes:
        input_dim: Input feature dimension from encoder.
        hidden_size: Hidden dimension of RNN layers.
        num_layers: Number of stacked RNN layers.
        output_dim: Output dimension (num_classes for classification).
        rnn_type: Type of RNN cell ("gru" or "lstm").
        bidirectional: Whether to use bidirectional RNN.
        dropout: Dropout rate between RNN layers (applied if num_layers > 1).
        pooling: Pooling strategy for sequence aggregation.
        task_type: Task type (classification or regression).
    """

    input_dim: int
    hidden_size: int = 256
    num_layers: int = 2
    output_dim: int = 2
    rnn_type: Literal["gru", "lstm"] = "gru"
    bidirectional: bool = True
    dropout: float = 0.1
    pooling: PoolingType = PoolingType.MEAN
    task_type: TaskType = TaskType.CLASSIFICATION


class AttentionPooling(nn.Module):
    """Attention-based pooling layer.

    Computes attention weights over the sequence and returns a weighted sum.

    Reference:
    - SpeechBrain attention pooling pattern
      (speechbrain/nnet/pooling.py - Apache-2.0 License)
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize attention pooling.

        Args:
            input_dim: Input feature dimension.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-weighted pooling.

        Args:
            x: Input tensor of shape (batch, time, features).
            lengths: Length tensor of shape (batch,).

        Returns:
            Pooled tensor of shape (batch, features).
        """
        batch_size, max_len, _ = x.shape

        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, time)

        # Create mask for valid positions
        # Reference: PyTorch sequence masking pattern
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(
            1
        )

        # Apply mask (set invalid positions to -inf before softmax)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # (batch, time)

        # Handle edge case where all positions are masked (shouldn't happen normally)
        weights = weights.masked_fill(torch.isnan(weights), 0.0)

        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, features)

        return pooled


class RNNHead(BaseHead):
    """RNN-based head for classification/regression tasks.

    This head processes encoder outputs through stacked RNN layers (GRU or LSTM)
    and applies a pooling strategy to obtain a fixed-size representation for
    classification or regression.

    Reference:
    - SpeechBrain RNN implementation for pack/pad handling
      (speechbrain/nnet/RNN.py - Apache-2.0 License)
    - PyTorch DataParallel FAQ for total_length in pad_packed_sequence
      (https://pytorch.org/docs/stable/notes/faq.html)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        dropout: float = 0.1,
        pooling: str = "mean",
        task_type: str = "classification",
    ) -> None:
        """Initialize the RNNHead.

        Args:
            input_dim: Input feature dimension from encoder.
            output_dim: Output dimension (number of classes for classification).
            hidden_size: Hidden dimension of RNN layers.
            num_layers: Number of stacked RNN layers.
            rnn_type: Type of RNN cell ("gru" or "lstm").
            bidirectional: Whether to use bidirectional RNN.
            dropout: Dropout rate between RNN layers.
            pooling: Pooling strategy ("last", "mean", or "attention").
            task_type: Task type ("classification" or "regression").
        """
        # Convert string task_type to enum
        task_type_enum = TaskType(task_type)
        if task_type_enum == TaskType.ORDINAL_REGRESSION:
            log.info(
                f"Using ORDINAL_REGRESSION task type: adjusting output_dim to {output_dim - 1}"
            )
            output_dim = output_dim - 1  # Adjust output dim for ordinal regression
        elif task_type_enum == TaskType.REGRESSION:
            log.info(f"Using REGRESSION task type: setting output_dim to 1")
            output_dim = 1  # For standard regression, output dim is 1
        elif task_type_enum == TaskType.GEOLOCATION:
            log.info(f"Using GEOLOCATION task type: setting output_dim to 3")
            output_dim = 3  # For geolocation, output dim is 3 (3D coordinates)

        super().__init__(task_type=task_type_enum, output_dim=output_dim)

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.pooling_type = PoolingType(pooling.lower())

        # Validate rnn_type
        if self.rnn_type not in ("gru", "lstm"):
            raise ValueError(f"rnn_type must be 'gru' or 'lstm', got {rnn_type}")

        # Build RNN
        # Reference: SpeechBrain RNN module initialization pattern
        rnn_class = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Calculate output dimension of RNN
        # Note: bidirectional doubles the hidden size
        rnn_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # Build pooling layer
        if self.pooling_type == PoolingType.ATTENTION:
            self.pooling = AttentionPooling(rnn_output_dim)
        else:
            self.pooling = None

        # Classifier/regressor
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(rnn_output_dim)
        self.classifier = nn.Linear(rnn_output_dim, output_dim)

    def _pool_last(
        self,
        rnn_out: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Pool using the last valid hidden state.

        For bidirectional RNNs, concatenates the last forward and first backward states.

        Args:
            rnn_out: RNN output of shape (batch, time, hidden * num_directions).
            lengths: Length tensor of shape (batch,).

        Returns:
            Pooled tensor of shape (batch, hidden * num_directions).
        """
        batch_size = rnn_out.size(0)

        if self.bidirectional:
            # Split forward and backward outputs
            # rnn_out: (batch, time, hidden * 2) -> forward: (batch, time, hidden), backward: (batch, time, hidden)
            forward_out, backward_out = rnn_out.chunk(2, dim=-1)

            # Get last valid timestep for forward direction
            # Reference: SpeechBrain approach for getting last timestep
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, forward_out.size(-1))
            last_forward = forward_out.gather(1, idx).squeeze(1)

            # First timestep for backward direction (contains info from last timestep)
            last_backward = backward_out[:, 0, :]

            pooled = torch.cat([last_forward, last_backward], dim=-1)
        else:
            # Get last valid timestep
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, rnn_out.size(-1))
            pooled = rnn_out.gather(1, idx).squeeze(1)

        return pooled

    def _pool_mean(
        self,
        rnn_out: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Pool using mean over valid timesteps.

        Args:
            rnn_out: RNN output of shape (batch, time, hidden * num_directions).
            lengths: Length tensor of shape (batch,).

        Returns:
            Pooled tensor of shape (batch, hidden * num_directions).
        """
        batch_size, max_len, hidden_dim = rnn_out.shape

        # Create mask for valid positions
        mask = torch.arange(max_len, device=rnn_out.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(rnn_out)

        # Zero out invalid positions and compute mean
        masked_out = rnn_out * mask.float()
        sum_out = masked_out.sum(dim=1)
        pooled = sum_out / lengths.unsqueeze(-1).float()

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

        # Ensure lengths are on CPU for pack_padded_sequence
        lengths_cpu = encoder_out_lengths.cpu()

        # Pack sequences for efficient RNN processing
        # Reference: PyTorch FAQ for DataParallel compatibility
        # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        packed_input = pack_padded_sequence(
            encoder_out,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,  # Allow unsorted sequences
        )

        # Pass through RNN
        packed_output, _ = self.rnn(packed_input)

        # Unpack sequences
        # Note: total_length ensures consistent output size across DataParallel devices
        rnn_out, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_len,
        )

        # Apply pooling
        if self.pooling_type == PoolingType.LAST:
            pooled = self._pool_last(rnn_out, encoder_out_lengths)
        elif self.pooling_type == PoolingType.MEAN:
            pooled = self._pool_mean(rnn_out, encoder_out_lengths)
        elif self.pooling_type == PoolingType.ATTENTION:
            pooled = self.pooling(rnn_out, encoder_out_lengths)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Apply layer norm and dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        # Classify
        logits = self.classifier(pooled)

        return logits
