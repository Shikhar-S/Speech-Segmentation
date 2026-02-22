"""IPA Embedding module for text-based L1 classification.

This module provides a simple embedding layer that converts IPA token IDs
to dense vectors. It follows the same interface as audio encoders (e.g., PowsmModel)
by providing an `encode` method that returns (encoder_out, encoder_out_lengths).

Usage:
    embedding = IPAEmbedding(vocab_size=100, embedding_dim=128)

    # Input: token IDs (B, T) and lengths (B,)
    encoder_out, encoder_out_lengths = embedding.encode(ipa_ids, lengths)
    # encoder_out: (B, T, embedding_dim)
    # encoder_out_lengths: (B,)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class IPAEmbedding(nn.Module):
    """IPA token embedding layer with encoder-compatible interface.

    This module embeds discrete IPA token IDs into continuous vectors.
    It provides an `encode` method compatible with audio encoders like PowsmModel,
    allowing seamless integration with downstream classification heads.

    Attributes:
        embedding: The embedding layer.
        embedding_dim: Output embedding dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the IPA Embedding module.

        Args:
            vocab_size: Size of the vocabulary (number of unique tokens).
            embedding_dim: Dimension of the embedding vectors.
            padding_idx: Index of the padding token (default: 0).
            dropout: Dropout probability applied after embedding.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, ipa_ids: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (alias for encode).

        Args:
            ipa_ids: Input token IDs of shape (batch, seq_len).
            lengths: Actual lengths of each sequence of shape (batch,).

        Returns:
            Tuple of:
                - encoder_out: Embedded tokens of shape (batch, seq_len, embedding_dim).
                - encoder_out_lengths: Same as input lengths.
        """
        return self.encode(ipa_ids, lengths)

    def encode(
        self, ipa_ids: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode IPA token IDs to embeddings.

        This method follows the same interface as audio encoders (e.g., PowsmModel.encode),
        returning (encoder_out, encoder_out_lengths).

        Args:
            ipa_ids: Input token IDs of shape (batch, seq_len).
            lengths: Actual lengths of each sequence of shape (batch,).

        Returns:
            Tuple of:
                - encoder_out: Embedded tokens of shape (batch, seq_len, embedding_dim).
                - encoder_out_lengths: Same as input lengths.
        """
        # Embed tokens: (B, T) -> (B, T, D)
        embedded = self.embedding(ipa_ids)
        embedded = self.dropout(embedded)

        return embedded, lengths

    def encoder_output_size(self) -> int:
        """Return the output dimension of the encoder.

        This method is provided for compatibility with other encoders.
        """
        return self.embedding_dim


def build_ipa_embedding(
    vocab_size: int,
    embedding_dim: int = 128,
    padding_idx: int = 0,
    dropout: float = 0.1,
) -> IPAEmbedding:
    """Factory function to build IPAEmbedding.

    This function is intended to be used as a Hydra target.

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of embeddings.
        padding_idx: Index of padding token.
        dropout: Dropout probability.

    Returns:
        Initialized IPAEmbedding module.
    """
    return IPAEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick sanity check
    print("Testing IPAEmbedding...")

    vocab_size = 100
    embedding_dim = 128
    batch_size = 4
    seq_len = 20

    model = IPAEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
    print(f"Model: {model}")
    print(f"Output size: {model.encoder_output_size()}")

    # Create dummy input
    ipa_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([20, 15, 10, 5])

    # Forward pass
    encoder_out, encoder_out_lengths = model.encode(ipa_ids, lengths)

    print(f"\nInput shape: {ipa_ids.shape}")
    print(f"Input lengths: {lengths}")
    print(f"Output shape: {encoder_out.shape}")
    print(f"Output lengths: {encoder_out_lengths}")

    assert encoder_out.shape == (batch_size, seq_len, embedding_dim)
    assert torch.equal(encoder_out_lengths, lengths)

    print("\nSanity check passed!")
