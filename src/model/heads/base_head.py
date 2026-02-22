"""Base head interface for downstream tasks.

This module defines the common interface that all downstream task heads must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn

# Type alias for input modes
InputType = Literal["audio", "ipa"]


class TaskType(Enum):
    """Enum for downstream task types.

    - CLASSIFICATION: output_dim is the number of classes.
    - REGRESSION: output_dim is the dimension of continuous target (e.g., 1 for scalar).
    - ORDINAL_REGRESSION: output_dim is the number of classes-1 (Coral style). Reference:https://arxiv.org/pdf/1901.07884
    - GEOLOCATION: output_dim is 3. head predicts 3D coordinates.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    GEOLOCATION = "geolocation"


class BaseHead(nn.Module, ABC):
    """Abstract base class for downstream task heads.

    Heads are responsible for transforming encoder outputs into task-specific logits.
    Loss computation, label mapping, and post-processing (argmax, etc.) are handled
    by the recipe (LightningModule), not the head.

    Attributes:
        task_type: The type of downstream task (classification or regression or ordinal regression).
        output_dim: The output dimension. For classification, this is the number of
            classes. For regression, this is the dimension of the target.
    """

    def __init__(self, task_type: TaskType, output_dim: int) -> None:
        """Initialize the base head.

        Args:
            task_type: The type of downstream task.
            output_dim: The output dimension of the head.
        """
        super().__init__()
        self._task_type = task_type
        self._output_dim = output_dim

    @property
    def task_type(self) -> TaskType:
        """Return the task type."""
        return self._task_type

    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self._output_dim

    @abstractmethod
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
        raise NotImplementedError
