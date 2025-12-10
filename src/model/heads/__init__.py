"""Head modules for downstream tasks."""

from src.model.heads.base_head import BaseHead, TaskType
from src.model.heads.rnn_head import RNNHead
from src.model.heads.transformer_head import TransformerHead

__all__ = ["BaseHead", "TaskType", "TransformerHead", "RNNHead"]

