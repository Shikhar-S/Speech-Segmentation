"""Base class for composable task heads."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric


class TaskHead(nn.Module):
    """Self-contained task head with parameters, criterion, and metrics.

    Subclasses implement ``forward()`` and optionally
    ``eval_metrics()``.  The hosting model composes heads via
    ``nn.ModuleDict``, iterating and summing
    ``weight * output["loss"]``.

    Attributes:
        weight: Scalar multiplier applied by the model when summing.
        log_name: Short identifier for metric names
            (e.g. ``"bce"`` -> ``"train/bce_loss"``).
        prog_bar_keys: Eval-metric keys shown in the progress bar.
        train_loss: Running mean of training loss values.
        val_loss: Running mean of validation loss values.
    """

    log_name: str = "loss"
    prog_bar_keys: frozenset[str] = frozenset()

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
        **ctx: Any,
    ) -> dict[str, Any]:
        """Compute loss.

        Args:
            features: Encoder output ``(B, T, D)``.
            feature_lens: Valid frame counts ``(B,)``.
            batch: Collated batch dict.
            **ctx: Extra context (e.g. ``net=...``).

        Returns:
            Dict with at least ``"loss"`` key.  May include
            auxiliary outputs consumed by ``eval_metrics()``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def eval_metrics(
        self,
        output: dict[str, Any],
        feature_lens: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """Expensive evaluation metrics from ``forward()`` output.

        Override in subclasses that produce metrics like P/R/F1.
        Default returns an empty dict.
        """
        return {}

    def log_output(
        self,
        pl_module: LightningModule,
        prefix: str,
        output: dict[str, Any],
        eval_metrics: dict[str, float] | None = None,
        on_step: bool = True,
        on_epoch: bool = True,
    ) -> None:
        """Update tracker and log loss + eval metrics.

        Args:
            pl_module: The Lightning module (for ``self.log``).
            prefix: Phase prefix (e.g. ``"train"``, ``"val_seg"``).
            output: Dict returned by ``forward()``.
            eval_metrics: Dict returned by ``eval_metrics()``.
            on_step: Log after each batch.
            on_epoch: Log epoch aggregate.
        """
        tracker = (
            self.train_loss if "train" in prefix else self.val_loss
        )
        tracker(output["loss"].detach())
        pl_module.log(
            f"{prefix}/{self.log_name}_loss",
            tracker,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
        )
        for k, v in (eval_metrics or {}).items():
            pl_module.log(
                f"{prefix}/{k}",
                v,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=(k in self.prog_bar_keys),
            )
